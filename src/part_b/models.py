"""Machine learning pipeline for lapse prediction, model tuning, evaluation, and retention list."""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import OUTPUTS_DIR, CLEAN_DATA_DIR
from src.common.io import save_json
from src.part_b.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_lapse_dataset,
    split_lapse_dataset,
)

LOGISTIC_PARAM_GRID = {"model__C": [0.01, 0.1, 1, 10, 100]}
FOREST_PARAM_GRID = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20],
}


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessor with numeric median-imputation/standard-scaling and categorical one-hot encoding."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


def evaluate_classifier(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute standard classification metrics: accuracy, precision, recall, F1, and confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def oversample_minority_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class (lapsed=1) strictly on the training set to evaluate class imbalance impact."""
    counts = y_train.value_counts()
    n_majority = counts[0]
    n_minority = counts[1]
    diff = n_majority - n_minority

    minority_indices = y_train[y_train == 1].index
    sampled_indices = y_train[y_train == 1].sample(n=diff, replace=True, random_state=random_state).index

    X_balanced = pd.concat([X_train, X_train.loc[sampled_indices]])
    y_balanced = pd.concat([y_train, y_train.loc[sampled_indices]])

    return X_balanced, y_balanced


def build_retention_list(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    segments_path: Path = OUTPUTS_DIR / "customer_segments.csv",
    top_n: int = 10,
) -> pd.DataFrame:
    """Generate high-risk retention list for premium customers in test set."""
    segments = pd.read_csv(segments_path)
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])

    eligible = X_test[X_test.index.isin(premium_ids)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["customer_id", "lapse_probability", "actually_lapsed"])

    probs = model.predict_proba(eligible)[:, 1]
    retention_df = pd.DataFrame({
        "customer_id": eligible.index,
        "lapse_probability": probs,
        "actually_lapsed": y_test.reindex(eligible.index).to_numpy(),
    })

    return retention_df.sort_values(
        by=["lapse_probability", "customer_id"],
        ascending=[False, True],
    ).head(top_n).reset_index(drop=True)


def train_and_evaluate_lapse_models(
    clean_dir: Path = CLEAN_DATA_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Execute complete lapse prediction modeling, tuning, evaluation, persistence, and verification."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    X, y = build_lapse_dataset(clean_dir=clean_dir)
    X_train, X_test, y_train, y_test = split_lapse_dataset(X, y)

    # 1. Baseline Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    baseline_metrics = evaluate_classifier(y_test, dummy_pred)

    # 2. Pipeline definitions
    preprocessor = build_preprocessor()

    log_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    rf_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    # 3. Hyperparameter search with 5-fold cross-validation on training data only
    log_search = GridSearchCV(log_pipeline, param_grid=LOGISTIC_PARAM_GRID, cv=5, scoring="f1", n_jobs=1)
    log_search.fit(X_train, y_train)

    rf_search = GridSearchCV(rf_pipeline, param_grid=FOREST_PARAM_GRID, cv=5, scoring="f1", n_jobs=1)
    rf_search.fit(X_train, y_train)

    # 4. Model selection based on highest training CV F1 score
    candidates = {
        "logistic": log_search,
        "forest": rf_search,
    }
    selected_name = max(candidates, key=lambda k: candidates[k].best_score_)
    selected_search = candidates[selected_name]
    best_estimator = selected_search.best_estimator_

    # 5. Evaluate tuned models on holdout test set
    test_evaluations = {}
    for name, searcher in candidates.items():
        estimator = searcher.best_estimator_
        preds = estimator.predict(X_test)
        test_evaluations[name] = {
            "best_params": searcher.best_params_,
            "cv_f1": float(searcher.best_score_),
            "test_metrics": evaluate_classifier(y_test, preds),
        }

    # 6. Class imbalance experiment on training set only
    X_train_bal, y_train_bal = oversample_minority_training(X_train, y_train)
    balanced_model = clone(candidates["logistic"].best_estimator_)
    balanced_model.fit(X_train_bal, y_train_bal)
    bal_preds = balanced_model.predict(X_test)
    oversampling_metrics = evaluate_classifier(y_test, bal_preds)

    # 7. Feature importances and coefficients
    rf_best = candidates["forest"].best_estimator_
    feature_names = rf_best.named_steps["preprocess"].get_feature_names_out()
    rf_importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_best.named_steps["model"].feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    log_best = candidates["logistic"].best_estimator_
    log_coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": log_best.named_steps["model"].coef_[0],
    }).sort_values(by="coefficient", ascending=False).reset_index(drop=True)

    # 8. Model persistence and reload validation
    model_path = outputs_dir / "lapse_model.joblib"
    joblib.dump(best_estimator, model_path)

    reloaded_model = joblib.load(model_path)
    original_preds = best_estimator.predict(X_test)
    reloaded_preds = reloaded_model.predict(X_test)
    np.testing.assert_array_equal(original_preds, reloaded_preds)

    original_probs = best_estimator.predict_proba(X_test)
    reloaded_probs = reloaded_model.predict_proba(X_test)
    np.testing.assert_allclose(original_probs, reloaded_probs)

    # 9. Retention list for high-risk premium customers in test set
    retention_df = build_retention_list(
        model=reloaded_model,
        X_test=X_test,
        y_test=y_test,
        segments_path=outputs_dir / "customer_segments.csv",
    )
    retention_df.to_csv(outputs_dir / "retention_list.csv", index=False)

    report = {
        "baseline": baseline_metrics,
        "selected_model": selected_name,
        "selected_cv_f1": float(selected_search.best_score_),
        "models": test_evaluations,
        "oversampling_experiment": {
            "before_oversampling": test_evaluations["logistic"]["test_metrics"],
            "after_oversampling": oversampling_metrics,
        },
        "top_features_rf": rf_importances.head(5).to_dict(orient="records"),
        "top_coefficients_logistic": log_coefs.head(5).to_dict(orient="records"),
        "retention_count": len(retention_df),
        "reload_verified": True,
    }

    save_json(report, outputs_dir / "lapse_report.json")
    return report
