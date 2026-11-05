"""RFM calculation, KMeans clustering (K=2..8), silhouette analysis, business profiles, and PCA."""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.common.config import OUTPUTS_DIR, CLEAN_DATA_DIR
from src.common.io import save_json

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")
RFM_FEATURES = ["recency", "frequency", "monetary"]
SEGMENT_COLUMNS = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]


def build_rfm(orders: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT_DATE) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary metrics for completed orders before snapshot."""
    df = orders.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    completed = df[(df["status"] == "completed") & (df["order_ts"] < snapshot)]

    rfm = completed.groupby("customer_id", sort=True).agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (snapshot - rfm.pop("last_order")).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[RFM_FEATURES].reset_index()
    return rfm


def assign_business_segments(rfm: pd.DataFrame, cluster_labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map K-Means cluster IDs to human-interpretable business segments based on profile characteristics."""
    df = rfm.assign(segment_id=cluster_labels)
    profiles = df.groupby("segment_id", sort=True).agg(
        customers=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )

    # 1. Premium: highest average monetary spend
    premium_id = int(profiles["mean_monetary"].idxmax())
    names = {
        premium_id: ("Premium", "Offer priority service and personalised high-value retention outreach.")
    }

    remaining = profiles.drop(index=premium_id)

    # 2. Frequent: highest average frequency among remaining
    if len(remaining) > 0:
        frequent_id = int(remaining["mean_frequency"].idxmax())
        names[frequent_id] = ("Frequent", "Reward repeat grocery shopping with replenishment reminders and loyalty benefits.")
        remaining = remaining.drop(index=frequent_id)

    # 3. Dormant: highest average recency (longest absence) among remaining
    if len(remaining) > 0:
        dormant_id = int(remaining["mean_recency"].idxmax())
        names[dormant_id] = ("Dormant", "Test a limited win-back offer and ask what stopped repeat purchases.")
        remaining = remaining.drop(index=dormant_id)

    # 4. Occasional: remaining clusters ordered by recency
    for idx, seg_id in enumerate(remaining.sort_values("mean_recency").index):
        name = "Occasional" if idx == 0 else f"Occasional {idx + 1}"
        names[int(seg_id)] = (name, "Encourage a second shopping occasion with relevant basket suggestions.")

    profiles["segment_name"] = [names[int(i)][0] for i in profiles.index]
    profiles["recommended_action"] = [names[int(i)][1] for i in profiles.index]
    profiles["is_premium"] = profiles.index == premium_id

    df["segment_name"] = df["segment_id"].map(profiles["segment_name"])
    df["is_premium"] = df["segment_id"] == premium_id

    return df[SEGMENT_COLUMNS], profiles.reset_index()


def perform_segmentation(
    clean_dir: Path = CLEAN_DATA_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Execute complete segmentation pipeline: RFM, K-Means grid (K=2..8), PCA, and artifact export."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(clean_dir / "orders.csv")

    rfm = build_rfm(orders)
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[RFM_FEATURES])

    # K-Means exploration for K=2..8
    kmeans_results = []
    models = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled_rfm)
        sil = float(silhouette_score(scaled_rfm, labels))
        kmeans_results.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": sil,
        })
        models[k] = km

    comparison_df = pd.DataFrame(kmeans_results)

    # Operational constraint: select highest silhouette among K in {3, 4, 5}
    candidates = comparison_df[comparison_df["k"].isin([3, 4, 5])]
    chosen_k = int(candidates.sort_values(by=["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    best_model = models[chosen_k]

    # Assign business segments
    segmented_df, profiles_df = assign_business_segments(rfm, best_model.labels_)

    # 2D PCA for visual projection and variance explanation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_rfm)
    pca_df = pd.DataFrame(coords, columns=["pc1", "pc2"])
    pca_df.insert(0, "customer_id", rfm["customer_id"])
    pca_df["segment_id"] = segmented_df["segment_id"]

    # Export customer_segments.csv
    segmented_df.to_csv(outputs_dir / "customer_segments.csv", index=False)

    summary = {
        "snapshot": str(SNAPSHOT_DATE),
        "customers": len(rfm),
        "recency_median": float(rfm["recency"].median()),
        "recency_min": int(rfm["recency"].min()),
        "recency_max": int(rfm["recency"].max()),
        "frequency_mean": float(rfm["frequency"].mean()),
        "frequency_max": int(rfm["frequency"].max()),
        "monetary_mean": float(rfm["monetary"].mean()),
        "monetary_max": float(rfm["monetary"].max()),
        "selected_k": chosen_k,
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "pca_variance_total": float(pca.explained_variance_ratio_.sum()),
        "kmeans_comparison": kmeans_results,
        "profiles": profiles_df.to_dict(orient="records"),
    }

    save_json(summary, outputs_dir / "segmentation_summary.json")
    return summary
