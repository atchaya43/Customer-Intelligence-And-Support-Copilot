"""Authoritative checkpoint validation script across Parts A, B, C, and D."""

import json
from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import (
    verify_supplied_files,
    CLEAN_DATA_DIR,
    RAW_DATA_DIR,
    OUTPUTS_DIR,
    STATE_DIR,
    DB_PATH,
    SQL_DIR,
)
from src.part_a.sql_analytics import verify_checkpoints as verify_part_a
from src.part_c.privacy import assert_no_pii

# Reference constants
PART_B_CLUSTER_SIZES = [138, 182, 419, 436]
PART_C_STATUSES = {
    "classified_json": 135,
    "classified_keyword": 12,
    "flagged_invalid_output": 9,
    "flagged_long": 4,
    "skipped_empty": 4,
}
PART_D_EXPECTED_SCENARIOS = [
    ("S1", "REFUND", 128.25),
    ("S2", "REFUND", 135.0),
    ("S3", "ESCALATE", 1999.0),
    ("S4", "DENIED", 0.0),
    ("S5", "DENIED", 0.0),
    ("S6", "NOT_FOUND", 0.0),
    ("S7", "FALLBACK_MAX_ITERATIONS", 0.0),
]


def validate_all() -> None:
    print("=" * 70)
    print("  CUSTOMER INTELLIGENCE & SUPPORT COPILOT - VALIDATION SUITE")
    print("=" * 70)

    # 0. Verify Supplied Files
    print("\n[Check 0] Supplied File Integrity...")
    verify_supplied_files(ROOT)
    print("          PASSED: Supplied scripts are byte-for-byte intact.")

    # 1. Part A Validation
    print("\n[Check 1] Validating Part A Data Foundation & SQL...")
    required_part_a_files = [
        CLEAN_DATA_DIR / "customers.csv",
        CLEAN_DATA_DIR / "products.csv",
        CLEAN_DATA_DIR / "orders.csv",
        DB_PATH,
        SQL_DIR / "queries.sql",
    ]
    for file_path in required_part_a_files:
        assert file_path.is_file(), f"Missing required Part A file: {file_path}"

    part_a_res = verify_part_a(db_path=DB_PATH, clean_dir=CLEAN_DATA_DIR)
    assert part_a_res["status"] == "PASS"
    print("          PASSED: Part A clean data, database, and all 7 SQL queries verified.")

    # 2. Part B Validation
    print("\n[Check 2] Validating Part B Customer Intelligence...")
    required_part_b_files = [
        OUTPUTS_DIR / "customer_segments.csv",
        OUTPUTS_DIR / "lapse_model.joblib",
        OUTPUTS_DIR / "retention_list.csv",
        OUTPUTS_DIR / "segmentation_summary.json",
        OUTPUTS_DIR / "lapse_report.json",
    ]
    for file_path in required_part_b_files:
        assert file_path.is_file(), f"Missing required Part B file: {file_path}"

    seg_df = pd.read_csv(OUTPUTS_DIR / "customer_segments.csv")
    assert len(seg_df) == 1175, f"Expected 1175 segmented customers, got {len(seg_df)}"
    assert sorted(seg_df["segment_id"].value_counts().values) == PART_B_CLUSTER_SIZES
    assert (seg_df["is_premium"] == True).sum() == 138

    seg_summary = json.loads((OUTPUTS_DIR / "segmentation_summary.json").read_text(encoding="utf-8"))
    assert seg_summary["selected_k"] == 4
    np.testing.assert_allclose(seg_summary["pca_variance_total"], 0.852, atol=0.01)

    lapse_rep = json.loads((OUTPUTS_DIR / "lapse_report.json").read_text(encoding="utf-8"))
    assert lapse_rep["reload_verified"] is True
    assert lapse_rep["selected_cv_f1"] >= 0.55

    ret_df = pd.read_csv(OUTPUTS_DIR / "retention_list.csv")
    assert len(ret_df) == 10
    assert list(ret_df.columns) == ["customer_id", "lapse_probability", "actually_lapsed"]
    print("          PASSED: Part B RFM, clustering, PCA, lapse model, and retention list verified.")

    # 3. Part C Validation
    print("\n[Check 3] Validating Part C Support Ticket Copilot...")
    required_part_c_files = [
        OUTPUTS_DIR / "triage_results.csv",
        OUTPUTS_DIR / "triage_budget_demo.csv",
        OUTPUTS_DIR / "triage_summary.json",
    ]
    for file_path in required_part_c_files:
        assert file_path.is_file(), f"Missing required Part C file: {file_path}"

    triage_df = pd.read_csv(OUTPUTS_DIR / "triage_results.csv", keep_default_na=False)
    assert len(triage_df) == 164
    assert_no_pii(triage_df.to_csv(index=False))

    status_counts = triage_df["status"].value_counts().to_dict()
    for status_key, expected_cnt in PART_C_STATUSES.items():
        assert status_counts.get(status_key) == expected_cnt, (
            f"Part C status mismatch for {status_key}: expected {expected_cnt}, got {status_counts.get(status_key)}"
        )

    triage_summary = json.loads((OUTPUTS_DIR / "triage_summary.json").read_text(encoding="utf-8"))
    assert triage_summary["final_run"]["accuracy"] == 1.0
    assert triage_summary["final_run"]["redaction_counts"] == {
        "email": 55,
        "phone": 91,
        "order_id": 132,
    }
    print("          PASSED: Part C privacy redaction, guardrails, token budget, and triage accuracy verified.")

    # 4. Part D Validation
    print("\n[Check 4] Validating Part D Policy RAG & Refund Agent...")
    required_part_d_files = [
        OUTPUTS_DIR / "rag_overlap_10.json",
        OUTPUTS_DIR / "rag_overlap_0.json",
        OUTPUTS_DIR / "agent_runs.jsonl",
        OUTPUTS_DIR / "refund_ledger.jsonl",
        OUTPUTS_DIR / "demonstrations.json",
    ]
    for file_path in required_part_d_files:
        assert file_path.is_file(), f"Missing required Part D file: {file_path}"

    rag_10 = json.loads((OUTPUTS_DIR / "rag_overlap_10.json").read_text(encoding="utf-8"))
    assert rag_10["chunks"] == 15
    assert rag_10["fact_correct"] == 12
    assert rag_10["unsupported_fallbacks"] == 2

    rag_0 = json.loads((OUTPUTS_DIR / "rag_overlap_0.json").read_text(encoding="utf-8"))
    assert rag_0["chunks"] == 13
    assert rag_0["fact_correct"] < 12  # Demonstrates chunk boundary penalty without overlap

    # Check agent runs
    agent_runs = []
    with open(OUTPUTS_DIR / "agent_runs.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                agent_runs.append(json.loads(line))

    assert len(agent_runs) == 7
    for actual, (exp_sid, exp_status, exp_refund) in zip(agent_runs, PART_D_EXPECTED_SCENARIOS):
        assert actual["scenario_id"] == exp_sid
        assert actual["status"] == exp_status, f"Scenario {exp_sid} status mismatch: expected {exp_status}, got {actual['status']}"
        assert abs(actual["refund_inr"] - exp_refund) < 1e-2, (
            f"Scenario {exp_sid} refund mismatch: expected {exp_refund}, got {actual['refund_inr']}"
        )

    # Check demonstrations
    demos = json.loads((OUTPUTS_DIR / "demonstrations.json").read_text(encoding="utf-8"))
    assert demos["token_limit"]["status"] == "FALLBACK_TOKEN_LIMIT"
    assert demos["timeout"]["status"] == "FALLBACK_TIMEOUT"
    assert demos["crash_recovery"]["crashed"] is True
    assert demos["crash_recovery"]["recovered_result"]["status"] == "REFUND"
    print("          PASSED: Part D Policy RAG, tool execution, 7 scenarios, and safety demos verified.")

    print("\n" + "=" * 70)
    print("  ALL CHECKPOINTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    validate_all()
