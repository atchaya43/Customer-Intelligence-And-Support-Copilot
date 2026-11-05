"""Master pipeline execution script running Parts A, B, C, and D sequentially."""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import verify_supplied_files
from src.part_a.cleaner import clean_all_data
from src.part_a.database import create_database, load_clean_data_to_db
from src.part_a.sql_analytics import run_sql_analytics
from src.part_b.rfm_segmentation import perform_segmentation
from src.part_b.models import train_and_evaluate_lapse_models
from src.part_c.triage_copilot import run_ticket_triage
from src.part_d.runner import run_part_d_pipeline


def run_all() -> None:
    print("=" * 70)
    print("  CUSTOMER INTELLIGENCE & SUPPORT COPILOT - COMPLETE EXECUTION")
    print("=" * 70)
    start_total = time.time()

    # Step 0: Integrity Check
    print("\n[0/4] Verifying supplied file integrity (SHA-256)...")
    verify_supplied_files(ROOT)
    print("      Supplied files (generate_data.py, mock_llm.py) verified intact.")

    # Step 1: Part A - Data Foundation
    print("\n[1/4] Running Part A: Data Cleaning, SQLite Database & SQL Analytics...")
    t0 = time.time()
    clean_all_data()
    create_database()
    load_clean_data_to_db()
    sql_res = run_sql_analytics()
    print(f"      Part A completed in {time.time() - t0:.2f}s. All 7 SQL queries executed.")

    # Step 2: Part B - Customer Intelligence
    print("\n[2/4] Running Part B: RFM Segmentation, KMeans, PCA & Lapse Modeling...")
    t0 = time.time()
    seg_summary = perform_segmentation()
    print(f"      RFM Segmentation completed (Selected K={seg_summary['selected_k']}).")
    lapse_report = train_and_evaluate_lapse_models()
    print(f"      Lapse Modeling completed (Selected Model={lapse_report['selected_model']}, CV F1={lapse_report['selected_cv_f1']:.4f}).")
    print(f"      Part B completed in {time.time() - t0:.2f}s.")

    # Step 3: Part C - Support Ticket Copilot
    print("\n[3/4] Running Part C: PII Redaction, Guardrails & Ticket Triage...")
    t0 = time.time()
    triage_report = run_ticket_triage()
    print(f"      Budget Demo tokens used: {triage_report['budget_demo']['tokens_used']}.")
    print(f"      Final Run accuracy: {triage_report['final_run']['accuracy'] * 100:.1f}%.")
    print(f"      Part C completed in {time.time() - t0:.2f}s.")

    # Step 4: Part D - Policy RAG & Refund Agent
    print("\n[4/4] Running Part D: Policy RAG, Tool Execution & ReAct Agent...")
    t0 = time.time()
    agent_report = run_part_d_pipeline()
    print(f"      RAG Overlap 10 Accuracy: {agent_report['rag_10_accuracy'] * 100:.1f}%.")
    print(f"      RAG Overlap 0 Accuracy: {agent_report['rag_0_accuracy'] * 100:.1f}%.")
    print(f"      All {agent_report['scenarios_completed']} agent scenarios evaluated.")
    print(f"      Part D completed in {time.time() - t0:.2f}s.")

    print("\n" + "=" * 70)
    print(f"  EXECUTION COMPLETE IN {time.time() - start_total:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
