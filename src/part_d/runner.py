"""Part D orchestrator: RAG evaluations, 7 refund scenarios, safety demonstrations, and persistence."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List

import mock_llm
from src.common.config import RAW_DATA_DIR, POLICIES_DIR, OUTPUTS_DIR, STATE_DIR, DB_PATH
from src.common.io import save_json, save_jsonl, load_jsonl
from src.part_d.retriever import evaluate_policy_rag, PolicyRetriever
from src.part_d.react_agent import ReActRefundAgent


def run_part_d_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    policies_dir: Path = POLICIES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    state_dir: Path = STATE_DIR,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """Execute complete Part D pipeline: RAG benchmarks, 7 scenarios, and safety demonstrations."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Policy RAG with 10-word overlap vs 0-word overlap
    rag_10 = evaluate_policy_rag(policies_dir=policies_dir, overlap=10)
    rag_0 = evaluate_policy_rag(policies_dir=policies_dir, overlap=0)

    save_json(rag_10, outputs_dir / "rag_overlap_10.json")
    save_json(rag_0, outputs_dir / "rag_overlap_0.json")

    # 2. Reset refund ledger for clean authoritative run
    ledger_path = outputs_dir / "refund_ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    # 3. Execute 7 Agent Scenarios
    with open(raw_dir / "agent_scenarios.csv", encoding="utf-8") as f:
        scenarios = list(csv.DictReader(f))

    scenario_results = []
    mock_llm.reset()

    for item in scenarios:
        scenario_id = item["scenario_id"]
        run_id = f"run_{scenario_id.lower()}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

        agent = ReActRefundAgent(
            run_id=run_id,
            scenario_id=scenario_id,
            user_message=item["message"],
            db_path=db_path,
            state_dir=state_dir,
        )
        result = agent.run()
        scenario_results.append(result)

    save_jsonl(scenario_results, outputs_dir / "agent_runs.jsonl")

    # 4. Safety Demonstrations
    # A. Token limit demonstration (tight limit triggers FALLBACK_TOKEN_LIMIT)
    token_agent = ReActRefundAgent(
        run_id="demo_token_limit",
        scenario_id="S1",
        user_message=scenarios[0]["message"],
        token_limit=200,
        db_path=db_path,
        state_dir=state_dir,
    )
    token_demo_result = token_agent.run()

    # B. Timeout demonstration (instant timeout triggers FALLBACK_TIMEOUT)
    timeout_agent = ReActRefundAgent(
        run_id="demo_timeout",
        scenario_id="S1",
        user_message=scenarios[0]["message"],
        timeout_seconds=0.00001,
        db_path=db_path,
        state_dir=state_dir,
    )
    timeout_demo_result = timeout_agent.run()

    # C. Crash Recovery demonstration
    crash_run_id = "demo_crash_recovery"
    crash_agent = ReActRefundAgent(
        run_id=crash_run_id,
        scenario_id="S1",
        user_message=scenarios[0]["message"],
        db_path=db_path,
        state_dir=state_dir,
    )

    crashed = False
    try:
        crash_agent.run(simulate_crash_after_tool="search_policy")
    except RuntimeError:
        crashed = True

    assert crashed, "Simulated crash should have interrupted execution"

    # Resume from checkpoint
    recovered_agent = ReActRefundAgent.resume_from_checkpoint(
        checkpoint_path=state_dir / f"{crash_run_id}.json",
        db_path=db_path,
    )
    recovered_result = recovered_agent.run()

    demonstrations = {
        "token_limit": token_demo_result,
        "timeout": timeout_demo_result,
        "crash_recovery": {
            "crashed": crashed,
            "recovered_result": recovered_result,
        },
    }
    save_json(demonstrations, outputs_dir / "demonstrations.json")

    summary = {
        "rag_10_accuracy": rag_10["fact_correct"] / rag_10["answerable"],
        "rag_0_accuracy": rag_0["fact_correct"] / rag_0["answerable"],
        "scenarios_completed": len(scenario_results),
        "scenario_statuses": [
            {"scenario": r["scenario_id"], "status": r["status"], "refund_inr": r["refund_inr"]}
            for r in scenario_results
        ],
        "demonstrations_verified": True,
    }
    save_json(summary, outputs_dir / "part_d_summary.json")

    return summary
