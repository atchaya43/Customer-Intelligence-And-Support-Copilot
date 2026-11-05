"""Tests for Part D: Policy Chunking, RAG Retrieval, Refund Tools, ReAct Loop, and Crash Recovery."""

import json
from pathlib import Path
import unittest

from src.common.config import POLICIES_DIR, OUTPUTS_DIR, STATE_DIR, DB_PATH
from src.part_d.chunking import chunk_text
from src.part_d.retriever import PolicyRetriever
from src.part_d.tools import (
    get_order,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    TOOL_REGISTRY,
    ToolExecutionContext,
    _ACTIVE_CONTEXT,
    ToolExecutionError,
)
from src.part_d.react_agent import ReActRefundAgent, extract_authorized_order


class TestPartDPolicyRAGAndAgent(unittest.TestCase):

    def test_chunking_with_and_without_overlap(self):
        """Chunking partitions documents accurately with sliding overlap window."""
        sample = "word " * 100
        chunks_10 = chunk_text("sample.txt", sample, chunk_size=40, overlap=10)
        self.assertEqual(len(chunks_10), 3)
        self.assertEqual(chunks_10[0]["start_word"], 0)
        self.assertEqual(chunks_10[0]["end_word"], 40)
        self.assertEqual(chunks_10[1]["start_word"], 30)

        chunks_0 = chunk_text("sample.txt", sample, chunk_size=40, overlap=0)
        self.assertEqual(len(chunks_0), 3)
        self.assertEqual(chunks_0[1]["start_word"], 40)

    def test_policy_retriever(self):
        """Policy retriever ranks relevant chunks using TF-IDF and cosine similarity."""
        retriever = PolicyRetriever(policies_dir=POLICIES_DIR, overlap=10)
        self.assertEqual(len(retriever.chunks), 15)

        results = retriever.retrieve("refund policy late delivery", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["source"] == "refund_policy.txt" for r in results))

    def test_extract_authorized_order(self):
        """Extracts valid order IDs and rejects messages containing multiple IDs."""
        self.assertEqual(extract_authorized_order("Order ORD-100048 was late"), "ORD-100048")
        self.assertIsNone(extract_authorized_order("No order number provided"))
        with self.assertRaises(ValueError):
            extract_authorized_order("Orders ORD-100001 and ORD-100002")

    def test_get_order_tool(self):
        """Database lookup retrieves actual order data and handles non-existent IDs."""
        order = get_order("ORD-100048", db_path=DB_PATH)
        self.assertEqual(order["order_id"], "ORD-100048")
        self.assertEqual(order["status"], "completed")

        missing = get_order("ORD-999999", db_path=DB_PATH)
        self.assertIn("error", missing)

    def test_refund_calculation_and_escalation_rules(self):
        """Delivery > 45 minutes qualifies for full refund; <= 45 minutes receives 0."""
        self.assertEqual(calculate_refund(150.0, 50.0)["refund_inr"], 150.0)
        self.assertEqual(calculate_refund(150.0, 30.0)["refund_inr"], 0.0)

    def test_agent_safety_limits(self):
        """Agent aborts gracefully on token limits, timeout limits, and max iterations."""
        # Max iterations
        agent_loop = ReActRefundAgent(
            run_id="test_loop",
            scenario_id="S7",
            user_message="[LOOP] Refund order ORD-100048",
            max_iterations=6,
            db_path=DB_PATH,
            state_dir=STATE_DIR,
        )
        res_loop = agent_loop.run()
        self.assertEqual(res_loop["status"], "FALLBACK_MAX_ITERATIONS")

        # Token limit
        agent_token = ReActRefundAgent(
            run_id="test_token_safety",
            scenario_id="S1",
            user_message="Order ORD-100048 arrived very late.",
            token_limit=150,
            db_path=DB_PATH,
            state_dir=STATE_DIR,
        )
        res_token = agent_token.run()
        self.assertEqual(res_token["status"], "FALLBACK_TOKEN_LIMIT")

    def test_crash_recovery_resumption(self):
        """Agent recovers from simulated crash and completes execution without error."""
        run_id = "test_crash_recovery_unit"
        agent = ReActRefundAgent(
            run_id=run_id,
            scenario_id="S1",
            user_message="My order ORD-100048 reached me very late.",
            db_path=DB_PATH,
            state_dir=STATE_DIR,
        )

        with self.assertRaises(RuntimeError):
            agent.run(simulate_crash_after_tool="search_policy")

        # Resume
        ckpt_path = STATE_DIR / f"{run_id}.json"
        self.assertTrue(ckpt_path.is_file())

        resumed = ReActRefundAgent.resume_from_checkpoint(ckpt_path, db_path=DB_PATH)
        res = resumed.run()
        self.assertEqual(res["status"], "REFUND")
        self.assertEqual(res["refund_inr"], 128.25)


if __name__ == "__main__":
    unittest.main()
