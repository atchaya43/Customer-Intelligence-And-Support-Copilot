"""Tests for Part C: Privacy redaction, guardrails, retries, JSON validation, and token budget."""

import unittest
from pathlib import Path
import random
import pandas as pd

from src.common.config import RAW_DATA_DIR
from src.part_c.privacy import redact_pii, assert_no_pii
from src.part_c.guardrails import check_ticket_guardrails
from src.part_c.router import route_ticket
from src.part_c.triage_copilot import parse_llm_response, TokenTracker


class TestPartCSupportCopilot(unittest.TestCase):

    def test_pii_redaction_and_assertion(self):
        """Emails, phone numbers, and order IDs are reliably sanitized."""
        sample = "Contact me at alice@test.com or call 9876543210 about ORD-123456."
        redacted, counts = redact_pii(sample)

        self.assertEqual(counts["email"], 1)
        self.assertEqual(counts["phone"], 1)
        self.assertEqual(counts["order_id"], 1)
        self.assertEqual(redacted, "Contact me at [EMAIL] or call [PHONE] about [ORDER_ID].")

        # Must not raise assertion error
        assert_no_pii(redacted)

        # Must raise assertion error when PII is present
        with self.assertRaises(AssertionError):
            assert_no_pii("Leaked email: support@instamart.in")

    def test_guardrails_empty_and_long(self):
        """Empty tickets and tickets over 400 tokens are intercepted before inference."""
        self.assertEqual(check_ticket_guardrails("   \n\t  "), "skipped_empty")
        self.assertIsNone(check_ticket_guardrails("Short regular support issue."))

        overlong_text = "word " * 500
        self.assertEqual(check_ticket_guardrails(overlong_text), "flagged_long")

    def test_llm_response_parsing_and_keyword_salvage(self):
        """Valid JSON is parsed; single allowed keyword is salvaged; malformed is flagged."""
        # Valid JSON
        status, cat = parse_llm_response('{"category": "fraud", "reason": "unauthorized"}')
        self.assertEqual(status, "classified_json")
        self.assertEqual(cat, "fraud")

        # Invalid JSON with single keyword salvage
        status, cat = parse_llm_response("The customer complaint is primarily about billing issues.")
        self.assertEqual(status, "classified_keyword")
        self.assertEqual(cat, "billing")

        # Multiple keywords -> ambiguous -> flagged
        status, cat = parse_llm_response("This is about billing and delivery at the same time.")
        self.assertEqual(status, "flagged_invalid_output")
        self.assertEqual(cat, "")

        # Valid JSON with invalid category
        status, cat = parse_llm_response('{"category": "marketing", "reason": "promotion"}')
        self.assertEqual(status, "flagged_invalid_output")

    def test_routing_rules(self):
        """Routing logic sends fraud to security, premium to premium desk, unclassified to human review."""
        self.assertEqual(route_ticket("classified_json", "fraud", False), ("security_team", "HIGH"))
        self.assertEqual(route_ticket("classified_json", "delivery", True), ("premium_desk", "HIGH"))
        self.assertEqual(route_ticket("classified_json", "delivery", False), ("support_queue", "NORMAL"))
        self.assertEqual(route_ticket("flagged_long", "", False), ("human_review", "NORMAL"))
        self.assertEqual(route_ticket("skipped_empty", "", True), ("human_review", "NORMAL"))

    def test_token_tracker_and_budget_exhaustion(self):
        """TokenTracker respects budget limits and records running totals."""
        tracker = TokenTracker(budget_limit=100)
        self.assertFalse(tracker.is_exhausted)

        class MockUsage:
            prompt_tokens = 50
            completion_tokens = 60
            total_tokens = 110

        class MockResp:
            usage = MockUsage()

        tracker.record_attempt("T1", 0, "response", MockResp())
        self.assertEqual(tracker.total_tokens, 110)
        self.assertTrue(tracker.is_exhausted)


if __name__ == "__main__":
    unittest.main()
