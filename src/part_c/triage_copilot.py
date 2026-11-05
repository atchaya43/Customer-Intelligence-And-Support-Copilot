"""Offline LLM triage batch processor with retries, validation, token budgeting, and routing."""

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
import time
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import mock_llm

from src.common.config import RAW_DATA_DIR, OUTPUTS_DIR
from src.common.io import save_json
from src.part_c.privacy import redact_pii, assert_no_pii
from src.part_c.guardrails import check_ticket_guardrails
from src.part_c.router import route_ticket, CLASSIFIED_STATUSES

VALID_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_RETRIES = 3
BASE_DELAY_SEC = 0.05

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]

ZERO_SHOT_PROMPT = (
    'CLASSIFY the support ticket. Categories: billing (charges or invoices), '
    'delivery (arrival or rider issues), technical (app or login problems), '
    'fraud (unauthorized account activity). Reply only with JSON containing '
    '"category" and "reason".'
)

FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + """
EXAMPLES:
Ticket: Someone hacked my account and charged my wallet.
Label: {"category": "fraud", "reason": "unauthorized account activity"}
Ticket: The app freezes on the payment page.
Label: {"category": "technical", "reason": "app failure"}
"""


@dataclass
class TokenTracker:
    budget_limit: Optional[int] = None
    total_tokens: int = 0
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    retries: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_exhausted(self) -> bool:
        return self.budget_limit is not None and self.total_tokens >= self.budget_limit

    def record_attempt(self, ticket_id: str, attempt: int, outcome: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            prompt_tokens, completion_tokens, tokens = 0, 0, 0
        else:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            tokens = usage.total_tokens

        self.total_tokens += tokens
        self.attempts.append({
            "ticket_id": ticket_id,
            "attempt": attempt,
            "outcome": outcome,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": tokens,
            "running_total": self.total_tokens,
        })


def parse_llm_response(text: Optional[str]) -> Tuple[str, str]:
    """Parse and validate LLM output. Returns (status, category)."""
    if not isinstance(text, str):
        return "flagged_invalid_output", ""

    # Attempt JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            category = data.get("category")
            if isinstance(category, str) and category.lower() in VALID_CATEGORIES:
                return "classified_json", category.lower()
        return "flagged_invalid_output", ""
    except json.JSONDecodeError:
        # Fallback keyword extraction: only if exactly 1 valid category keyword appears
        words = set(re.findall(r"[a-z]+", text.lower()))
        matched = words & set(VALID_CATEGORIES)
        if len(matched) == 1:
            return "classified_keyword", matched.pop()
        return "flagged_invalid_output", ""


def invoke_llm_with_retries(
    ticket_id: str,
    messages: List[Dict[str, str]],
    tracker: TokenTracker,
    rng: random.Random,
    sleeper_fn=lambda _: None,
) -> Tuple[str, Optional[Any]]:
    """Invoke mock LLM with exponential backoff and jitter upon transient errors."""
    for attempt in range(MAX_RETRIES + 1):
        if tracker.is_exhausted:
            return "budget_exhausted", None

        try:
            response = mock_llm.chat(messages)
            tracker.record_attempt(ticket_id, attempt, "response", response)
            return "response", response
        except mock_llm.TransientLLMError as err:
            tracker.record_attempt(ticket_id, attempt, "transient_error", err)
            if tracker.is_exhausted:
                return "budget_exhausted", None
            if attempt == MAX_RETRIES:
                return "flagged_permanent_error", None

            # Exponential backoff with jitter
            delay = BASE_DELAY_SEC * (2 ** attempt) * rng.uniform(0.9, 1.15)
            tracker.retries.append({
                "ticket_id": ticket_id,
                "retry_number": attempt,
                "wait_seconds": delay,
            })
            sleeper_fn(delay)

    return "flagged_permanent_error", None


def process_ticket_batch(
    tickets_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    system_prompt: str,
    token_budget: Optional[int] = None,
    sleeper_fn=lambda _: None,
) -> Tuple[pd.DataFrame, Dict[str, Any], TokenTracker]:
    """Process a batch of tickets through privacy redaction, guardrails, LLM inference, and routing."""
    mock_llm.reset()
    tracker = TokenTracker(budget_limit=token_budget)
    rng = random.Random(42)

    # Segment mapping
    segment_lookup = {}
    for row in segments_df.itertuples():
        segment_lookup[row.customer_id] = (str(row.segment_name), bool(row.is_premium))

    assert_no_pii(system_prompt)
    rows = []
    redaction_counts = Counter()
    classified_count = 0
    correct_count = 0

    for ticket in tickets_df.itertuples():
        # Check guardrails
        guard_status = check_ticket_guardrails(ticket.text)
        redacted_text, r_counts = redact_pii(ticket.text)
        redaction_counts.update(r_counts)

        tokens_before = tracker.total_tokens
        status = guard_status
        category = ""

        if status is None:
            # Ticket is eligible for inference
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": redacted_text},
            ]
            outcome, response = invoke_llm_with_retries(
                ticket.ticket_id, messages, tracker, rng, sleeper_fn
            )
            if outcome == "response":
                try:
                    content = response.choices[0].message.content
                except (AttributeError, IndexError):
                    content = None
                status, category = parse_llm_response(content)
            else:
                status = outcome

        tokens_used = tracker.total_tokens - tokens_before
        seg_name, is_prem = segment_lookup.get(ticket.customer_id, ("", False))
        route, priority = route_ticket(status, category, is_prem)

        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "segment_name": seg_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
        })

        if status in CLASSIFIED_STATUSES:
            classified_count += 1
            if category == ticket.true_category:
                correct_count += 1

    results_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    assert_no_pii(results_df.to_csv(index=False))

    summary = {
        "total_tickets": len(tickets_df),
        "classified_count": classified_count,
        "correct_count": correct_count,
        "accuracy": (correct_count / classified_count) if classified_count > 0 else 0.0,
        "tokens_used": tracker.total_tokens,
        "status_counts": results_df["status"].value_counts().to_dict(),
        "routing_counts": results_df["route"].value_counts().to_dict(),
        "priority_counts": results_df["priority"].value_counts().to_dict(),
        "redaction_counts": dict(redaction_counts),
        "retries_count": len(tracker.retries),
        "attempts_count": len(tracker.attempts),
    }

    return results_df, summary, tracker


def run_ticket_triage(
    raw_dir: Path = RAW_DATA_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Execute both triage budget demo and final triage run."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(raw_dir / "tickets.csv", keep_default_na=False)
    segments = pd.read_csv(outputs_dir / "customer_segments.csv")

    # 1. Budget demo with 6,000 token limit
    budget_df, budget_summary, _ = process_ticket_batch(
        tickets, segments, ZERO_SHOT_PROMPT, token_budget=6000
    )
    budget_df.to_csv(outputs_dir / "triage_budget_demo.csv", index=False)

    # 2. Final triage run with few-shot prompt and generous budget
    final_df, final_summary, _ = process_ticket_batch(
        tickets, segments, FEW_SHOT_PROMPT, token_budget=100000
    )
    final_df.to_csv(outputs_dir / "triage_results.csv", index=False)

    report = {
        "budget_demo": budget_summary,
        "final_run": final_summary,
    }
    save_json(report, outputs_dir / "triage_summary.json")
    return report
