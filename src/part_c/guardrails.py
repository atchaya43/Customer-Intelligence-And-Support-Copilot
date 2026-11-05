"""Pre-inference ticket input guardrails."""

from typing import Optional
import mock_llm

MAX_INPUT_TOKENS = 400


def check_ticket_guardrails(text: str) -> Optional[str]:
    """Check whether a ticket should be bypassed before LLM inference.
    
    Returns 'skipped_empty', 'flagged_long', or None if eligible.
    """
    if not text.strip():
        return "skipped_empty"
    if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None
