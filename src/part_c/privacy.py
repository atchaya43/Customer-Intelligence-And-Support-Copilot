"""Privacy protection, regex redactions, and PII leak detection."""

import re
from typing import Tuple, Dict

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

PII_RULES = [
    ("email", EMAIL_PATTERN, "[EMAIL]"),
    ("phone", PHONE_PATTERN, "[PHONE]"),
    ("order_id", ORDER_PATTERN, "[ORDER_ID]"),
]


def assert_no_pii(text: str) -> None:
    """Assert that raw emails, phone numbers, or order IDs do not exist in the string."""
    for name, pattern, _ in PII_RULES:
        if pattern.search(text):
            raise AssertionError(f"PII leak detected! Text contains unredacted pattern: {name}")


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Redact email addresses, phone numbers, and order IDs from text."""
    redacted = text
    counts = {}
    for name, pattern, replacement in PII_RULES:
        redacted, count = pattern.subn(replacement, redacted)
        counts[name] = count

    assert_no_pii(redacted)
    return redacted, counts
