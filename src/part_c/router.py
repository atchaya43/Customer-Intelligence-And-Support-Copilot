"""Support ticket queue routing and SLA priority assignment."""

from typing import Tuple

CLASSIFIED_STATUSES = {"classified_json", "classified_keyword"}


def route_ticket(status: str, category: str, is_premium: bool) -> Tuple[str, str]:
    """Determine routing queue and priority level for a processed ticket.
    
    Returns (route, priority).
    """
    if status not in CLASSIFIED_STATUSES:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"
