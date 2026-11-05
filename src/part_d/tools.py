"""Allowlisted refund tools, database integration, authorization, and financial ledger."""

from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from typing import Dict, Any, Optional, Callable

from src.common.config import DB_PATH, OUTPUTS_DIR, ROOT_DIR
from src.common.db import get_db_connection
from src.part_d.retriever import PolicyRetriever

_ACTIVE_CONTEXT: ContextVar[Optional["ToolExecutionContext"]] = ContextVar("active_tool_context", default=None)


class ToolExecutionError(ValueError):
    """Raised when a tool call fails parameter validation or authorization."""


def parse_money(val: Any) -> float:
    """Validate and round a monetary amount to 2 decimal places."""
    if isinstance(val, bool) or not isinstance(val, (int, float, Decimal)):
        raise ToolExecutionError("Monetary amount must be a finite number")
    num = float(val)
    if not math.isfinite(num) or num < 0:
        raise ToolExecutionError("Monetary amount must be non-negative and finite")
    return round(num, 2)


def get_order(order_id: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Retrieve verified order record from SQLite database."""
    if not isinstance(order_id, str) or not re.fullmatch(r"ORD-\d{6}", order_id):
        raise ToolExecutionError(f"Invalid order identifier: {order_id}")

    ctx = _ACTIVE_CONTEXT.get()
    if ctx is not None:
        ctx.verify_order_authorization(order_id)
        db_path = ctx.db_path

    with get_db_connection(db_path, read_only=True) as conn:
        cursor = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": float(row[3]) if row[3] is not None else None,
    }


def search_policy(query: str) -> Dict[str, Any]:
    """Search policy knowledge base for relevant policy chunks."""
    if not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("Policy search query cannot be empty")

    ctx = _ACTIVE_CONTEXT.get()
    retriever = ctx.retriever if ctx is not None else PolicyRetriever()
    results = retriever.retrieve(query, top_k=3)
    return {"results": results}


def calculate_refund(order_amount: float, delivery_minutes: float) -> Dict[str, Any]:
    """Compute refund amount based on delivery duration threshold (> 45 minutes)."""
    amt = parse_money(order_amount)
    if not isinstance(delivery_minutes, (int, float)) or not math.isfinite(delivery_minutes) or delivery_minutes < 0:
        raise ToolExecutionError("Delivery minutes must be a non-negative finite number")

    ctx = _ACTIVE_CONTEXT.get()
    if ctx is not None:
        verified_order = ctx.get_verified_order()
        if amt != round(verified_order["order_amount"], 2):
            raise ToolExecutionError("Provided amount does not match verified database record")
        if delivery_minutes != verified_order["delivery_minutes"]:
            raise ToolExecutionError("Provided delivery duration does not match verified database record")

    refund = amt if delivery_minutes > 45 else 0.0
    return {"refund_inr": refund}


def issue_refund(order_id: str, refund_inr: float) -> Dict[str, Any]:
    """Issue automated refund for eligible orders <= INR 1000."""
    ctx = _ACTIVE_CONTEXT.get()
    if ctx is None:
        raise ToolExecutionError("Financial mutation requires active run context")
    return ctx.record_ledger_entry("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> Dict[str, Any]:
    """Escalate eligible refund > INR 1000 to human support lead."""
    ctx = _ACTIVE_CONTEXT.get()
    if ctx is None:
        raise ToolExecutionError("Financial mutation requires active run context")
    return ctx.record_ledger_entry("escalate_to_human", order_id, refund_inr)


TOOL_REGISTRY: Dict[str, Callable] = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Read the requested order from the verified database.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three most relevant policy chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate_refund",
        "description": "Compute the verified order's full late-delivery refund or zero.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
            "additionalProperties": False,
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible simulated refund of at most INR 1000.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record an eligible refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
            "additionalProperties": False,
        },
    },
]


class ToolExecutionContext:
    """Execution context binding an agent run to verified orders and durable financial ledger."""

    def __init__(
        self,
        run_id: str,
        scenario_id: str,
        authorized_order_id: Optional[str] = None,
        db_path: Path = DB_PATH,
        ledger_path: Path = OUTPUTS_DIR / "refund_ledger.jsonl",
        retriever: Optional[PolicyRetriever] = None,
    ):
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.authorized_order_id = authorized_order_id
        self.db_path = db_path
        self.ledger_path = Path(ledger_path)
        self.retriever = retriever or PolicyRetriever()

    def verify_order_authorization(self, order_id: str) -> None:
        if self.authorized_order_id is not None and order_id != self.authorized_order_id:
            raise ToolExecutionError(
                f"Unauthorized order access: {order_id} does not match authorized {self.authorized_order_id}"
            )

    def get_verified_order(self) -> Dict[str, Any]:
        if not self.authorized_order_id:
            raise ToolExecutionError("No order authorized for this session")
        order = get_order(self.authorized_order_id, db_path=self.db_path)
        if "error" in order:
            raise ToolExecutionError("Authorized order not found in database")
        return order

    def record_ledger_entry(self, action_name: str, order_id: str, refund_inr: float) -> Dict[str, Any]:
        self.verify_order_authorization(order_id)
        verified = self.get_verified_order()

        amount = parse_money(refund_inr)
        expected_amt = parse_money(verified["order_amount"])

        if verified["status"] != "completed":
            raise ToolExecutionError("Only completed orders are eligible for refund actions")
        if verified["delivery_minutes"] is None or verified["delivery_minutes"] <= 45:
            raise ToolExecutionError("Order delivered within 45 minutes is ineligible for refund")
        if amount != expected_amt:
            raise ToolExecutionError("Requested refund amount differs from verified order amount")

        # Threshold check: issue_refund <= 1000, escalate_to_human > 1000
        if action_name == "issue_refund" and amount > 1000:
            raise ToolExecutionError("Refunds over INR 1000 must use escalate_to_human")
        if action_name == "escalate_to_human" and amount <= 1000:
            raise ToolExecutionError("Refunds under or equal to INR 1000 must use issue_refund")

        action_id = f"{self.run_id}:{action_name}:{order_id}"
        entry = {
            "action_id": action_id,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "action": action_name,
            "tool_name": action_name,
            "order_id": order_id,
            "refund_inr": amount,
        }

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Idempotency check: load existing ledger lines
        existing = []
        if self.ledger_path.exists():
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))

        # Check if already present
        for past in existing:
            if past.get("action_id") == action_id:
                return {"refund_inr": amount}
            if past.get("run_id") == self.run_id and past.get("order_id") == order_id:
                raise ToolExecutionError("Conflict: run has already performed an action on this order")

        # Append new entry with UTC ISO timestamp
        new_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

        return {"refund_inr": amount}
