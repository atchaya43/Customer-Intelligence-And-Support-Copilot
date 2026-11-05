"""ReAct agent loop with tool orchestration, checkpoints, safety limits, and crash recovery."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Dict, Any, List, Optional, Tuple

import mock_llm
from src.common.config import STATE_DIR, OUTPUTS_DIR, DB_PATH
from src.common.io import save_json, load_json
from src.part_d.tools import (
    TOOL_REGISTRY,
    TOOLS_SPEC,
    ToolExecutionContext,
    _ACTIVE_CONTEXT,
    ToolExecutionError,
)

AGENT_SYSTEM_PROMPT = (
    "AGENT. Use the provided tools to process this simulated refund. "
    "Never invent observations. Follow verified eligibility and amount checks. "
    "Return a final status and concise explanation."
)


def extract_authorized_order(user_message: str) -> Optional[str]:
    """Extract single authorized order ID from user message."""
    matches = sorted(set(re.findall(r"ORD-\d{6}", user_message)))
    if len(matches) > 1:
        raise ValueError("A single refund request cannot authorize multiple orders")
    return matches[0] if matches else None


class ReActRefundAgent:
    """Multi-step ReAct agent executing allowlisted refund tools under safety guardrails."""

    def __init__(
        self,
        run_id: str,
        scenario_id: str,
        user_message: str,
        max_iterations: int = 6,
        token_limit: int = 6000,
        timeout_seconds: float = 30.0,
        db_path: Path = DB_PATH,
        state_dir: Path = STATE_DIR,
        client: Any = mock_llm,
    ):
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.user_message = user_message
        self.max_iterations = max_iterations
        self.token_limit = token_limit
        self.timeout_seconds = timeout_seconds
        self.db_path = db_path
        self.state_dir = Path(state_dir)
        self.client = client

        self.authorized_order_id = extract_authorized_order(user_message)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        self.steps: List[Dict[str, Any]] = []
        self.tools_executed: List[str] = []
        self.llm_calls: int = 0
        self.tokens_used: int = 0
        self.execution_state: str = "running"
        self.result: Optional[Dict[str, Any]] = None

    def _save_checkpoint(self, elapsed_seconds: float) -> None:
        """Persist current execution state to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": 1,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "system_prompt": AGENT_SYSTEM_PROMPT,
            "user_message": self.user_message,
            "authorized_order_id": self.authorized_order_id,
            "execution_state": self.execution_state,
            "limits": {
                "max_iterations": self.max_iterations,
                "token_limit": self.token_limit,
                "timeout_seconds": self.timeout_seconds,
            },
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
            "step_number": len(self.steps),
            "elapsed_seconds": elapsed_seconds,
            "tools_executed": self.tools_executed,
            "messages": self.messages,
            "steps": self.steps,
            "result": self.result,
        }
        save_json(checkpoint, self.state_dir / f"{self.run_id}.json")

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: Path,
        db_path: Path = DB_PATH,
        client: Any = mock_llm,
    ) -> "ReActRefundAgent":
        """Reconstruct agent state from saved checkpoint to resume interrupted runs."""
        state = load_json(checkpoint_path)
        agent = cls(
            run_id=state["run_id"],
            scenario_id=state["scenario_id"],
            user_message=state["user_message"],
            max_iterations=state["limits"]["max_iterations"],
            token_limit=state["limits"]["token_limit"],
            timeout_seconds=state["limits"]["timeout_seconds"],
            db_path=db_path,
            state_dir=checkpoint_path.parent,
            client=client,
        )
        agent.messages = state["messages"]
        agent.steps = state["steps"]
        agent.tools_executed = state["tools_executed"]
        agent.llm_calls = state["llm_calls"]
        agent.tokens_used = state["tokens_used"]
        agent.execution_state = state["execution_state"]
        agent.result = state.get("result")
        return agent

    def run(self, simulate_crash_after_tool: Optional[str] = None) -> Dict[str, Any]:
        """Execute ReAct cycle until resolution or safety limit trigger."""
        if self.execution_state == "finished" and self.result is not None:
            return self.result

        start_time = time.time()
        ctx = ToolExecutionContext(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            authorized_order_id=self.authorized_order_id,
            db_path=self.db_path,
        )
        ctx_token = _ACTIVE_CONTEXT.set(ctx)

        try:
            while self.execution_state == "running":
                elapsed = time.time() - start_time

                # Safety checks
                if elapsed > self.timeout_seconds:
                    self.execution_state = "finished"
                    self.result = {
                        "run_id": self.run_id,
                        "scenario_id": self.scenario_id,
                        "status": "FALLBACK_TIMEOUT",
                        "final_answer": "Execution timed out before resolution.",
                        "steps": self.steps,
                        "llm_calls": self.llm_calls,
                        "tokens_used": self.tokens_used,
                        "refund_inr": 0.0,
                    }
                    self._save_checkpoint(elapsed)
                    return self.result

                if self.tokens_used > self.token_limit:
                    self.execution_state = "finished"
                    self.result = {
                        "run_id": self.run_id,
                        "scenario_id": self.scenario_id,
                        "status": "FALLBACK_TOKEN_LIMIT",
                        "final_answer": "Token limit exceeded before resolution.",
                        "steps": self.steps,
                        "llm_calls": self.llm_calls,
                        "tokens_used": self.tokens_used,
                        "refund_inr": 0.0,
                    }
                    self._save_checkpoint(elapsed)
                    return self.result

                if self.llm_calls >= self.max_iterations:
                    self.execution_state = "finished"
                    self.result = {
                        "run_id": self.run_id,
                        "scenario_id": self.scenario_id,
                        "status": "FALLBACK_MAX_ITERATIONS",
                        "final_answer": "Maximum step iteration limit reached.",
                        "steps": self.steps,
                        "llm_calls": self.llm_calls,
                        "tokens_used": self.tokens_used,
                        "refund_inr": 0.0,
                    }
                    self._save_checkpoint(elapsed)
                    return self.result

                # Call LLM
                response = self.client.chat(self.messages, tools=TOOLS_SPEC)
                self.llm_calls += 1
                if hasattr(response, "usage") and response.usage:
                    self.tokens_used += response.usage.total_tokens

                choice = response.choices[0]
                message = choice.message

                # Tool calling pathway
                if choice.finish_reason == "tool_calls" and message.tool_calls:
                    formatted_tool_calls = []
                    extracted_calls = []

                    for tc in message.tool_calls:
                        tc_id = tc["id"] if isinstance(tc, dict) else getattr(tc, "id")
                        tc_name = tc["name"] if isinstance(tc, dict) else getattr(tc, "name")
                        tc_args = tc["arguments"] if isinstance(tc, dict) else getattr(tc, "arguments")
                        formatted_tool_calls.append({
                            "id": tc_id,
                            "name": tc_name,
                            "arguments": tc_args,
                        })
                        extracted_calls.append((tc_id, tc_name, tc_args))

                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": formatted_tool_calls,
                    }
                    self.messages.append(assistant_msg)

                    for call_id, tool_name, args in extracted_calls:
                        # Execute registered tool
                        tool_fn = TOOL_REGISTRY[tool_name]
                        observation = tool_fn(**args)

                        # Record step
                        self.steps.append({
                            "tool_call_id": call_id,
                            "tool": tool_name,
                            "arguments": args,
                            "observation": observation,
                        })
                        self.tools_executed.append(tool_name)

                        # Append tool response message
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": json.dumps(observation),
                        })

                        elapsed = time.time() - start_time
                        self._save_checkpoint(elapsed)

                        # Simulated crash demonstration support
                        if simulate_crash_after_tool and tool_name == simulate_crash_after_tool:
                            raise RuntimeError(f"Simulated process crash after executing {tool_name}")

                else:
                    # Final answer reached
                    final_text = message.content or ""
                    self.messages.append({"role": "assistant", "content": final_text})

                    # Parse final status and refund amount
                    status = "NEEDS_INFO"
                    refund_inr = 0.0

                    if "REFUND:" in final_text:
                        status = "REFUND"
                        # Extract from executed financial/calculation steps
                        for step in reversed(self.steps):
                            if step["tool"] in ("issue_refund", "calculate_refund"):
                                obs = step.get("observation", {})
                                if "refund_inr" in obs:
                                    refund_inr = float(obs["refund_inr"])
                                    break
                        if refund_inr == 0.0:
                            amt_match = re.search(r"INR\s+([\d.]+)", final_text)
                            if amt_match:
                                refund_inr = float(amt_match.group(1))

                    elif "ESCALATE:" in final_text:
                        status = "ESCALATE"
                        for step in reversed(self.steps):
                            if step["tool"] in ("escalate_to_human", "calculate_refund"):
                                obs = step.get("observation", {})
                                if "refund_inr" in obs:
                                    refund_inr = float(obs["refund_inr"])
                                    break

                    elif "DENIED:" in final_text:
                        status = "DENIED"
                        refund_inr = 0.0
                    elif "NOT_FOUND:" in final_text:
                        status = "NOT_FOUND"
                        refund_inr = 0.0

                    self.execution_state = "finished"
                    self.result = {
                        "run_id": self.run_id,
                        "scenario_id": self.scenario_id,
                        "status": status,
                        "final_answer": final_text,
                        "steps": self.steps,
                        "llm_calls": self.llm_calls,
                        "tokens_used": self.tokens_used,
                        "refund_inr": refund_inr,
                    }
                    self._save_checkpoint(time.time() - start_time)
                    return self.result

        finally:
            _ACTIVE_CONTEXT.reset(ctx_token)

        return self.result
