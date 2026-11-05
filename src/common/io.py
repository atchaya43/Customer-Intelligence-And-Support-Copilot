"""Input/Output helper utilities for JSON, JSONL, and CSV artifacts."""

import json
from pathlib import Path
from typing import Any, List, Dict


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """Save an object to a formatted JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Path) -> Any:
    """Load an object from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict[str, Any]], file_path: Path) -> None:
    """Save a list of dictionaries as line-delimited JSON (JSONL)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from a line-delimited JSON (JSONL) file."""
    if not file_path.is_file():
        return []
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
