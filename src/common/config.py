"""Project configuration, paths, and integrity verification."""

import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
POLICIES_DIR = DATA_DIR / "policies"
DB_PATH = DATA_DIR / "instamart.db"

OUTPUTS_DIR = ROOT_DIR / "outputs"
STATE_DIR = ROOT_DIR / "state"
SQL_DIR = ROOT_DIR / "sql"

# Official reference SHA-256 hashes of supplied generator and mock
EXPECTED_HASHES = {
    "generate_data.py": "2ed34dc68e76d38b737d7d869f037e5f1412e2926985b9b2095496ddf3d04a6d",
    "mock_llm.py": "b579136594236181854a9d956f87528e99ef7e60490f509a7c6230642ba85378",
}


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_supplied_files(root: Path = ROOT_DIR) -> None:
    """Assert that generate_data.py and mock_llm.py have not been modified."""
    for filename, expected_hash in EXPECTED_HASHES.items():
        target = root / filename
        if not target.is_file():
            raise FileNotFoundError(f"Required supplied file is missing: {filename}")
        actual_hash = compute_sha256(target)
        if actual_hash.lower() != expected_hash.lower():
            raise AssertionError(
                f"Supplied file {filename} hash mismatch! "
                f"Expected: {expected_hash}, Got: {actual_hash}. "
                "Supplied files must not be modified."
            )
