"""Document chunking with configurable window size and overlap."""

from typing import List, Dict, Any

DEFAULT_CHUNK_SIZE = 40
DEFAULT_CHUNK_OVERLAP = 10


def chunk_text(
    source_filename: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Divide text into fixed word-count chunks with sliding-window overlap."""
    if not (0 <= overlap < chunk_size):
        raise ValueError("Overlap must satisfy: 0 <= overlap < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "chunk_id": f"{source_filename}:{len(chunks):03d}",
            "source": source_filename,
            "start_word": start,
            "end_word": end,
            "text": " ".join(chunk_words),
        })
        if end == len(words):
            break

    return chunks
