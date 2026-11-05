"""TF-IDF policy retrieval, chunk ranking, and grounded answering evaluation."""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

from src.common.config import POLICIES_DIR, RAW_DATA_DIR, ROOT_DIR
from src.part_d.chunking import chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

POLICY_SYSTEM_PROMPT = (
    "POLICY ASSISTANT. Answer only from the supplied context. "
    "If the context does not contain the answer, say you could not find the information. "
    "Do not use general knowledge or invent policy rules."
)

UNSUPPORTED_FALLBACK = "I could not find that information in the policy documents."


class PolicyRetriever:
    """Retriever indexing policy documents with TF-IDF and cosine similarity."""

    def __init__(self, policies_dir: Path = POLICIES_DIR, overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.policies_dir = Path(policies_dir)
        self.overlap = overlap
        self.policy_paths = sorted(self.policies_dir.glob("*.txt"), key=lambda p: p.name)

        if not self.policy_paths:
            raise FileNotFoundError(f"No policy text files found in {self.policies_dir}")

        self.chunks: List[Dict[str, Any]] = []
        for path in self.policy_paths:
            content = path.read_text(encoding="utf-8")
            doc_chunks = chunk_text(
                source_filename=path.name,
                text=content,
                chunk_size=DEFAULT_CHUNK_SIZE,
                overlap=self.overlap,
            )
            self.chunks.extend(doc_chunks)

        self.vectorizer = TfidfVectorizer()
        corpus = [c["text"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks for a query string."""
        if not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).ravel()

        # Sort descending by score, tie-break by chunk_id
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: (-scores[idx], self.chunks[idx]["chunk_id"]),
        )

        results = []
        for idx in ranked_indices[:top_k]:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(scores[idx]),
                "text": self.chunks[idx]["text"],
            })
        return results


def ground_response(text: Optional[str], retrieved_chunks: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]]) -> str:
    """Validate that answer text is strictly grounded in retrieved chunk tokens."""
    if not isinstance(text, str) or not text.strip() or text.startswith("I could not find"):
        return UNSUPPORTED_FALLBACK

    # Gather available tokens and positions from retrieved chunks
    tokens, positions = [], []
    for ret in retrieved_chunks:
        matching_chunk = next(
            (c for c in all_chunks if c["source"] == ret["source"] and c["text"] == ret["text"]),
            None,
        )
        if matching_chunk is None:
            return UNSUPPORTED_FALLBACK
        words = ret["text"].split()
        tokens.extend(words)
        positions.extend((matching_chunk["source"], matching_chunk["start_word"] + i) for i in range(len(words)))

    wanted = text.split()
    for start in range(len(tokens) - len(wanted) + 1):
        if tokens[start:start + len(wanted)] != wanted:
            continue
        span = positions[start:start + len(wanted)]
        source, first = span[0]
        last, accepted = first - 1, []
        for word, (curr_source, offset) in zip(wanted, span):
            if curr_source != source or offset < first or offset > last + 1:
                break
            if offset == last + 1:
                accepted.append(word)
                last = offset
        else:
            return " ".join(accepted)

    return UNSUPPORTED_FALLBACK


def answer_policy_question(
    question: str,
    retriever: PolicyRetriever,
    client: Any = mock_llm,
) -> Dict[str, Any]:
    """Retrieve top context and query Policy Assistant for grounded answer."""
    retrieved = retriever.retrieve(question, top_k=3)
    context_lines = [f"[{r['source']}]\n{r['text']}" for r in retrieved]
    context_str = "\n".join(context_lines)

    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"},
    ]

    response = client.chat(messages)
    raw_answer = response.choices[0].message.content
    grounded_answer = ground_response(raw_answer, retrieved, retriever.chunks)

    return {
        "answer": grounded_answer,
        "raw_answer": raw_answer,
        "sources": [r["source"] for r in retrieved],
        "retrieved": retrieved,
        "tokens_used": int(response.usage.total_tokens),
    }


def evaluate_policy_rag(
    policies_dir: Path = POLICIES_DIR,
    eval_csv_path: Path = RAW_DATA_DIR / "policy_eval.csv",
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """Run evaluation benchmark across all test questions for given chunk overlap."""
    retriever = PolicyRetriever(policies_dir=policies_dir, overlap=overlap)
    with open(eval_csv_path, encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))

    mock_llm.reset()
    details = []

    for item in questions:
        res = answer_policy_question(item["question"], retriever=retriever)
        is_answerable = item["expected_doc"] != "NONE"
        source_correct = is_answerable and (item["expected_doc"] in res["sources"])
        raw_fact_correct = is_answerable and (item["expected_fact"] in res["raw_answer"])
        fact_correct = is_answerable and (item["expected_fact"] in res["answer"])
        unsupported_fallback = (not is_answerable) and res["answer"].startswith("I could not find")

        details.append({
            **item,
            **res,
            "answerable": is_answerable,
            "source_correct": source_correct,
            "fact_correct": fact_correct,
            "raw_fact_correct": raw_fact_correct,
            "unsupported_fallback": unsupported_fallback,
        })

    summary = {
        "overlap": overlap,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_counts": dict(sorted(Counter(c["source"] for c in retriever.chunks).items())),
        "chunks": len(retriever.chunks),
        "questions": len(details),
        "answerable": sum(d["answerable"] for d in details),
        "source_correct": sum(d["source_correct"] for d in details),
        "fact_correct": sum(d["fact_correct"] for d in details),
        "raw_fact_correct": sum(d["raw_fact_correct"] for d in details),
        "unanswerable": sum(not d["answerable"] for d in details),
        "unsupported_fallbacks": sum(d["unsupported_fallback"] for d in details),
        "total_tokens": sum(d["tokens_used"] for d in details),
        "details": details,
    }

    return summary
