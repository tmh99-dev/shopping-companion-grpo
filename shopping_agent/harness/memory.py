"""harness.memory: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from collections import defaultdict
from shopping_agent.data.tasks import truth_items
from shopping_agent.tools.search import memory_terms
from typing import Any
import hashlib


def memory_id(conversation_id: str, question_id: str, ordinal: int) -> str:
    value = hashlib.sha256(f"{conversation_id}:{question_id}:{ordinal}".encode()).hexdigest()[:10]
    return f"memory_{value}"


def build_memories(all_tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    memories: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_tasks:
        info = row["extra_info"]
        for ordinal, item in enumerate(truth_items(row)):
            # Only prior user utterances are exposed; target IDs, labels, product
            # names and assistant-written preference summaries are excluded.
            user_turns = [message["content"] for message in item.get("dialogue", [])
                          if message.get("role") == "user"]
            if not user_turns:
                continue
            memories[info["conversation_id"]].append({
                "id": memory_id(info["conversation_id"], info["question_id"], ordinal),
                "text": "\n".join(f"User: {turn}" for turn in user_turns),
            })
    return memories


def retrieve_memories(question: str, docs: list[dict[str, str]], top_k: int = 5) -> list[dict[str, str]]:
    if not docs:
        return []
    query = set(memory_terms(question))
    scores = []
    for doc in docs:
        document = set(memory_terms(doc["text"]))
        score = 0.0
        for query_token in query:
            best = 0.0
            for document_token in document:
                if query_token == document_token:
                    best = 3.0
                    break
                if (len(query_token) >= 4 and len(document_token) >= 4 and
                        (query_token in document_token or document_token in query_token)):
                    best = max(best, 2.0)
            score += best
        scores.append(score)
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), docs[i]["id"]))
    positive = [i for i in order if float(scores[i]) > 0]
    return [{**docs[i], "score": round(float(scores[i]), 4)} for i in positive[:top_k]]
