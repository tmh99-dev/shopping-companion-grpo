"""tools.search: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import math
import re


TOKEN_RE = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold()) or ["empty"]


MEMORY_STOPWORDS = {
    "a", "an", "and", "are", "for", "from", "i", "in", "is", "it", "my",
    "of", "on", "or", "preference", "preferences", "the", "to", "user", "with",
}


def memory_terms(text: str) -> list[str]:
    """Content words with a conservative singular variant."""
    output = []
    for token in TOKEN_RE.findall(text.casefold()):
        if token in MEMORY_STOPWORDS:
            continue
        output.append(token)
        if len(token) > 3 and token.endswith("s"):
            output.append(token[:-1])
    return output or ["empty"]


class BM25Okapi:
    """Small dependency-free BM25 implementation for the 10k pilot corpus."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.frequencies = []
        document_frequency: dict[str, int] = defaultdict(int)
        for document in corpus:
            counts: dict[str, int] = defaultdict(int)
            for token in document:
                counts[token] += 1
            self.frequencies.append(counts)
            for token in counts:
                document_frequency[token] += 1
        size = len(corpus)
        self.idf = {token: math.log(1 + (size - count + 0.5) / (count + 0.5))
                    for token, count in document_frequency.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for document, counts in zip(self.corpus, self.frequencies):
            length_norm = self.k1 * (1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9))
            score = 0.0
            for token in query:
                frequency = counts.get(token, 0)
                if frequency:
                    score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / (frequency + length_norm)
            scores.append(score)
        return scores


class ProductSearchMixin:
    @staticmethod
    def matched_fields(row: dict[str, Any], query: str) -> list[dict[str, Any]]:
        """Explain lexical matches using the catalog's original fields and values."""
        query_words = set(terms(query)) - MEMORY_STOPWORDS
        product = row["product"]
        sources = [("product_name", product.get("product_name", "")),
                   ("category", product.get("category", ""))]
        for key, values in (product.get("attributes") or {}).items():
            for value in values if isinstance(values, list) else [values]:
                sources.append((str(key), str(value)))
        for option in product.get("options", []):
            if isinstance(option, dict):
                for key, values in option.items():
                    for value in values if isinstance(values, list) else [values]:
                        sources.append((str(key), str(value)))
        matches = []
        for field, value in sources:
            hit = sorted(query_words & set(terms(str(value))))
            if hit:
                matches.append({"field": field, "value": value, "matched_terms": hit})
        return sorted(matches, key=lambda item: (-len(item["matched_terms"]), item["field"]))[:5]

    def search(self, arguments: dict[str, Any], bundle_question: str | None = None) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"error": "query must be non-empty"}
        filters = arguments.get("filters") or {}
        if not isinstance(filters, dict):
            return {"error": "filters must be an object"}
        supported_filters = {"max_price", "min_price", "seller_id", "category"}
        unsupported_filters = sorted(set(filters) - supported_filters)
        if unsupported_filters:
            return {
                "error": f"Unsupported product filters: {unsupported_filters}. "
                         "Supported filters: max_price, min_price, seller_id, category. "
                         "Describe the product and preferences in query; inspect product details before selection."
            }
        price_preference = None
        safe_hard_max = None
        proposed = arguments.get("preferred_max_list_price")
        if bundle_question is not None:
            # Versioned bundle-search policy: a model-suggested price is a
            # ranking preference, never a proof that a discounted basket is
            # infeasible. The hard bound is derived only from public terms.
            from shopping_agent.tools.voucher import parse_public_voucher
            public = parse_public_voucher(bundle_question)
            safe_hard_max = public["budget"] + public["cap"]
            if proposed is None:
                proposed = filters.get("max_price")  # old model calls stay soft in this version
            filters = {key: value for key, value in filters.items() if key != "max_price"}
        if proposed is not None:
            try:
                price_preference = float(proposed)
            except (TypeError, ValueError):
                return {"error": "preferred_max_list_price must be a nonnegative number"}
            if not math.isfinite(price_preference) or price_preference < 0:
                return {"error": "preferred_max_list_price must be a nonnegative number"}
        try:
            top_k = max(1, min(int(arguments.get("top_k", 8)), 10))
        except (TypeError, ValueError):
            top_k = 8
        # Category names differ in granularity between marketplaces. Keep the
        # requested product terms dominant; an inferred category is only a hint.
        category = str(filters.get("category", "")).strip()
        query_scores = self.index.get_scores(terms(query))
        category_scores = self.index.get_scores(terms(category)) if category else [0.0] * len(self.rows)
        scores = [float(base) + 0.1 * float(hint) for base, hint in zip(query_scores, category_scores)]
        if price_preference is not None:
            for index, row in enumerate(self.rows):
                try:
                    if float(row["product"].get("price")) <= price_preference:
                        scores[index] += 0.05  # soft preference; never excludes a product
                except (TypeError, ValueError):
                    pass
        ranked = sorted((i for i, base in enumerate(query_scores) if float(base) > 0),
                        key=lambda i: (-scores[i], str(self.rows[i]["id"])))
        output = []
        for index in ranked:
            row = self.rows[index]
            product = row["product"]
            try:
                price = float(product.get("price"))
            except (TypeError, ValueError):
                continue
            if safe_hard_max is not None and price > float(safe_hard_max):
                continue
            if filters.get("max_price") is not None and price > float(filters["max_price"]):
                continue
            if filters.get("min_price") is not None and price < float(filters["min_price"]):
                continue
            if filters.get("seller_id") is not None and str(product.get("seller_id")) != str(filters["seller_id"]):
                continue
            item = self.public_summary(row)
            item["search_score"] = round(float(scores[index]), 4)
            item["matched_fields"] = self.matched_fields(row, query)
            output.append(item)
            if len(output) >= top_k:
                break
        return {
            "results": output,
            "applied_filters": {"max_price": filters.get("max_price"),
                                "min_price": filters.get("min_price"), "seller_id": filters.get("seller_id")},
            "notice": ("matched_fields shows literal query-word matches in original catalog fields; "
                       "it does not prove a preference or variant is satisfied. Category is a soft hint. "
                       "Inspect products and their options before selection."),
            **({"bundle_price_policy": {
                "preferred_max_list_price": price_preference,
                "hard_max_list_price": float(safe_hard_max),
                "rule": "Hard bound is public final budget plus maximum discount; quoted basket decides affordability.",
            }} if bundle_question is not None else {}),
            **({"price_preference": {"preferred_max_list_price": price_preference,
                                     "effect": "soft ranking only"}}
               if bundle_question is None and price_preference is not None else {}),
        }
