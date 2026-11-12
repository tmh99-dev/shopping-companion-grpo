"""tools.finish: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from typing import Any


def selections_from_quoted_ids(arguments: dict[str, Any],
                               constraints: list[dict[str, Any]],
                               requested_items: list[str],
                               viewed: dict[str, dict[str, Any]],
                               successful_calculations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Resolve only IDs the model explicitly chose and already quoted.

    Item ownership comes solely from the model's source-verified checks. This
    function never consults target IDs or preference labels.
    """
    raw_ids = arguments.get("quoted_product_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return None, "finish requires non-empty quoted_product_ids"
    ids = [str(value) for value in raw_ids]
    if len(ids) != len(requested_items) or len(set(ids)) != len(ids):
        return None, "quoted_product_ids must contain one unique ID per requested item"
    if any(product_id not in viewed for product_id in ids):
        return None, "inspect every quoted product before finishing"
    if not any(set(quote.get("product_ids", [])) == set(ids)
               and quote.get("within_budget") is True
               for quote in successful_calculations):
        return None, "quoted_product_ids need a matching within-budget calculate_bundle result"
    variant_map = arguments.get("variant_indices_by_product") or {}
    if not isinstance(variant_map, dict) or any(str(key) not in ids for key in variant_map):
        return None, "variant_indices_by_product must map only quoted product IDs"
    selections = []
    for item in requested_items:
        matching = []
        for product_id in ids:
            check_lists = [(entry.get("candidate_checks") or {}).get(product_id, [])
                           for entry in constraints if isinstance(entry, dict)
                           and str(entry.get("item", "")).casefold() == item.casefold()]
            if any(check.get("evidence_verified") is True and check.get("status") in
                   {"supported", "available_unselected"}
                   for checks in check_lists for check in checks if isinstance(check, dict)):
                matching.append((product_id, [check for checks in check_lists for check in checks
                                              if isinstance(check, dict)]))
        if len(matching) != 1:
            return None, f"record source-verified checks for exactly one quoted ID under {item}"
        product_id, checks = matching[0]
        needs_variant = {check.get("variant_index") for check in checks
                         if check.get("status") == "available_unselected"
                         and check.get("evidence_verified") is True}
        selection: dict[str, Any] = {"item": item, "product_id": product_id}
        if product_id in variant_map:
            index = variant_map[product_id]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                return None, "variant_index must be a non-negative integer"
            if needs_variant and index not in needs_variant:
                return None, f"selected variant_index for {product_id} does not match verified option"
            selection["variant_index"] = index
        elif needs_variant:
            return None, f"choose a verified variant_index for {product_id}"
        selections.append(selection)
    if len({selection["product_id"] for selection in selections}) != len(ids):
        return None, "each quoted ID must map to one different requested item"
    return selections, None


def evaluate(submitted: list[str], truth: dict[str, Any], viewed: set[str]) -> dict[str, Any]:
    target_ids = [str(item["product_id"]) for item in truth.get("preferences", [truth])]
    submitted_set, target_set = set(submitted), set(target_ids)
    overlap = len(submitted_set & target_set)
    return {
        "target_count": len(target_set),
        "submitted_count": len(submitted_set),
        "matched_targets": overlap,
        "target_recall": overlap / len(target_set) if target_set else 0.0,
        "exact_product_match": submitted_set == target_set,
        "all_submitted_inspected": bool(submitted_set) and submitted_set <= viewed,
    }
