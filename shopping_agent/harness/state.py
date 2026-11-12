"""harness.state: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from typing import Any


def repeated_action_feedback(name: str, arguments: dict[str, Any],
                             calculations: list[dict[str, Any]],
                             viewable_ids: list[str] | None = None,
                             neutral_quote_feedback: bool = False) -> dict[str, Any]:
    """Refer back to an observed quote without calling it a successful offer."""
    if name == "calculate_bundle":
        ids = [str(value) for value in arguments.get("product_ids", [])]
        quote = next((row for row in reversed(calculations)
                      if sorted(str(value) for value in row.get("product_ids", []))
                      == sorted(ids)), None)
        if quote is not None:
            if neutral_quote_feedback:
                return {
                    "error": "This exact bundle already has a quote.",
                    "prior_quote": {key: quote.get(key) for key in
                                    ("product_ids", "threshold_met", "shop_rule_met",
                                     "voucher_applicable", "voucher_inapplicable_reasons",
                                     "within_budget", "final_price", "budget",
                                     "seller_id_by_product")},
                }
            return {
                "error": "This exact bundle was already quoted. Review its outcome before calculating again.",
                "prior_quote": {key: quote.get(key) for key in
                                ("product_ids", "threshold_met", "shop_rule_met",
                                 "voucher_applicable", "within_budget", "final_price",
                                 "budget", "seller_id_by_product")},
                "next_step": ("The final price is within budget. Check product preferences, "
                              "inspect any missing products, then finish with these selections "
                              "if they meet the request."
                              if quote.get("within_budget")
                              else "The final price exceeds budget. Change the product combination "
                                   "before recalculating; a voucher may lower the price."),
            }
    if name == "view_product":
        return {
            "error": "This product was already inspected and cannot be viewed again.",
            "viewable_product_ids": (viewable_ids or [])[:10],
            "next_step": ("Use a different uninspected candidate ID, or search for another requested item. "
                          "A candidate marked × for a required feature cannot satisfy that item."
                          if viewable_ids else
                          "Search for a different candidate or calculate/finish using already inspected products."),
        }
    if name == "search_products":
        return {"error": "This search already returned candidates; inspect one or change the query."}
    return {"error": "This identical action already completed; choose a different next action."}


def viewable_candidate_ids(candidates: dict[str, dict[str, Any]],
                           viewed: dict[str, dict[str, Any]]) -> list[str]:
    """Only a search-result product not yet inspected can be opened."""
    return [product_id for product_id in candidates if product_id not in viewed]


def uninspected_record_check_feedback(product_id: str,
                                      candidates: dict[str, dict[str, Any]],
                                      viewed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reject an unsupported check and point only to model-visible next actions."""
    in_search_results = product_id in candidates
    return {
        "error": "record_check requires an inspected product_id",
        "attempted_product_id": product_id,
        "viewed_product_ids": list(viewed)[-8:],
        "next_step": (
            f"No checks were saved for {product_id}. It is only a search candidate. "
            f"Call view_product with product_id={product_id} first; then cite fields "
            "from that page in record_check. You can instead finish an already "
            "verified inspected product if it meets the public request."
            if in_search_results else
            f"No checks were saved for {product_id}. This ID is not in the retained "
            "search results or opened product pages. Search for it first, or finish "
            "an already verified inspected product if it meets the public request."
        ),
    }
