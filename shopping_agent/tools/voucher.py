"""Public coupon parsing, deterministic pricing and versioned tool feedback.

Legacy nested-budget calls use calculate_public_bundle_v1; the separate-budget
calculator preserves older feedback. calculate_public_bundle is the neutral
feedback variant. ProductQuoteMixin keeps the existing runtime switch.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any
import math
import re

_MONEY = r"\$\s*([0-9]+(?:\.[0-9]+)?)"


_FIELDS = ("scope", "threshold", "threshold_operator", "discount_percent",
           "discount_cap", "budget")


def parse_public_voucher(question: str) -> dict:
    start = question.lower().find("voucher")
    if start < 0:
        raise ValueError("No public voucher clause")
    clause = question[start:].lower()
    threshold = re.search(r"(?:exceeds|over|>|min spend|minimum spend)\s*" + _MONEY, clause)
    cap = re.search(r"(?:capped at|up to|with|cap of|cap)\s*" + _MONEY, clause)
    budget = re.search(r"\bbudget\s*:?\s*" + _MONEY, clause)
    rate = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", clause)
    if not all((threshold, cap, budget, rate)):
        raise ValueError("Unsupported public voucher wording")
    strict = "exceeds" in clause or bool(re.search(r"\bover\s*\$|>\s*\$", clause))
    if not strict and not ("min spend" in clause or "minimum spend" in clause):
        raise ValueError("Missing threshold operator")
    return {
        "voucher_type": "shop" if re.search(r"same[ -]shop|same shop", clause) else "platform",
        "threshold": Decimal(threshold.group(1)),
        "threshold_strict": strict,
        "discount_rate": Decimal(rate.group(1)) / 100,
        "cap": Decimal(cap.group(1)),
        "budget": Decimal(budget.group(1)),
    }


def validate_model_coupon(question: str, proposed: dict) -> dict:
    """Return mismatched field names without revealing the expected values."""
    if not isinstance(proposed, dict):
        return {"valid": False, "error_fields": list(_FIELDS), "unexpected_fields": []}
    public = parse_public_voucher(question)
    expected = {
        "scope": "same_shop" if public["voucher_type"] == "shop" else "platform",
        "threshold": public["threshold"],
        "threshold_operator": ">" if public["threshold_strict"] else ">=",
        "discount_percent": public["discount_rate"] * 100,
        "discount_cap": public["cap"],
        "budget": public["budget"],
    }
    errors = []
    for field in _FIELDS:
        value = proposed.get(field)
        if field in ("scope", "threshold_operator"):
            correct = value == expected[field]
        else:
            try:
                correct = not isinstance(value, bool) and Decimal(str(value)) == expected[field]
            except (InvalidOperation, TypeError, ValueError):
                correct = False
        if not correct:
            errors.append(field)
    unexpected = sorted(set(proposed) - set(_FIELDS))
    return {"valid": not errors and not unexpected,
            "error_fields": errors, "unexpected_fields": unexpected}


def calculate_public_bundle_v1(arguments: dict, question: str, products_by_id: dict) -> dict:
    """Use original listed prices; compare the raw sum to the public threshold."""
    ids = arguments.get("product_ids")
    if not isinstance(ids, list) or not ids or len(set(map(str, ids))) != len(ids):
        return {"error": "product_ids must be a non-empty list of unique IDs"}
    ids = [str(value) for value in ids]
    unknown = [value for value in ids if value not in products_by_id]
    if unknown:
        return {"error": "unknown product_ids", "product_ids": unknown}
    check = validate_model_coupon(question, arguments.get("voucher"))
    if not check["valid"]:
        return {"error": "voucher fields do not match the public request",
                "error_fields": check["error_fields"],
                "unexpected_fields": check["unexpected_fields"]}
    public = parse_public_voucher(question)
    prices = [Decimal(str(products_by_id[value]["product"]["price"])) for value in ids]
    raw_subtotal = sum(prices)
    # Match the released benchmark's rounding only after raw-price eligibility.
    subtotal = raw_subtotal.to_integral_value(rounding=ROUND_CEILING)
    seller_by_product = {value: str(products_by_id[value]["product"].get("seller_id", ""))
                         for value in ids}
    same_shop = len(set(seller_by_product.values())) == 1
    threshold_met = (raw_subtotal > public["threshold"] if public["threshold_strict"]
                     else raw_subtotal >= public["threshold"])
    shop_rule_met = public["voucher_type"] != "shop" or same_shop
    applicable = threshold_met and shop_rule_met
    discount = min(subtotal * public["discount_rate"], public["cap"]) if applicable else Decimal(0)
    final_price = (subtotal - discount).to_integral_value(rounding=ROUND_CEILING)
    return {
        "product_ids": ids,
        "raw_subtotal": float(raw_subtotal), "subtotal": int(subtotal),
        "same_shop": same_shop, "seller_id_by_product": seller_by_product,
        "voucher_type": public["voucher_type"],
        "threshold": float(public["threshold"]), "threshold_met": threshold_met,
        "shop_rule_met": shop_rule_met, "voucher_applicable": applicable,
        "discount": float(discount), "final_price": int(final_price),
        "budget": float(public["budget"]),
        "within_budget": final_price <= public["budget"],
    }


def calculate_public_bundle_separate_budget(arguments: dict, question: str, products_by_id: dict) -> dict:
    voucher = arguments.get("voucher")
    if not isinstance(voucher, dict):
        return {"error": "voucher must be an object"}
    if "budget" in voucher:
        return {"error": "budget is a separate task constraint, not a voucher field",
                "error_fields": ["budget"]}
    # The v1 calculator independently verifies every model field against the
    # public request. Move the separately supplied budget into that verifier's
    # expected shape; never read reward-model metadata here.
    verified_arguments = {
        "product_ids": arguments.get("product_ids"),
        "voucher": {**voucher, "budget": arguments.get("budget")},
    }
    result = calculate_public_bundle_v1(verified_arguments, question, products_by_id)
    if "error" in result:
        return result
    result["voucher_schema"] = "public_coupon_budget_separate_v2"
    if result["voucher_type"] == "shop" and not result["same_shop"]:
        result["next_step"] = (
            "This same-shop voucher cannot apply to sellers in this bundle. "
            "Compare seller_id in observed candidates, or search another item "
            "using a seller_id already seen; inspect any replacement before finishing."
        )
    return result


def enrich_scope_error_feedback(result: dict, arguments: dict, question: str) -> dict:
    """Explain only a mismatch derivable from the public coupon wording."""
    if "error" in result:
        if "scope" in result.get("error_fields", []) and "scope_correction" not in result:
            public = parse_public_voucher(question)
            required = "same_shop" if public["voucher_type"] == "shop" else "platform"
            submitted = (arguments.get("voucher") or {}).get("scope")
            result["scope_correction"] = {
                "submitted": submitted,
                "required_from_public_request": required,
                "public_rule": ("The request explicitly limits this voucher to products from the same shop."
                                if required == "same_shop" else
                                "The request does not state a same-shop restriction."),
            }
            result["next_step"] = (
                f"No quote was recorded. If keeping the same product IDs, call calculate_bundle "
                f"again with voucher.scope={required}; check any other error_fields as well. "
                "The model still chooses whether to keep or change the products."
            )
    return result


def calculate_public_bundle(arguments: dict, question: str, products_by_id: dict) -> dict:
    result = enrich_scope_error_feedback(
        calculate_public_bundle_separate_budget(arguments, question, products_by_id), arguments, question)
    if "error" in result:
        return result
    result.pop("next_step", None)
    result["voucher_schema"] = "public_coupon_budget_facts_v3"
    result["voucher_inapplicable_reasons"] = (
        (["threshold_not_met"] if not result["threshold_met"] else [])
        + (["same_shop_rule_not_met"] if not result["shop_rule_met"] else [])
    )
    return result


def voucher_threshold_is_strict(question: str) -> bool:
    """Read the public voucher wording, never the answer product metadata."""
    clause = question.split("Voucher:", 1)[-1].split("Budget:", 1)[0].lower()
    if "exceeds" in clause or re.search(r"\bover\s*\$|>\s*\$", clause):
        return True
    if "min spend" in clause or "minimum spend" in clause:
        return False
    raise ValueError(f"Unrecognized voucher threshold wording: {clause[:120]}")


class ProductQuoteMixin:
    def calculate(self, arguments: dict[str, Any], task_truth: dict[str, Any]) -> dict[str, Any]:
        ids = arguments.get("product_ids")
        if not isinstance(ids, list) or not ids:
            return {"error": "product_ids must be a non-empty list"}
        ids = [str(value) for value in ids]
        if len(ids) != len(set(ids)):
            return {"error": "product_ids must be unique"}
        unknown = [value for value in ids if value not in self.by_id]
        if unknown:
            return {"error": f"unknown product_ids: {unknown}"}
        expected_count = int(task_truth.get("n", 1))
        if len(ids) != expected_count:
            return {"error": f"This task requires exactly {expected_count} unique product_id(s), got {len(ids)}."}
        products = [self.by_id[value]["product"] for value in ids]
        prices = [float(product["price"]) for product in products]
        raw_subtotal = sum(prices)
        # The released task generator represents bundle totals as the ceiling
        # of the sum of listed decimal prices, and similarly ceilings the
        # post-discount amount. Match that observable task convention.
        subtotal = math.ceil(raw_subtotal - 1e-9)
        voucher = task_truth.get("voucher")
        if not voucher:
            return {
                "product_ids": ids,
                "raw_subtotal": round(raw_subtotal, 2),
                "subtotal": subtotal,
                "final_price": subtotal,
                "within_budget": None,
                "notice": "This is a single-product task; no voucher applies.",
            }
        seller_ids = [str(product.get("seller_id", "")) for product in products]
        seller_by_product = dict(zip(ids, seller_ids))
        same_shop = len(set(seller_ids)) == 1
        # Eligibility uses the public wording and listed prices before the
        # benchmark's ceiling convention.  $14.28 does not "exceed $15".
        threshold = float(voucher["threshold"])
        threshold_met = (raw_subtotal > threshold + 1e-9 if task_truth.get("_voucher_threshold_strict", True)
                         else raw_subtotal + 1e-9 >= threshold)
        shop_rule_met = task_truth.get("voucher_type") != "shop" or same_shop
        applicable = threshold_met and shop_rule_met
        match = re.search(r"([0-9.]+)", str(voucher["discount"]))
        rate = float(match.group(1)) / 100 if match else 0.0
        discount = min(subtotal * rate, float(voucher["cap"])) if applicable else 0.0
        final_price = math.ceil(subtotal - discount - 1e-9)
        return {
            "product_ids": ids,
            "raw_subtotal": round(raw_subtotal, 2),
            "subtotal": subtotal,
            "same_shop": same_shop,
            "seller_id_by_product": seller_by_product,
            "voucher_type": task_truth.get("voucher_type"),
            "threshold": voucher["threshold"],
            "threshold_met": threshold_met,
            "shop_rule_met": shop_rule_met,
            "voucher_applicable": applicable,
            "discount": round(discount, 2),
            "final_price": final_price,
            "budget": voucher["budget"],
            "within_budget": final_price <= float(voucher["budget"]),
        }

    def calculate_public(self, arguments: dict[str, Any], question: str,
                         expected_count: int,
                         neutral_quote_feedback: bool = False) -> dict[str, Any]:
        """Versioned quote path: model reads the coupon; labels supply no coupon fields."""
        if neutral_quote_feedback:
            calculator = calculate_public_bundle
        else:
            calculator = calculate_public_bundle_separate_budget

        if expected_count == 1 and not re.search(r"\bvoucher\b", question, re.I):
            return {"error": "calculate_bundle is unavailable for a single product without a voucher; inspect and finish the product"}
        ids = arguments.get("product_ids")
        if not isinstance(ids, list) or len(ids) != expected_count:
            return {"error": f"This task requires exactly {expected_count} product_id(s)."}
        result = calculator(arguments, question, self.by_id)
        return (result if neutral_quote_feedback else
                enrich_scope_error_feedback(result, arguments, question))
