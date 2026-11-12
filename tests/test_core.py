"""Small CPU-only release check: python -m pytest tests/test_core.py.

All products/dialogues below are synthetic fixtures. Full historical regression
tests are kept locally, outside the publication package.
"""
import json

from shopping_agent.data.catalog import ProductStore
from shopping_agent.tools.voucher import calculate_public_bundle_v1 as calculate_public_bundle
from shopping_agent.tools.finish import selections_from_quoted_ids
from shopping_agent.tools.requirements import (
    merge_constraint_updates, requested_bundle_items, user_quote_is_visible,
)
from shopping_agent.harness.context import state_text
from shopping_agent.harness.protocol import parse_native_tool_call
from shopping_agent.harness.progress import recent_useful_progress
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v12



def test_category_is_only_a_soft_search_hint():
    class Index:
        def get_scores(self, words):
            return [2.0, 1.0, 0.0] if words == ["fan", "cover"] else [0.0, 5.0, 30.0]

    store = ProductStore.__new__(ProductStore)
    store.rows = [
        {"id": "cover", "product": {"product_name": "fan cover", "price": 3}, "contents": "fan cover"},
        {"id": "fan", "product": {"product_name": "fan", "price": 4}, "contents": "fan"},
        {"id": "wrong", "product": {"product_name": "heater", "price": 5}, "contents": "heater"},
    ]
    store.index = Index()
    result = store.search({"query": "fan cover", "filters": {"category": "appliances"}, "top_k": 3})
    assert [item["product_id"] for item in result["results"]] == ["cover", "fan"]


def test_native_tool_call_requires_one_complete_valid_call():
    valid = '<tool_call>\n{"name":"view_product","arguments":{"product_id":"123"}}\n</tool_call>'
    assert parse_native_tool_call(valid)["action"]["arguments"]["product_id"] == "123"
    assert parse_native_tool_call(valid + valid) is None
    assert parse_native_tool_call('<tool_call>{"name":"view_product","arguments":{"product_id":"123}}}</tool_call>') is None


def test_slot_evidence_checks_only_released_user_words():
    docs = [{"user_evidence_text": "High waist is wanted. I find it stays put better.\n"
             "Spandex works well for me—it stretches on runs."}]
    assert user_quote_is_visible("high waist is wanted", docs)
    assert user_quote_is_visible("Spandex works well for me", docs)
    assert not user_quote_is_visible("the assistant recommends silk", docs)
    assert not user_quote_is_visible("high waist", docs)


def test_sparse_updates_preserve_other_items_and_verify_visible_evidence():
    question = ("Product bundle includes: solar light, belt, and shoulder bag. "
                "Voucher applies to same-shop products only, valid when total exceeds $9.")
    assert requested_bundle_items(question) == ["solar light", "belt", "shoulder bag"]
    initial = [
        {"item": "solar light", "requirement": "solar-powered and dimmable", "evidence": []},
        {"item": "belt", "requirement": "black leather", "evidence": ["Black is a must"]},
        {"item": "shoulder bag", "requirement": "leather with laptop pocket", "evidence": []},
    ]
    viewed = {"p1": {"product_id": "p1", "name": "Genuine Leather Belt",
                      "attributes": {"belt_material": ["Leather"]},
                      "options": [{"color_family": ["Brown"]}, {"color_family": ["Black"]}]}}
    persisted = merge_constraint_updates([], initial, viewed)
    updated = merge_constraint_updates(persisted, [{
        "item": "belt", "product_id": "p1", "checks": [
            {"need": "Leather", "status": "supported",
             "source": "attributes.belt_material", "value": "Leather"},
            {"need": "Black", "status": "supported",
             "source": "options[1].color_family", "value": "Black"},
        ]}], viewed)
    assert len(updated) == 3
    assert [item["item"] for item in updated] == ["solar light", "belt", "shoulder bag"]
    assert updated[1]["requirement"] == "black leather"
    assert updated[1]["evidence"] == ["Black is a must"]
    checks = updated[1]["candidate_checks"]["p1"]
    assert checks[0]["status"] == "supported" and checks[0]["evidence_verified"]
    assert checks[1]["status"] == "available_unselected" and checks[1]["evidence_verified"]
    assert persisted[1]["candidate_checks"] == {}

    multiple_needs = merge_constraint_updates([], [
        {"item": "belt", "requirement": "Black color", "evidence": ["Black is a must"]},
        {"item": "belt", "requirement": "Leather material", "evidence": ["Leather is wanted"]},
    ], viewed)
    assert multiple_needs[0]["requirement"] == "Black color; Leather material"
    assert multiple_needs[0]["evidence"] == ["Black is a must", "Leather is wanted"]
    repeated_summary = merge_constraint_updates(multiple_needs, [
        {"item": "belt", "requirement": "Black color; Leather material"}], viewed)
    assert repeated_summary[0]["requirement"] == "Black color; Leather material"

    state = state_text(question, [], updated, {}, viewed, [], None,
                       "add_on_deals", 3, 20, stage2_context=True, compact=True,
                       minimal=True)
    assert '"status": "option_available"' in state
    assert '"item_submitted": false' in state
    assert '"option_index_for_finish": 1' in state
    assert "Opened product details (viewed pages; viewing alone does not verify required features):" in state
    assert "Required-feature checks by requested item and product ID" in state
    assert 'Selectable specifications (available options, not submitted items; use original variant_index in finish): [{"item": "belt", "product_id": "p1", "variant_index_for_finish": 1}]' in state
    assert "Public requested items missing from checklist: []" in state


def test_finish_shortcut_requires_explicit_verified_variant_index():
    constraints = [{"item": "belt", "candidate_checks": {"belt1": [
        {"status": "available_unselected", "evidence_verified": True,
         "variant_index": 1}]}}]
    viewed = {"belt1": {"product_id": "belt1"}}
    quotes = [{"product_ids": ["belt1"], "within_budget": True}]
    args = {"quoted_product_ids": ["belt1"]}
    assert selections_from_quoted_ids(args, constraints, ["belt"], viewed, quotes)[0] is None
    args["variant_indices_by_product"] = {"belt1": 0}
    assert selections_from_quoted_ids(args, constraints, ["belt"], viewed, quotes)[0] is None
    args["variant_indices_by_product"] = {"belt1": 1}
    selections, error = selections_from_quoted_ids(args, constraints, ["belt"], viewed, quotes)
    assert error is None
    assert selections[0]["variant_index"] == 1


def test_atomic_evidence_slots_do_not_inherit_a_composite_check_on_replacement():
    question = "Product bundle includes: Grinding wheel and Voltage regulator."
    constraints = [{
        "item": "Grinding wheel", "requirement": "intended for wood grinding and shaping",
        "evidence": ["User confirms \"power_tool_material_uses: wood\".",
                     "User confirms \"power_tool_accessory_use: Grinding & Sharpening\"."],
        "candidate_checks": {
            "old": [{"need": "intended for wood grinding and shaping", "status": "supported",
                     "source": "attributes.power_tool_material_uses", "value": "wood",
                     "evidence_verified": True},
                    {"need": "Grinding & Sharpening", "status": "supported",
                     "source": "attributes.power_tool_accessory_use",
                     "value": "Grinding & Sharpening", "evidence_verified": True}],
            "new": [{"need": "intended for wood grinding and shaping", "status": "supported",
                     "source": "attributes.power_tool_material_uses", "value": "wood",
                     "evidence_verified": True}],
        },
    }]
    viewed = {pid: {"product_id": pid, "search_query": "Grinding wheel",
                    "name": "Grinding wheel", "attributes": {}, "options": []}
              for pid in ("old", "new")}
    state = state_text(question, [], constraints, {}, viewed, [], None,
                       "add_on_deals", 10, 20, stage2_context=True,
                       atomic_evidence_slots=True)
    assert '"id": "old"' in state and '"preference_gate": "recorded_checks_pass"' in state
    assert '"id": "new"' in state and '"preference_gate": "incomplete_checks"' in state
    assert '"unresolved_model_declared_fields_for_this_id": ["power_tool_accessory_use: Grinding & Sharpening"]' in state


def test_single_product_cannot_crash_public_voucher_calculator():
    store = ProductStore.__new__(ProductStore)
    result = store.calculate_public({"product_ids": ["123"], "voucher": {}},
                                    "Current Date: 2025/01/01\nmosquito killer lamp", 1)
    assert "error" in result
    assert "single product" in result["error"]


def test_quote_uses_raw_prices_for_strict_threshold_and_rejects_cross_shop():
    question = ("Voucher: applies only to products from the same shop; "
                "valid when total price exceeds $15; 10% off capped at $10. Budget: $14.")
    voucher = {"scope": "same_shop", "threshold": 15, "threshold_operator": ">",
               "discount_percent": 10, "discount_cap": 10, "budget": 14}
    products = {
        "a": {"product": {"price": 7.14, "seller_id": "one"}},
        "b": {"product": {"price": 7.14, "seller_id": "one"}},
        "c": {"product": {"price": 9.00, "seller_id": "two"}},
    }
    below = calculate_public_bundle({"product_ids": ["a", "b"], "voucher": voucher},
                                    question, products)
    assert below["raw_subtotal"] == 14.28
    assert below["subtotal"] == 15
    assert below["threshold_met"] is False
    assert below["voucher_applicable"] is False
    cross = calculate_public_bundle({"product_ids": ["a", "c"], "voucher": voucher},
                                    question, products)
    assert cross["threshold_met"] is True
    assert cross["shop_rule_met"] is False
    assert cross["discount"] == 0


def test_quote_rejects_wrong_model_coupon_without_revealing_public_values():
    question = "Voucher: total exceeds $15; 20% off capped at $7. Budget: $13."
    products = {"a": {"product": {"price": 20, "seller_id": "one"}}}
    wrong = {"scope": "platform", "threshold": 15, "threshold_operator": ">=",
             "discount_percent": 20, "discount_cap": 7, "budget": 13}
    result = calculate_public_bundle({"product_ids": ["a"], "voucher": wrong},
                                     question, products)
    assert result["error_fields"] == ["threshold_operator"]
    assert "15" not in str(result)


def progress_event(step, name, arguments, observation):
    return {"step": step, "action": {"name": name, "arguments": arguments},
            "observation": observation}


def test_repeated_or_invalid_actions_do_not_extend():
    trace = [
        progress_event(17, "search_products", {"query": "belt"},
              {"results": [{"product_id": "a"}]}),
        progress_event(18, "view_product", {"product_id": "a"},
              {"error": "already inspected"}),
        progress_event(19, "search_products", {"query": "belt"},
              {"results": [{"product_id": "a"}]}),
        progress_event(20, "calculate_bundle", {"product_ids": ["a"]},
              {"error": "already quoted"}),
    ]
    assert recent_useful_progress(trace, 20) == []


def event(name, arguments, observation):
    return {"action": {"name": name, "arguments": arguments}, "observation": observation}


def test_v12_checks_public_item_use_without_reference_category_gate():
    def single(item, title, category, feature, product_id="alternative"):
        episode = {"question_type": "single_product", "question":
                   f"Current Date: 2025/07/05 (Sat)\n{item}",
                   "finish_reason": "model_finish", "submitted_product_ids": [product_id],
                   "trajectory": [event("view_product", {"product_id": product_id}, {
                       "product_id": product_id, "name": title, "category": category,
                       "attributes": {feature.split(":")[0]: [feature.split(":")[1].strip()]},
                       "options": []})], "provided_memories": [], "final_constraints": []}
        kwargs = {"truth_items": [{"product_id": "different_reference",
                                    "category": "unrelated_reference_category",
                                    "wanted_features": [feature]}],
                  "source_products_by_id": {product_id: {"product": {}}},
                  "expected_memory_ids": ["history"]}
        return score_episode_grounded_open_world_v12(episode, **kwargs)

    vacuum = single("leaf blower", "Mite removal vacuum cleaner Cordless",
                    "Household Vacuum Cleaners", "cordless: Cordless")
    blower = single("leaf blower", "Portable Leaf Blower Cordless",
                    "Power Tools", "cordless: Cordless")
    coffee = single("coffee", "Arabica Coffee Powder", "Groceries - Coffee",
                    "ingredient: coffee")
    assert vacuum["valid_finish"] and not vacuum["complete"]
    assert vacuum["reward"] < 1.0
    assert vacuum["public_item_fit_v12"][0]["missing_terms"] == ["leaf", "blower"]
    assert blower["complete"] and blower["reward"] == 1.0
    assert coffee["complete"] and coffee["reward"] == 1.0
