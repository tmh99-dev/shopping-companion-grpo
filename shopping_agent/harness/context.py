"""harness.context: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from shopping_agent.harness.prompts import STAGE2_HISTORY_ID
from shopping_agent.harness.state import viewable_candidate_ids
from shopping_agent.tools.requirements import _model_declared_field_values
from shopping_agent.tools.requirements import _normalized_check_text
from shopping_agent.tools.requirements import has_repeated_public_item_types
from shopping_agent.tools.requirements import missing_public_requirement_slots
from shopping_agent.tools.requirements import requested_bundle_items
from shopping_agent.tools.requirements import unused_public_history_for_slots
from shopping_agent.tools.search import terms
from typing import Any
import itertools
import json
import re


def stage2_history_document(row: dict[str, Any]) -> dict[str, str]:
    """Use only the released Stage-2 input, never reward-side annotations."""
    messages = row.get("prompt") or []
    matches = [str(message.get("content", "")) for message in messages
               if message.get("role") == "user" and str(message.get("content", "")).startswith(
                   "The most relevant user dialogue memories are as follows:")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one released Stage-2 history message")
    raw = matches[0]
    user_lines = [line[5:].strip() for line in raw.splitlines()
                  if line.lower().startswith("user:")]
    if not user_lines:
        raise ValueError("Stage-2 history has no user statements to verify preferences")
    return {"id": STAGE2_HISTORY_ID, "text": raw,
            "user_evidence_text": "\n".join(user_lines)}


def task_card(question_type: str) -> str:
    if question_type == "add_on_deals":
        return ("Task type: bundle shopping. Select one product per requested type and verify its preferences. "
                "Voucher form A, platform: without explicit same-shop wording, products may have different sellers. "
                "Voucher form B, same-shop: only explicit same-shop wording requires one seller for the voucher. "
                "Check the quoted threshold and final budget for the selected products. A bundle can fit the "
                "budget even when its voucher is inapplicable. Do not impose a seller_id filter for a platform "
                "voucher; after an empty same-shop seller search, change or remove that filter.")
    return "Task type: single-product shopping. Select exactly one inspected product that satisfies the known requirements."


def state_text(question: str, memories: list[dict[str, str]], constraints: list[dict[str, Any]],
               candidates: dict[str, dict[str, Any]], viewed: dict[str, dict[str, Any]],
               recent_actions: list[dict[str, Any]], last_observation: dict[str, Any] | None,
               question_type: str, step: int, max_steps: int,
               stage2_context: bool = False, compact: bool = False,
               successful_calculations: list[dict[str, Any]] | None = None,
               minimal: bool = False, tight: bool = False,
               check_as_tool: bool = False,
               history_as_checklist: bool = False,
               native_tools: bool = False, quote_focus: bool = False,
               bundle_decision_state: bool = False,
               neutral_quote_feedback: bool = False,
               bundle_choice_focus: bool = False,
               bundle_loop_focus: bool = False,
               quote_evidence_gaps: bool = False,
               atomic_evidence_slots: bool = False,
               quote_focus_on_replacement: bool = False,
               ) -> str:
    if (quote_focus and stage2_context and successful_calculations
            and last_observation is successful_calculations[-1]
            and (not quote_focus_on_replacement or
                 (len(successful_calculations) >= 2 and
                  set(map(str, successful_calculations[-1].get("product_ids", []))) !=
                  set(map(str, successful_calculations[-2].get("product_ids", [])))))):
        # After a quote, put the exact quoted choice and its observed evidence
        # in one small decision view. This is a display of model-visible state,
        # not a recommendation derived from reference products or reward labels.
        quote = successful_calculations[-1]
        quoted_ids = [str(value) for value in quote.get("product_ids", [])]
        quoted = []
        for product_id in quoted_ids:
            product = viewed.get(product_id, {})
            checks = [
                {"item": item.get("item"), "checks": (item.get("candidate_checks") or {}).get(product_id, [])}
                for item in constraints if isinstance(item, dict)
                and product_id in (item.get("candidate_checks") or {})
            ]
            quoted.append({"product_id": product_id, "name": product.get("name"),
                           "seller_id": product.get("seller_id"), "price": product.get("price"),
                           "recorded_checks": checks})
            if quote_evidence_gaps:
                field_checks = []
                for item in constraints:
                    if (not isinstance(item, dict) or product_id not in
                            (item.get("candidate_checks") or {})):
                        continue
                    for field, wanted in _model_declared_field_values(item):
                        attribute = next((value for key, value in
                                          (product.get("attributes") or {}).items()
                                          if key.casefold() == field.casefold()), None)
                        values = attribute if isinstance(attribute, list) else [attribute]
                        if any(value is not None and
                               _normalized_check_text(value) == _normalized_check_text(wanted)
                               for value in values):
                            status = "observed_attribute"
                            option_indices = []
                        else:
                            option_indices = [index for index, option in
                                              enumerate(product.get("options") or [])
                                              if isinstance(option, dict)
                                              for key, value in option.items()
                                              if key.casefold() == field.casefold()
                                              and any(_normalized_check_text(candidate) ==
                                                      _normalized_check_text(wanted)
                                                      for candidate in
                                                      (value if isinstance(value, list) else [value]))]
                            status = ("available_option_not_selected" if option_indices else
                                      "missing_from_this_product_page")
                        field_checks.append({"item": item.get("item"),
                                             "field": field, "wanted": wanted,
                                             "status": status,
                                             **({"original_variant_indices": option_indices}
                                                if option_indices else {})})
                quoted[-1]["user_confirmed_field_checks_for_this_id"] = field_checks
        requirements = [{"item": item.get("item"), "requirement": item.get("requirement"),
                         "user_quote": str((item.get("evidence") or [""])[0])[:120]}
                        for item in constraints if isinstance(item, dict)]
        other_ids = [str(product_id) for product_id in viewed if product_id not in quoted_ids]
        feedback = last_observation or {}
        return f"""CURRENT REQUEST
{question}

MODEL-AUTHORED REQUIREMENTS FROM THE USER HISTORY
{json.dumps(requirements, ensure_ascii=False)}

POST-QUOTE DECISION STATE (all checks are the model's claims; source fields and final reward are verified separately)
Step: {step}/{max_steps}
Exact quoted products and their recorded checks: {json.dumps(quoted, ensure_ascii=False)}
Quote result: {json.dumps(quote, ensure_ascii=False)}
Other inspected product IDs: {json.dumps(other_ids[-15:], ensure_ascii=False)}
Uninspected candidate IDs: {json.dumps(viewable_candidate_ids(candidates, viewed)[:10], ensure_ascii=False)}
Last tool feedback: {json.dumps(feedback, ensure_ascii=False)}

Decide now: if this exact quoted bundle satisfies the requested product types and your recorded preferences, and the quote is within budget, call finish with exactly one inspected product per requested type and the original variant_index for selected options. A voucher can be inapplicable while the purchase is still within budget. If a requirement is missing or the quote is over budget, search or inspect a different candidate and quote the changed bundle. Do not inherit a previous product's checks when replacing it. Choose exactly one native tool call."""
    memory_text = "\n\n".join(
        f"[{doc['id']}]\n{doc.get('user_evidence_text', doc['text']) if minimal and stage2_context else doc['text']}"
        for doc in memories) or "(none)"
    if stage2_context and has_repeated_public_item_types(question):
        missing_slots = missing_public_requirement_slots(question, constraints)
        uncited_history = unused_public_history_for_slots(
            memories, constraints, len(missing_slots))
        if uncited_history is not None:
            memory_text = uncited_history
    memory_label = ("HISTORICAL USER STATEMENTS (assistant turns omitted for context budget)"
                    if minimal and stage2_context else
                    "RELEVANT HISTORICAL DIALOGUE PROVIDED WITH THIS STAGE-2 TASK"
                    if stage2_context else "MEMORY EXCERPTS RETRIEVED BY YOUR search_memory CALLS")
    if (history_as_checklist and stage2_context and constraints and not
            (has_repeated_public_item_types(question) and
             missing_public_requirement_slots(question, constraints))):
        memory_text = "\n".join(
            f"{item.get('item')}: {item.get('requirement')} | user quote: "
            f"{str((item.get('evidence') or [''])[0])[:160]}"
            for item in constraints[:10] if isinstance(item, dict))
        memory_label = "MODEL-AUTHORED SUMMARY OF THE HISTORY ALREADY READ ON THE FIRST TURN"
    relevant_words = set(terms(question + " " + " ".join(
        str(item.get("requirement", "")) for item in constraints if isinstance(item, dict))))
    same_shop_required = bool(re.search(r"\bsame(?:-|\s+)shop\b", question, re.I))
    viewed_rows = []
    # Persist every inspected product. A bundle can involve several searches;
    # dropping earlier views made the agent forget items it had already checked.
    viewed_values = list(viewed.values())
    for view_index, value in enumerate(viewed_values):
        if tight and stage2_context and view_index < len(viewed_values) - 3:
            viewed_rows.append({"product_id": value.get("product_id"),
                                "name": str(value.get("name", ""))[:35],
                                "detail": "Older details condensed; see saved checks below."})
            continue
        item = {k: (value.get(k)[:45 if compact and stage2_context else 90]
                    if k == "name" and isinstance(value.get(k), str)
                    else value.get(k)[:22]
                    if k == "search_query" and compact and stage2_context
                    and isinstance(value.get(k), str)
                    else value.get(k))
                for k in ("product_id", "search_query", "name", "category", "price", "seller_id", "attributes", "options")}
        if compact and stage2_context:
            attributes = item.get("attributes") or {}
            item["attributes"] = {key: raw for key, raw in attributes.items()
                                  if relevant_words & set(terms(f"{key} {raw}"))}
            options = item.get("options") or []
            relevant_options = []
            for variant_index, option in enumerate(options):
                if not isinstance(option, dict):
                    continue
                fields = {key: raw for key, raw in option.items()
                          if relevant_words & set(terms(f"{key} {raw}"))}
                if fields:
                    relevant_options.append({"variant_index": variant_index, "fields": fields})
            item["options"] = relevant_options
            if tight:
                # Large original product pages can expose hundreds of size or
                # color variants. Rank by the model's visible request/checklist,
                # never by hidden target IDs, and retain original variant indices.
                def value_words(raw):
                    if isinstance(raw, (list, tuple)):
                        return set(terms(" ".join(map(str, raw))))
                    return set(terms(str(raw)))

                def brief_value(raw):
                    if isinstance(raw, (list, tuple)):
                        matches = [value for value in raw
                                   if relevant_words & value_words(value)]
                        chosen = (matches or list(raw))[:5]
                        return {"values": [str(value)[:65] for value in chosen],
                                "total_values": len(raw)}
                    return str(raw)[:110]

                scored_attributes = sorted(
                    item["attributes"].items(),
                    key=lambda pair: len(relevant_words & value_words(pair[1])),
                    reverse=True,
                )
                item["attributes"] = {
                    key: brief_value(raw) for key, raw in scored_attributes[:8]
                }
                scored_options = sorted(
                    relevant_options,
                    key=lambda option: len(relevant_words & value_words(
                        list(option["fields"].values()))),
                    reverse=True,
                )
                item["options_total"] = len(options)
                item["options"] = [
                    {"variant_index": option["variant_index"],
                     "fields": {key: brief_value(raw)
                                for key, raw in list(option["fields"].items())[:6]}}
                    for option in scored_options[:8]
                ]
        viewed_rows.append(item)
    candidate_keys = (("product_id", "search_query", "name", "price", "seller_id")
                      if compact and stage2_context and same_shop_required else
                      ("product_id", "search_query", "name", "price") if compact and stage2_context
                      else ("product_id", "search_query", "name", "category", "price", "seller_id")
                      if stage2_context else
                      ("product_id", "name", "category", "price", "seller_id", "matched_fields"))
    if stage2_context:
        # Keep the leading candidates from *each* search, rather than only the
        # final search (which is usually the last item in a bundle).
        grouped: dict[str, list[dict[str, Any]]] = {}
        for value in candidates.values():
            if str(value.get("product_id", "")) in viewed:
                continue
            query = str(value.get("search_query", ""))
            grouped.setdefault(query, []).append(value)
        retained_groups = list(grouped.values())[-3:] if tight else grouped.values()
        retained = [value for group in retained_groups for value in group[:2 if tight else 3]]
    else:
        retained = list(candidates.values())[-8:]
    candidate_rows = [{k: (value.get(k)[:35 if compact and stage2_context else 70]
                           if k == "name" and isinstance(value.get(k), str) else value.get(k))
                       for k in candidate_keys}
                      for value in retained]
    shopping_progress = []
    if stage2_context:
        # Group only model-visible searches, views and model-authored checks.
        # This is a navigation index, not an inferred preference match.
        requested = requested_bundle_items(question)
        item_slots = [(index + 1, item) for index, item in enumerate(requested)]
        if not item_slots:
            item_slots = [(index + 1, str(c.get("item", "")))
                          for index, c in enumerate(constraints)
                          if isinstance(c, dict) and c.get("item")]
        for slot_index, item_name in item_slots[:10]:
            words = set(terms(item_name))
            related = [value for value in candidates.values()
                       if words and words <= set(terms(str(value.get("search_query", ""))))]
            matching_views = [value for value in viewed.values()
                              if words and words <=
                              set(terms(str(value.get("search_query", ""))))]
            item_constraint = next((c for c in constraints if isinstance(c, dict) and
                                    (c.get("slot_index") == slot_index or
                                     (c.get("slot_index") is None and
                                      requested.count(item_name) <= 1)) and
                                    str(c.get("item", "")).casefold() == item_name.casefold()), {})
            checks_by_id = item_constraint.get("candidate_checks") or {}
            declared_fields = (_model_declared_field_values(item_constraint)
                               if atomic_evidence_slots else [])
            requirement_text = str(item_constraint.get("requirement", ""))
            # Model-authored prose may contain several explicit field needs.
            # Split only at field boundaries, not descriptive phrases such as
            # "grip and gentleness" within one need.
            if re.search(r"\b[a-zA-Z_][\w]*\s*:", requirement_text):
                requirement_parts = re.split(
                    r"\s*,?\s+and\s+(?=[a-zA-Z_][\w]*\s*:)", requirement_text)
            else:
                requirement_parts = re.split(r"\s+and\s+", requirement_text)
            required_parts = [part.strip() for section in requirement_parts
                              for part in section.split(";") if part.strip()]
            matching_views.extend(value for value in viewed.values()
                                  if str(value.get("product_id", "")) in checks_by_id and
                                  value not in matching_views)
            inspected = []
            for value in matching_views:
                product_id = str(value.get("product_id", ""))
                product_checks = [check for check in checks_by_id.get(product_id, [])[:5]
                                  if isinstance(check, dict)]
                generic_words = {"a", "an", "and", "color", "feature", "is", "material",
                                 "of", "shape", "the", "type", "wanted", "with",
                                 "are", "required", "require", "must", "have", "need",
                                 "needs", "should", "be", "for"}
                def content_words(text):
                    return set(terms(str(text))) - generic_words

                def labeled_need(text):
                    match = re.match(r"\s*([a-zA-Z_][\w]*)\s*:\s*(.+)", str(text))
                    if not match:
                        return None
                    value = re.split(r"\s+(?:for|because|so that)\s+", match.group(2),
                                     maxsplit=1)[0]
                    return match.group(1).casefold(), content_words(value)

                def declared_required(check):
                    labeled_parts = [labeled_need(part) for part in required_parts]
                    if any(labeled_parts):
                        label = labeled_need(check.get("need", ""))
                        return bool(label and any(
                            part and label[0] == part[0] and part[1] <= label[1]
                            for part in labeled_parts))
                    need_words = content_words(check.get("need", ""))
                    return bool(need_words and any(need_words & content_words(part)
                                                   for part in required_parts))

                def possible_visible_support(check):
                    need_words = content_words(check.get("need", ""))
                    if not need_words:
                        return False
                    visible_values = [value.get("name", ""), value.get("description_excerpt", "")]
                    visible_values.extend((value.get("attributes") or {}).values())
                    visible_values.extend(field_value for option in (value.get("options") or [])
                                          if isinstance(option, dict)
                                          for field_value in option.values())
                    return any(need_words <= content_words(visible)
                               for visible in visible_values)

                def supports_declared_field(check, field, wanted):
                    source = str(check.get("source", ""))
                    source_match = re.fullmatch(
                        r"(?:attributes\.([A-Za-z_]\w*)|options\[\d+\]\.([A-Za-z_]\w*))",
                        source)
                    return bool(source_match and
                                (source_match.group(1) or source_match.group(2)).casefold()
                                == field.casefold() and
                                check.get("evidence_verified") is True and
                                check.get("status") in {"supported", "available_unselected"} and
                                _normalized_check_text(check.get("value", "")) ==
                                _normalized_check_text(wanted))

                required_checks = [check for check in product_checks if
                                   (any(supports_declared_field(check, field, wanted)
                                        for field, wanted in declared_fields)
                                    if declared_fields else declared_required(check))]
                failed = any(check.get("status") == "contradicted" or
                             (check.get("status") == "unknown" and
                              not possible_visible_support(check))
                             for check in required_checks)
                recheck = any(check.get("status") == "unknown" and
                              possible_visible_support(check) for check in required_checks)
                supported_words = set().union(*(
                    content_words(check.get("need", "")) for check in required_checks
                    if check.get("status") in {"supported", "available_unselected"}
                    and check.get("evidence_verified") is True))
                supported_checks = [check for check in required_checks
                                    if check.get("status") in
                                    {"supported", "available_unselected"}
                                    and check.get("evidence_verified") is True]
                passed = (all(any(supports_declared_field(check, field, wanted)
                                  for check in product_checks)
                              for field, wanted in declared_fields)
                          if declared_fields else bool(required_parts) and all(
                    (any(labeled_need(check.get("need", "")) and
                         labeled_need(check.get("need", ""))[0] == labeled_need(part)[0] and
                         labeled_need(part)[1] <=
                         labeled_need(check.get("need", ""))[1]
                         for check in supported_checks)
                     if labeled_need(part) else
                     bool(content_words(part) and content_words(part) <= supported_words))
                    for part in required_parts))
                unresolved_fields = [f"{field}: {wanted}" for field, wanted in declared_fields
                                     if not any(supports_declared_field(check, field, wanted)
                                                for check in product_checks)]
                needs_variant_choice = any(check.get("status") == "available_unselected"
                                           for check in required_checks)
                inspected.append({
                    "id": product_id, "seller": value.get("seller_id"),
                    "price": value.get("price"), "item_submitted": False,
                    "checks": [{"need": check.get("need"),
                                "status": ("option_available" if check.get("status") == "available_unselected"
                                           else check.get("status")),
                                "mark": ({"supported": "✓", "available_unselected": "✓",
                                          "contradicted": "×", "unknown":
                                          "?" if possible_visible_support(check) else "×"}
                                         .get(check.get("status"), "?"))
                                if check in required_checks else "—",
                                **({"option_index_for_finish": check.get("variant_index")}
                                   if check.get("status") == "available_unselected" else {})}
                               for check in product_checks],
                    **({"unresolved_model_declared_fields_for_this_id": unresolved_fields}
                       if declared_fields else {}),
                    "preference_gate": "fails_required_check" if failed else
                                       "recheck_saved_evidence" if recheck else
                                       "ready_to_submit_with_variant" if passed and needs_variant_choice else
                                       "recorded_checks_pass" if passed else "incomplete_checks",
                })
            # Keep a previously verified candidate visible when later browsing
            # would otherwise push it out of the short progress window.
            # The candidate was checked against a page the model actually saw.
            inspected_limit = 5
            if len(inspected) > inspected_limit:
                recent_inspected = inspected[-inspected_limit:]
                ready_inspected = [entry for entry in inspected if entry["preference_gate"] in
                                   {"recorded_checks_pass", "ready_to_submit_with_variant"}]
                if ready_inspected and ready_inspected[-1]["id"] not in {
                        entry["id"] for entry in recent_inspected}:
                    retained_ids = {ready_inspected[-1]["id"]} | {
                        entry["id"] for entry in recent_inspected[-(inspected_limit - 1):]}
                    inspected = [entry for entry in inspected if entry["id"] in retained_ids]
                else:
                    inspected = recent_inspected
            retained_ids = {entry["id"] for entry in inspected}
            shopping_progress.append({
                "item": item_name,
                "slot_index": slot_index,
                "inspected": inspected,
                "other_inspected_ids": [str(value.get("product_id", ""))
                                        for value in matching_views
                                        if str(value.get("product_id", "")) not in retained_ids],
                "uninspected_candidate_ids": [str(value.get("product_id", ""))
                                              for value in related
                                              if str(value.get("product_id", "")) not in viewed][:5],
            })
    # A short public-state navigation summary near the next action prevents
    # long inspected-product pages from obscuring which requested type is still
    # missing. It does not assert that any candidate meets a preference.
    selectable_options = [
        {"item": row["item"], "product_id": entry["id"],
         "variant_index_for_finish": check["option_index_for_finish"]}
        for row in shopping_progress for entry in row["inspected"]
        for check in entry["checks"] if "option_index_for_finish" in check
        and check.get("mark") == "✓"
    ]
    requested_progress = [row for row in shopping_progress
                          if not requested or row["item"] in requested] if stage2_context else []
    def slot_key(row):
        return (str(row["slot_index"]) if requested.count(row["item"]) > 1
                else row["item"])

    def constraint_for_slot(row):
        return next((c for c in constraints if isinstance(c, dict) and
                     (c.get("slot_index") == row.get("slot_index") or
                      (c.get("slot_index") is None and
                       requested.count(row["item"]) <= 1)) and
                     str(c.get("item", "")).casefold() == row["item"].casefold()), {})

    missing_inspected = [{"slot_index": row["slot_index"], "item": row["item"]}
                         for row in requested_progress if not row["inspected"]]
    ready_by_item = {
        slot_key(row): [entry["id"] for entry in row["inspected"]
                      if entry["preference_gate"] in {"recorded_checks_pass", "ready_to_submit_with_variant"}]
        for row in requested_progress
    }
    missing_verified = [row for row in requested_progress
                        if row["inspected"] and not ready_by_item[slot_key(row)]]
    latest_quote = (successful_calculations or [])[-1] if successful_calculations else None
    quoted_ids = [str(value) for value in latest_quote.get("product_ids", [])] if latest_quote else []
    if (bundle_loop_focus and stage2_context and question_type == "add_on_deals"
            and requested_progress and viewed
            and all(constraint_for_slot(row).get("requirement")
                    for row in requested_progress)
            and not (check_as_tool and isinstance(last_observation, dict)
                                            and "product_id" in last_observation
                                            and "attributes" in last_observation)):
        # A compact, evidence-bound loop state. It does not choose products or
        # infer matches from labels; the model's verified checks remain separate
        # for every product ID.
        slots = []
        for row in requested_progress:
            item_constraint = constraint_for_slot(row)
            checks_by_id = item_constraint.get("candidate_checks") or {}
            slots.append({
                "item": row["item"], "slot_index": row["slot_index"],
                "requirement": item_constraint.get("requirement"),
                "verified_candidate_ids": ready_by_item[slot_key(row)],
                "inspected": [{
                    "product_id": entry["id"], "seller_id": entry["seller"],
                    "list_price": entry["price"], "gate": entry["preference_gate"],
                    "checks": [{"need": check.get("need"),
                                "status": check.get("status"),
                                "evidence_verified": check.get("evidence_verified")}
                               for check in checks_by_id.get(entry["id"], [])
                               if isinstance(check, dict)]
                } for entry in row["inspected"]],
                "uninspected_candidate_ids": row["uninspected_candidate_ids"],
            })
        missing_slots = [slot["item"] for slot in slots if not slot["verified_candidate_ids"]]
        # Do not obscure a newly opened product page: the next turn must still
        # have its source paths and literal values for record_check.
        if missing_slots and all(slot["inspected"] for slot in slots):
            return f"""CURRENT REQUEST
{question}

MODEL-RECORDED SHOPPING SLOTS (checks are per product, never inherited by another ID)
Step: {step}/{max_steps}
{json.dumps(slots, ensure_ascii=False)}
Missing verified slots: {json.dumps(missing_slots, ensure_ascii=False)}
Last tool feedback: {json.dumps(last_observation, ensure_ascii=False) if isinstance(last_observation, dict) and 'error' in last_observation else '(see slot state above)'}

Requirements are already recorded. Choose one available native tool call. A missing slot may need record_check on an inspected product or a different candidate; choose using the product fields you actually saw."""
        if not missing_slots and latest_quote:
            quote_facts = {key: latest_quote.get(key) for key in
                           ("product_ids", "seller_id_by_product", "voucher_applicable",
                            "threshold_met", "shop_rule_met", "final_price", "budget",
                            "within_budget")}
            return f"""CURRENT REQUEST
{question}

QUOTED BUNDLE DECISION STATE
Step: {step}/{max_steps}
Model-checked candidates by requested item: {json.dumps(slots, ensure_ascii=False)}
Existing quote for exact IDs: {json.dumps(quote_facts, ensure_ascii=False)}
The exact IDs above already have a quote. A different choice needs its own quote. No items have been submitted.
Last tool feedback: {json.dumps(last_observation, ensure_ascii=False) if isinstance(last_observation, dict) and 'error' in last_observation else '(quote facts above)'}

Choose one available native tool call. You may finish the quoted IDs if they meet the request, choose a different combination, or investigate a specific unresolved condition."""
    if (bundle_choice_focus and stage2_context and question_type == "add_on_deals"
            and requested_progress and all(ready_by_item.values())
            and not (check_as_tool and isinstance(last_observation, dict)
                     and "product_id" in last_observation
                     and "attributes" in last_observation)):
        # Replace the long browsing state at the decision point. All rows
        # come from inspected products and source-verified model checks; the
        # policy still selects IDs and may use any available tool.
        item_rows = []
        for row in requested_progress:
            candidate_rows_for_item = []
            item_constraint = constraint_for_slot(row)
            for entry in row["inspected"]:
                checks = (item_constraint.get("candidate_checks") or {}).get(entry["id"], [])
                unresolved = [check.get("need") for check in checks
                              if check.get("status") not in
                              {"supported", "available_unselected"}
                              or check.get("evidence_verified") is not True]
                if (not unresolved and entry["preference_gate"] not in
                        {"recorded_checks_pass", "ready_to_submit_with_variant"}):
                    unresolved = [str(item_constraint.get("requirement", ""))]
                candidate_rows_for_item.append({
                    "product_id": entry["id"], "seller_id": entry["seller"],
                    "list_price": entry["price"],
                    "verified_needs": [check.get("need") for check in checks
                                       if check.get("status") in
                                       {"supported", "available_unselected"}
                                       and check.get("evidence_verified") is True],
                    "unresolved_needs": unresolved,
                    "preference_gate": entry["preference_gate"],
                    "variant_indices_for_finish": [check["variant_index"]
                                                   for check in checks
                                                   if check.get("evidence_verified") is True
                                                   and "variant_index" in check],
                })
            item_rows.append({"item": row["item"], "slot_index": row["slot_index"],
                              "candidates": candidate_rows_for_item,
                              "verified_candidate_ids": ready_by_item[slot_key(row)]})
        complete_sellers: dict[str, dict[str, list[str]]] = {}
        if same_shop_required:
            for row in item_rows:
                for entry in row["candidates"]:
                    if (entry["product_id"] in row["verified_candidate_ids"]
                            and entry["seller_id"] is not None):
                        complete_sellers.setdefault(str(entry["seller_id"]), {}).setdefault(
                            (str(row["slot_index"]) if requested.count(row["item"]) > 1
                             else row["item"]), []).append(entry["product_id"])
            complete_sellers = {seller: items for seller, items in complete_sellers.items()
                                if len(items) == len(item_rows)}
        quote_facts = ({key: latest_quote.get(key) for key in
                        ("product_ids", "voucher_applicable", "threshold_met",
                         "shop_rule_met", "final_price", "budget", "within_budget")}
                       if latest_quote else None)
        requirements = [{"item": item.get("item"), "requirement": item.get("requirement"),
                         "user_evidence": (item.get("evidence") or [])[:2]}
                        for item in constraints if isinstance(item, dict)]
        return f"""CURRENT REQUEST
{question}

MODEL-RECORDED REQUIREMENTS
{json.dumps(requirements, ensure_ascii=False)}

SHOPPING DECISION STATE (candidate checks were verified against inspected product fields)
Step: {step}/{max_steps}
Quote-ready categories: {json.dumps([row['item'] for row in item_rows], ensure_ascii=False)}
Inspected candidates by requested item: {json.dumps(item_rows, ensure_ascii=False)}
Same-seller groups among verified candidates: {json.dumps(complete_sellers, ensure_ascii=False)}
Latest quote for exact product IDs: {json.dumps(quote_facts, ensure_ascii=False)}
Other ID combinations have no matching quote yet. No product IDs have been selected by the harness.
Last tool feedback: {json.dumps(last_observation, ensure_ascii=False) if isinstance(last_observation, dict) and 'error' in last_observation else '(see checked candidates and quote facts above)'}

Choose the next action with one native tool call. Search, inspect, quote and finish remain available; select product IDs yourself."""
    bundle_decision = ""
    if bundle_decision_state and stage2_context and question_type == "add_on_deals":
        # Summarize only inspected products and the model's recorded checks.
        # The policy still chooses the exact IDs for calculate_bundle/finish.
        ready_rows = [
            {"item": row["item"], "slot_index": row["slot_index"], "candidates": [
                {"id": entry["id"], "seller_id": entry["seller"],
                 "list_price": entry["price"]}
                for entry in row["inspected"]
                if entry["id"] in ready_by_item[slot_key(row)]
            ]}
            for row in requested_progress
        ]
        seller_groups: dict[str, dict[str, list[str]]] = {}
        if same_shop_required:
            for row in ready_rows:
                for entry in row["candidates"]:
                    if entry["seller_id"] is not None:
                        seller_groups.setdefault(str(entry["seller_id"]), {}).setdefault(
                            (str(row["slot_index"]) if requested.count(row["item"]) > 1
                             else row["item"]), []).append(entry["id"])
            seller_groups = {seller: items for seller, items in seller_groups.items()
                             if len(items) == len(ready_rows) and ready_rows}
        new_same_shop_choice = bool(
            seller_groups and any(
                any(product_id not in quoted_ids for ids in items.values() for product_id in ids)
                for items in seller_groups.values()))
        if latest_quote:
            quote_status = ("The latest quote applies only to product_ids "
                            + json.dumps(quoted_ids, ensure_ascii=False)
                            + "; it has not priced any replacement combination.")
        else:
            quote_status = "No product combination has been quoted yet."
        if all(row["candidates"] for row in ready_rows) and ready_rows:
            if (same_shop_required and latest_quote
                    and latest_quote.get("voucher_applicable") is False
                    and new_same_shop_choice):
                decision = ("The earlier quote did not apply the same-shop voucher. "
                            "New checked candidates include a same-seller choice. "
                            "Choose one inspected ID per requested item and quote those exact IDs "
                            "with calculate_bundle before finishing. You may keep the old choice "
                            "if its price and all requirements already satisfy the request.")
            elif latest_quote and latest_quote.get("within_budget") is True:
                decision = ("If the exact quoted IDs satisfy every requirement, call finish "
                            "with those IDs. If you choose different IDs, calculate_bundle "
                            "for the new combination first.")
            else:
                decision = ("Choose one inspected ID per requested item and call "
                            "calculate_bundle for those exact IDs now. Continue searching "
                            "only if a specific requirement is still unmet.")
        else:
            decision = "Inspect and check a candidate for each missing requested item."
        if ready_rows and all(row["candidates"] for row in ready_rows):
            bundle_decision = ("BUNDLE DECISION STATE (only source-verified checks; "
                               "no pair is selected for you): "
                               + json.dumps({"checked_candidates": ready_rows,
                                             "same_seller_groups": seller_groups,
                                             "quote_status": quote_status,
                                             "next_step": decision}, ensure_ascii=False))
    quote_covers_ready_items = bool(
        requested_progress and len(quoted_ids) == len(requested_progress)
        and len(set(quoted_ids)) == len(quoted_ids)
        and any(all(product_id in ready_by_item[slot_key(row)]
                    for row, product_id in zip(requested_progress, assignment))
                for assignment in itertools.permutations(quoted_ids)))
    if (question_type == "add_on_deals" and quote_covers_ready_items
            and latest_quote.get("within_budget") is True):
        next_action_hint = (
            "Already quoted this model-checked bundle within budget: "
            + json.dumps(quoted_ids, ensure_ascii=False)
            + ". The recorded checks for these IDs support the model's stated requirements. "
              "If those requirements cover the public request, call finish now with exactly "
              "one item/product_id pair per requested type. Further viewing does not improve "
              "this quoted choice; do not calculate the same bundle again."
        )
    elif (requested_progress and all(ready_by_item.values())
            and question_type == "add_on_deals"):
        next_action_hint = (
            "Model-checked candidates with visible supporting fields: "
            + json.dumps(ready_by_item, ensure_ascii=False)
            + ". Choose one inspected ID per requested item and calculate_bundle now; "
              "explore alternatives only if the quote or remaining constraints require it."
        )
    elif (requested_progress and all(ready_by_item.values())
            and question_type == "single_product"):
        next_action_hint = (
            "SINGLE-PRODUCT DECISION: Source-checked candidates for the requested item: "
            + json.dumps(ready_by_item, ensure_ascii=False)
            + ". If the recorded requirement covers every preference in the public request "
              "and a candidate also meets any price limit, choose one of these inspected IDs "
              "and call finish with selections=[{item, product_id}] now. Include its original "
              "variant_index if a required option is available but not yet selected. "
              "Search further only if a public requirement or price limit remains unresolved; "
              "do not reopen the checked product."
        )
    elif missing_inspected:
        next_action_hint = ("No inspected candidate yet for: "
                            + json.dumps(missing_inspected, ensure_ascii=False)
                            + ". Search or inspect those types before reopening an inspected ID."
                            " Missing model checklist marks do not make a saved product page disappear.")
    elif missing_verified:
        next_action_hint = (
            "No preference-verified candidate yet for: "
            + json.dumps([{"item": row["item"],
                           "uninspected_candidate_ids": row["uninspected_candidate_ids"]}
                          for row in missing_verified], ensure_ascii=False)
            + ". Inspect an uninspected candidate for those items, or check its saved page if"
              " a needed field was missed. Opening a page alone does not satisfy its required"
              " features; a quote may estimate price, but it is not a ready-to-submit bundle.")
    elif requested_progress:
        next_action_hint = ("Each requested type has an inspected candidate. Use saved product details "
                            "to compare preferences; for a bundle, quote one candidate per type with "
                            "calculate_bundle, then finish if its final price fits. Do not reopen IDs "
                            "in Opened product details. A missing checklist mark does not block a quote.")
    else:
        next_action_hint = ""
    if stage2_context and has_repeated_public_item_types(question):
        missing_slots = missing_public_requirement_slots(question, constraints)
        if missing_slots:
            next_action_hint = (
                "First complete the public product inventory. Call set_requirement_slot for "
                + json.dumps(missing_slots[0], ensure_ascii=False)
                + " now; one call covers all confirmed features of this whole product occurrence, "
                  "not one feature. Keep repeated item names in separate numbered slots. "
                  "Search after all public slots are filled.")
    if (neutral_quote_feedback and question_type == "add_on_deals"
            and latest_quote):
        next_action_hint = (
            "Latest quote facts: "
            + json.dumps({"product_ids": quoted_ids,
                          "voucher_applicable": latest_quote.get("voucher_applicable"),
                          "within_budget": latest_quote.get("within_budget"),
                          "final_price": latest_quote.get("final_price"),
                          "budget": latest_quote.get("budget")}, ensure_ascii=False)
            + ". Any different product IDs have no matching quote yet."
        )
    if (check_as_tool and isinstance(last_observation, dict)
            and "product_id" in last_observation and "attributes" in last_observation):
        next_action_hint = "Call record_check for the last inspected product before another search or view."
    if (bundle_decision_state and bundle_decision
            and not (check_as_tool and isinstance(last_observation, dict)
                     and "product_id" in last_observation
                     and "attributes" in last_observation)):
        # The explicit decision state supersedes the older hint, which could
        # foreground an affordable but coupon-ineligible quote after new
        # same-seller products had been inspected and checked.
        next_action_hint = bundle_decision
    if (isinstance(last_observation, dict) and
            last_observation.get("error") == "status_evidence_conflict"):
        conflict_id = str(last_observation.get("product_id", ""))
        next_action_hint = (
            "CHECK CORRECTION PENDING for inspected product ID " + conflict_id
            + ". The previous record_check saved no checks. Its cited field and value"
              " literally match the stated need, but the model marked it unknown."
              " If this visible value satisfies the user's need, call record_check"
              " again for the same ID with all required checks and a consistent"
              " status; the saved page is available without reopening it."
              " If the relation is genuinely uncertain, you may inspect a"
              " different candidate instead."
        )
    if (isinstance(last_observation, dict) and
            last_observation.get("error") ==
            "record_check requires an inspected product_id"):
        next_action_hint = (
            "CHECK BLOCKED: " + str(last_observation.get("next_step", ""))
            + " Do not call record_check again for that ID until its page has"
              " been opened."
        )
    quote_keys = ("product_ids", "seller_id_by_product", "voucher_applicable",
                  "within_budget", "final_price") if same_shop_required else (
                      "product_ids", "voucher_applicable", "within_budget", "final_price")
    quotes = [{k: q.get(k) for k in quote_keys}
              for q in (successful_calculations or [])[-1 if tight and stage2_context else -3:]]
    feedback = last_observation
    if stage2_context and isinstance(last_observation, dict):
        if isinstance(last_observation.get("results"), list):
            # The candidate list above already preserves the search results.
            feedback = {"result_count": len(last_observation["results"]),
                        "query": last_observation.get("query"),
                        "notice": "Candidate details are retained in Search candidates above."}
        elif "product_id" in last_observation and "attributes" in last_observation:
            # Opened product details above preserve attributes and selectable options.
            feedback = {"product_id": last_observation["product_id"],
                        "description_excerpt": str(last_observation.get("description_excerpt", ""))[:350],
                        "notice": last_observation.get("notice")}
    display_constraints = (
        [{"item": c.get("item"), "slot_index": c.get("slot_index"),
          "requirement": c.get("requirement"),
          "evidence_count": len(c.get("evidence", []))}
         for c in constraints if isinstance(c, dict)]
        if stage2_context else constraints)
    if (stage2_context and has_repeated_public_item_types(question) and
            missing_public_requirement_slots(question, constraints)):
        # During initial extraction, old slot text invites the small model to
        # copy it into the next same-named slot. The full model-authored list
        # remains in harness memory and reappears once every slot is filled.
        display_constraints = [
            {"slot_index": c.get("slot_index"), "item": c.get("item"),
             "recorded": True, "evidence_count": len(c.get("evidence", []))}
            for c in constraints if isinstance(c, dict)]
    display_recent_actions = (
        [] if tight and stage2_context else
        recent_actions[-1:] if compact and stage2_context else recent_actions[-4:])
    if (stage2_context and has_repeated_public_item_types(question) and
            missing_public_requirement_slots(question, constraints)):
        display_recent_actions = [
            {"name": action.get("name"),
             "slot_index": (action.get("arguments") or {}).get("slot_index")}
            if action.get("name") == "set_requirement_slot" else action
            for action in display_recent_actions]
    if tight and stage2_context:
        display_constraints = [
            {"item": str(c.get("item", ""))[:50],
             "requirement": str(c.get("requirement", ""))[:140],
             "evidence_count": c.get("evidence_count", 0),
             }
            for c in display_constraints[:10] if isinstance(c, dict)
        ]
        if isinstance(feedback, dict) and "description_excerpt" in feedback:
            feedback = {**feedback, "description_excerpt": str(feedback["description_excerpt"])[:120]}
        if not same_shop_required and isinstance(feedback, dict):
            # Seller maps do not affect a platform voucher; preserve the quote
            # arithmetic while dropping redundant seller IDs in tight state.
            feedback = {key: value for key, value in feedback.items()
                        if key != "seller_id_by_product"}
            if isinstance(feedback.get("prior_quote"), dict):
                feedback["prior_quote"] = {
                    key: value for key, value in feedback["prior_quote"].items()
                    if key != "seller_id_by_product"}
    return f"""CURRENT REQUEST
{question}

TASK CARD
{task_card(question_type)}

{memory_label}
{memory_text}

HARNESS STATE
Step: {step}/{max_steps}
Last valid model constraints: {json.dumps(display_constraints, ensure_ascii=False)}
Public requested item slots (repeated names are separate purchases): {json.dumps([{'slot_index': i + 1, 'item': item} for i, item in enumerate(requested_bundle_items(question))], ensure_ascii=False)}
Public requested items missing from checklist: {json.dumps(missing_public_requirement_slots(question, constraints), ensure_ascii=False)}
Search candidates retained by harness: {json.dumps(candidate_rows, ensure_ascii=False)}
Opened product details (viewed pages; viewing alone does not verify required features): {json.dumps(viewed_rows, ensure_ascii=False)}
Viewable product IDs (uninspected retained candidates only): {json.dumps([str(row.get('product_id')) for row in candidate_rows if str(row.get('product_id')) not in viewed], ensure_ascii=False)}
{('Blocked view_product IDs (already inspected): ' + json.dumps(list(viewed)[-10:], ensure_ascii=False)) if not tight else 'All IDs in Opened product details are blocked from view_product.'}
{('Older candidate details are omitted to fit context; search or inspect again if needed.' if tight and stage2_context else '')}
Required-feature checks by requested item and product ID (✓ = verified page evidence; × = contradicted or unsupported; ? = recheck saved evidence; no check means detail opened but not verified): {json.dumps(shopping_progress, ensure_ascii=False)}
Selectable specifications (available options, not submitted items; use original variant_index in finish): {json.dumps(selectable_options, ensure_ascii=False)}
Prior bundle quotes (may fail voucher or budget): {json.dumps(quotes, ensure_ascii=False)}
Recent actions: {json.dumps(display_recent_actions, ensure_ascii=False)}
Last tool feedback: {json.dumps(feedback, ensure_ascii=False)}
{(('Call record_check next for inspected product ID ' + str(last_observation['product_id']) + ' using only its visible fields above.') if check_as_tool else 'The last viewed product needs your brief attribute check in constraints on this turn; cite only fields visible above. Do not repeat the whole checklist.') if stage2_context and isinstance(last_observation, dict) and 'product_id' in last_observation and 'attributes' in last_observation else ''}
{next_action_hint}

{('Choose the next action by calling exactly one available tool. Return a native <tool_call> only.' if native_tools else 'Choose the next action. Output JSON only.')}"""
