"""harness.runner: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from shopping_agent.data.tasks import truth_items
from shopping_agent.harness.context import stage2_history_document
from shopping_agent.harness.context import state_text
from shopping_agent.harness.progress import recent_useful_progress
from shopping_agent.harness.prompts import STAGE2_HISTORY_ID
from shopping_agent.harness.prompts import SYSTEM_PROMPT
from shopping_agent.harness.prompts import SYSTEM_PROMPT_STAGE2
from shopping_agent.harness.prompts import SYSTEM_PROMPT_STAGE2_CHECK_TOOL
from shopping_agent.harness.prompts import SYSTEM_PROMPT_STAGE2_NATIVE
from shopping_agent.harness.protocol import parse_json_object
from shopping_agent.harness.protocol import parse_native_tool_call
from shopping_agent.harness.protocol import render_policy_prompt
from shopping_agent.harness.state import repeated_action_feedback
from shopping_agent.harness.state import uninspected_record_check_feedback
from shopping_agent.harness.state import viewable_candidate_ids
from shopping_agent.rewards.scoring import score_episode
from shopping_agent.rewards.scoring import score_episode_grounded
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v10
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v11
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v12
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v6
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v7
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v8
from shopping_agent.rewards.scoring import score_episode_grounded_open_world_v9
from shopping_agent.rewards.scoring import score_episode_grounded_progress
from shopping_agent.rewards.scoring import score_episode_grounded_progress_v2
from shopping_agent.tools.finish import evaluate
from shopping_agent.tools.finish import selections_from_quoted_ids
from shopping_agent.harness.memory import memory_id
from shopping_agent.harness.memory import retrieve_memories
from shopping_agent.tools.requirements import has_repeated_public_item_types
from shopping_agent.tools.requirements import merge_constraint_updates
from shopping_agent.tools.requirements import missing_public_requirement_slots
from shopping_agent.tools.requirements import public_history_section_index
from shopping_agent.tools.requirements import record_check_status_conflicts
from shopping_agent.tools.requirements import requested_bundle_items
from shopping_agent.tools.requirements import user_quote_is_visible
from shopping_agent.tools.requirements import validate_public_bundle_count
from shopping_agent.tools.requirements import validate_public_requirements
from shopping_agent.tools.voucher import voucher_threshold_is_strict
import json
import torch


def _episode_coroutine(row, store, memory_docs, seed: int, max_steps: int,
                grounded_reward: bool = False, stage2_context: bool = False,
                progress_reward: bool = False, progress_reward_v2: bool = False,
                tokenizer=None, max_input_tokens: int = 6144,
                 model_voucher: bool = False, open_world_reward: bool = False,
                 incremental_reward_v7: bool = False,
                 premature_finish_reward_v8: bool = False,
                 annotated_features_reward_v9: bool = False,
                 max_consecutive_errors: int = 3,
                 check_as_tool: bool = False,
                 native_tools: bool = False,
                 quote_focus: bool = False,
                 repair_native_calls: bool = False,
                 governed_bundle_price: bool = False,
                 bundle_decision_state: bool = False,
                 neutral_quote_feedback: bool = False,
                 bundle_choice_focus: bool = False,
                 bundle_loop_focus: bool = False,
                 finish_quoted_ids: bool = False,
                 quote_decision_gate: bool = False,
                 quote_evidence_gaps: bool = False,
                 atomic_evidence_slots: bool = False,
                 quote_focus_on_replacement: bool = False,
                 strong_ready_repeat_penalty_v10: bool = False,
                 false_negative_check_penalty_v11: bool = False,
                 public_item_fit_v12: bool = False,
                 adaptive_step_extension: bool = False):
    info = row["extra_info"]
    truth = json.loads(row["reward_model"]["ground_truth"])
    if stage2_context and info.get("question_type") == "add_on_deals":
        validate_public_bundle_count(info["question"], int(truth.get("n", 1)))
    if truth.get("voucher"):
        truth["_voucher_threshold_strict"] = voucher_threshold_is_strict(info["question"])
    memories = []
    provided_memories = [stage2_history_document(row)] if stage2_context else []
    constraints = []
    candidates = {}
    viewed = {}
    trajectory = []
    training_rows = []
    submitted = []
    finish_reason = "max_steps"
    successful_action_keys = set()
    successful_calculations = []
    pending_quote = None
    consecutive_errors = 0
    recent_actions = []
    last_observation = None
    current_step_limit = max_steps
    step_extensions = []

    for step in range(1, max_steps + (10 if adaptive_step_extension else 0) + 1):
        if step > current_step_limit:
            recent_progress = recent_useful_progress(trajectory, current_step_limit)
            if not recent_progress or current_step_limit >= max_steps + 10:
                break
            current_step_limit += 5
            step_extensions.append({"at_step": step - 1,
                                    "reason": "new_observed_information_in_last_three_steps",
                                    "progress_steps": recent_progress,
                                    "new_limit": current_step_limit})
        def decision_state(text):
            if quote_decision_gate and pending_quote is not None:
                return ("QUOTE_DECISION_PENDING: The last quote used exactly "
                        + json.dumps(pending_quote.get("product_ids", []))
                        + ". Choose finish for a model-selected inspected bundle, "
                        "or continue_shopping with a concrete reason to reopen browsing. "
                        "The harness will not choose or submit for you.\n" + text)
            return text
        state = state_text(
            info["question"], provided_memories if stage2_context else memories,
            constraints, candidates, viewed, recent_actions, last_observation,
            info["question_type"], step, current_step_limit, stage2_context,
            successful_calculations=successful_calculations,
            check_as_tool=check_as_tool or native_tools,
            history_as_checklist=native_tools, native_tools=native_tools,
            quote_focus=quote_focus, bundle_decision_state=bundle_decision_state,
            neutral_quote_feedback=neutral_quote_feedback,
            bundle_choice_focus=bundle_choice_focus,
            bundle_loop_focus=bundle_loop_focus,
            quote_evidence_gaps=quote_evidence_gaps,
            atomic_evidence_slots=atomic_evidence_slots,
            quote_focus_on_replacement=quote_focus_on_replacement,
        )
        state = decision_state(state)
        system_prompt = (SYSTEM_PROMPT_STAGE2_NATIVE if native_tools else
                         SYSTEM_PROMPT_STAGE2_CHECK_TOOL if check_as_tool else
                         SYSTEM_PROMPT_STAGE2) if stage2_context else SYSTEM_PROMPT
        if model_voucher and not native_tools:
            bundle_schema = ('calculate_bundle({"product_ids": [string, ...], "voucher": '
                '{"scope": "platform" or "same_shop", "threshold": number, '
                '"threshold_operator": ">" or ">=", "discount_percent": number, '
                '"discount_cap": number}, "budget": number})')
            system_prompt = system_prompt.replace(
                '3. calculate_bundle({"product_ids": [string, ...]})',
                '3. ' + bundle_schema).replace(
                '4. calculate_bundle({"product_ids": [string, ...]})',
                '4. ' + bundle_schema)
            system_prompt += ("\nFor a bundle, read all voucher fields from the current request "
                              "and include them in calculate_bundle. Budget is a separate task "
                              "constraint and belongs in its own argument. The tool checks your reading "
                              "against the public request and reports mismatched field names. "
                              "Use same_shop only if the voucher explicitly requires products "
                              "from the same shop; otherwise use platform. Compare the raw total "
                              "with the threshold exactly as worded: 'exceeds $10' means "
                              "threshold_operator='>' (a total of exactly $10 does not qualify); "
                              "'min spend $10' or 'at least $10' means '>=' (exactly $10 qualifies). "
                              "If the tool reports a wrong field, change that field based on the "
                              "current request before retrying; never resend the same rejected object. "
                              "The tool never supplies the correct values.")
        elif model_voucher and native_tools:
            system_prompt += ("\nUse voucher.scope=same_shop only for explicit same-shop wording. "
                              "'exceeds $10' requires threshold_operator='>'; "
                              "'min spend $10' or 'at least $10' requires '>='. "
                              "Budget is separate from the voucher threshold.")
        if governed_bundle_price:
            system_prompt += ("\nFor bundle searches, preferred_max_list_price only ranks cheaper listed prices "
                              "slightly higher; it never excludes more expensive candidates. "
                              "The search tool applies a safe upper bound from the public final budget "
                              "and maximum coupon discount. The final budget applies to the whole quoted "
                              "basket, not each listed item. Use calculate_bundle for exact affordability.")
        if finish_quoted_ids:
            system_prompt += ("\nFor a previously quoted bundle, finish can use quoted_product_ids "
                              "instead of repeating item/product_id selections. Supply exactly the IDs "
                              "you chose and quoted. If a wanted option has several variants, include "
                              "variant_indices_by_product for the selected original index. The tool "
                              "requires your own source-verified record_check for every quoted ID; "
                              "it never chooses a product or specification for you.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state},
        ]
        context_compacted = False
        compaction_tier = "none"
        if stage2_context:
            full_prompt = render_policy_prompt(tokenizer, messages, native_tools,
                                                governed_bundle_price, finish_quoted_ids,
                                                quote_decision_gate)
            full_length = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
            if full_length > max_input_tokens:
                state = state_text(
                    info["question"], provided_memories, constraints, candidates, viewed,
                    recent_actions, last_observation, info["question_type"], step,
                    current_step_limit, stage2_context=True, compact=True,
                    successful_calculations=successful_calculations,
                    check_as_tool=check_as_tool or native_tools,
                    history_as_checklist=native_tools, native_tools=native_tools,
                    quote_focus=quote_focus, bundle_decision_state=bundle_decision_state,
                    neutral_quote_feedback=neutral_quote_feedback,
                    bundle_choice_focus=bundle_choice_focus,
                    bundle_loop_focus=bundle_loop_focus,
                    quote_evidence_gaps=quote_evidence_gaps,
                    atomic_evidence_slots=atomic_evidence_slots,
                    quote_focus_on_replacement=quote_focus_on_replacement,
                        )
                messages[1]["content"] = decision_state(state)
                context_compacted = True
                compaction_tier = "compact"
                compact_prompt = render_policy_prompt(tokenizer, messages, native_tools,
                                                       governed_bundle_price, finish_quoted_ids,
                                                       quote_decision_gate)
                if len(tokenizer(compact_prompt, add_special_tokens=False)["input_ids"]) > max_input_tokens:
                    state = state_text(
                        info["question"], provided_memories, constraints, candidates, viewed,
                        recent_actions, last_observation, info["question_type"], step,
                        current_step_limit, stage2_context=True, compact=True,
                        successful_calculations=successful_calculations, minimal=True,
                        check_as_tool=check_as_tool or native_tools,
                        history_as_checklist=native_tools, native_tools=native_tools,
                        quote_focus=quote_focus, bundle_decision_state=bundle_decision_state,
                        neutral_quote_feedback=neutral_quote_feedback,
                        bundle_choice_focus=bundle_choice_focus,
                        bundle_loop_focus=bundle_loop_focus,
                        quote_evidence_gaps=quote_evidence_gaps,
                        atomic_evidence_slots=atomic_evidence_slots,
                        quote_focus_on_replacement=quote_focus_on_replacement,
                                )
                    messages[1]["content"] = decision_state(state)
                    compaction_tier = "minimal"
                    minimal_prompt = render_policy_prompt(tokenizer, messages, native_tools,
                                                           governed_bundle_price, finish_quoted_ids,
                                                           quote_decision_gate)
                    if len(tokenizer(minimal_prompt, add_special_tokens=False)["input_ids"]) > max_input_tokens:
                        state = state_text(
                            info["question"], provided_memories, constraints, candidates, viewed,
                            recent_actions, last_observation, info["question_type"], step,
                            current_step_limit, stage2_context=True, compact=True,
                            successful_calculations=successful_calculations,
                            minimal=True, tight=True,
                            check_as_tool=check_as_tool or native_tools,
                            history_as_checklist=native_tools, native_tools=native_tools,
                            quote_focus=quote_focus, bundle_decision_state=bundle_decision_state,
                            neutral_quote_feedback=neutral_quote_feedback,
                            bundle_choice_focus=bundle_choice_focus,
                            bundle_loop_focus=bundle_loop_focus,
                            quote_evidence_gaps=quote_evidence_gaps,
                            atomic_evidence_slots=atomic_evidence_slots,
                            quote_focus_on_replacement=quote_focus_on_replacement,
                                        )
                        messages[1]["content"] = decision_state(state)
                        compaction_tier = "tight"
                        tight_prompt = render_policy_prompt(tokenizer, messages, native_tools,
                                                             governed_bundle_price, finish_quoted_ids,
                                                             quote_decision_gate)
                        tight_length = len(tokenizer(
                            tight_prompt, add_special_tokens=False)["input_ids"])
                        if tight_length > max_input_tokens:
                            if not trajectory:
                                raise RuntimeError(
                                    f"Initial Stage-2 prompt has {tight_length} tokens, "
                                    f"above {max_input_tokens}; audit this task before training"
                                )
                            finish_reason = "context_overflow"
                            trajectory.append({
                                "step": step,
                                "prompt_tokens": tight_length,
                                "context_compacted": True,
                                "observation": {"error": "Tight state still exceeds the context budget"},
                                "terminated": "context_overflow; no model action generated",
                            })
                            break
        raw, prompt_ids, response_ids = yield messages, seed + step
        # Tool validity is filled after execution. Invalid/repeated actions
        # never inherit a positive episode advantage.
        training_rows.append((prompt_ids, response_ids, False))
        parsed = (parse_native_tool_call(raw, repair_syntax=repair_native_calls)
                  if native_tools else parse_json_object(raw))
        event = {
            "step": step,
            "raw_model_output": raw,
            "parsed": parsed,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "context_compacted": context_compacted,
            "compaction_tier": compaction_tier,
        }
        if parsed and parsed.get("syntax_repair"):
            event["syntax_repair"] = parsed["syntax_repair"]
        if not parsed or not isinstance(parsed.get("action"), dict):
            observation = {"error": "Output must match the required JSON schema."}
        else:
            if not native_tools and isinstance(parsed.get("constraints"), list):
                constraints = merge_constraint_updates(constraints, parsed["constraints"], viewed)
            action = parsed["action"]
            name = action.get("name")
            arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
            event["action"] = {"name": name, "arguments": arguments}
            recent_actions.append(event["action"])
            action_key = json.dumps(event["action"], sort_keys=True, ensure_ascii=False)
            if (quote_decision_gate and pending_quote is not None and
                    name not in {"finish", "continue_shopping"}):
                observation = {"error": "Choose finish for the quoted products or continue_shopping to reopen browsing; no purchase was made."}
            elif action_key in successful_action_keys or (
                name == "calculate_bundle" and
                any(sorted(str(value) for value in quote.get("product_ids", [])) ==
                    sorted(str(value) for value in arguments.get("product_ids", []))
                    for quote in successful_calculations)
            ) or (
                name == "view_product" and str(arguments.get("product_id", "")) in viewed
            ):
                observation = repeated_action_feedback(
                    name, arguments, successful_calculations,
                    viewable_candidate_ids(candidates, viewed),
                    neutral_quote_feedback=neutral_quote_feedback)
            elif name == "continue_shopping" and quote_decision_gate:
                reason = str(arguments.get("reason", "")).strip()
                if pending_quote is None:
                    observation = {"error": "continue_shopping is only available after a successful within-budget quote"}
                elif not reason:
                    observation = {"error": "continue_shopping requires a non-empty reason"}
                else:
                    pending_quote = None
                    observation = {"browsing_reopened": True, "reason": reason,
                                   "notice": "The prior quote covers only its exact product IDs; quote a new combination before finishing."}
            elif name == "search_memory" and stage2_context:
                observation = {"error": "search_memory is unavailable: relevant history is already in the task input."}
            elif name == "search_memory":
                query = str(arguments.get("query", "")).strip()
                if not query:
                    observation = {"error": "query must be non-empty"}
                else:
                    try:
                        top_k = max(1, min(int(arguments.get("top_k", 3)), 5))
                    except (TypeError, ValueError):
                        top_k = 3
                    results = retrieve_memories(query, memory_docs, top_k=top_k)
                    known = {item["id"] for item in memories}
                    memories.extend(item for item in results if item["id"] not in known)
                    observation = {
                        "query": query,
                        "results": results,
                        "notice": "These are prior user statements. Extract only explicit preferences and cite memory IDs.",
                    }
            elif name == "set_requirements" and native_tools:
                items = arguments.get("items")
                if constraints:
                    observation = {"error": "requirements were already recorded; use search_products or another tool"}
                elif has_repeated_public_item_types(info["question"]):
                    observation = {"error": "bundle_requires_numbered_slots",
                                   "public_item_slots": [{"slot_index": i + 1, "item": item}
                                                         for i, item in enumerate(
                                                             requested_bundle_items(info["question"]))],
                                   "next_step": "Call set_requirement_slot for one numbered product occurrence at a time. Put all wanted features for that occurrence in one requirement."}
                else:
                    validated, error = validate_public_requirements(info["question"], items)
                    if error:
                        observation = error
                    else:
                        constraints = merge_constraint_updates(constraints, validated, viewed)
                        observation = {"requirements_recorded": len(constraints),
                                       "items": [{"slot_index": entry.get("slot_index"),
                                                  "item": entry.get("item")}
                                                 for entry in constraints]}
            elif name == "set_requirement_slot" and native_tools:
                public_slots = requested_bundle_items(info["question"])
                raw_slot = arguments.get("slot_index")
                slot_index = (raw_slot if isinstance(raw_slot, int) and
                              not isinstance(raw_slot, bool) else None)
                item = str(arguments.get("item", "")).strip()
                requirement = str(arguments.get("requirement", "")).strip()
                evidence = arguments.get("evidence")
                missing = missing_public_requirement_slots(info["question"], constraints)
                if not has_repeated_public_item_types(info["question"]):
                    observation = {"error": "set_requirement_slot is not active for this task; use set_requirements"}
                elif not missing:
                    observation = {"error": "all public item slots already have requirements"}
                elif slot_index != missing[0]["slot_index"] or item.casefold() != missing[0]["item"].casefold():
                    observation = {"error": "fill_next_public_slot_in_order",
                                   "next_public_slot": missing[0]}
                elif not requirement or not isinstance(evidence, list) or not evidence:
                    observation = {"error": "slot requires one combined requirement and non-empty user evidence"}
                elif any(not isinstance(quote, str) or
                         not user_quote_is_visible(quote, provided_memories)
                         for quote in evidence):
                    observation = {"error": "slot_evidence_not_in_released_user_words",
                                   "next_public_slot": missing[0],
                                   "next_step": "Use brief exact excerpts from the released user turns for this product occurrence; do not cite assistant suggestions or invent a quote."}
                elif any(str(entry.get("item", "")).casefold() == item.casefold() and
                         str(entry.get("requirement", "")).casefold() == requirement.casefold() and
                         entry.get("evidence") == evidence for entry in constraints):
                    observation = {"error": "repeated_slot_reuses_previous_requirement_and_evidence",
                                   "next_public_slot": missing[0],
                                   "history_navigation": public_history_section_index(
                                       provided_memories, constraints),
                                   "next_step": "This is a separate product occurrence. The history index shows which dated dialogue supplied earlier slots. Re-read a different relevant user dialogue and fill this slot with its own confirmed preferences and exact quotes."}
                else:
                    constraints = merge_constraint_updates(constraints, [{
                        "slot_index": slot_index, "item": public_slots[slot_index - 1],
                        "requirement": requirement, "evidence": evidence}], viewed)
                    observation = {"requirements_recorded": len(constraints),
                                   "recorded_slot": {"slot_index": slot_index, "item": item},
                                   "remaining_public_slots": missing_public_requirement_slots(
                                       info["question"], constraints)}
            elif name == "record_check" and (check_as_tool or native_tools):
                product_id = str(arguments.get("product_id", "")).strip()
                item = str(arguments.get("item", "")).strip()
                checks = arguments.get("checks")
                matching_slots = [entry for entry in constraints
                                  if str(entry.get("item", "")).casefold() == item.casefold()]
                raw_slot = arguments.get("slot_index")
                slot_index = (raw_slot if isinstance(raw_slot, int) and not isinstance(raw_slot, bool)
                              else matching_slots[0].get("slot_index")
                              if len(matching_slots) == 1 else None)
                if product_id not in viewed:
                    observation = uninspected_record_check_feedback(
                        product_id, candidates, viewed)
                elif not item or not isinstance(checks, list) or not checks:
                    observation = {"error": "record_check requires item and non-empty checks"}
                elif (not any(entry.get("slot_index") == slot_index for entry in matching_slots)):
                    observation = {"error": "ambiguous_or_unknown_item_slot",
                                   "matching_public_slots": [entry.get("slot_index") for entry in matching_slots],
                                   "next_step": "For a repeated product type, supply its numbered slot_index in record_check. Checks for one slot do not verify another slot."}
                else:
                    conflicts = record_check_status_conflicts(checks, viewed[product_id])
                    if conflicts:
                        observation = {
                            "error": "status_evidence_conflict", "product_id": product_id,
                            "conflicts": conflicts,
                            "next_step": "No checks from this call were recorded. The cited field and value visibly match the stated need, so an unknown or contradicted status cannot be verified. Reconsider the status and call record_check again using the saved page; do not reopen the product. A broader semantic need may remain unknown if its relation to the value is unresolved.",
                        }
                    else:
                        constraints = merge_constraint_updates(
                            constraints, [{"item": item, "slot_index": slot_index,
                                           "product_id": product_id,
                                           "checks": checks}], viewed)
                        matching = next((entry for entry in constraints
                                         if entry.get("slot_index") == slot_index), {})
                        observation = {"recorded_product_id": product_id, "item": item,
                                       "slot_index": slot_index,
                                       "checks": matching.get("candidate_checks", {}).get(product_id, [])}
            elif name == "search_products":
                if (native_tools and has_repeated_public_item_types(info["question"])
                        and missing_public_requirement_slots(info["question"], constraints)):
                    observation = {"error": "complete_public_item_slots_before_search",
                                   "remaining_public_slots": missing_public_requirement_slots(
                                       info["question"], constraints)}
                else:
                    observation = store.search(
                    arguments,
                    bundle_question=(info["question"] if governed_bundle_price
                                     and info["question_type"] == "add_on_deals" else None),
                    )
                for result in observation.get("results", []):
                    candidates[result["product_id"]] = {
                        **result, "search_query": str(arguments.get("query", ""))}
            elif name == "view_product":
                if str(arguments.get("product_id", "")) not in candidates:
                    observation = {"error": "view_product requires an ID from search results",
                                   "viewable_product_ids": viewable_candidate_ids(candidates, viewed)[:10]}
                else:
                    observation = store.view(arguments)
                if "product_id" in observation and "error" not in observation:
                    viewed[observation["product_id"]] = {
                        **observation,
                        "search_query": candidates.get(observation["product_id"], {}).get("search_query", "")}
            elif name == "calculate_bundle":
                observation = (store.calculate_public(arguments, info["question"], int(truth.get("n", 1)),
                                                      neutral_quote_feedback=neutral_quote_feedback)
                               if model_voucher else store.calculate(arguments, truth))
                if "error" not in observation:
                    successful_calculations.append(observation)
                    if quote_decision_gate and observation.get("within_budget") is True:
                        pending_quote = observation
            elif name == "finish":
                selections = arguments.get("selections")
                shortcut_error = None
                if (finish_quoted_ids and not isinstance(selections, list)
                        and "quoted_product_ids" in arguments):
                    selections, shortcut_error = selections_from_quoted_ids(
                        arguments, constraints, requested_bundle_items(info["question"]),
                        viewed, successful_calculations)
                    if shortcut_error is None:
                        arguments = {**arguments, "selections": selections}
                        event["action"]["arguments"] = arguments
                if shortcut_error is not None:
                    observation = {"error": shortcut_error}
                elif not isinstance(selections, list):
                    observation = {"error": "finish requires selections list with item and product_id"}
                elif any(not isinstance(value, dict) or not str(value.get("item", "")).strip()
                         or not str(value.get("product_id", "")).strip() for value in selections):
                    observation = {"error": "every selection requires non-empty item and product_id"}
                else:
                    values = [str(value["product_id"]) for value in selections]
                    expected = int(truth.get("n", 1))
                    if len(values) != expected:
                        observation = {"error": f"finish requires exactly {expected} selections, got {len(values)}"}
                    elif len(set(values)) != len(values):
                        observation = {"error": "finish product_ids must be unique"}
                    elif any(value not in store.by_id for value in values):
                        observation = {"error": "finish contains an unknown product_id"}
                    elif any(store.invalid_variant_selection(value) for value in selections):
                        observation = {"error": "finish variant_index must refer to an inspected product option (zero-based)"}
                    elif any(value not in viewed for value in values):
                        missing = [value for value in values if value not in viewed]
                        observation = {"error": f"inspect every submitted product first; missing {missing}"}
                    elif truth.get("voucher") and not any(
                        set(quote.get("product_ids", [])) == set(values)
                        and quote.get("voucher_applicable") in (True, False)
                        and quote.get("within_budget") is True
                        for quote in successful_calculations
                    ):
                        observation = {"error": "run calculate_bundle on these exact selections and keep the final price within budget; the voucher may be inapplicable"}
                    else:
                        submitted = values
                        finish_reason = "model_finish"
                        observation = {"finished": True, "selections": selections}
                        event["observation"] = observation
                        trajectory.append(event)
                        training_rows[-1] = (prompt_ids, response_ids,
                                             not bool(parsed.get("syntax_repair")))
                        break
            else:
                observation = {"error": f"unknown action {name}"}
            if "error" not in observation:
                successful_action_keys.add(action_key)
        event["observation"] = observation
        trajectory.append(event)
        last_observation = observation
        training_rows[-1] = (prompt_ids, response_ids,
                             "error" not in observation and not bool(
                                 parsed and parsed.get("syntax_repair")))
        consecutive_errors = consecutive_errors + 1 if "error" in observation else 0
        if max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
            finish_reason = "no_progress_guard"
            event["terminated"] = "three consecutive invalid or repeated actions"
            break

    metrics = evaluate(submitted, truth, set(viewed))
    episode = {
        "question_id": info["question_id"],
        "conversation_id": info["conversation_id"],
        "question_type": info["question_type"],
        "question": info["question"],
        "retrieved_memories": memories,
        "provided_memories": provided_memories,
        "final_constraints": constraints,
        "checklist_state": ("native_record_check_v1" if native_tools else
                            "model_record_check_v1" if check_as_tool else "model_sparse_merge_v2"),
        "quote_decision_gate": quote_decision_gate,
        "quote_evidence_gaps": quote_evidence_gaps,
        "atomic_evidence_slots": atomic_evidence_slots,
        "quote_focus_on_replacement": quote_focus_on_replacement,
        "history_context": ("model_summary_after_set_requirements_v1" if native_tools
                            else "released_history_with_budgeted_compaction"),
        "submitted_product_ids": submitted,
        "finish_reason": finish_reason,
        "step_limit_initial": max_steps,
        "step_limit_final": current_step_limit,
        "step_extensions": step_extensions,
        "metrics": metrics,
        "target_product_ids_audit_only": [str(item["product_id"]) for item in truth_items(row)],
        "reward_rule": ("grounded_preferences_v12" if public_item_fit_v12 else
                        "grounded_preferences_v11" if false_negative_check_penalty_v11 else
                        "grounded_preferences_v10" if strong_ready_repeat_penalty_v10 else
                        "grounded_preferences_v9" if annotated_features_reward_v9 else
                        "grounded_preferences_v8" if premature_finish_reward_v8 else
                        "grounded_preferences_v7" if incremental_reward_v7 else
                        "grounded_preferences_v6" if open_world_reward else
                        "grounded_preferences_v5" if progress_reward_v2 else
                        "grounded_preferences_v4" if progress_reward else
                        "grounded_preferences_v3" if grounded_reward else "legacy"),
        "trajectory": trajectory,
    }
    expected_memory_ids = ([STAGE2_HISTORY_ID] * len(truth_items(row)) if stage2_context
                           else [memory_id(info["conversation_id"], info["question_id"], ordinal)
                                 for ordinal, _ in enumerate(truth_items(row))])
    if grounded_reward:
        scorer = (score_episode_grounded_open_world_v12 if public_item_fit_v12 else
                  score_episode_grounded_open_world_v11 if false_negative_check_penalty_v11 else
                  score_episode_grounded_open_world_v10 if strong_ready_repeat_penalty_v10 else
                  score_episode_grounded_open_world_v9 if annotated_features_reward_v9 else
                  score_episode_grounded_open_world_v8 if premature_finish_reward_v8 else
                  score_episode_grounded_open_world_v7 if incremental_reward_v7 else
                  score_episode_grounded_open_world_v6 if open_world_reward else
                  score_episode_grounded_progress_v2 if progress_reward_v2 else
                  score_episode_grounded_progress if progress_reward else score_episode_grounded)
        reward_breakdown = scorer(
            episode, truth_items=truth_items(row),
            source_products_by_id=store.by_id,
            expected_memory_ids=expected_memory_ids,
        )
    else:
        reward_breakdown = score_episode(
            episode,
            target_product_ids=episode["target_product_ids_audit_only"],
            expected_memory_ids=expected_memory_ids,
            required_features_by_product_id={
                str(item["product_id"]): item.get("wanted_features", [])
                for item in truth_items(row)
            },
        )
    episode["reward_breakdown"] = reward_breakdown
    episode["training_reward"] = reward_breakdown["reward"]
    return episode, training_rows


def run_episode(row, policy, store, memory_docs, seed: int, max_steps: int,
                grounded_reward: bool = False, stage2_context: bool = False,
                progress_reward: bool = False, progress_reward_v2: bool = False,
                model_voucher: bool = False, open_world_reward: bool = False,
                incremental_reward_v7: bool = False,
                premature_finish_reward_v8: bool = False,
                annotated_features_reward_v9: bool = False,
                max_consecutive_errors: int = 3,
                check_as_tool: bool = False,
                native_tools: bool = False,
                quote_focus: bool = False,
                repair_native_calls: bool = False,
                governed_bundle_price: bool = False,
                bundle_decision_state: bool = False,
                neutral_quote_feedback: bool = False,
                bundle_choice_focus: bool = False,
                bundle_loop_focus: bool = False,
                finish_quoted_ids: bool = False,
                quote_decision_gate: bool = False,
                quote_evidence_gaps: bool = False,
                atomic_evidence_slots: bool = False,
                quote_focus_on_replacement: bool = False,
                strong_ready_repeat_penalty_v10: bool = False,
                false_negative_check_penalty_v11: bool = False,
                public_item_fit_v12: bool = False,
                adaptive_step_extension: bool = False):
    """Keep the serial API for old experiments and deterministic regression tests."""
    worker = _episode_coroutine(
        row, store, memory_docs, seed, max_steps, grounded_reward,
        stage2_context, progress_reward, progress_reward_v2,
        policy.tokenizer, policy.max_input_tokens, model_voucher, open_world_reward,
        incremental_reward_v7, premature_finish_reward_v8, annotated_features_reward_v9,
        max_consecutive_errors, check_as_tool, native_tools, quote_focus,
        repair_native_calls, governed_bundle_price, bundle_decision_state,
        neutral_quote_feedback, bundle_choice_focus, bundle_loop_focus,
        finish_quoted_ids, quote_decision_gate, quote_evidence_gaps,
        atomic_evidence_slots,
        quote_focus_on_replacement, strong_ready_repeat_penalty_v10,
        false_negative_check_penalty_v11, public_item_fit_v12,
        adaptive_step_extension,
    )
    try:
        request = next(worker)
        while True:
            messages, generation_seed = request
            request = worker.send(policy.generate(messages, generation_seed))
    except StopIteration as done:
        return done.value


def run_episodes_batched(row, policy, store, memory_docs, seeds: list[int],
                         max_steps: int, grounded_reward: bool = False,
                         stage2_context: bool = False, progress_reward: bool = False,
                         progress_reward_v2: bool = False, batch_size: int = 4,
                          model_voucher: bool = False, open_world_reward: bool = False,
                          incremental_reward_v7: bool = False,
                          premature_finish_reward_v8: bool = False,
                          annotated_features_reward_v9: bool = False,
                          max_consecutive_errors: int = 3,
                          check_as_tool: bool = False,
                          native_tools: bool = False,
                          quote_focus: bool = False,
                          repair_native_calls: bool = False,
                          governed_bundle_price: bool = False,
                          bundle_decision_state: bool = False,
                          neutral_quote_feedback: bool = False,
                          bundle_choice_focus: bool = False,
                          bundle_loop_focus: bool = False,
                          finish_quoted_ids: bool = False,
                          quote_decision_gate: bool = False,
                          quote_evidence_gaps: bool = False,
                          atomic_evidence_slots: bool = False,
                          quote_focus_on_replacement: bool = False,
                          strong_ready_repeat_penalty_v10: bool = False,
                          false_negative_check_penalty_v11: bool = False,
                          public_item_fit_v12: bool = False,
                          adaptive_step_extension: bool = False):
    """Advance independent episodes together, batching only their model calls."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    def generate_with_backoff(requests):
        try:
            return policy.generate_many(requests)
        except torch.cuda.OutOfMemoryError:
            if len(requests) == 1:
                raise
            torch.cuda.empty_cache()
            middle = len(requests) // 2
            return (generate_with_backoff(requests[:middle])
                    + generate_with_backoff(requests[middle:]))

    results = []
    for offset in range(0, len(seeds), batch_size):
        chunk = seeds[offset:offset + batch_size]
        workers = [_episode_coroutine(
            row, store, memory_docs, seed, max_steps, grounded_reward,
            stage2_context, progress_reward, progress_reward_v2,
            policy.tokenizer, policy.max_input_tokens, model_voucher, open_world_reward,
            incremental_reward_v7, premature_finish_reward_v8, annotated_features_reward_v9,
            max_consecutive_errors, check_as_tool, native_tools, quote_focus,
            repair_native_calls, governed_bundle_price, bundle_decision_state,
            neutral_quote_feedback, bundle_choice_focus, bundle_loop_focus,
            finish_quoted_ids, quote_decision_gate, quote_evidence_gaps,
            atomic_evidence_slots,
            quote_focus_on_replacement, strong_ready_repeat_penalty_v10,
            false_negative_check_penalty_v11, public_item_fit_v12,
            adaptive_step_extension,
            ) for seed in chunk]
        active = {}
        completed = [None] * len(chunk)
        for index, worker in enumerate(workers):
            try:
                active[index] = next(worker)
            except StopIteration as done:
                completed[index] = done.value
        while active:
            indices = list(active)
            requests = [active[index] for index in indices]
            generations = generate_with_backoff(requests)
            if len(generations) != len(indices):
                raise RuntimeError("generate_many returned a different number of episodes")
            for index, generated in zip(indices, generations):
                try:
                    active[index] = workers[index].send(generated)
                except StopIteration as done:
                    completed[index] = done.value
                    del active[index]
        results.extend(completed)
    return results
