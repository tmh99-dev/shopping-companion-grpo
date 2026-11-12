"""Auditable reward for Shopping Companion episodes.

The module is intentionally independent of the rollout and trainer.  It can be
used for offline audits or imported by a future trainer.  Hidden target IDs and
the expected memory IDs are reward-side data only and must never be exposed in
the Agent observation.
"""

from __future__ import annotations

from collections.abc import Collection
from itertools import permutations
import re
from typing import Any


def _action(event: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    action = event.get("action") or {}
    arguments = action.get("arguments")
    return action.get("name"), arguments if isinstance(arguments, dict) else {}


def _observation(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("observation")
    return value if isinstance(value, dict) else {}


def _viewed_ids(episode: dict[str, Any]) -> set[str]:
    viewed = set()
    for event in episode.get("trajectory", []):
        name, _ = _action(event)
        observation = _observation(event)
        if name == "view_product" and "error" not in observation and observation.get("product_id") is not None:
            viewed.add(str(observation["product_id"]))
    return viewed


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _verified_preference_ids(episode: dict[str, Any],
                             required_features: dict[str, Collection[str]]) -> tuple[set[str], dict[str, list[str]]]:
    """Check hidden wanted attributes against product details actually delivered to the Agent."""
    observations = {}
    for event in episode.get("trajectory", []):
        name, _ = _action(event)
        observation = _observation(event)
        if name == "view_product" and "error" not in observation and observation.get("product_id") is not None:
            observations[str(observation["product_id"])] = observation
    verified = set()
    missing = {}
    for product_id, features in required_features.items():
        observation = observations.get(str(product_id))
        if observation is None:
            missing[str(product_id)] = list(features)
            continue
        options = observation.get("options", [])
        fields = {}
        for option in options if isinstance(options, list) else []:
            if isinstance(option, dict):
                for key, values in option.items():
                    bucket = fields.setdefault(_normalized(key), set())
                    for value in values if isinstance(values, list) else [values]:
                        bucket.add(_normalized(value))
        absent = []
        for feature in features:
            key, separator, value = str(feature).partition(":")
            if not separator or _normalized(value) not in fields.get(_normalized(key), set()):
                absent.append(str(feature))
        if absent:
            missing[str(product_id)] = absent
        else:
            verified.add(str(product_id))
    return verified, missing


def _valid_quotes(episode: dict[str, Any], expected_count: int,
                  viewed: set[str]) -> list[set[str]]:
    quotes = []
    for event in episode.get("trajectory", []):
        name, arguments = _action(event)
        if name != "calculate_bundle":
            continue
        observation = _observation(event)
        if "error" in observation:
            continue
        raw_ids = arguments.get("product_ids")
        if not isinstance(raw_ids, list):
            continue
        ids = [str(value) for value in raw_ids]
        id_set = set(ids)
        if len(ids) != expected_count or len(id_set) != expected_count:
            continue
        if not id_set <= viewed:
            continue
        # A voucher is conditional. An in-budget purchase is still possible
        # at the undiscounted price when the voucher does not apply.
        if observation.get("voucher_applicable") not in (True, False):
            continue
        if observation.get("within_budget") is not True:
            continue
        quotes.append(id_set)
    return quotes


def score_episode(
    episode: dict[str, Any],
    *,
    target_product_ids: Collection[str] | None = None,
    expected_memory_ids: Collection[str] = (),
    required_features_by_product_id: dict[str, Collection[str]] | None = None,
) -> dict[str, Any]:
    """Return a bounded reward and a complete audit breakdown.

    Exact, fully verified completion is always 1.  Otherwise the score is a
    combination of valid terminal target recall and small process components.
    Invalid/unfinished episodes are capped at 0.35.
    """
    target_values = (target_product_ids if target_product_ids is not None
                     else episode.get("target_product_ids_audit_only", []))
    target_list = [str(value) for value in target_values]
    target_set = set(target_list)
    if not target_set:
        raise ValueError("At least one target product ID is required")
    expected_count = len(target_set)

    submitted_raw = episode.get("submitted_product_ids", [])
    submitted_raw = [str(value) for value in submitted_raw] if isinstance(submitted_raw, list) else []
    submitted_set = set(submitted_raw)
    viewed = _viewed_ids(episode)
    count_exact = len(submitted_raw) == expected_count
    unique = len(submitted_set) == len(submitted_raw)
    all_submitted_inspected = bool(submitted_raw) and submitted_set <= viewed
    finished = episode.get("finish_reason") == "model_finish"

    question_type = episode.get("question_type")
    bundle = question_type == "add_on_deals"
    valid_quotes = _valid_quotes(episode, expected_count, viewed) if bundle else []
    quote_matches_submission = (not bundle) or submitted_set in valid_quotes
    valid_finish = bool(
        finished and count_exact and unique and all_submitted_inspected
        and quote_matches_submission
    )

    overlap = len(submitted_set & target_set)
    target_recall = overlap / expected_count
    required_features = required_features_by_product_id or {}
    verified_ids, missing_features = _verified_preference_ids(episode, required_features)
    supported_targets = target_set & verified_ids if required_features_by_product_id is not None else target_set
    supported_recall = len(submitted_set & supported_targets) / expected_count
    exact_valid_completion = valid_finish and submitted_set == target_set and supported_recall == 1.0

    expected_memories = {str(value) for value in expected_memory_ids}
    retrieved_memories = {
        str(item.get("id")) for item in episode.get("retrieved_memories", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    cited_memories = {
        str(evidence)
        for constraint in episode.get("final_constraints", [])
        if isinstance(constraint, dict)
        for evidence in constraint.get("evidence", [])
    }
    useful_memory_ids = expected_memories & retrieved_memories & cited_memories
    memory_denominator = len(expected_memories) or expected_count
    memory_use = len(useful_memory_ids) / memory_denominator

    target_view = len(viewed & supported_targets) / expected_count
    valid_calc = 1.0 if valid_quotes else 0.0

    repeat_count = 0
    other_invalid_count = 0
    for event in episode.get("trajectory", []):
        error = _observation(event).get("error")
        if error is None:
            continue
        if str(error).startswith("Identical action already succeeded"):
            repeat_count += 1
        else:
            other_invalid_count += 1

    terminal_component = 0.60 * supported_recall if valid_finish else 0.0
    memory_component = 0.10 * memory_use
    target_view_component = 0.15 * target_view
    valid_calc_component = 0.10 * valid_calc
    repeat_penalty = min(0.10, 0.01 * repeat_count)
    invalid_penalty = min(0.05, 0.01 * other_invalid_count)

    if exact_valid_completion:
        reward = 1.0
    else:
        raw_reward = (
            terminal_component + memory_component + target_view_component
            + valid_calc_component - repeat_penalty - invalid_penalty
        )
        reward = max(0.0, min(0.99, raw_reward))
        if not valid_finish:
            reward = min(0.35, reward)

    return {
        "reward": float(reward),
        "exact_valid_completion": bool(exact_valid_completion),
        "valid_finish": bool(valid_finish),
        "hard_checks": {
            "finished": bool(finished),
            "count_exact": bool(count_exact),
            "unique": bool(unique),
            "all_submitted_inspected": bool(all_submitted_inspected),
            "quote_matches_submission": bool(quote_matches_submission),
        },
        "components": {
            "target_recall": float(target_recall),
            "preference_supported_recall": float(supported_recall),
            "unverified_wanted_features": missing_features,
            "terminal": float(terminal_component),
            "memory_use": float(memory_use),
            "memory": float(memory_component),
            "target_view": float(target_view),
            "target_view_reward": float(target_view_component),
            "valid_calc": float(valid_calc),
            "valid_calc_reward": float(valid_calc_component),
            "repeat_penalty": float(repeat_penalty),
            "invalid_penalty": float(invalid_penalty),
        },
        "counts": {
            "target": expected_count,
            "submitted_raw": len(submitted_raw),
            "submitted_unique": len(submitted_set),
            "matched": overlap,
            "viewed": len(viewed),
            "relevant_memories_used": len(useful_memory_ids),
            "repeat_errors": repeat_count,
            "other_invalid_errors": other_invalid_count,
        },
    }


def _field_values(product: dict[str, Any], field: str) -> list[str]:
    attributes = product.get("attributes") or {}
    if not isinstance(attributes, dict):
        return []
    return [str(value) for key, values in attributes.items()
            if _normalized(key) == _normalized(field)
            for value in (values if isinstance(values, list) else [values])]


def _selected_variant(episode: dict[str, Any], product_id: str) -> int | None:
    for event in reversed(episode.get("trajectory", [])):
        name, arguments = _action(event)
        if name != "finish" or _observation(event).get("finished") is not True:
            continue
        for selection in arguments.get("selections", []):
            if not isinstance(selection, dict) or str(selection.get("product_id")) != product_id:
                continue
            try:
                return int(selection["variant_index"]) if "variant_index" in selection else None
            except (TypeError, ValueError):
                return None
    return None


def _feature_status(product: dict[str, Any], feature: str,
                    variant_index: int | None) -> dict[str, Any]:
    field, separator, wanted = str(feature).partition(":")
    result = {"feature": str(feature), "status": "unknown", "source": None,
              "observed_values": []}
    if not separator or not _normalized(wanted):
        return result
    wanted_norm = _normalized(wanted)
    static = _field_values(product, field)
    options = product.get("options") or []
    if not isinstance(options, list):
        options = []
    variants = []
    for option in options:
        values = []
        if isinstance(option, dict):
            for key, raw in option.items():
                if _normalized(key) == _normalized(field):
                    values.extend(str(v) for v in (raw if isinstance(raw, list) else [raw]))
        variants.append(values)
    if any(variants) and variant_index is not None:
        if not 0 <= variant_index < len(variants):
            return result
        chosen = variants[variant_index]
        if chosen:
            result.update(source=f"options[{variant_index}].{field}", observed_values=chosen,
                          status="supported" if wanted_norm in {_normalized(v) for v in chosen}
                          else "contradicted")
        return result
    if any(variants):
        if all(values and wanted_norm in {_normalized(v) for v in values} for values in variants):
            result.update(source=f"options[*].{field}", observed_values=variants[0], status="supported")
        else:
            result.update(source=f"options[unselected].{field}",
                          observed_values=sorted({v for values in variants for v in values}))
        return result
    if static:
        result.update(source=f"attributes.{field}", observed_values=static,
                      status="supported" if wanted_norm in {_normalized(v) for v in static}
                      else "contradicted")
    return result


def _open_world_feature_status(product: dict[str, Any], observation: dict[str, Any],
                               feature: str, variant_index: int | None) -> dict[str, Any]:
    """Use public evidence actually delivered by product_view, with cautious text fallback.

    A title can prove an explicit, non-variant feature such as Rechargeable.
    Text cannot resolve a selectable size/color or override a conflicting field.
    """
    result = _feature_status(product, feature, variant_index)
    if result["status"] == "supported":
        return result
    field, separator, wanted = str(feature).partition(":")
    if not separator:
        return result
    field_norm, wanted_norm = _normalized(field), _normalized(wanted)
    if result["source"] and str(result["source"]).startswith("options["):
        return result
    if result["source"] and result["status"] == "contradicted":
        # A warm-white bulb satisfies a request for warm light, while the
        # reverse is not necessarily true. Keep the original field as evidence.
        values = [_normalized(v) for v in result["observed_values"]]
        if field_norm == "light bulb colour" and wanted_norm == "warm" and any(
                value == "warm white" for value in values):
            return {**result, "status": "supported", "source": "semantic_attributes.light_bulb_colour"}
        return result
    if result["source"] is not None:
        return result
    if field_norm in {"size", "color", "colour", "color family", "colour family",
                      "basket capacity", "bag capacity range"} or not wanted_norm:
        return result
    # A full source description may contain text the policy never saw. Only
    # the title and the bounded excerpt returned by product_view are eligible.
    for source, raw in (("view.name", observation.get("name", "")),
                        ("view.description_excerpt", observation.get("description_excerpt", ""))):
        raw = str(raw)
        pattern = r"\b" + r"\W+".join(map(re.escape, wanted_norm.split())) + r"\b"
        match = re.search(pattern, raw, flags=re.I)
        if not match:
            continue
        preceding = _normalized(raw[max(0, match.start() - 35):match.start()]).split()[-4:]
        if any(word in {"no", "not", "non", "without", "unavailable"} for word in preceding):
            continue
        return {**result, "status": "supported", "source": source,
                "observed_values": [raw[max(0, match.start() - 45):match.end() + 45]]}
    return result


def _memory_feature_status(feature: str, constraints: list[dict[str, Any]],
                           retrieved: dict[str, str], expected_id: str) -> dict[str, Any]:
    _, separator, wanted = str(feature).partition(":")
    def canonical(text: str) -> str:
        # Simple morphology shared by the requested value and a user's wording,
        # e.g. Portable in a product field vs Portability in a dialogue turn.
        return re.sub(r"\b([a-z]+)ability\b", r"\1able", _normalized(text))

    value = canonical(wanted) if separator else ""
    user_text = retrieved.get(expected_id, "")
    user_sentences = [sentence.strip() for sentence in re.split(r"[.!?\n]+", user_text)
                      if sentence.strip()]
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        evidence = constraint.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        requirement = canonical(constraint.get("requirement", ""))
        if not value or not re.search(r"\b" + re.escape(value) + r"\b", requirement):
            continue
        # A negated requirement must not count as a recovered wanted preference.
        prefix = requirement[:requirement.find(value)].split()[-3:]
        suffix = requirement[requirement.find(value) + len(value):].split()[:4]
        if (any(word in {"not", "without", "avoid", "irrelevant"} for word in prefix)
                or any(word in {"not", "irrelevant", "optional", "unimportant"} for word in suffix)
                or "does not matter" in " ".join(suffix)):
            continue
        cited_sentences = []
        if expected_id in evidence:
            cited_sentences = user_sentences
        else:
            # A model may cite an exact user utterance instead of the history
            # document ID. Accept only text actually present in user turns.
            for citation in evidence:
                quoted = re.sub(r"^user:\s*", "", str(citation).strip(), flags=re.I)
                normalized_quote = _normalized(quoted)
                if normalized_quote and any(normalized_quote in _normalized(line)
                                            for line in user_text.splitlines()):
                    cited_sentences.append(quoted)
        for sentence in cited_sentences:
            text = canonical(sentence)
            if not re.search(r"\b" + re.escape(value) + r"\b", text):
                continue
            if re.search(r"\b(doesn t|don t|not|no|does not|do not|doesnt|dont)\b.{0,35}\b(matter|want|need|important)\b", text):
                continue
            if not re.search(r"\b(want|wanted|need|important|prefer|must|stick to|definitely|exactly)\b", text):
                continue
            return {"feature": feature, "status": "recalled", "memory_id": expected_id,
                    "memory_excerpt": sentence.strip()[:180]}
    return {"feature": feature, "status": "not_verified", "memory_id": expected_id}


def score_episode_grounded(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
    open_world: bool = False,
) -> dict[str, Any]:
    """Preference reward for new runs with independently sourced product records.

    Unknown fields or unselected variants never count as satisfied. The
    versioned open-world path also checks bounded text shown by product_view.
    """
    if not truth_items:
        raise ValueError("At least one preference slot is required")
    submitted = [str(x) for x in episode.get("submitted_product_ids", [])]
    viewed = _viewed_ids(episode)
    viewed_observations = {
        str(_observation(event)["product_id"]): _observation(event)
        for event in episode.get("trajectory", [])
        if _action(event)[0] == "view_product" and "error" not in _observation(event)
        and _observation(event).get("product_id") is not None
    }
    slot_count = len(truth_items)
    finished = episode.get("finish_reason") == "model_finish"
    count_exact = len(submitted) == slot_count
    unique = len(set(submitted)) == len(submitted)
    inspected = bool(submitted) and set(submitted) <= viewed
    bundle = episode.get("question_type") == "add_on_deals"
    quote_ok = not bundle or set(submitted) in _valid_quotes(episode, slot_count, viewed)
    valid_finish = finished and count_exact and unique and inspected and quote_ok

    provided = episode.get("provided_memories", [])
    retrieved = {str(m.get("id")): str(m.get("user_evidence_text", m.get("text", "")))
                 for m in [*episode.get("retrieved_memories", []), *provided]
                 if isinstance(m, dict)}
    constraints = episode.get("final_constraints", [])
    if provided:
        # The released Stage-2 dialogue is already given in full. Its broad
        # document ID alone does not identify which user statement supports a
        # particular preference. Only a verifiable user quote earns credit.
        provided_ids = {str(m.get("id")) for m in provided if isinstance(m, dict)}
        constraints_for_memory = [
            {**constraint, "evidence": [citation for citation in constraint.get("evidence", [])
                                        if citation not in provided_ids]}
            if isinstance(constraint, dict) and isinstance(constraint.get("evidence"), list)
            else constraint for constraint in constraints
        ]
    else:
        constraints_for_memory = constraints
    expected_ids = list(expected_memory_ids)
    if len(expected_ids) != slot_count:
        raise ValueError("Expected one memory ID per preference slot")
    memory_checks = [
        _memory_feature_status(feature, constraints_for_memory, retrieved, str(expected_ids[i]))
        for i, item in enumerate(truth_items)
        for feature in item.get("wanted_features", [])
    ]
    memory_rate = (sum(c["status"] == "recalled" for c in memory_checks) / len(memory_checks)
                   if memory_checks else 0.0)

    def assess(slot: dict[str, Any], product_id: str) -> dict[str, Any]:
        source = source_products_by_id.get(product_id)
        if source is None:
            checks = [{"feature": f, "status": "unknown", "source": "original_record_missing",
                       "observed_values": []} for f in slot.get("wanted_features", [])]
            category = "unknown"
        else:
            product = source.get("product", source)
            if open_world:
                observation = viewed_observations.get(product_id, {})
                checks = [_open_world_feature_status(product, observation, f,
                                                     _selected_variant(episode, product_id))
                          for f in slot.get("wanted_features", [])]
            else:
                checks = [_feature_status(product, f, _selected_variant(episode, product_id))
                          for f in slot.get("wanted_features", [])]
            wanted_category = _normalized(slot.get("category", ""))
            actual_category = _normalized(product.get("category", ""))
            category = ("supported" if wanted_category and wanted_category in actual_category
                        else "unknown" if not wanted_category else "contradicted")
        supported = sum(c["status"] == "supported" for c in checks)
        rate = supported / len(checks) if checks else 0.0
        if category != "supported":
            rate = 0.0
        return {"product_id": product_id, "reference_product_id": str(slot.get("product_id")),
                "category_status": category, "features": checks, "satisfaction": rate}

    best: list[dict[str, Any]] = []
    if count_exact and unique:
        for order in permutations(submitted):
            candidate = [assess(slot, pid) for slot, pid in zip(truth_items, order)]
            if not best or sum(x["satisfaction"] for x in candidate) > sum(x["satisfaction"] for x in best):
                best = candidate
    preference_rate = (sum(x["satisfaction"] for x in best) / slot_count if best else 0.0)
    target_set = {str(item.get("product_id")) for item in truth_items}
    reference_recall = len(set(submitted) & target_set) / slot_count if valid_finish else 0.0
    complete = bool(valid_finish and (open_world or set(submitted) == target_set) and best and all(
        item["category_status"] == "supported" and item["features"]
        and all(feature["status"] == "supported" for feature in item["features"])
        for item in best))
    # Memory credit requires value-level recall, not an ID intersection alone.
    memory_reward = 0.15 * memory_rate
    verified_view_rate = sum(pid in viewed and pid in source_products_by_id for pid in set(submitted)) / slot_count
    view_reward = 0.05 * min(1.0, verified_view_rate)
    # Historical versions use reference IDs as a small floor when fields are
    # incomplete. Open-world scoring keeps ID overlap diagnostic only.
    reference_floor = 0.0 if open_world else 0.10 * reference_recall
    preference_value = 0.50 * preference_rate if valid_finish else 0.0
    selection_reward = max(reference_floor, preference_value)
    completion_reward = 0.30 if complete else 0.0
    errors = [str(_observation(e).get("error")) for e in episode.get("trajectory", [])
              if _observation(e).get("error") is not None]
    repeat_penalty = min(0.05, 0.01 * sum(e.startswith("Identical action already succeeded") for e in errors))
    invalid_penalty = min(0.05, 0.01 * sum(not e.startswith("Identical action already succeeded") for e in errors))
    reward = max(0.0, min(1.0, memory_reward + view_reward + selection_reward + completion_reward
                           - repeat_penalty - invalid_penalty))
    if not valid_finish:
        reward = min(0.20, reward)
    result = {"reward": reward, "valid_finish": bool(valid_finish), "complete": complete,
            "exact_valid_completion": bool(complete and set(submitted) == target_set),
            "hard_checks": {"finished": finished, "count_exact": count_exact, "unique": unique,
                            "inspected": inspected, "quote_matches_submission": quote_ok},
            "components": {"grounded_preference": memory_reward, "memory": memory_reward,
                           "verified_view": view_reward,
                           "selection": selection_reward, "completion": completion_reward,
                           "repeat_penalty": repeat_penalty, "invalid_penalty": invalid_penalty},
            "selection_diagnostics": {"reference_recall": reference_recall,
                                      "reference_floor": reference_floor,
                                      "preference_value": preference_value},
            "memory_checks": memory_checks, "product_checks": best,
            "scoring_rule": ("grounded_preferences_open_world_v6" if open_world else
                             "grounded_preferences_stage2_context_v1" if provided
                             else "grounded_preferences_v3")}
    if open_world:
        result["grounded_completion"] = complete
        result["selection_diagnostics"]["reference_exact_match"] = bool(set(submitted) == target_set)
    return result


def score_episode_grounded_progress(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
    open_world: bool = False,
) -> dict[str, Any]:
    """Versioned Stage-2 reward with bounded, source-grounded inspection credit.

    This progress credit does not depend on the reference product ID. It only
    checks products the Agent actually inspected, and distinct products are
    assigned to distinct requested slots. A selectable option earns half the
    feature credit until the Agent chooses a variant in a valid finish.
    """
    result = score_episode_grounded(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids,
        open_world=open_world,
    )
    if not episode.get("provided_memories"):
        raise ValueError("Progress reward requires the supplied Stage-2 history")
    progress = 0.0
    if not result["valid_finish"]:
        viewed = sorted(_viewed_ids(episode) & source_products_by_id.keys())
        if viewed:
            slot_rates: list[dict[str, float]] = []
            for slot in truth_items:
                wanted_category = _normalized(slot.get("category", ""))
                features = slot.get("wanted_features", [])
                rates = {}
                for product_id in viewed:
                    source = source_products_by_id[product_id]
                    product = source.get("product", source)
                    actual_category = _normalized(product.get("category", ""))
                    if not wanted_category or wanted_category not in actual_category or not features:
                        continue
                    options = product.get("options") or []
                    if not isinstance(options, list):
                        options = []
                    observation = next((o for e in episode.get("trajectory", [])
                                        if _action(e)[0] == "view_product"
                                        and (o := _observation(e)).get("product_id") == product_id
                                        and "error" not in o), {})
                    feature_status = (lambda f, i: _open_world_feature_status(product, observation, f, i)
                                      if open_world else _feature_status(product, f, i))
                    direct = sum(feature_status(feature, None)["status"] == "supported"
                                 for feature in features)
                    possible = max((sum(feature_status(feature, index)["status"] == "supported"
                                        for feature in features)
                                    for index in range(len(options))), default=direct)
                    # Do not grant full credit for an option that has not been selected.
                    rates[product_id] = (direct + 0.5 * max(0, possible - direct)) / len(features)
                slot_rates.append(rates)

            def best_assignment(slot_index: int, used: frozenset[str]) -> float:
                if slot_index == len(slot_rates):
                    return 0.0
                best = best_assignment(slot_index + 1, used)
                for product_id, rate in slot_rates[slot_index].items():
                    if product_id not in used:
                        best = max(best, rate + best_assignment(slot_index + 1, used | {product_id}))
                return best

            progress = 0.10 * best_assignment(0, frozenset()) / len(truth_items)
    result["components"]["viewed_progress"] = progress
    result["reward"] = min(1.0, result["reward"] + progress)
    if not result["valid_finish"]:
        result["reward"] = min(0.20, result["reward"])
    result["scoring_rule"] = "grounded_preferences_stage2_progress_v1"
    return result


def score_episode_grounded_progress_v2(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
    open_world: bool = False,
) -> dict[str, Any]:
    """Progress reward with efficiency penalties reserved for complete tasks.

    Failed trajectories can receive bounded credit for grounded memory and
    inspected preference matches, but merely avoiding one duplicate action
    cannot create a GRPO advantage when every attempt failed in the same way.
    """
    result = score_episode_grounded_progress(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids, open_world=open_world,
    )
    if not result["complete"]:
        components = result["components"]
        positive = (components["grounded_preference"] + components["verified_view"]
                    + components["selection"] + components["completion"]
                    + components["viewed_progress"])
        result["reward"] = min(0.30 if not result["valid_finish"] else 0.99, positive)
        components["effective_repeat_penalty"] = 0.0
        components["effective_invalid_penalty"] = 0.0
    else:
        components = result["components"]
        components["effective_repeat_penalty"] = components["repeat_penalty"]
        components["effective_invalid_penalty"] = components["invalid_penalty"]
    result["scoring_rule"] = ("grounded_preferences_open_world_v6" if open_world
                              else "grounded_preferences_stage2_progress_v2")
    return result


def score_episode_grounded_open_world_v6(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """Versioned reward: verified alternatives can complete without a reference ID match."""
    return score_episode_grounded_progress_v2(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids, open_world=True,
    )


def score_episode_grounded_open_world_v7(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """Reward verified category and feature progress; reserve 1.0 for completion.

    Reference IDs remain diagnostics only. A recovered history checklist helps
    partial trajectories, but cannot reduce a fully verified purchase below 1.
    """
    result = score_episode_grounded_open_world_v6(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids,
    )
    memory_rate = (sum(check["status"] == "recalled" for check in result["memory_checks"])
                   / len(result["memory_checks"]) if result["memory_checks"] else 0.0)
    valid_finish = result["valid_finish"]
    complete = result["grounded_completion"]
    if valid_finish:
        checks = result["product_checks"]
        category_rate = sum(item["category_status"] == "supported" for item in checks) / len(truth_items)
        feature_rate = sum(item["satisfaction"] for item in checks) / len(truth_items)
        category_credit = 0.10 * category_rate
        feature_credit = 0.55 * feature_rate
        # A syntactically valid submission of the wrong category is not
        # shopping progress. The quote/finish credit requires a matched slot.
        finish_credit = 0.10 if category_rate > 0 else 0.0
        memory_credit = 0.05 * memory_rate
        # A complete purchase is the outcome we optimize. Checklist citations
        # are useful diagnostics, not an additional hard requirement.
        reward = 1.0 if complete else min(0.80, category_credit + feature_credit
                                          + finish_credit + memory_credit)
    else:
        viewed = _viewed_ids(episode) & source_products_by_id.keys()
        matched_slots = set()
        used_products = set()
        for slot_index, slot in enumerate(truth_items):
            wanted = _normalized(slot.get("category", ""))
            for product_id in sorted(viewed):
                product = source_products_by_id[product_id]
                product = product.get("product", product)
                if (wanted and wanted in _normalized(product.get("category", ""))
                        and product_id not in used_products):
                    matched_slots.add(slot_index)
                    used_products.add(product_id)
                    break
        category_credit = 0.05 * len(matched_slots) / len(truth_items)
        # v6's viewed_progress checks actual view observations, source fields,
        # and distinct slot assignments, with at most half credit for an
        # unselected option. Scale its 0.10 range to 0.20.
        feature_credit = 2.0 * result["components"]["viewed_progress"]
        finish_credit = 0.0
        memory_credit = 0.05 * memory_rate
        reward = min(0.30, category_credit + feature_credit + memory_credit)
    result["reward"] = reward
    result["components"]["category_progress_v7"] = category_credit
    result["components"]["feature_progress_v7"] = feature_credit
    result["components"]["valid_finish_v7"] = finish_credit
    result["components"]["memory_process_v7"] = memory_credit
    result["components"]["outcome_bonus_v7"] = 1.0 - min(1.0, category_credit + feature_credit
                                                             + finish_credit + memory_credit) if complete else 0.0
    result["scoring_rule"] = "grounded_preferences_open_world_v7"
    return result


def score_episode_grounded_open_world_v8(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """Distinguish verified completion from repeated premature submissions.

    Only a successful, fully grounded episode gets this process penalty. Failed
    episodes retain v7's evidence-based partial score so invalid-action noise
    alone cannot create a GRPO advantage.
    """
    result = score_episode_grounded_open_world_v7(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids,
    )
    premature = sum(
        _action(event)[0] == "finish"
        and "inspect every submitted product first" in str(_observation(event).get("error", ""))
        for event in episode.get("trajectory", [])
    )
    penalty = min(0.10, 0.02 * premature) if result["grounded_completion"] else 0.0
    result["reward"] = max(0.0, result["reward"] - penalty)
    result["components"]["premature_finish_count_v8"] = premature
    result["components"]["premature_finish_penalty_v8"] = penalty
    result["scoring_rule"] = "grounded_preferences_open_world_v8"
    return result


def score_episode_grounded_open_world_v9(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """Score only annotated preferences; reference categories are audit data.

    A field such as Types_of_Plaster_&_Bandages still needs that actual product
    field. The word "adhesive" in a tire patch title cannot prove it is a
    medical plaster. Other non-variant features may use visible title text.
    """
    if not truth_items:
        raise ValueError("At least one preference slot is required")
    submitted = [str(value) for value in episode.get("submitted_product_ids", [])]
    viewed_observations = {
        str(observation["product_id"]): observation
        for event in episode.get("trajectory", [])
        if _action(event)[0] == "view_product"
        and "error" not in (observation := _observation(event))
        and observation.get("product_id") is not None
    }
    viewed = set(viewed_observations)
    count_exact = len(submitted) == len(truth_items)
    unique = len(set(submitted)) == len(submitted)
    inspected = bool(submitted) and set(submitted) <= viewed
    bundle = episode.get("question_type") == "add_on_deals"
    quote_ok = not bundle or set(submitted) in _valid_quotes(episode, len(truth_items), viewed)
    finished = episode.get("finish_reason") == "model_finish"
    valid_finish = bool(finished and count_exact and unique and inspected and quote_ok)

    provided = episode.get("provided_memories", [])
    retrieved = {
        str(memory.get("id")): str(memory.get("user_evidence_text", memory.get("text", "")))
        for memory in [*episode.get("retrieved_memories", []), *provided]
        if isinstance(memory, dict)
    }
    constraints = episode.get("final_constraints", [])
    if provided:
        provided_ids = {str(memory.get("id")) for memory in provided if isinstance(memory, dict)}
        constraints = [
            {**constraint, "evidence": [citation for citation in constraint.get("evidence", [])
                                        if citation not in provided_ids]}
            if isinstance(constraint, dict) and isinstance(constraint.get("evidence"), list)
            else constraint for constraint in constraints
        ]
    expected_ids = list(expected_memory_ids)
    if len(expected_ids) != len(truth_items):
        raise ValueError("Expected one memory ID per preference slot")
    memory_checks = [
        _memory_feature_status(feature, constraints, retrieved, str(expected_ids[index]))
        for index, item in enumerate(truth_items)
        for feature in item.get("wanted_features", [])
    ]
    memory_rate = (sum(check["status"] == "recalled" for check in memory_checks)
                   / len(memory_checks) if memory_checks else 0.0)

    def feature_status(product_id: str, feature: str, variant_index: int | None) -> dict[str, Any]:
        source = source_products_by_id.get(product_id)
        if source is None:
            return {"feature": feature, "status": "unknown", "source": "original_record_missing",
                    "observed_values": []}
        observation = viewed_observations.get(product_id, {})
        # The original record establishes provenance, but only fields actually
        # delivered by view_product may justify a model's choice.
        product = {"attributes": observation.get("attributes", {}),
                   "options": observation.get("options", [])}
        result = _open_world_feature_status(
            product, observation, feature, variant_index)
        field = _normalized(str(feature).partition(":")[0])
        if field.startswith("types of ") and str(result.get("source", "")).startswith("view."):
            return {"feature": feature, "status": "unknown", "source": None,
                    "observed_values": []}
        return result

    def assess(slot: dict[str, Any], product_id: str, *, selected: bool) -> dict[str, Any]:
        features = slot.get("wanted_features", [])
        variant_index = _selected_variant(episode, product_id) if selected else None
        checks = [feature_status(product_id, feature, variant_index) for feature in features]
        rate = sum(check["status"] == "supported" for check in checks) / len(checks) if checks else 0.0
        return {"product_id": product_id, "reference_product_id_audit_only": str(slot.get("product_id")),
                "features": checks, "satisfaction": rate}

    product_checks: list[dict[str, Any]] = []
    if count_exact and unique:
        # Maximize supported user features, never overlap with the reference ID.
        for order in permutations(submitted):
            candidate = [assess(slot, product_id, selected=True)
                         for slot, product_id in zip(truth_items, order)]
            if not product_checks or sum(item["satisfaction"] for item in candidate) > sum(
                    item["satisfaction"] for item in product_checks):
                product_checks = candidate
    feature_rate = (sum(item["satisfaction"] for item in product_checks) / len(truth_items)
                    if product_checks else 0.0)
    complete = bool(valid_finish and product_checks and all(
        item["features"] and all(check["status"] == "supported" for check in item["features"])
        for item in product_checks))

    if valid_finish:
        feature_credit = 0.65 * feature_rate
        finish_credit = 0.10 if feature_rate > 0 else 0.0
        memory_credit = 0.05 * memory_rate
        reward = 1.0 if complete else min(0.80, feature_credit + finish_credit + memory_credit)
    else:
        # Inspecting a matching field earns bounded progress even if the Agent
        # never submits. Each viewed product can fill at most one requested slot.
        slot_rates: list[dict[str, float]] = []
        for slot in truth_items:
            rates = {}
            for product_id in sorted(viewed & source_products_by_id.keys()):
                direct = assess(slot, product_id, selected=False)["satisfaction"]
                options = viewed_observations[product_id].get("options") or []
                features = slot.get("wanted_features", [])
                possible = direct
                if isinstance(options, list) and options and features:
                    possible = max(
                        sum(feature_status(product_id, feature, index)["status"] == "supported"
                            for feature in features) / len(features)
                        for index in range(len(options))
                    )
                rates[product_id] = direct + 0.5 * max(0.0, possible - direct)
            slot_rates.append(rates)

        def best_progress(index: int, used: frozenset[str]) -> float:
            if index == len(slot_rates):
                return 0.0
            best = best_progress(index + 1, used)
            for product_id, rate in slot_rates[index].items():
                if product_id not in used:
                    best = max(best, rate + best_progress(index + 1, used | {product_id}))
            return best

        feature_credit = 0.25 * best_progress(0, frozenset()) / len(truth_items)
        finish_credit = 0.0
        memory_credit = 0.05 * memory_rate
        reward = min(0.30, feature_credit + memory_credit)

    premature = sum(
        _action(event)[0] == "finish"
        and "inspect every submitted product first" in str(_observation(event).get("error", ""))
        for event in episode.get("trajectory", [])
    )
    premature_penalty = min(0.10, 0.02 * premature) if complete else 0.0
    reward = max(0.0, reward - premature_penalty)
    target_ids = {str(item.get("product_id")) for item in truth_items}
    return {
        "reward": reward, "valid_finish": valid_finish, "complete": complete,
        "grounded_completion": complete,
        "exact_valid_completion": bool(complete and set(submitted) == target_ids),
        "hard_checks": {"finished": finished, "count_exact": count_exact,
                        "unique": unique, "inspected": inspected,
                        "quote_matches_submission": quote_ok},
        "components": {"annotated_feature_credit": feature_credit,
                       "valid_finish_credit": finish_credit,
                       "memory_process_credit": memory_credit,
                       "completion_bonus": (1.0 - min(1.0, feature_credit + finish_credit
                                                      + memory_credit)) if complete else 0.0,
                       "premature_finish_count": premature,
                       "premature_finish_penalty": premature_penalty},
        "selection_diagnostics": {"reference_recall": len(set(submitted) & target_ids)
                                  / len(truth_items) if valid_finish else 0.0,
                                  "reference_exact_match": bool(set(submitted) == target_ids),
                                  "annotated_feature_rate": feature_rate},
        "memory_checks": memory_checks, "product_checks": product_checks,
        "scoring_rule": "grounded_preferences_open_world_v9",
    }


def score_episode_grounded_open_world_v10(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """Penalize repeated work on an already verified, source-supported product.

    New searches, new product views, and checks on an incomplete product are
    never penalized here. An idle terminal state after all slots are ready or
    after an affordable quote costs one additional unit. Only observed pages,
    verified checks and public item slots are used, never reference IDs.
    """
    result = score_episode_grounded_open_world_v9(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids,
    )
    viewed: dict[str, dict[str, Any]] = {}
    recorded: dict[tuple[int, str], list[dict[str, Any]]] = {}
    ready_ids: set[str] = set()
    ready_by_slot: list[set[str]] = [set() for _ in truth_items]
    redundant_steps: list[int] = []
    completed_quote_sets: set[frozenset[str]] = set()
    ready_at_step: int | None = None
    affordable_quote_at_step: int | None = None
    quote_replacement_seen = False
    explored_after_ready = False
    attempted_finish_after_ready = False
    attempted_finish_after_quote = False
    attempted_quote_after_ready = False

    def has_distinct_ready_bundle() -> bool:
        def fill(slot: int, used: set[str]) -> bool:
            if slot == len(ready_by_slot):
                return True
            return any(fill(slot + 1, used | {pid}) for pid in ready_by_slot[slot] - used)
        return fill(0, set())

    def quoted_ids_are_ready(ids: Any) -> bool:
        if not isinstance(ids, list) or len(ids) != len(ready_by_slot):
            return False
        ids = [str(pid) for pid in ids]
        return len(set(ids)) == len(ids) and any(
            all(pid in ready_by_slot[slot] for slot, pid in enumerate(order))
            for order in permutations(ids))

    public_items_match = re.search(
        r"Product bundle includes:\s*(.*?)\.\s*Voucher\b",
        str(episode.get("question", "")), re.I | re.S)
    public_items = ([item.strip() for item in re.split(
        r",\s*(?:and\s+)?|\s+and\s+", public_items_match.group(1))
        if item.strip()] if public_items_match else [])

    def slot_for_check(item: Any) -> int | None:
        if len(truth_items) == 1 and not public_items:
            return 0
        matches = [index for index, name in enumerate(public_items)
                   if _normalized(name) == _normalized(item)]
        return matches[0] if len(matches) == 1 and matches[0] < len(truth_items) else None

    def ready(slot_index: int, product_id: str) -> bool:
        if product_id not in viewed or product_id not in source_products_by_id:
            return False
        features = truth_items[slot_index].get("wanted_features", [])
        checks = recorded.get((slot_index, product_id), [])
        if not features or not checks:
            return False
        observation = viewed[product_id]
        product = {"attributes": observation.get("attributes", {}),
                   "options": observation.get("options", [])}
        for feature in features:
            if (not _normalized(feature) or
                    _open_world_feature_status(product, observation, feature, None)["status"] != "supported"):
                return False
            if not any(
                check.get("evidence_verified") is True and
                check.get("status") == "supported" and
                _normalized(feature) == _normalized(check.get("need", ""))
                for check in checks
            ):
                return False
        return True

    for ordinal, event in enumerate(episode.get("trajectory", []), 1):
        name, arguments = _action(event)
        observation = _observation(event)
        product_id = str(arguments.get("product_id", ""))
        error = str(observation.get("error", ""))
        step = int(event.get("step", ordinal))
        if ready_at_step is not None and name == "finish":
            attempted_finish_after_ready = True
        if affordable_quote_at_step is not None and name == "finish":
            attempted_finish_after_quote = True
        if (ready_at_step is not None and name == "search_products" and not error):
            explored_after_ready = True
            if affordable_quote_at_step is not None:
                quote_replacement_seen = True
        if (ready_at_step is not None and name == "calculate_bundle" and
                quoted_ids_are_ready(arguments.get("product_ids"))):
            attempted_quote_after_ready = True
        if (name == "view_product" and product_id in ready_ids and error and
                event.get("compaction_tier") in {None, "none", "compact"}):
            redundant_steps.append(step)
        elif (name == "record_check" and product_id in ready_ids and
              "identical action already completed" in error.casefold() and
              event.get("compaction_tier") in {None, "none", "compact"}):
            redundant_steps.append(step)
        elif (name == "calculate_bundle" and error and
              any(phrase in error.casefold() for phrase in
                  ("already quoted", "identical action already completed")) and
              isinstance(arguments.get("product_ids"), list) and
              frozenset(str(pid) for pid in arguments["product_ids"]) in completed_quote_sets):
            redundant_steps.append(step)
        if name == "view_product" and "error" not in observation and observation.get("product_id") is not None:
            viewed[str(observation["product_id"])] = observation
            if ready_at_step is not None:
                explored_after_ready = True
            if affordable_quote_at_step is not None:
                quote_replacement_seen = True
        elif name == "record_check" and "error" not in observation:
            slot = slot_for_check(observation.get("item", arguments.get("item")))
            if slot is not None and str(observation.get("recorded_product_id", product_id)) == product_id:
                recorded[(slot, product_id)] = observation.get("checks", [])
                if ready(slot, product_id):
                    ready_ids.add(product_id)
                    ready_by_slot[slot].add(product_id)
                else:
                    ready_ids.discard(product_id)
                    ready_by_slot[slot].discard(product_id)

        if ready_at_step is None and has_distinct_ready_bundle():
            ready_at_step = step
        if (name == "calculate_bundle" and not error and
                isinstance(arguments.get("product_ids"), list)):
            completed_quote_sets.add(frozenset(
                str(pid) for pid in arguments["product_ids"]))
        if (name == "calculate_bundle" and not error and
                observation.get("within_budget") is True and
                quoted_ids_are_ready(arguments.get("product_ids"))):
            affordable_quote_at_step = step
            quote_replacement_seen = False

    last_step = len(episode.get("trajectory", []))
    stalled_before_quote = bool(
        episode.get("question_type") == "add_on_deals" and
        ready_at_step is not None and last_step > ready_at_step and
        not attempted_quote_after_ready and not explored_after_ready and
        not attempted_finish_after_ready and
        not result["valid_finish"])
    stalled_after_quote = bool(
        affordable_quote_at_step is not None and last_step > affordable_quote_at_step and
        not quote_replacement_seen and not attempted_finish_after_quote and
        not result["valid_finish"])
    stalled_single = bool(
        episode.get("question_type") != "add_on_deals" and
        ready_at_step is not None and last_step > ready_at_step and
        not explored_after_ready and not attempted_finish_after_ready and
        not result["valid_finish"])
    terminal_stall_count = int(stalled_before_quote or stalled_after_quote or stalled_single)
    penalty = min(1.0, 0.20 * (len(redundant_steps) + terminal_stall_count))
    result["reward"] = max(0.0, result["reward"] - penalty)
    result["components"]["ready_repeat_count_v10"] = len(redundant_steps)
    result["components"]["ready_repeat_penalty_v10"] = penalty
    result["components"]["terminal_stall_count_v10"] = terminal_stall_count
    result["components"]["stalled_before_quote_v10"] = stalled_before_quote
    result["components"]["stalled_after_quote_v10"] = stalled_after_quote
    result["components"]["stalled_single_v10"] = stalled_single
    result["ready_repeat_steps_v10"] = redundant_steps
    result["scoring_rule"] = "grounded_preferences_open_world_v10"
    return result


def score_episode_grounded_open_world_v11(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """v10 plus a small penalty for clear, model-authored false negatives.

    Only a tool-reported conflict between a contradicted check and a visible,
    unambiguously matching value counts. Unknown/ambiguous evidence does not.
    Repeating the same mistake after feedback costs more; v10 repeat steps are
    excluded so one action is not charged twice.
    """
    result = score_episode_grounded_open_world_v10(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids,
    )
    seen: set[tuple[str, str]] = set()
    charged_steps: list[int] = []
    penalty = 0.0
    already_charged = set(result.get("ready_repeat_steps_v10", []))
    for step_index, event in enumerate(episode.get("trajectory", []), 1):
        observation = event.get("observation") or {}
        step = event.get("step", step_index)
        if (observation.get("error") != "status_evidence_conflict" or
                step in already_charged):
            continue
        product_id = str(observation.get("product_id", ""))
        keys = {(product_id, str(item.get("need", "")).casefold())
                for item in observation.get("conflicts", [])
                if isinstance(item, dict) and item.get("reported_status") == "contradicted"
                and item.get("need") and product_id}
        if not keys:
            continue
        penalty += 0.10 if keys.issubset(seen) else 0.05
        seen.update(keys)
        charged_steps.append(step)
    penalty = round(min(0.20, penalty), 4)
    result["reward"] = max(0.0, result["reward"] - penalty)
    result["components"]["false_negative_check_penalty_v11"] = penalty
    result["false_negative_check_steps_v11"] = charged_steps
    result["scoring_rule"] = "grounded_preferences_open_world_v11"
    return result


def _public_requested_items(episode: dict[str, Any], expected_count: int | None = None) -> list[str]:
    """Read item types from the public request, never from reference products."""
    question = str(episode.get("question", ""))
    if episode.get("question_type") != "add_on_deals":
        lines = [line.strip() for line in question.splitlines() if line.strip()]
        return [lines[-1]] if lines else []
    match = re.search(r"Product bundle includes:\s*(.*?)\.\s*Voucher\b",
                      question, re.I | re.S)
    if not match:
        return []
    # A conjunction inside an item name ("wallet and belt set") must not
    # become an extra item slot. Commas are the primary public delimiters.
    parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
    parts = [re.sub(r"^and\s+", "", part, flags=re.I).strip() for part in parts]
    if expected_count is not None and len(parts) < expected_count and parts:
        last = re.split(r"\s+and\s+", parts[-1], maxsplit=1, flags=re.I)
        if len(last) == 2:
            parts = [*parts[:-1], *[part.strip() for part in last]]
    return parts


def _public_item_fit(item: str, observation: dict[str, Any]) -> dict[str, Any]:
    """Conservative visible type/use check; no reference category comparison.

    Attribute values alone do not establish the product's purpose. When the
    visible title/category/description does not establish the requested item,
    return unknown rather than declaring the product wrong.
    """
    tokens = re.findall(r"[a-z0-9]+", item.casefold())
    tokens = [token for token in tokens if token not in {"a", "an", "the", "of", "for"}]
    visible = " ".join(str(observation.get(key) or "") for key in
                       ("name", "category", "description_excerpt"))
    words = set(re.findall(r"[a-z0-9]+", visible.casefold()))
    words.update(re.findall(r"[a-z0-9]+", re.sub(
        r"(?<=[a-z])(?=[A-Z])", " ", visible).casefold()))
    def matches(token: str) -> bool:
        variants = {token}
        if token.endswith("ies"):
            variants.add(token[:-3] + "y")
        elif token.endswith("y"):
            variants.add(token[:-1] + "ies")
        elif token.endswith("s"):
            variants.add(token[:-1])
        else:
            variants.add(token + "s")
        if token == "leaf":
            variants.add("leaves")
        if token == "blower":
            variants.update({"blow", "blowing"})
        if token == "watch":
            variants.add("watches")
        if token == "swim":
            variants.add("swimming")
        return bool(variants & words) or any(
            word.endswith(token) and len(word) - len(token) <= 5 for word in words)
    matched = [token for token in tokens if matches(token)]
    return {"requested_item": item,
            "status": "supported" if tokens and len(matched) == len(tokens) else "unknown",
            "matched_terms": matched, "missing_terms": [token for token in tokens if token not in matched],
            "source": "view_product.name/category/description_excerpt"}


def score_episode_grounded_open_world_v12(
    episode: dict[str, Any], *, truth_items: list[dict[str, Any]],
    source_products_by_id: dict[str, dict[str, Any]],
    expected_memory_ids: Collection[str],
) -> dict[str, Any]:
    """v11 with a public item-purpose gate on the completion bonus.

    This is an opt-in offline rule. Unknown purpose keeps attribute credit but
    cannot receive full completion credit. It does not require reference IDs or
    categories and does not change the Agent's tools or observations.
    """
    result = score_episode_grounded_open_world_v11(
        episode, truth_items=truth_items,
        source_products_by_id=source_products_by_id,
        expected_memory_ids=expected_memory_ids,
    )
    items = _public_requested_items(episode, len(truth_items))
    checks = result["product_checks"]
    observed = {str(observation["product_id"]): observation
                for event in episode.get("trajectory", [])
                if _action(event)[0] == "view_product"
                and "error" not in (observation := _observation(event))
                and observation.get("product_id") is not None}
    fit = []
    if len(items) == len(checks) == len(truth_items):
        fit = [{"product_id": check["product_id"],
                **_public_item_fit(item, observed.get(check["product_id"], {}))}
               for item, check in zip(items, checks)]
    elif result["complete"]:
        fit = [{"status": "unparsed", "source": "public_request",
                "reason": "public item slots could not be aligned with submitted products"}]
    result["public_item_fit_v12"] = fit
    if result["complete"] and fit and any(item["status"] != "supported" for item in fit):
        components = result["components"]
        partial = min(0.80, components["annotated_feature_credit"] +
                      components["valid_finish_credit"] +
                      components["memory_process_credit"])
        deductions = (components["premature_finish_penalty"] +
                      components["ready_repeat_penalty_v10"] +
                      components["false_negative_check_penalty_v11"])
        result["reward"] = max(0.0, partial - deductions)
        result["complete"] = False
        result["grounded_completion"] = False
        result["exact_valid_completion"] = False
        components["completion_bonus"] = 0.0
    result["scoring_rule"] = "grounded_preferences_open_world_v12"
    return result

