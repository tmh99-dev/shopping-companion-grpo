"""tools.requirements: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from decimal import Decimal
from typing import Any
import copy
import re


def requested_bundle_items(question: str) -> list[str]:
    """Extract only the item names explicitly stated in the public request."""
    match = re.search(r"Product bundle includes:\s*(.*?)\.\s*Voucher\b", question, re.I | re.S)
    if not match:
        return []
    return [item.strip() for item in re.split(r",\s*(?:and\s+)?|\s+and\s+", match.group(1))
            if item.strip()]


def validate_public_requirements(question: str, items: Any) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Keep every public product occurrence, including repeated product types.

    This checks only the current request's item names and order. Historical
    preferences remain model-authored and are never supplied from labels.
    """
    requested = requested_bundle_items(question)
    if not isinstance(items, list) or not items or any(not isinstance(x, dict) for x in items):
        return None, {"error": "set_requirements requires non-empty item entries"}
    if not requested:
        lines = [line.strip() for line in question.splitlines() if line.strip()]
        if len(lines) == 2 and lines[0].startswith("Current Date:"):
            if len(items) != 1:
                return None, {"error": "single_product_requires_one_item_slot",
                              "public_item": lines[1],
                              "submitted_item_count": len(items),
                              "next_step": "Call set_requirements again with one item entry. Combine all wanted features of this single product in its requirement, not one entry per feature."}
            validated = [{**items[0], "slot_index": 1}]
        else:
            validated = items
    else:
        named = [str(x.get("item", "")).strip() for x in items]
        if len(named) != len(requested) or any(a.casefold() != b.casefold()
                                               for a, b in zip(named, requested)):
            return None, {
                "error": "requirements_do_not_cover_public_item_slots",
                "public_item_slots": [{"slot_index": i + 1, "item": item}
                                      for i, item in enumerate(requested)],
                "submitted_items": named,
                "next_step": "Call set_requirements again as {\"items\":[...]}, with exactly one entry per numbered public product slot in order. Put all confirmed features of one product occurrence inside its single requirement; do not make one entry per feature. Repeated product names still need separate entries and separate user evidence.",
            }
        validated = [{**item, "slot_index": i + 1} for i, item in enumerate(items)]
    if any(not str(item.get("requirement", "")).strip() or
           not isinstance(item.get("evidence"), list) or not item["evidence"]
           for item in items):
        return None, {"error": "requirement_or_user_evidence_missing",
                      "next_step": "For every public item, write its confirmed wanted features and at least one user quote. If the user truly expressed no preference, cite that user statement instead of leaving evidence empty."}
    return validated, None


def missing_public_requirement_slots(question: str, constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requested = requested_bundle_items(question)
    missing = []
    for index, name in enumerate(requested, 1):
        if not any(isinstance(entry, dict) and
                   str(entry.get("item", "")).casefold() == name.casefold() and
                   (entry.get("slot_index") == index or
                    (entry.get("slot_index") is None and requested.count(name) == 1))
                   for entry in constraints):
            missing.append({"slot_index": index, "item": name})
    return missing


def has_repeated_public_item_types(question: str) -> bool:
    items = [item.casefold() for item in requested_bundle_items(question)]
    return len(items) != len(set(items))


def user_quote_is_visible(quote: str, history_docs: list[dict[str, Any]]) -> bool:
    """Verify a short model quote against released user turns, not labels."""
    phrase = re.findall(r"[a-z0-9]+", quote.casefold())
    if len(phrase) < 3:
        return False
    for doc in history_docs:
        for line in str(doc.get("user_evidence_text", "")).splitlines():
            words = re.findall(r"[a-z0-9]+", line.casefold())
            if any(words[i:i + len(phrase)] == phrase
                   for i in range(len(words) - len(phrase) + 1)):
                return True
    return False


def public_history_section_index(history_docs: list[dict[str, Any]],
                                 constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A navigation index of released dialogue blocks; no preference extraction."""
    rows = []
    for doc in history_docs:
        raw = str(doc.get("text", ""))
        blocks = re.split(r"(?=^\[Date:\s*)", raw, flags=re.M)
        for block in blocks:
            date = re.match(r"\[Date:\s*([^\]]+)\]", block)
            user_lines = re.findall(r"(?im)^user:\s*(.+)$", block)
            if not date or not user_lines:
                continue
            words = re.findall(r"[a-z0-9]+", " ".join(user_lines).casefold())
            used = []
            for entry in constraints:
                for quote in entry.get("evidence", []):
                    phrase = re.findall(r"[a-z0-9]+", str(quote).casefold())
                    if len(phrase) >= 3 and any(
                        words[i:i + len(phrase)] == phrase
                        for i in range(len(words) - len(phrase) + 1)):
                        used.append(entry.get("slot_index"))
                        break
            rows.append({"date": date.group(1),
                         "opening_user_words": user_lines[0][:115],
                         "used_by_slots": sorted(set(index for index in used if isinstance(index, int)))})
    return rows[:10]


def unused_public_history_for_slots(history_docs: list[dict[str, Any]],
                                    constraints: list[dict[str, Any]],
                                    remaining_slot_count: int) -> str | None:
    """Hide already cited dialogue blocks only while other public slots need filling.

    This uses model-authored exact quotes and released dialogue text. If too few
    uncited blocks remain, keep the complete history instead of guessing a map.
    """
    if not constraints or not remaining_slot_count:
        return None
    remaining = []
    for doc in history_docs:
        blocks = re.split(r"(?=^\[Date:\s*)", str(doc.get("text", "")), flags=re.M)
        for block in blocks:
            if not re.match(r"\[Date:\s*", block):
                continue
            user_lines = re.findall(r"(?im)^user:\s*(.+)$", block)
            words = re.findall(r"[a-z0-9]+", " ".join(user_lines).casefold())
            cited = any(
                len(phrase) >= 3 and any(words[i:i + len(phrase)] == phrase
                                         for i in range(len(words) - len(phrase) + 1))
                for entry in constraints for quote in entry.get("evidence", [])
                for phrase in [re.findall(r"[a-z0-9]+", str(quote).casefold())]
            )
            if not cited:
                remaining.append(block.strip())
    if len(remaining) < remaining_slot_count:
        return None
    return "\n\n".join(remaining)


def validate_public_bundle_count(question: str, expected: int) -> None:
    """Reject a bundle label that requires more items than the request names."""
    items = requested_bundle_items(question)
    if not items or expected > len(items):
        raise ValueError(
            f"Bundle label requires {expected} products but the public request names "
            f"only {len(items)}; exclude or audit this task before rollout")


def _leather_title_evidence(observation: dict[str, Any]) -> bool:
    title = str(observation.get("name", ""))
    mentions = list(re.finditer(r"\bleather\b", title, re.I))
    disqualifiers = {"faux", "pu", "synthetic", "vegan", "imitation", "fake"}
    return any(not disqualifiers.intersection(
        re.findall(r"[a-z]+", title[:match.start()].lower())[-2:])
        for match in mentions)


def _normalized_check_text(value: Any) -> str:
    """Treat hyphenated and spaced spelling alike when comparing a cited value."""
    return " ".join(re.sub(r"[-‐‑‒–—]", " ", str(value).casefold()).split())


def _literal_need_matches_value(need: str, value: str) -> bool:
    """Flag only literal agreement, not a model's broader semantic judgment."""
    wanted = re.fullmatch(r"\s*[A-Za-z_]\w*\s*:\s*(.+?)\s*", need)
    literal = wanted.group(1) if wanted else need
    return bool(value and _normalized_check_text(literal) == _normalized_check_text(value))


def _literal_need_matches_observed(need: str, observed: Any) -> bool:
    """A required value may be one exact member of a visible attribute list."""
    values = observed if isinstance(observed, list) else [observed]
    return any(value is not None and _literal_need_matches_value(need, str(value))
               for value in values)


def _clear_need_matches_observed(need: str, source: str, observed: Any) -> bool:
    """Recognize only unambiguous visible agreement, including a 4 MP field value of 4."""
    if _literal_need_matches_observed(need, observed):
        return True
    if source.casefold() != "attributes.megapixels":
        return False
    match = re.fullmatch(
        r"\s*(?:(?:megapixels?|mp)\s*:\s*)?(\d+(?:\.\d+)?)\s*"
        r"(?:mp|megapixels?)?\s*", need, re.I)
    if not match:
        return False
    wanted = Decimal(match.group(1))
    values = observed if isinstance(observed, list) else [observed]
    return any(value is not None and
               re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(value)) and
               Decimal(str(value).strip()) == wanted for value in values)


def _model_declared_field_values(item: dict[str, Any]) -> list[tuple[str, str]]:
    """Read only literal FIELD: value spans the model quoted as wanted evidence."""
    pairs = []
    for declared in item.get("evidence") or []:
        raw = str(declared)
        literal = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*:\s*(.+?)\s*", raw)
        quoted = re.search(r'["“]([A-Za-z_][\w]*\s*:\s*[^"”]+)["”]', raw)
        match = literal or (re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*:\s*(.+?)\s*",
                                          quoted.group(1)) if quoted else None)
        if match:
            pair = (match.group(1), match.group(2))
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def _visible_check(update: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Keep a model's judgment only when its cited field/value was visible."""
    need = str(update.get("need", "")).strip()[:100]
    status = str(update.get("status", "unknown")).strip().casefold()
    source = str(update.get("source", "")).strip()[:100]
    value = str(update.get("value", "")).strip()[:120]
    if status not in {"supported", "available_unselected", "contradicted", "unknown"}:
        status = "unknown"
    result = {"need": need, "status": status, "source": source, "value": value,
              "evidence_verified": False}
    if not need:
        result["status"] = "unknown"
        return result
    if status == "unknown" and need.casefold() == "leather" and _leather_title_evidence(observation):
        result.update(status="supported", source="name", value="Leather",
                      evidence_verified=True, evidence_recovered_from="name")
        return result
    observed: Any = None
    option_source = False
    if source.startswith("attributes."):
        field = source[len("attributes."):]
        observed = next((v for k, v in (observation.get("attributes") or {}).items()
                         if k.casefold() == field.casefold()), None)
    elif (match := (re.fullmatch(r"options\[(\d+)\]\.fields\.([^\s]+)", source) or
                    re.fullmatch(r"options\[(\d+)\]\.([^\s]+)", source) or
                    re.fullmatch(r"options\.variant_index\.(\d+)\.fields\.([^\s]+)", source))):
        index, field = int(match.group(1)), match.group(2)
        option_source = True
        result["source"] = f"options[{index}].{field}"
        options = observation.get("options") or []
        if isinstance(options, list) and 0 <= index < len(options) and isinstance(options[index], dict):
            observed = next((v for k, v in options[index].items()
                             if k.casefold() == field.casefold()), None)
        # A displayed option is not yet a selected specification.
        if status == "supported":
            result["status"] = "available_unselected"
        elif status == "unknown":
            result["status"] = "available_unselected"
        elif status == "contradicted":
            # One mismatching option does not disqualify a product that may
            # offer the requested value in another selectable variant.
            result["status"] = "unknown"
    elif source in {"name", "description_excerpt"}:
        observed = observation.get(source)
    if isinstance(observed, list):
        verified = any(str(item).casefold() == value.casefold() for item in observed)
        if not verified and len(observed) > 1:
            # The visible page displays list-valued attributes together. A
            # model may cite the complete list as one comma-separated value;
            # accept it only when every listed element matches in order.
            verified = _normalized_check_text(value) == _normalized_check_text(
                ", ".join(map(str, observed)))
    else:
        verified = bool(value and observed is not None
                        and value.casefold() in str(observed).casefold())
    if status == "unknown" and not option_source:
        if verified and _clear_need_matches_observed(need, source, observed):
            result["status_evidence_conflict"] = True
            result["cited_field_found"] = True
            result["observed_value"] = next(
                (str(item) for item in observed if str(item).casefold() == value.casefold()),
                value) if isinstance(observed, list) else str(observed)
            result["evidence_error"] = "visible cited value literally matches the need, but status is unknown"
        return result
    if not verified:
        if (status == "supported" and need.casefold() == "leather" and
                _leather_title_evidence(observation)):
            result.update(status="supported", source="name", value="Leather",
                          evidence_verified=True, evidence_recovered_from="name")
            return result
        result["status"] = "unknown"
        result["evidence_error"] = "cited field/value was not in the inspected product"
    elif result["status"] in {"supported", "available_unselected"} and not (
            _normalized_check_text(need) in _normalized_check_text(value) or
            _normalized_check_text(value) in _normalized_check_text(need) or
            (isinstance(observed, list) and
             _literal_need_matches_observed(need, observed))):
        result["status"] = "unknown"
        result["evidence_error"] = "cited value does not match the claimed need"
    elif result["status"] == "contradicted" and _clear_need_matches_observed(
            need, source, observed):
        result["status"] = "unknown"
        result["status_evidence_conflict"] = True
        result["cited_field_found"] = True
        result["observed_value"] = value
        result["evidence_error"] = "cited value does not contradict the claimed need"
    else:
        result["evidence_verified"] = True
        if option_source and result["status"] == "available_unselected":
            result["variant_index"] = index
    return result


def record_check_status_conflicts(checks: list[Any], observation: dict[str, Any]) -> list[dict[str, str]]:
    """Explain literal status/citation conflicts before recording model checks."""
    conflicts = []
    for check in checks[:8]:
        if not isinstance(check, dict):
            continue
        verified = _visible_check(check, observation)
        if verified.get("status_evidence_conflict"):
            conflicts.append({
                **{key: str(verified[key]) for key in
                   ("need", "source", "observed_value")},
                "reported_status": str(check.get("status", "unknown")).casefold(),
            })
    return conflicts


def merge_constraint_updates(existing: list[dict[str, Any]], updates: list[Any],
                             viewed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge model-authored slot and product checks without erasing other slots."""
    merged = copy.deepcopy([item for item in existing if isinstance(item, dict)])
    def key_for(item: dict[str, Any]) -> tuple[str, Any]:
        slot = item.get("slot_index")
        return ("slot", int(slot)) if isinstance(slot, int) and slot > 0 else (
            "item", str(item.get("item", "")).strip().casefold())

    by_item = {key_for(item): item for item in merged}
    for update in updates[:10]:
        if not isinstance(update, dict):
            continue
        item_name = str(update.get("item", "")).strip()[:100]
        if not item_name:
            continue
        key = key_for(update)
        if key not in by_item:
            if len(merged) >= 10:
                break
            current = {"item": item_name, "requirement": "", "evidence": [], "candidate_checks": {}}
            if key[0] == "slot":
                current["slot_index"] = key[1]
            by_item[key] = current
            merged.append(current)
        current = by_item[key]
        if isinstance(update.get("requirement"), str) and update["requirement"].strip():
            new_requirements = [part.strip()[:120] for part in
                                update["requirement"].split(";") if part.strip()]
            known = [part.strip() for part in str(current.get("requirement", "")).split(";")
                     if part.strip()]
            for new_requirement in new_requirements:
                if new_requirement.casefold() not in {part.casefold() for part in known}:
                    known.append(new_requirement)
            current["requirement"] = "; ".join(known)[:500]
        if isinstance(update.get("evidence"), list) and update["evidence"]:
            known_evidence = [str(value) for value in current.get("evidence", [])]
            for value in update["evidence"]:
                if isinstance(value, str) and value not in known_evidence:
                    known_evidence.append(value)
            current["evidence"] = known_evidence[:15]
        product_id = str(update.get("product_id", "")).strip()
        checks = update.get("checks")
        if not product_id or not isinstance(checks, list):
            continue
        candidates = current.setdefault("candidate_checks", {})
        if product_id not in candidates and len(candidates) >= 5:
            continue
        prior = {str(check.get("need", "")).casefold(): check
                 for check in candidates.get(product_id, []) if isinstance(check, dict)}
        observation = viewed.get(product_id, {})
        for check in checks[:8]:
            if not isinstance(check, dict) or not str(check.get("need", "")).strip():
                continue
            normalized = _visible_check(check, observation)
            need_key = normalized["need"].casefold()
            previous = prior.get(need_key, {})
            # A later invalid citation must not erase a check already backed
            # by the product page saved in this same episode.
            if previous.get("evidence_verified") and not normalized.get("evidence_verified"):
                continue
            prior[need_key] = normalized
        candidates[product_id] = list(prior.values())[:8]
    return merged
