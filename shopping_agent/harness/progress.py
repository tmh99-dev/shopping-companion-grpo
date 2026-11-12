"""Observable information gain used by the bounded step extension gate."""


def useful_progress_steps(trajectory: list[dict]) -> list[int]:
    """Return steps that revealed new information, without target IDs or labels."""
    seen_search_ids: set[str] = set()
    seen_views: set[str] = set()
    seen_checks: set[tuple[str, str, str, str]] = set()
    seen_quotes: set[tuple[str, ...]] = set()
    progress = []
    for ordinal, event in enumerate(trajectory, 1):
        action = event.get("action") or {}
        arguments = action.get("arguments") or {}
        observation = event.get("observation") or {}
        if "error" in observation:
            continue
        name = action.get("name")
        useful = False
        if name == "search_products":
            ids = {str(row["product_id"]) for row in observation.get("results", [])
                   if isinstance(row, dict) and row.get("product_id") is not None}
            useful = bool(ids - seen_search_ids)
            seen_search_ids |= ids
        elif name == "view_product":
            pid = str(observation.get("product_id", ""))
            useful = bool(pid and pid not in seen_views)
            if pid:
                seen_views.add(pid)
        elif name == "record_check":
            pid = str(observation.get("recorded_product_id", arguments.get("product_id", "")))
            item = str(observation.get("item", arguments.get("item", "")))
            checks = {
                (pid, item, str(check.get("need", "")), str(check.get("status", "")))
                for check in observation.get("checks", []) if isinstance(check, dict)
                and check.get("evidence_verified") is True
                and check.get("status") in {"supported", "contradicted"}
            }
            useful = bool(checks - seen_checks)
            seen_checks |= checks
        elif name == "calculate_bundle":
            ids = observation.get("product_ids")
            if (isinstance(ids, list) and ids and
                    observation.get("voucher_applicable") in (True, False) and
                    observation.get("within_budget") in (True, False)):
                key = tuple(sorted(str(pid) for pid in ids))
                useful = key not in seen_quotes
                seen_quotes.add(key)
        if useful:
            progress.append(int(event.get("step", ordinal)))
    return progress


def recent_useful_progress(trajectory: list[dict], step_limit: int) -> list[int]:
    """Evidence for one more five-step block at the current deadline."""
    return [step for step in useful_progress_steps(trajectory)
            if step_limit - 2 <= step <= step_limit]
