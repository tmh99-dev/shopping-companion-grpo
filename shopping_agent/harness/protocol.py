"""harness.protocol: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from typing import Any
import json
import re


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    # A sampled policy sometimes doubles only the opening quote of an
    # evidence string: [""Black ..."]. Repair that narrow syntax error rather
    # than dropping an otherwise complete tool action. Valid JSON above is
    # never changed, and all other malformed output remains invalid.
    repaired = re.sub(r'(\[\s*|,\s*)""(?=[A-Za-z])', r'\1"', text)
    if repaired != text:
        try:
            value = json.loads(repaired)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def parse_native_tool_call(text: str, repair_syntax: bool = False) -> dict[str, Any] | None:
    """Accept one Qwen tool call; optionally repair audited syntax-only slips."""
    match = re.fullmatch(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", text, re.S)
    repair = None
    if not match:
        # A complete single JSON object can be used even if the model omitted
        # only the closing tool tag. A second started call is never ignored.
        if not repair_syntax:
            return None
        match = re.fullmatch(r"\s*<tool_call>\s*(\{.*\})\s*", text, re.S)
        if not match:
            return None
        repair = "append_missing_tool_end_tag"
    body = match.group(1)
    try:
        action = json.loads(body)
    except json.JSONDecodeError:
        if not repair_syntax:
            return None
        # The whole result must parse after an action-specific, audited edit.
        # Never guess a truncated value, product ID, voucher field, or need.
        if (re.match(r'\{\s*"name"\s*:\s*"set_requirements"\s*,', body)
                and body.endswith(']}]}]}')):
            candidate = body[:-2] + body[-1]  # extra penultimate ']'
            repair = "remove_extra_terminal_items_bracket"
        elif re.match(r'\{\s*"name"\s*:\s*"view_product"\s*,', body):
            candidate = re.sub(r'("product_id"\s*:\s*"[0-9]+)(\}\s*\})$',
                               r'\1"\2', body, count=1)
            repair = "close_numeric_product_id_quote" if candidate != body else None
        elif re.match(r'\{\s*"name"\s*:\s*"record_check"\s*,', body):
            candidate = re.sub(r'(\},\s*)("need"\s*:)', r'\1{\2', body, count=1)
            repair = "open_next_check_object" if candidate != body else None
        elif (re.match(r'\{\s*"name"\s*:\s*"set_requirements"\s*,', body)
              and re.search(r',\s*\}\s*$', body)):
            candidate = re.sub(r',\s*\}\s*$', '', body)
            repair = "remove_extra_trailing_comma_and_brace"
        else:
            return None
        if repair is None:
            return None
        try:
            action = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    if (repair_syntax and isinstance(action, dict) and action.get("name") == "set_requirements"
            and isinstance(action.get("arguments"), list)
            and all(isinstance(item, dict) and {"item", "requirement", "evidence"} <= item.keys()
                    for item in action["arguments"])):
        action = {**action, "arguments": {"items": action["arguments"]}}
        repair = "wrap_requirement_items_argument"
    if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not isinstance(action.get("arguments"), dict):
        return None
    return {"action": action, **({"syntax_repair": repair} if repair else {})}


def native_shopping_tools(governed_bundle_price: bool = False,
                          finish_quoted_ids: bool = False,
                          quote_decision_gate: bool = False) -> list[dict[str, Any]]:
    """Public tool schemas for Qwen's native tool-call chat template."""
    def tool(name: str, description: str, properties: dict[str, Any], required: list[str]):
        return {"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties,
                               "required": required, "additionalProperties": False}}}

    text = {"type": "string"}
    tools = [
        tool("set_requirements", "Once at start: one ordered entry per public item occurrence, including repeated types; cite confirmed user preferences.",
             {"items": {"type": "array", "items": {"type": "object", "properties": {
                 "item": text, "requirement": text,
                 "evidence": {"type": "array", "items": text}},
                 "required": ["item", "requirement", "evidence"]}}}, ["items"]),
        tool("set_requirement_slot", "For repeated product types only: fill one numbered public product slot with its complete confirmed preferences and brief user evidence; one slot per call.",
             {"slot_index": {"type": "integer", "minimum": 1}, "item": text,
              "requirement": text, "evidence": {"type": "array", "items": text}},
             ["slot_index", "item", "requirement", "evidence"]),
        tool("search_products", "Search public product catalog. Attribute words belong in query, not filters.",
             {"query": text, **({"preferred_max_list_price": {"type": "number", "description":
                 "Soft listed-price preference; does not exclude more expensive candidates."}}
                 if governed_bundle_price else {}),
              "filters": {"type": "object", "properties": {
                 **({} if governed_bundle_price else {"max_price": {"type": "number"}}),
                 "min_price": {"type": "number"},
                 "seller_id": text, "category": text}},
              "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
        tool("view_product", "Open one uninspected ID from search results.",
             {"product_id": text}, ["product_id"]),
        tool("record_check", "After viewing, record visible evidence for this product's wanted features. Include slot_index when a product type occurs more than once in the public request.",
             {"item": text, "slot_index": {"type": "integer", "minimum": 1},
              "product_id": text,
              "checks": {"type": "array", "items": {"type": "object", "properties": {
                  "need": text, "status": {"type": "string", "enum": ["supported", "contradicted", "unknown"]},
                  "source": {"type": "string", "description": "Full visible path such as attributes.power_tool_material_uses or options[1].color_family"},
                  "value": {"type": "string", "description": "Literal value at that source"}},
                  "required": ["need", "status", "source", "value"]}}},
             ["item", "product_id", "checks"]),
        tool("calculate_bundle", "Quote exact product IDs with public voucher and budget fields.",
             {"product_ids": {"type": "array", "items": text},
              "voucher": {"type": "object", "properties": {
                  "scope": {"type": "string", "enum": ["platform", "same_shop"]},
                  "threshold": {"type": "number"},
                  "threshold_operator": {"type": "string", "enum": [">", ">="]},
                  "discount_percent": {"type": "number"},
                  "discount_cap": {"type": "number"}},
                  "required": ["scope", "threshold", "threshold_operator",
                               "discount_percent", "discount_cap"]},
              "budget": {"type": "number"}}, ["product_ids", "voucher", "budget"]),
        tool("finish", "Submit inspected products. For a previously quoted bundle, quoted_product_ids is a shorter alternative to selections when each ID has a recorded item check." if finish_quoted_ids else
             "Submit one inspected product per requested item after quoting a bundle.",
             {"selections": {"type": "array", "items": {"type": "object", "properties": {
                 "item": text, "product_id": text,
                 "variant_index": {"type": "integer", "minimum": 0}},
                 "required": ["item", "product_id"]}},
              **({"quoted_product_ids": {"type": "array", "items": text},
                  "variant_indices_by_product": {"type": "object",
                      "additionalProperties": {"type": "integer", "minimum": 0}}}
                 if finish_quoted_ids else {}),
              "reason": text}, [] if finish_quoted_ids else ["selections"]),
    ]
    if quote_decision_gate:
        tools.append(tool("continue_shopping", "Decline the current quote and resume browsing; give a concrete reason from the task or observed products.",
                          {"reason": text}, ["reason"]))
    return tools


def render_policy_prompt(tokenizer, messages, native_tools: bool,
                         governed_bundle_price: bool = False,
                         finish_quoted_ids: bool = False,
                         quote_decision_gate: bool = False):
    tools = (native_shopping_tools(governed_bundle_price, finish_quoted_ids,
                                  quote_decision_gate) if native_tools else [])
    if quote_decision_gate and native_tools:
        if "QUOTE_DECISION_PENDING:" in messages[-1]["content"]:
            tools = [tool for tool in tools if tool["function"]["name"] in
                     {"finish", "continue_shopping"}]
        else:
            tools = [tool for tool in tools if tool["function"]["name"] !=
                     "continue_shopping"]
    kwargs = {"tools": tools} if native_tools else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)
