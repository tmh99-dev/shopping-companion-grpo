"""Independently check saved bundle quotes against public wording and source prices."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import pyarrow.parquet as pq


def ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def check(run_dir: Path, project_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tasks = project_dir / config["tasks"]
    rows = {r["extra_info"]["question_id"]: r for split in ("train", "test")
            for r in pq.read_table(tasks / f"{split}.parquet").to_pylist()
            if r["ability"] == "stage_2"}
    products = {str(p["id"]): p["product"] for p in
                (json.loads(line) for line in (project_dir / config["catalog"]).open(encoding="utf-8"))}
    checked = 0
    mismatches = []
    trace_paths = (list((run_dir / "traces").glob("*.json"))
                   + list(run_dir.glob("trace_*.json"))
                   + list(run_dir.glob("sample_*.json")))
    for path in sorted(trace_paths):
        episode = json.loads(path.read_text(encoding="utf-8"))
        row = rows[episode["question_id"]]
        truth = json.loads(row["reward_model"]["ground_truth"])
        if not truth.get("voucher"):
            continue
        clause = row["extra_info"]["question"].split("Voucher:", 1)[-1].split("Budget:", 1)[0].lower()
        strict = "exceeds" in clause or bool(re.search(r"\bover\s*\$|>\s*\$", clause))
        if not strict and not ("min spend" in clause or "minimum spend" in clause):
            raise ValueError(f"Unrecognized voucher wording: {episode['question_id']}")
        voucher = truth["voucher"]
        threshold = Decimal(str(voucher["threshold"]))
        cap = Decimal(str(voucher["cap"]))
        rate = Decimal(str(voucher["discount"]).rstrip("%")) / 100
        budget = Decimal(str(voucher["budget"]))
        for event in episode["trajectory"]:
            action = event.get("action") or {}
            quote = event.get("observation") or {}
            if action.get("name") != "calculate_bundle" or "error" in quote:
                continue
            ids = [str(x) for x in action.get("arguments", {}).get("product_ids", [])]
            if len(ids) != truth["n"] or len(set(ids)) != len(ids):
                continue
            raw = sum((Decimal(str(products[x]["price"])) for x in ids), Decimal(0))
            subtotal = ceil(raw)  # Published reference arithmetic convention.
            sellers = [str(products[x].get("seller_id", "")) for x in ids]
            same_shop = len(set(sellers)) == 1
            threshold_met = raw > threshold if strict else raw >= threshold
            shop_rule_met = truth.get("voucher_type") != "shop" or same_shop
            applicable = threshold_met and shop_rule_met
            discount = min(Decimal(subtotal) * rate, cap) if applicable else Decimal(0)
            final_price = ceil(Decimal(subtotal) - discount)
            expected = {"raw_subtotal": float(raw), "subtotal": subtotal,
                        "same_shop": same_shop, "threshold_met": threshold_met,
                        "shop_rule_met": shop_rule_met, "voucher_applicable": applicable,
                        "discount": float(discount), "final_price": final_price,
                        "within_budget": Decimal(final_price) <= budget}
            differences = {key: {"saved": quote.get(key), "recomputed": value}
                           for key, value in expected.items()
                           if key not in quote or (abs(float(quote[key]) - float(value)) > 1e-8
                                                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                                                   else quote[key] != value)}
            checked += 1
            if differences:
                mismatches.append({"trace": path.name, "question_id": episode["question_id"],
                                   "step": event["step"], "differences": differences})
    return {"passed": not mismatches, "checked_quotes": checked,
            "mismatch_count": len(mismatches), "mismatches": mismatches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = check(args.run_dir, args.project_dir)
    (args.run_dir / "quote_arithmetic_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "checked_quotes", "mismatch_count")}))
