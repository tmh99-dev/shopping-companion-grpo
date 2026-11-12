"""Offline checks for task visibility, product evidence and voucher rules.

Reference labels are used only for auditing; they are never added to Agent prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from shopping_agent.data.catalog import ProductStore
from shopping_agent.data.tasks import load_tasks, truth_items
from shopping_agent.harness.context import stage2_history_document
from shopping_agent.rewards.scoring import _feature_status, _normalized


BUNDLE_RE = re.compile(
    r"Product bundle includes:\s*(.+?)\.\s*Voucher\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_visible_items(question: str, question_type: str) -> list[str]:
    """Extract explicitly requested item phrases without semantic guessing."""
    if question_type != "add_on_deals":
        lines = [line.strip() for line in question.splitlines() if line.strip()]
        return [lines[-1]] if lines else []
    match = BUNDLE_RE.search(question)
    if not match:
        return []
    text = match.group(1).strip()
    if "," in text:
        # Commas are authoritative separators. This preserves product phrases
        # such as "transmitter and receiver, Apple Pencil case" as two items.
        text = re.sub(r",\s*and\s+", ", ", text, flags=re.IGNORECASE)
        return [item.strip() for item in text.split(",") if item.strip()]
    # Two-item lists have no comma in the released mini split.
    return [item.strip() for item in re.split(r"\s+and\s+", text, maxsplit=1,
                                              flags=re.IGNORECASE) if item.strip()]


def audit_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    info = row["extra_info"]
    truth = json.loads(row["reward_model"]["ground_truth"])
    preferences = truth.get("preferences")
    preference_count = len(preferences) if isinstance(preferences, list) else 1
    truth_n = truth.get("n")
    expected_count = int(truth_n) if truth_n is not None else preference_count
    visible_items = parse_visible_items(info["question"], info["question_type"])
    reasons = []
    if not visible_items:
        reasons.append("visible_items_unparseable")
    elif len(visible_items) != expected_count:
        reasons.append("visible_item_count_mismatch")
    if truth_n is not None and int(truth_n) != preference_count:
        reasons.append("truth_n_preference_count_mismatch")
    return {
        "split": split,
        "question_id": info["question_id"],
        "conversation_id": info["conversation_id"],
        "question_type": info["question_type"],
        "visible_items": visible_items,
        "visible_item_count": len(visible_items),
        "truth_n": truth_n,
        "preference_count": preference_count,
        "expected_count": expected_count,
        "reasons": reasons,
    }


def build_audit(data_dir: Path) -> dict[str, Any]:
    rows = []
    for split in ("train", "test"):
        path = data_dir / f"{split}.parquet"
        for row in pq.read_table(path).to_pylist():
            if row["ability"] == "stage_2":
                rows.append(audit_row(row, split))
    rows.sort(key=lambda item: (item["split"], item["question_id"]))
    clean = [row for row in rows if not row["reasons"]]
    flagged = [row for row in rows if row["reasons"]]
    return {
        "schema_version": 1,
        "scope": "offline stage_2 visible-item-count versus hidden-truth audit",
        "warning": "Hidden truth is used only for data-integrity auditing and must not enter Agent context.",
        "summary": {
            "stage_2_rows": len(rows),
            "clean": len(clean),
            "flagged": len(flagged),
            "train_flagged": sum(row["split"] == "train" for row in flagged),
            "test_flagged": sum(row["split"] == "test" for row in flagged),
        },
        "clean_question_ids": [row["question_id"] for row in clean],
        "flagged": flagged,
    }


def audit_catalog(tasks: Path, catalog_path: Path) -> dict:
    catalog = {str(row["id"]): row["product"] for row in
               (json.loads(line) for line in catalog_path.open(encoding="utf-8"))}
    output = {"criteria": {
        "source_selectable": "Each target product has one source option (or no selection) that supports all its labeled features jointly.",
        "literal_user_value": "Each labeled feature value occurs literally in released Stage-2 user turns after simple normalization; synonyms can fail this conservative check.",
        "without_variant": "Every labeled feature is supported without a variant selection; failures can still be solvable by selecting an option.",
    }, "splits": {}}
    for split in ("train", "test"):
        rows = []
        for task in load_tasks(tasks / f"{split}.parquet"):
            user_text = _normalized(stage2_history_document(task)["user_evidence_text"])
            features = []
            slots_jointly_selectable = []
            for slot in truth_items(task):
                pid = str(slot["product_id"])
                product = catalog.get(pid)
                if product is None:
                    slots_jointly_selectable.append(False)
                else:
                    options = product.get("options") or []
                    slots_jointly_selectable.append(any(
                        all(_feature_status(product, feature, index)["status"] == "supported"
                            for feature in slot.get("wanted_features", []))
                        for index in [None, *range(len(options))]))
                for feature in slot.get("wanted_features", []):
                    if product is None:
                        supported = static = False
                    else:
                        options = product.get("options") or []
                        statuses = [_feature_status(product, feature, index)["status"]
                                    for index in [None, *range(len(options))]]
                        static = statuses[0] == "supported"
                        supported = "supported" in statuses
                    wanted = _normalized(str(feature).partition(":")[2])
                    features.append({"product_id": pid, "feature": feature,
                                     "source_selectable": supported,
                                     "without_variant": static,
                                     "literal_user_value": bool(wanted and wanted in user_text)})
            rows.append({"question_id": task["extra_info"]["question_id"],
                         "question_type": task["extra_info"]["question_type"],
                         "features": features,
                         "source_selectable": all(slots_jointly_selectable),
                         "literal_user_value": all(f["literal_user_value"] for f in features),
                         "without_variant": all(f["without_variant"] for f in features)})
        output["splits"][split] = {
            "total": len(rows),
            "source_selectable": sum(r["source_selectable"] for r in rows),
            "literal_user_value": sum(r["literal_user_value"] for r in rows),
            "both": sum(r["source_selectable"] and r["literal_user_value"] for r in rows),
            "without_variant": sum(r["without_variant"] for r in rows),
            "rows": rows,
        }
    return output


def audit_vouchers(tasks: Path, catalog_path: Path) -> dict:
    catalog = {str(row["id"]): row["product"] for row in
               (json.loads(line) for line in catalog_path.open(encoding="utf-8"))}
    splits = {}
    excluded = []
    metadata_mismatches = []
    for split in ("train", "test"):
        records = []
        for row in pq.read_table(tasks / f"{split}.parquet").to_pylist():
            if row["ability"] != "stage_2" or row["extra_info"]["question_type"] != "add_on_deals":
                continue
            truth = json.loads(row["reward_model"]["ground_truth"])
            voucher = truth["voucher"]
            ids = [str(item["product_id"]) for item in truth["preferences"]]
            raw_subtotal = sum(float(catalog[product_id]["price"]) for product_id in ids)
            rounded_subtotal = math.ceil(raw_subtotal - 1e-9)
            threshold = float(voucher["threshold"])
            question = str(row["extra_info"]["question"])
            voucher_clause = question.split("Voucher:", 1)[-1].split("Budget:", 1)[0].lower()
            public_voucher_text = question[question.lower().find("voucher"):].lower()
            amount = r"\$\s*([0-9]+(?:\.[0-9]+)?)"
            public_threshold = re.search(r"(?:exceeds|over|>|min spend|minimum spend)\s*" + amount,
                                         public_voucher_text)
            public_cap = re.search(r"(?:capped at|up to|with|cap of|cap)\s*" + amount,
                                   public_voucher_text)
            public_budget = re.search(r"\bbudget\s*:?\s*" + amount, public_voucher_text)
            public_discount = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", public_voucher_text)
            if not all((public_threshold, public_cap, public_budget, public_discount)):
                raise ValueError(f"Could not parse public voucher amounts: {row['extra_info']['question_id']}")
            public_shop = bool(re.search(r"same[ -]shop|same shop", public_voucher_text))
            if (float(public_threshold.group(1)) != float(voucher["threshold"])
                    or float(public_cap.group(1)) != float(voucher["cap"])
                    or float(public_budget.group(1)) != float(voucher["budget"])
                    or float(public_discount.group(1)) != float(str(voucher["discount"]).rstrip("%"))
                    or public_shop != (truth.get("voucher_type") == "shop")):
                metadata_mismatches.append(row["extra_info"]["question_id"])
            strict = ("exceeds" in voucher_clause
                      or bool(re.search(r"\bover\s*\$|>\s*\$", voucher_clause)))
            if not strict and not ("min spend" in voucher_clause or "minimum spend" in voucher_clause):
                raise ValueError(f"Voucher wording not audited: {row['extra_info']['question_id']}")
            rate = float(str(voucher["discount"]).rstrip("%")) / 100
            reference_final = math.ceil(
                rounded_subtotal - min(rounded_subtotal * rate, float(voucher["cap"])) - 1e-9)
            if reference_final != voucher["price_after_voucher"]:
                raise ValueError(f"Reference arithmetic differs: {row['extra_info']['question_id']}")
            conflict = (raw_subtotal <= threshold + 1e-9 if strict
                        else raw_subtotal + 1e-9 < threshold)
            record = {
                "question_id": row["extra_info"]["question_id"],
                "raw_subtotal": round(raw_subtotal, 2),
                "rounded_subtotal": rounded_subtotal,
                "threshold": threshold,
                "threshold_operator": ">" if strict else ">=",
                "reference_price_after_voucher": voucher["price_after_voucher"],
                "conflict": conflict,
            }
            records.append(record)
            if conflict:
                excluded.append(record["question_id"])
        splits[split] = {"count": len(records), "conflicts": [r for r in records if r["conflict"]]}
    return {
        "reason": "The reference bundle's listed-price sum does not meet the public voucher threshold wording.",
        "tasks_sha256": {split: hashlib.sha256((tasks / f"{split}.parquet").read_bytes()).hexdigest()
                         for split in ("train", "test")},
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "splits": splits,
        "metadata_mismatches": metadata_mismatches,
        "excluded_question_ids": sorted(excluded),
    }


def oracle_check(tasks: list[dict[str, Any]], store: ProductStore) -> dict[str, Any]:
    missing = []
    bundle_quotes = []
    for row in tasks:
        truth = json.loads(row["reward_model"]["ground_truth"])
        ids = [str(item["product_id"]) for item in truth.get("preferences", [truth])]
        missing.extend(value for value in ids if value not in store.by_id)
        if truth.get("preferences"):
            bundle_quotes.append(store.calculate({"product_ids": ids}, truth))
    return {
        "tasks": len(tasks), "target_ids": sum(len(truth_items(row)) for row in tasks),
        "missing_target_ids": sorted(set(missing)), "bundle_quotes": bundle_quotes,
    }


def validate_data(tasks, catalog, require_literal=True):
    """Offline checks only: reference labels never enter Agent observations."""
    tasks, catalog = Path(tasks), Path(catalog)
    splits = {s: pq.read_table(tasks / (s + '.parquet')).to_pylist()
              for s in ('train', 'test')}
    ids, conversations = {}, {}
    for split, rows in splits.items():
        if not rows or any(r['ability'] != 'stage_2' for r in rows):
            raise ValueError('Expected a nonempty prepared Stage-2 split: ' + split)
        ids[split] = {r['extra_info']['question_id'] for r in rows}
        conversations[split] = {r['extra_info']['conversation_id'] for r in rows}
        if len(ids[split]) != len(rows):
            raise ValueError('Duplicate question IDs: ' + split)
    if ids['train'] & ids['test'] or conversations['train'] & conversations['test']:
        raise ValueError('Training and test questions/conversations overlap')
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    for split in splits:
        metadata = pq.read_schema(tasks / (split + '.parquet')).metadata or {}
        expected = metadata.get(b'shopping_catalog_sha256')
        if expected and expected.decode() != catalog_hash:
            raise ValueError('Catalog differs from the one associated with this dataset')
    product_ids = set()
    with catalog.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            pid = str(row['id'])
            if pid in product_ids or not isinstance(row['product'], dict):
                raise ValueError('Duplicate or malformed product: ' + pid)
            product_ids.add(pid)
    observable = build_audit(tasks)
    if observable['flagged']:
        raise ValueError('Public item counts disagree with labels')
    support = audit_catalog(tasks, catalog)
    invalid = [r['question_id'] for split in support['splits'].values()
               for r in split['rows'] if not r['source_selectable']
               or (require_literal and not r['literal_user_value'])]
    if invalid:
        raise ValueError('Unsupported product/history requirements: ' + ', '.join(invalid))
    vouchers = audit_vouchers(tasks, catalog)
    if vouchers['metadata_mismatches'] or vouchers['excluded_question_ids']:
        raise ValueError('Public voucher rules conflict with reference data')
    return {'counts': {s: len(rows) for s, rows in splits.items()},
            'products': len(product_ids), 'conversation_overlap': 0,
            'observability': observable, 'catalog_support': support,
            'vouchers': vouchers,
            'note': 'Structural checks cannot resolve ambiguous wording; curate source data before import.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('data'))
    parser.add_argument('--catalog', type=Path, default=Path('data/products.jsonl'))
    parser.add_argument('--check', choices=('all', 'visibility', 'catalog', 'voucher'), default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.check == 'visibility':
        result = build_audit(args.tasks)
    elif args.check == 'catalog':
        result = audit_catalog(args.tasks, args.catalog)
    elif args.check == 'voucher':
        result = audit_vouchers(args.tasks, args.catalog)
    else:
        result = validate_data(args.tasks, args.catalog)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'check': args.check, 'report': str(args.output) if args.output else None,
                      'counts': result.get('counts', result.get('summary'))}))


if __name__ == '__main__':
    main()
