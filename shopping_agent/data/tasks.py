"""data.tasks: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pyarrow.parquet as pq


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [row for row in pq.read_table(path).to_pylist() if row["ability"] == "stage_2"]


def truth_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    truth = json.loads(row["reward_model"]["ground_truth"])
    return truth.get("preferences", [truth])


def load_dataset_quality_exclusions(tasks_dir: Path,
                                    available_question_ids: set[str]) -> set[str]:
    """Load audited task exclusions without changing the pinned source Parquet."""
    path = tasks_dir / "quality_exclusions.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("excluded_question_ids")
    if not isinstance(ids, list) or any(not isinstance(qid, str) or not qid for qid in ids):
        raise ValueError(f"Invalid excluded_question_ids in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate excluded_question_ids in {path}")
    unknown = set(ids) - available_question_ids
    if unknown:
        raise ValueError(f"Quality exclusion IDs absent from dataset: {sorted(unknown)}")
    return set(ids)


def select_conversations(tasks, count: int, offset: int = 0):
    conversation_ids = sorted({row["extra_info"]["conversation_id"] for row in tasks})
    chosen = conversation_ids[offset:offset + count]
    if len(chosen) != count:
        raise ValueError(f"Requested {count} conversations at offset {offset}, only {len(chosen)} available")
    selected = [row for row in tasks if row["extra_info"]["conversation_id"] in chosen]
    selected.sort(key=lambda row: (row["extra_info"]["conversation_id"], row["extra_info"]["question_type"]))
    return selected, chosen


def select_task_mix(tasks, bundle_count: int, single_count: int,
                    bundle_offset: int = 0, single_offset: int = 0):
    """Deterministically select a declared task mix without reading rewards."""
    bundles = sorted(
        (row for row in tasks if row["extra_info"]["question_type"] == "add_on_deals"),
        key=lambda row: row["extra_info"]["question_id"],
    )
    singles = sorted(
        (row for row in tasks if row["extra_info"]["question_type"] == "single_product"),
        key=lambda row: row["extra_info"]["question_id"],
    )
    selected_bundles = bundles[bundle_offset:bundle_offset + bundle_count]
    selected_singles = singles[single_offset:single_offset + single_count]
    if len(selected_bundles) != bundle_count or len(selected_singles) != single_count:
        raise ValueError("Requested task mix exceeds the available split")
    selected = selected_bundles + selected_singles
    selected.sort(key=lambda row: (row["extra_info"]["question_type"], row["extra_info"]["question_id"]))
    return selected
