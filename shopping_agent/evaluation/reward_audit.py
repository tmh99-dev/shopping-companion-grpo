"""Replay Stage-2 grounded rewards against the released tasks and source catalog."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from shopping_agent.harness.prompts import STAGE2_HISTORY_ID
from shopping_agent.harness.context import stage2_history_document
from shopping_agent.data.tasks import truth_items
from shopping_agent.rewards.scoring import score_episode_grounded, score_episode_grounded_progress, score_episode_grounded_progress_v2, score_episode_grounded_open_world_v6, score_episode_grounded_open_world_v7, score_episode_grounded_open_world_v8, score_episode_grounded_open_world_v9, score_episode_grounded_open_world_v10, score_episode_grounded_open_world_v11, score_episode_grounded_open_world_v12


def same_structure(left, right) -> bool:
    """Compare replayed numerical diagnostics with ordinary float tolerance."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_structure(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_structure(a, b) for a, b in zip(left, right))
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-8)
    return left == right


def audit(run_dir: Path, project_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not config.get("stage2_context") or config.get("reward_rule") not in (
            "grounded_preferences_v3", "grounded_preferences_v4", "grounded_preferences_v5",
            "grounded_preferences_v6", "grounded_preferences_v7",
            "grounded_preferences_v8", "grounded_preferences_v9",
            "grounded_preferences_v10", "grounded_preferences_v11",
            "grounded_preferences_v12"):
        raise ValueError("This audit only applies to grounded Stage-2 runs")
    scorer = ({"grounded_preferences_v3": score_episode_grounded,
               "grounded_preferences_v4": score_episode_grounded_progress,
               "grounded_preferences_v5": score_episode_grounded_progress_v2,
               "grounded_preferences_v6": score_episode_grounded_open_world_v6,
               "grounded_preferences_v7": score_episode_grounded_open_world_v7,
               "grounded_preferences_v8": score_episode_grounded_open_world_v8,
               "grounded_preferences_v9": score_episode_grounded_open_world_v9,
               "grounded_preferences_v10": score_episode_grounded_open_world_v10,
               "grounded_preferences_v11": score_episode_grounded_open_world_v11,
               "grounded_preferences_v12": score_episode_grounded_open_world_v12}
              [config["reward_rule"]])
    tasks_dir = project_dir / config["tasks"]
    rows = [r for part in ("train", "test") for r in
            pq.read_table(tasks_dir / f"{part}.parquet").to_pylist()
            if r["ability"] == "stage_2"]
    lookup = {r["extra_info"]["question_id"]: r for r in rows}
    catalog = {str(p["id"]): p for p in
               (json.loads(s) for s in (project_dir / config["catalog"]).open(encoding="utf-8"))}
    checks: list[dict] = []
    errors: list[str] = []
    expected = {"before": [qid for qid in config["eval_question_ids"]
                           for _ in range(config.get("eval_samples", 1))],
                "train": [qid for qid in config["train_question_ids"] for _ in range(config["samples"])],
                "after": [qid for qid in config["eval_question_ids"]
                          for _ in range(config.get("eval_samples", 1))]}
    actual = {phase: [] for phase in expected}
    for path in sorted((run_dir / "traces").glob("*.json")):
        e = json.loads(path.read_text(encoding="utf-8"))
        phase = path.name.split("_", 1)[0]
        qid = e["question_id"]
        row = lookup.get(qid)
        if phase not in actual or row is None:
            errors.append(f"unexpected trace or question: {path.name}")
            continue
        actual[phase].append(qid)
        history = stage2_history_document(row)
        history_ok = e.get("provided_memories") == [history]
        no_memory_search = all(
            event.get("action", {}).get("name") != "search_memory"
            for event in e.get("trajectory", []))
        score = scorer(
            e, truth_items=truth_items(row), source_products_by_id=catalog,
            expected_memory_ids=[STAGE2_HISTORY_ID] * len(truth_items(row)),
        )
        same_reward = math.isclose(float(e["training_reward"]), score["reward"], abs_tol=1e-8)
        same_breakdown = same_structure(e.get("reward_breakdown"), score)
        if not all((history_ok, no_memory_search, same_reward, same_breakdown)):
            errors.append(f"{path.name}: history={history_ok}, no_memory_search={no_memory_search}, "
                          f"reward={same_reward}, breakdown={same_breakdown}")
        checks.append({"trace": path.name, "phase": phase, "question_id": qid,
                       "reward": score["reward"], "complete": score["complete"],
                       "finish_reason": e.get("finish_reason"),
                       "history_matches_released_input": history_ok,
                       "no_memory_search": no_memory_search,
                       "reward_replays": same_reward and same_breakdown})
    for phase, qids in expected.items():
        if sorted(actual[phase]) != sorted(qids):
            errors.append(f"{phase} question/sample counts differ: {actual[phase]}")
    if set(config["train_question_ids"]) & set(config["eval_question_ids"]):
        errors.append("train/eval question overlap")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    if not metrics_path.exists() and config["train_question_ids"]:
        errors.append("training metrics missing")
    if len(metrics) != len(config["train_question_ids"]):
        errors.append("optimizer step count differs from train question count")
    for index, metric in enumerate(metrics):
        if metric.get("question_id") != config["train_question_ids"][index]:
            errors.append(f"update {index + 1} question order differs")
        if len(metric.get("rewards", [])) != config["samples"]:
            errors.append(f"update {index + 1} sample count differs")
        if metric.get("observation_loss_tokens") != 0:
            errors.append(f"update {index + 1} observation tokens entered loss")
        should_save = ((index + 1) % config.get("save_every", 1) == 0
                       or index + 1 == len(metrics))
        if should_save and not (run_dir / f"adapter_step_{index + 1}" / "COMPLETE").exists():
            errors.append(f"update {index + 1} checkpoint incomplete")
    params = json.loads((run_dir / "parameter_comparison.json").read_text(encoding="utf-8"))
    summary = {phase: {"count": len([c for c in checks if c["phase"] == phase]),
                       "complete": sum(c["complete"] for c in checks if c["phase"] == phase),
                       "mean_reward": (sum(c["reward"] for c in checks if c["phase"] == phase) /
                                       max(1, len([c for c in checks if c["phase"] == phase])))}
               for phase in expected}
    return {"passed": not errors, "errors": errors, "summary": summary,
            "groups_seen": len(metrics),
            "optimizer_steps": sum(not m.get("skipped", False) for m in metrics),
            "changed_tensors": params["changed_tensors"],
            "traces": checks,
            "limitations": (f"{len(config['train_question_ids'])} training tasks and "
                            f"{len(config['eval_question_ids'])} evaluation tasks; "
                            "derived catalog. Check run provenance before calling an evaluation blind.")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = audit(args.run_dir, args.project_dir)
    out = args.run_dir / "integrity_stage2.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": report["errors"],
                      "summary": report["summary"], "optimizer_steps": report["optimizer_steps"],
                      "changed_tensors": report["changed_tensors"]}, ensure_ascii=False))
