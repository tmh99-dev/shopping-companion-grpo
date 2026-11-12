"""Summarize an audited Stage-2 v11 run from saved trajectories."""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def phase_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "exact": sum(bool(row["reward_breakdown"].get("exact_valid_completion")) for row in rows),
        "grounded": sum(bool(row["reward_breakdown"].get("complete")) for row in rows),
        "valid_finish": sum(bool(row["reward_breakdown"].get("valid_finish")) for row in rows),
        "mean_reward": statistics.mean(row["training_reward"] for row in rows) if rows else 0.0,
        "quote_attempts": sum(any((event.get("action") or {}).get("name") == "calculate_bundle"
                                  for event in row["trajectory"]) for row in rows),
    }


def short_observation(event: dict) -> dict:
    observation = event.get("observation") or {}
    keys = ("error", "next_step", "product_id", "name", "seller_id", "price", "final_price",
            "threshold_met", "shop_rule_met", "voucher_applicable", "within_budget",
            "finished", "recorded_product_id", "checks", "product_ids")
    out = {key: observation[key] for key in keys if key in observation}
    if "results" in observation:
        out["results_first_five"] = observation["results"][:5]
    return out


def main(run: Path) -> None:
    config, result = load(run / "config.json"), load(run / "result.json")
    integrity, quotes = load(run / "integrity_stage2.json"), load(run / "quote_arithmetic_audit.json")
    metrics, params = load(run / "metrics.json"), load(run / "parameter_comparison.json")
    rows = {phase: [load(path) for path in sorted((run / "traces").glob(f"{phase}_*.json"))]
            for phase in ("before", "train", "after")}
    by_type = {phase: {kind: phase_summary([r for r in rows[phase] if r["question_type"] == kind])
                       for kind in ("single_product", "add_on_deals")}
               for phase in ("before", "after")}
    per_question = []
    for qid in config["eval_question_ids"]:
        a = [r for r in rows["after"] if r["question_id"] == qid]
        b = [r for r in rows["before"] if r["question_id"] == qid]
        per_question.append({"question_id": qid, "type": b[0]["question_type"],
                             "before": phase_summary(b), "after": phase_summary(a)})
    trained = [m for m in metrics if not m.get("skipped")]
    grad = [m["metrics"]["actor/grad_norm"][0] for m in trained]
    train_by_type = {kind: phase_summary([r for r in rows["train"] if r["question_type"] == kind])
                     for kind in ("single_product", "add_on_deals")}
    train_errors = {}
    for kind in train_by_type:
        subset = [r for r in rows["train"] if r["question_type"] == kind]
        train_errors[kind] = {
            "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in subset)),
            "check_uninspected": sum("requires an inspected product_id" in
                                     str((event.get("observation") or {}).get("error", ""))
                                     for r in subset for event in r["trajectory"]),
            "repeat_view": sum("already inspected" in
                               str((event.get("observation") or {}).get("error", ""))
                               for r in subset for event in r["trajectory"]),
        }
    contrast = sum(0 < sum(bool(r["reward_breakdown"].get("complete")) for r in rows["train"]
                           if r["question_id"] == m["question_id"]) < config["samples"]
                   for m in metrics)
    stats = {
        "integrity_passed": integrity["passed"], "quote_audit_passed": quotes["passed"],
        "quotes_checked": quotes["checked_quotes"], "traces": {k: len(v) for k, v in rows.items()},
        "by_type": by_type, "train_by_type": train_by_type, "train_errors": train_errors,
        "per_question": per_question, "groups": len(metrics),
        "optimizer_updates": len(trained), "skipped_groups": len(metrics) - len(trained),
        "success_failure_contrast_groups": contrast,
        "nonzero_gradient_groups": sum(value > 0 for value in grad),
        "gradient_norm_range": [min(grad), max(grad)] if grad else [],
        "observation_loss_tokens": sum(m.get("observation_loss_tokens", 0) for m in metrics),
        "positive_invalid_tokens_masked": sum(m.get("positive_invalid_tokens_masked", 0) for m in metrics),
        "changed_lora_tensors": params["changed_tensors"],
        "context_overflow_episodes": sum(r["finish_reason"] == "context_overflow"
                                         for phase in rows.values() for r in phase),
        "step_extensions": sum(len(r.get("step_extensions") or [])
                               for phase in rows.values() for r in phase),
        "false_negative_check_penalty_sum": sum(
            r["reward_breakdown"].get("components", {}).get("false_negative_check_penalty_v11", 0)
            for phase in rows.values() for r in phase),
    }
    (run / "analysis.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for row in metrics:
        step = row["update"]
        axes[0].scatter([step] * len(row["rewards"]), row["rewards"], alpha=0.4, s=11)
        if row.get("skipped"):
            axes[1].axvline(step, color="gray", alpha=0.12)
            axes[2].axvline(step, color="gray", alpha=0.12)
            continue
        pg = row["metrics"].get("actor/pg_loss", [])
        if pg:
            axes[1].scatter(step, statistics.mean(pg), color="tab:orange", s=14)
        axes[2].scatter(step, row["metrics"]["actor/grad_norm"][0], color="tab:green", s=14)
    for axis, title in zip(axes, ("Rollout reward", "Policy loss", "Gradient norm")):
        axis.set_title(title)
        axis.set_xlabel("Training group")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "reward_loss_grad.png", dpi=160)
    plt.close(fig)

    def fraction(item: dict, field: str) -> str:
        return f"{item[field]}/{item['n']}"

    before, after = phase_summary(rows["before"]), phase_summary(rows["after"])
    lines = ["# Shopping Companion v11 实验报告", "",
             f"原始 Qwen3-4B，新 LoRA；{len(config['train_question_ids'])} 道训练题每题 {config['samples']} 条轨迹，"
             f"{len(config['eval_question_ids'])} 道开发题训练前后各 {config['eval_samples']} 条轨迹。",
             "开发题曾用于调试，不是盲测。训练和测试会话不重叠。", "",
             "| 指标 | 训练前 | 训练后 |", "|---|---:|---:|",
             f"| 严格满分 | {fraction(before, 'exact')} | {fraction(after, 'exact')} |",
             f"| 源字段支持的完成 | {fraction(before, 'grounded')} | {fraction(after, 'grounded')} |",
             f"| 有效提交 | {fraction(before, 'valid_finish')} | {fraction(after, 'valid_finish')} |",
             f"| 平均奖励 | {before['mean_reward']:.4f} | {after['mean_reward']:.4f} |",
             "", "## 单品与组合", "",
             "| 类型 | 前满分 | 后满分 | 前有效提交 | 后有效提交 | 前均奖 | 后均奖 |",
             "|---|---:|---:|---:|---:|---:|---:|" ]
    for kind, label in (("single_product", "单品"), ("add_on_deals", "组合")):
        b, a = by_type["before"][kind], by_type["after"][kind]
        lines.append(f"| {label} | {fraction(b, 'exact')} | {fraction(a, 'exact')} | "
                     f"{fraction(b, 'valid_finish')} | {fraction(a, 'valid_finish')} | "
                     f"{b['mean_reward']:.4f} | {a['mean_reward']:.4f} |")
    lines += ["", "## 每题开发结果", "",
              "| 题 ID | 类型 | 前满分 | 后满分 | 前均奖 | 后均奖 |",
              "|---|---|---:|---:|---:|---:|"]
    for q in per_question:
        b, a = q["before"], q["after"]
        lines.append(f"| `{q['question_id']}` | {q['type']} | {fraction(b, 'exact')} | "
                     f"{fraction(a, 'exact')} | {b['mean_reward']:.3f} | {a['mean_reward']:.3f} |")
    lines += ["", "## 训练与核算", "",
              f"独立奖励复算：**{integrity['passed']}**；报价复算：**{quotes['passed']}**，"
              f"检查 {quotes['checked_quotes']} 次有效报价。轨迹数 {stats['traces']}。",
              f"{len(metrics)} 组中 {len(trained)} 组实际更新，{len(metrics)-len(trained)} 组零差异跳过；"
              f"有真实完成／失败对照的组数 {contrast}。",
              f"非零梯度组 {stats['nonzero_gradient_groups']}；LoRA 张量变化 {params['changed_tensors']}；"
              f"观察 token 进入 loss 数 {stats['observation_loss_tokens']}。",
              f"上下文超限轨迹 {stats['context_overflow_episodes']}；额外步数延长事件 {stats['step_extensions']}。",
              f"单品训练轨迹有效提交 {train_by_type['single_product']['valid_finish']}/{train_by_type['single_product']['n']}，"
              f"组合 {train_by_type['add_on_deals']['valid_finish']}/{train_by_type['add_on_deals']['n']}。"
              f"单品对未查看商品调用 record_check 的工具错误 {train_errors['single_product']['check_uninspected']} 次；"
              f"组合 {train_errors['add_on_deals']['check_uninspected']} 次。",
              "奖励、参数更新与开发题的小样本差异都不能单独证明泛化提升。", "",
              "## 训练后低分轨迹逐步检查", ""]
    low = sorted(rows["after"], key=lambda r: (r["training_reward"], len(r["trajectory"])))
    selected, seen = [], set()
    for row in low:
        if row["question_id"] not in seen:
            selected.append(row)
            seen.add(row["question_id"])
        if len(selected) == 3:
            break
    for row in selected:
        name = next(path.name for path in sorted((run / "traces").glob("after_*.json"))
                    if load(path)["question_id"] == row["question_id"] and
                    load(path)["training_reward"] == row["training_reward"])
        lines += [f"### `{row['question_id']}` · {row['question_type']} · {row['training_reward']:.4f}", "",
                  f"完整轨迹：[traces/{name}](traces/{name})。结束原因：`{row['finish_reason']}`。", "",
                  "原题：", "", row.get("question", ""), "", "历史偏好材料：", ""]
        lines.extend(str(mem.get("text", "")) for mem in row.get("provided_memories", []))
        lines += ["", "逐步动作与工具反馈：", ""]
        for event in row["trajectory"]:
            action = event.get("action") or {}
            args = json.dumps(action.get("arguments", {}), ensure_ascii=False)
            obs = json.dumps(short_observation(event), ensure_ascii=False)
            lines.append(f"{event.get('step')}. `{action.get('name', 'invalid_output')}` {args[:500]} → {obs[:850]}")
        lines += ["", "最终失败类型需结合完整轨迹、商品原始字段与核验反馈判读；此处保留逐步证据，不按奖励值自动归因。", ""]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"before": before, "after": after, "by_type": by_type,
                      "integrity_passed": integrity["passed"],
                      "quote_audit_passed": quotes["passed"],
                      "optimizer_updates": len(trained)}, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
