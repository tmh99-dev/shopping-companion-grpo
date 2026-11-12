"""Independently audit rewards and coupon arithmetic in an existing run."""
import argparse
import json
from pathlib import Path

from shopping_agent.evaluation.quote_audit import check
from shopping_agent.evaluation.reward_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project-dir", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report", action="store_true",
                        help="Generate the comparison report and curves (requires matplotlib)")
    args = parser.parse_args()
    rewards = audit(args.run_dir, args.project_dir)
    quotes = check(args.run_dir, args.project_dir)
    for filename, result in (("integrity_stage2.json", rewards),
                             ("quote_arithmetic_audit.json", quotes)):
        (args.run_dir / filename).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rewards_passed": rewards["passed"],
                      "quotes_passed": quotes["passed"],
                      "summary": rewards["summary"]}, ensure_ascii=False))
    if not rewards["passed"] or not quotes["passed"]:
        raise SystemExit("Audit failed; inspect saved audit files before interpreting results.")
    if args.report:
        from shopping_agent.evaluation.report import main as report
        report(args.run_dir)
