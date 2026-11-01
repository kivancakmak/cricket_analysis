"""AI-vs-coach agreement report (markdown) from labels/labels.csv.

Usage:
    python scripts/agreement_report.py [--labels labels/labels.csv] [--out outputs/agreement_report.md]
                                       [--shot forward_defence] [--coach coach1]

Prints the report and writes it to ``--out``. Statistics are explained at the
bottom of the report and in ``src/agreement.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agreement import build_report
from src.labels import LABELS_CSV, load_labels


def main() -> None:
    """Parse the command line, build the report and write it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS_CSV, help="labels CSV (default: labels/labels.csv)")
    parser.add_argument("--out", type=Path, default=Path("outputs/agreement_report.md"),
                        help="where to write the markdown report")
    parser.add_argument("--shot", help="only this shot type, e.g. forward_defence")
    parser.add_argument("--coach", help="only this coach id")
    args = parser.parse_args()

    report = build_report(load_labels(args.labels), source=str(args.labels), shot=args.shot, coach=args.coach)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
