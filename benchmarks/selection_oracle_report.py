"""Confusion matrix of the Light Frame QC gate against the counterfactual oracle.

Reads the ``qc/manifest.json`` and ``qc/selection.json`` of one or more E2E
run directories (or project run directories under ``runs/*``) produced with a
selection policy that records counterfactual evidence (``include-all`` is the
diagnostic policy that keeps every guard-allowed frame at full weight), and
tabulates, per frame:

* the legacy gate disposition (PASS / REVIEW / HARD_FAIL),
* the selection action and confidence,
* the leave-one-out deltas (depth, background, FWHM proxy) with their CIs,
* the oracle verdict: HARMFUL (removing the frame improves the master with the
  CI excluding zero), NEUTRAL, or BENEFICIAL.

The confusion matrix answers "which frames the gate would have excluded that
the oracle says were worth keeping" (false exclusions) and the converse.

    .venv/bin/python benchmarks/selection_oracle_report.py <run-dir>... \
        --output benchmarks/results/<machine>-selection-oracle-<date>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


def _load(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((run / "qc" / "manifest.json").read_text(encoding="utf-8"))
    selection_path = run / "qc" / "selection.json"
    if not selection_path.is_file():
        raise SystemExit(f"{run}: no qc/selection.json (run with an unattended selection policy)")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return manifest, selection


def _verdict(frame: dict[str, Any], harmful_depth: float, harmful_background: float, beneficial_depth: float) -> str:
    depth_ci = frame.get("deltaDepthCi") or [None, None]
    background_ci = frame.get("deltaBackgroundCi") or [None, None]
    if (depth_ci[0] is not None and depth_ci[0] > harmful_depth) or (
        background_ci[0] is not None and background_ci[0] > harmful_background
    ):
        return "HARMFUL"
    if depth_ci[1] is not None and depth_ci[1] < -beneficial_depth:
        return "BENEFICIAL"
    return "NEUTRAL"


def summarize(run: Path) -> dict[str, Any]:
    manifest, selection = _load(run)
    parameters = selection["parameters"]
    gate_by_path = {
        item["path"]: (item.get("qualityGate") or {}).get("disposition", "HARD_FAIL")
        for item in manifest["frames"]
    }
    rows: list[dict[str, Any]] = []
    counterfactual = selection.get("counterfactual") or {}
    frames_by_path: dict[str, dict[str, Any]] = {}
    for group, report in counterfactual.items():
        for frame in report["frames"]:
            frames_by_path[frame["path"]] = {**frame, "group": group}
    for decision in selection["frames"]:
        path = decision["path"]
        oracle = frames_by_path.get(path)
        verdict = None
        if oracle is not None:
            verdict = _verdict(
                oracle,
                parameters["harmfulDepthMag"],
                parameters["harmfulBackgroundSigma"],
                parameters["beneficialDepthMag"],
            )
        rows.append(
            {
                "path": path,
                "name": Path(path).name,
                "gate": gate_by_path.get(path, "UNKNOWN"),
                "action": decision["action"],
                "confidence": decision["confidence"],
                "reasons": [item["code"] for item in decision["reasons"]],
                "oracle": verdict,
                "deltaDepthMag": oracle.get("deltaDepthMag") if oracle else None,
                "deltaDepthCi": oracle.get("deltaDepthCi") if oracle else None,
                "deltaBackgroundSigma": oracle.get("deltaBackgroundSigma") if oracle else None,
                "deltaBackgroundCi": oracle.get("deltaBackgroundCi") if oracle else None,
                "deltaFwhmPx": oracle.get("deltaFwhmPx") if oracle else None,
                "group": oracle.get("group") if oracle else None,
            }
        )
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = matrix.setdefault(row["gate"], {"HARMFUL": 0, "NEUTRAL": 0, "BENEFICIAL": 0, "NOT_MEASURED": 0})
        cell[row["oracle"] or "NOT_MEASURED"] += 1
    false_exclusions = [row for row in rows if row["gate"] != "PASS" and row["oracle"] == "BENEFICIAL"]
    missed = [row for row in rows if row["gate"] == "PASS" and row["oracle"] == "HARMFUL"]
    return {
        "run": str(run),
        "policy": parameters["policy"],
        "frames": rows,
        "confusion": matrix,
        "gateWouldExcludeButOracleBeneficial": [row["name"] for row in false_exclusions],
        "gatePassButOracleHarmful": [row["name"] for row in missed],
        "counterfactualGroups": {
            group: {key: report[key] for key in ("tilesUsed", "sigmaBlockAll", "backgroundRmsAllSigma", "fwhmAllPx")}
            for group, report in counterfactual.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="E2E run directories (each with qc/selection.json)")
    parser.add_argument("--output", required=True, help="new JSON report path (must not exist)")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    runs = []
    for value in args.runs:
        root = Path(value)
        if (root / "qc" / "selection.json").is_file():
            runs.append(root)
        else:
            for runs_dir in (root / "runs", root / "details" / "runs"):
                runs.extend(
                    sorted(path for path in runs_dir.glob("*") if (path / "qc" / "selection.json").is_file())
                )
    if not runs:
        parser.error("no run directory with qc/selection.json found")
    report = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-selection-oracle-report-v1",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [summarize(run) for run in runs],
    }
    for run in report["runs"]:
        print(f"== {run['run']} ({run['policy']})")
        for gate, cell in sorted(run["confusion"].items()):
            print(f"   {gate:9s} " + "  ".join(f"{key}={value}" for key, value in cell.items()))
        print("   gate-excluded but beneficial:", run["gateWouldExcludeButOracleBeneficial"])
        print("   gate-pass but harmful:", run["gatePassButOracleHarmful"])
        worst = sorted((row for row in run["frames"] if row["deltaDepthMag"] is not None), key=lambda row: -row["deltaDepthMag"])[:5]
        for row in worst:
            print(f"   {row['name']:40s} gate={row['gate']:9s} dDepth={row['deltaDepthMag']:+.4f} dBg={row['deltaBackgroundSigma'] if row['deltaBackgroundSigma'] is None else round(row['deltaBackgroundSigma'], 3)} {row['oracle']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
