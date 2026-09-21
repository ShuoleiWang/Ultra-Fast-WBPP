"""Tolerance gate between two product sets of the same run (baseline vs candidate).

The bit-identity check (equal SHA-256 of the solved masters) is the gate for
changes that must not touch a pixel.  A change of the numerical evaluation
itself — the deterministic Lanczos-3 weight table of kernel v3 replaced six
libm calls per output pixel — moves a small fraction of registered pixels by
a unit in the last place of Float32 and can flip a handful of rejection
decisions at the clip boundary.  This gate states what such a change is
allowed to do to the masters and to the integration evidence, and reports
it:

- identical geometry, identical NaN masks;
- the bulk: at least ``--bulk-fraction`` (default 99.99 %) of the finite
  pixels within ``--ulps`` (default 16) Float32 units of the baseline or
  within ``--bulk-sigma`` (default 0.002) times the baseline's robust
  per-pixel noise, whichever is larger.  Last-ulp changes of the registered
  pixels move the robust statistics of the normalization (medians and
  quantiles jump by one order statistic, about noise / tile samples), so
  whole tiles shift by ~1e-3 of the noise: invisible, but far more than a
  few ulps;
- the outliers: at most ``--outlier-fraction`` (default 1e-6) of the pixels
  outside the bulk, each within ``--outlier-sigma`` (default 2.0) times the
  noise (a rejection decision at the clip boundary flips and moves one
  pixel by up to the clipped sample's deviation / N);
- the accepted-sample counts of the integration maps within
  ``--count-fraction`` (default 0.01 %);
- optionally, ``evaluate_masters.py`` reports of both product sets against
  the same reference (``--evaluation baseline.json candidate.json``,
  repeatable): every per-metric PASS/WARN/FAIL status identical, and the
  headline numbers (SNR gains G_4 and G_8, depth Δm) within
  ``--evaluation-tolerance`` (default 0.002, an order of magnitude under the
  evaluator's own confidence intervals).  The per-filter label
  (EQUIVALENT / INCONCLUSIVE / ...) is read from the sibling ``summary.md``
  and reported; a label that flips while the statuses and numbers are
  unchanged is the evaluator's hard threshold being crossed by a rounding
  (e.g. G_8 = 1.019993 → 1.020002), not a change of the master, and does
  not fail the gate.

    .venv/bin/python benchmarks/master_tolerance_gate.py \\
        --baseline ~/Astro/Results/baseline --candidate ~/Astro/Results/candidate \\
        --output build/tolerance-gate.json

Exit status 0 = PASS, 1 = FAIL, 2 = usage error.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from astropy.io import fits

PRODUCT_NAMES = ("L", "R", "G", "B")


def _read(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def _robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    sample = finite if finite.size <= 4_000_000 else finite[:: -(-finite.size // 4_000_000)]
    median = float(np.median(sample))
    return 1.4826 * float(np.median(np.abs(sample - median)))


def compare_master(
    baseline: Path,
    candidate: Path,
    *,
    ulps: float,
    bulk_sigma: float,
    bulk_fraction: float,
    outlier_fraction: float,
    outlier_sigma: float,
) -> dict[str, Any]:
    a = _read(baseline)
    b = _read(candidate)
    record: dict[str, Any] = {"baseline": str(baseline), "candidate": str(candidate), "shape": list(a.shape)}
    if a.shape != b.shape:
        record.update(status="FAIL", reason=f"shape {a.shape} vs {b.shape}")
        return record
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    if not np.array_equal(finite_a, finite_b):
        record.update(status="FAIL", reason="NaN masks differ", nanMismatch=int(np.count_nonzero(finite_a != finite_b)))
        return record
    both = finite_a
    av = a[both].astype(np.float64)
    bv = b[both].astype(np.float64)
    difference = np.abs(av - bv)
    spacing = np.spacing(np.abs(a[both])).astype(np.float64)
    sigma = _robust_sigma(a)
    bulk_limit = np.maximum(ulps * spacing, bulk_sigma * sigma if np.isfinite(sigma) else 0.0)
    in_bulk = difference <= bulk_limit
    identical = difference == 0.0
    outliers = ~in_bulk
    max_outlier = float(difference[outliers].max()) if np.any(outliers) else 0.0
    percentiles = np.percentile(difference, [99.0, 99.9, 99.99, 99.999]) if difference.size else [0.0] * 4
    record.update(
        finitePixels=int(av.size),
        identicalFraction=float(identical.mean()),
        withinUlpsFraction=float((difference <= ulps * spacing).mean()),
        bulkFraction=float(in_bulk.mean()),
        differencePercentiles={"p99": float(percentiles[0]), "p99.9": float(percentiles[1]), "p99.99": float(percentiles[2]), "p99.999": float(percentiles[3])},
        maxAbsoluteDifference=float(difference.max()),
        maxOutlierDifference=max_outlier,
        outlierPixels=int(np.count_nonzero(outliers)),
        outlierFraction=float(outliers.mean()),
        baselineRobustSigma=sigma,
        maxOutlierInSigma=(max_outlier / sigma) if sigma and np.isfinite(sigma) and sigma > 0 else None,
        sha256Identical=bool(identical.all()),
    )
    failures = []
    if record["bulkFraction"] < bulk_fraction:
        failures.append(f"only {record['bulkFraction']:.6%} of pixels within {ulps:g} ulps / {bulk_sigma:g} sigma")
    if record["outlierFraction"] > outlier_fraction:
        failures.append(f"{record['outlierPixels']} outlier pixels ({record['outlierFraction']:.2e} > {outlier_fraction:g})")
    if np.any(outliers) and sigma > 0 and max_outlier > outlier_sigma * sigma:
        failures.append(f"outlier {max_outlier:.4g} exceeds {outlier_sigma:g} sigma ({outlier_sigma * sigma:.4g})")
    record["status"] = "FAIL" if failures else "PASS"
    if failures:
        record["reason"] = "; ".join(failures)
    return record


def _integration_counts(root: Path) -> dict[str, int]:
    """Accepted-sample totals of every integration map under the run."""

    counts: dict[str, int] = {}
    for path in sorted(glob.glob(str(root / "details" / "runs" / "*" / "coverage" / "*_acceptedSampleCount.fits"))):
        with fits.open(path, memmap=False) as hdul:
            counts[os.path.basename(path)] = int(np.nansum(np.asarray(hdul[0].data, dtype=np.float64)))
    return counts


def compare_counts(baseline: Path, candidate: Path, *, count_fraction: float) -> dict[str, Any]:
    a = _integration_counts(baseline)
    b = _integration_counts(candidate)
    record: dict[str, Any] = {"maps": {}, "status": "PASS"}
    if set(a) != set(b):
        record.update(status="FAIL", reason="integration maps differ", baselineMaps=sorted(a), candidateMaps=sorted(b))
        return record
    for name in sorted(a):
        base, cand = a[name], b[name]
        relative = abs(cand - base) / base if base else 0.0
        record["maps"][name] = {"baseline": base, "candidate": cand, "relativeDifference": relative}
        if relative > count_fraction:
            record["status"] = "FAIL"
            record["reason"] = f"{name}: accepted samples differ by {relative:.3%}"
    if not a:
        record.update(status="NOT_APPLICABLE", reason="no integration maps found")
    return record


def _summary_verdict(report_path: Path, filter_name: str | None) -> str | None:
    """The per-filter label of the ``summary.md`` next to a report, if any."""

    if not filter_name:
        return None
    summary = report_path.parent / "summary.md"
    if not summary.exists():
        return None
    marker = f"**{filter_name} verdict: "
    for line in summary.read_text(encoding="utf-8").splitlines():
        if line.startswith(marker):
            return line[len(marker):].split("**", 1)[0].strip()
    return None


def _headline_numbers(report: dict[str, Any]) -> dict[str, float | None]:
    info = report.get("info", {})
    gains = info.get("noise", {}).get("G", {})
    depth = info.get("depth", {})

    def first(value: Any) -> float | None:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        return float(value) if isinstance(value, (int, float)) else None

    return {"G_4": first(gains.get("4")), "G_8": first(gains.get("8")), "depthDm": first(depth.get("dm"))}


def compare_evaluations(pairs: list[tuple[Path, Path]], *, tolerance: float = 0.002) -> dict[str, Any]:
    record: dict[str, Any] = {"status": "PASS", "tolerance": tolerance, "reports": []}
    for base_path, cand_path in pairs:
        base = json.loads(base_path.read_text(encoding="utf-8"))
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        base_verdicts = {(m["family"], m["metric"]): m["status"] for m in base.get("metrics", [])}
        cand_verdicts = {(m["family"], m["metric"]): m["status"] for m in cand.get("metrics", [])}
        changed = sorted(f"{family}/{metric}: {base_verdicts.get((family, metric))} -> {cand_verdicts.get((family, metric))}"
                         for family, metric in set(base_verdicts) | set(cand_verdicts)
                         if base_verdicts.get((family, metric)) != cand_verdicts.get((family, metric)))
        base_numbers = _headline_numbers(base)
        cand_numbers = _headline_numbers(cand)
        shifts = {
            name: (None if base_numbers[name] is None or cand_numbers[name] is None else cand_numbers[name] - base_numbers[name])
            for name in base_numbers
        }
        exceeded = sorted(name for name, shift in shifts.items() if shift is not None and abs(shift) > tolerance)
        filter_name = base.get("filter") or cand.get("filter")
        base_label = _summary_verdict(base_path, filter_name)
        cand_label = _summary_verdict(cand_path, filter_name)
        entry: dict[str, Any] = {
            "filter": filter_name,
            "baseline": str(base_path),
            "candidate": str(cand_path),
            "changedVerdicts": changed,
            "headline": {"baseline": base_numbers, "candidate": cand_numbers, "shift": shifts},
            "exceededTolerance": exceeded,
            "label": {"baseline": base_label, "candidate": cand_label},
        }
        if base_label != cand_label and not changed and not exceeded:
            entry["note"] = (
                f"per-filter label {base_label} -> {cand_label} with identical metric statuses and headline "
                f"shifts within {tolerance:g}: a hard threshold of the label rule crossed by a rounding, not a change of the master"
            )
        record["reports"].append(entry)
        if changed or exceeded:
            record["status"] = "FAIL"
    if not pairs:
        record["status"] = "NOT_APPLICABLE"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="baseline project output directory")
    parser.add_argument("--candidate", required=True, help="candidate project output directory")
    parser.add_argument("--products", nargs="*", default=list(PRODUCT_NAMES), help="master names to compare (default L R G B)")
    parser.add_argument("--ulps", type=float, default=16.0)
    parser.add_argument("--bulk-sigma", type=float, default=0.002)
    parser.add_argument("--bulk-fraction", type=float, default=0.9999)
    parser.add_argument("--outlier-fraction", type=float, default=1e-6)
    parser.add_argument("--outlier-sigma", type=float, default=2.0)
    parser.add_argument("--count-fraction", type=float, default=1e-4)
    parser.add_argument("--evaluation", nargs=2, action="append", default=[], metavar=("BASELINE_JSON", "CANDIDATE_JSON"))
    parser.add_argument("--evaluation-tolerance", type=float, default=0.002, help="allowed shift of the evaluator's G_4, G_8 and depth Δm")
    parser.add_argument("--output", help="write the gate report JSON here")
    args = parser.parse_args(argv)
    baseline = Path(args.baseline).expanduser().resolve()
    candidate = Path(args.candidate).expanduser().resolve()
    masters = {}
    for name in args.products:
        base_path = baseline / f"{name}.fits"
        cand_path = candidate / f"{name}.fits"
        if not base_path.exists() or not cand_path.exists():
            masters[name] = {"status": "FAIL", "reason": "product missing", "baseline": str(base_path), "candidate": str(cand_path)}
            continue
        masters[name] = compare_master(
            base_path,
            cand_path,
            ulps=args.ulps,
            bulk_sigma=args.bulk_sigma,
            bulk_fraction=args.bulk_fraction,
            outlier_fraction=args.outlier_fraction,
            outlier_sigma=args.outlier_sigma,
        )
    counts = compare_counts(baseline, candidate, count_fraction=args.count_fraction)
    evaluations = compare_evaluations(
        [(Path(a).resolve(), Path(b).resolve()) for a, b in args.evaluation], tolerance=args.evaluation_tolerance
    )
    statuses = [record["status"] for record in masters.values()] + [counts["status"], evaluations["status"]]
    verdict = "FAIL" if "FAIL" in statuses else "PASS"
    report = {
        "verdict": verdict,
        "policy": {
            "ulps": args.ulps,
            "bulkSigma": args.bulk_sigma,
            "bulkFraction": args.bulk_fraction,
            "outlierFraction": args.outlier_fraction,
            "outlierSigma": args.outlier_sigma,
            "countFraction": args.count_fraction,
            "evaluationTolerance": args.evaluation_tolerance,
        },
        "masters": masters,
        "integrationCounts": counts,
        "evaluations": evaluations,
    }
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"tolerance gate: {verdict}")
    for name, record in masters.items():
        if "bulkFraction" in record:
            sigma_text = "n/a" if record["maxOutlierInSigma"] is None else f"{record['maxOutlierInSigma']:.2f} sigma"
            print(
                f"  {name}: {record['status']} — identical {record['identicalFraction']:.2%}, bulk {record['bulkFraction']:.5%}, "
                f"p99.99 |d| {record['differencePercentiles']['p99.99']:.2e} ADU, outliers {record['outlierPixels']} "
                f"(max {record['maxOutlierDifference']:.3g} ADU = {sigma_text})"
                + (f"; {record['reason']}" if record.get("reason") else "")
            )
        else:
            print(f"  {name}: {record['status']} — {record.get('reason')}")
    print(f"  integration accepted-sample counts: {counts['status']}" + (f" — {counts.get('reason')}" if counts.get("reason") else ""))
    print(f"  evaluation verdicts: {evaluations['status']}")
    for entry in evaluations.get("reports", []):
        shifts = ", ".join(f"{name} {shift:+.2e}" for name, shift in entry["headline"]["shift"].items() if shift is not None)
        labels = entry["label"]
        label_text = f"{labels['baseline']} -> {labels['candidate']}" if labels["baseline"] != labels["candidate"] else (labels["baseline"] or "n/a")
        print(f"    {entry['filter'] or '?'}: label {label_text}; {shifts}"
              + (f"; changed {entry['changedVerdicts']}" if entry["changedVerdicts"] else "")
              + (f"; exceeded {entry['exceededTolerance']}" if entry["exceededTolerance"] else "")
              + ("; NOTE: label boundary crossing" if entry.get("note") else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
