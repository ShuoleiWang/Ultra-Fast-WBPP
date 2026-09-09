from __future__ import annotations

import csv
from html import escape
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .models import Decision, FrameResult, GateDisposition, RunResult


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _csv_safe(value: Any) -> Any:
    """Neutralize spreadsheet formulas while preserving JSON's raw value."""

    if isinstance(value, str) and value.lstrip()[:1] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def _decision_badge(decision: Decision) -> str:
    return f'<span class="badge {escape(decision.value.lower())}">{escape(decision.value)}</span>'


def _gate_badge(frame: FrameResult) -> str:
    if frame.quality_gate is None:
        return '<span class="gate gate-unset">GATE —</span>'
    disposition = frame.quality_gate.disposition.value
    return (
        f'<span class="gate gate-{escape(disposition.lower().replace("_", "-"))}">'
        f'GATE {escape(disposition)}</span>'
    )


def _heatmap(frame: FrameResult) -> str:
    completeness = frame.grid.get("completeness")
    missing = frame.grid.get("missingMask")
    if not completeness or not missing:
        return '<div class="no-map">无空间图</div>'
    rows = len(completeness)
    columns = len(completeness[0]) if rows else 0
    cells: list[str] = []
    for row in range(rows):
        for column in range(columns):
            value = completeness[row][column]
            is_missing = bool(missing[row][column])
            if value is None:
                color = "#26313a"
                title = "无参考星"
            else:
                clamped = max(0.0, min(1.0, float(value)))
                red = round(205 * (1.0 - clamped) + 25)
                green = round(165 * clamped + 35)
                color = f"rgb({red},{green},60)"
                title = f"星点完整率 {clamped:.1%}"
            outline = "inset 0 0 0 1px #ff6b6b" if is_missing else "none"
            cells.append(
                f'<span title="{escape(title)}" style="background:{color};box-shadow:{outline}"></span>'
            )
    return (
        f'<div class="heatmap" style="grid-template-columns:repeat({columns},1fr)">'
        + "".join(cells)
        + "</div>"
    )


def _thumbnail(frame: FrameResult, output_directory: Path) -> str:
    if not frame.thumbnail_path:
        return _heatmap(frame)
    path = Path(frame.thumbnail_path)
    try:
        source = path.resolve().relative_to(output_directory.resolve()).as_posix()
    except ValueError:
        return _heatmap(frame)
    return (
        '<div class="preview">'
        f'<img loading="lazy" src="{escape(source)}" alt="preview">'
        f'{_heatmap(frame)}'
        "</div>"
    )


def _frame_row(frame: FrameResult, output_directory: Path) -> str:
    features = frame.features
    metadata = frame.metadata
    reasons = ", ".join(frame.reasons) or "—"
    warnings = ", ".join(frame.warnings)
    gate_codes = (
        ", ".join(item.code for item in frame.quality_gate.evidence)
        if frame.quality_gate is not None
        else ""
    )
    gate_value = (
        frame.quality_gate.disposition.value
        if frame.quality_gate is not None
        else "NOT_EVALUATED"
    )
    name = Path(frame.path).name
    return f"""
      <tr data-decision="{escape(frame.decision.value)}" data-gate="{escape(gate_value)}" data-filter="{escape(metadata.filter_name)}">
        <td>{_decision_badge(frame.decision)} {_gate_badge(frame)}<div class="confidence">{escape(frame.confidence.value)}</div></td>
        <td class="file"><span title="{escape(frame.path)}">{escape(name)}</span><small>{escape(frame.group_id)}</small></td>
        <td>{escape(metadata.filter_name)}<small>{_fmt(metadata.exposure_seconds, 1)} s</small></td>
        <td>{frame.star_count}<small>匹配 {frame.registration.matched_stars}, RMS {_fmt(frame.registration.rms_pixels, 2)} px · HFR {_fmt(features.nina_hfr_pixels, 2)} · FWHM {_fmt(features.median_fwhm_native_pixels, 2)} native px · e {_fmt(features.median_eccentricity, 3)} · elong {_fmt(features.elongated_fraction, 3)}</small></td>
        <td>{_fmt(features.transparency_ratio)}<small>额外消光 {_fmt(features.extra_extinction_mag)} mag</small></td>
        <td>{_fmt(features.star_completeness)}<small>检出星比 {_fmt(features.detected_source_ratio)} · 局部变暗 P90 {_fmt(features.spatial_dimming_p90_mag)} / 变亮 P90 {_fmt(features.spatial_brightening_p90_mag)} mag</small></td>
        <td>{_fmt(features.largest_missing_region)}<small>共同视场 {_fmt(features.overlap_fraction)} · 边界 {_fmt(features.boundary_support)}</small></td>
        <td><strong>C {features.cloud_score}</strong> / <strong>O {features.occlusion_score}</strong> / <strong>S {features.shape_score}</strong><small>背景 z {_fmt(features.background_z, 2)} · 噪声 z {_fmt(features.noise_z, 2)} · 夜间消光残差 {_fmt(features.nightly_extinction_residual, 3)}</small><small>{escape(reasons)}</small>{f'<small class="gate-codes">{escape(gate_codes)}</small>' if gate_codes else ''}{f'<small class="warning">{escape(warnings)}</small>' if warnings else ''}</td>
        <td>{_thumbnail(frame, output_directory)}</td>
      </tr>
    """


def render_html(run: RunResult) -> str:
    counts = {
        decision: sum(frame.decision == decision for frame in run.frames)
        for decision in Decision
    }
    decision_cards = "".join(
        f'<button data-kind="decision" data-filter="{decision.value}"><span>{escape(decision.value)}</span><b>{count}</b></button>'
        for decision, count in counts.items()
    )
    gate_counts = {
        gate: sum(
            frame.quality_gate is not None
            and frame.quality_gate.disposition is gate
            for frame in run.frames
        )
        for gate in GateDisposition
    }
    gate_cards = "".join(
        f'<button data-kind="gate" data-filter="{gate.value}"><span>GATE {escape(gate.value)}</span><b>{count}</b></button>'
        for gate, count in gate_counts.items()
    )
    rows = "".join(_frame_row(frame, Path(run.output_directory)) for frame in run.frames)
    warning_block = "".join(f"<li>{escape(item)}</li>" for item in run.warnings)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Light Frame QC</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1015; --panel:#121a22; --line:#26313a; --text:#e8eef3; --muted:#91a2af; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text) }}
header {{ padding:28px 32px 18px; position:sticky; top:0; z-index:4; background:linear-gradient(180deg,#0b1015 80%,transparent) }}
h1 {{ margin:0 0 4px; font-size:26px }}
.subtitle,small,.confidence {{ color:var(--muted) }}
.cards {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:16px }}
.cards button {{ border:1px solid var(--line); background:var(--panel); color:var(--text); border-radius:10px; padding:8px 12px; cursor:pointer }}
.cards button.active {{ outline:2px solid #6eb5ff }} .cards b {{ margin-left:8px; font-size:18px }}
main {{ padding:0 24px 36px; overflow:auto }}
table {{ width:100%; border-collapse:separate; border-spacing:0; min-width:1350px; background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden }}
th {{ position:sticky; top:118px; z-index:3; text-align:left; background:#16212a; color:#b9c7d1; padding:10px; border-bottom:1px solid var(--line) }}
td {{ padding:10px; vertical-align:top; border-bottom:1px solid var(--line) }} tr:last-child td {{ border-bottom:0 }}
td small,.file small {{ display:block; max-width:240px; overflow-wrap:anywhere; margin-top:3px }}
.file span {{ display:block; max-width:260px; overflow-wrap:anywhere }}
.badge {{ display:inline-block; font-weight:700; font-size:11px; padding:4px 7px; border-radius:999px }}
.gate {{ display:inline-block; font-weight:700; font-size:10px; padding:3px 6px; border-radius:999px; margin-top:4px }}
.gate-pass {{ background:#17422d; color:#8aebba }} .gate-review {{ background:#4a3b12; color:#ffda74 }}
.gate-hard-fail {{ background:#5a2028; color:#ff9ca7 }} .gate-unset {{ background:#303840; color:#b8c3cb }}
.keep {{ background:#143d2a; color:#7de2ad }} .review {{ background:#493b12; color:#ffd86b }}
.reject_cloud {{ background:#243d59; color:#84c5ff }} .reject_occlusion {{ background:#58252b; color:#ff9ba5 }}
.unassessable {{ background:#313941; color:#bbc5cc }} .warning {{ color:#ffac73 }} .gate-codes {{ color:#8fd3ff }}
.preview {{ display:flex; gap:6px; align-items:center }} .preview img {{ width:128px; max-height:96px; object-fit:contain; background:#05080a }}
.heatmap {{ width:128px; aspect-ratio:1.33; display:grid; gap:1px; background:#05080a; padding:2px }} .heatmap span {{ min-width:2px }}
.no-map {{ color:var(--muted); width:128px }}
.warnings {{ margin:0 24px 18px; color:#ffbf8d }}
@media(max-width:800px) {{ header {{ padding-left:16px }} main {{ padding-left:8px }} }}
</style>
</head>
<body>
<header>
  <h1>Light Frame QC</h1>
  <div class="subtitle">{escape(run.generated_at.isoformat())} · {len(run.frames)} 帧 · 算法 {escape(run.algorithm_version)}</div>
  <div class="cards"><button data-kind="all" data-filter="ALL" class="active"><span>ALL</span><b>{len(run.frames)}</b></button>{gate_cards}{decision_cards}</div>
</header>
{f'<ul class="warnings">{warning_block}</ul>' if warning_block else ''}
<main>
<table>
<thead><tr><th>结论</th><th>文件</th><th>通道</th><th>星点</th><th>透明度</th><th>空间一致性</th><th>缺失区域</th><th>证据</th><th>预览/缺星图</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</main>
<script>
for (const button of document.querySelectorAll('.cards button')) {{
  button.addEventListener('click', () => {{
    document.querySelectorAll('.cards button').forEach(b => b.classList.remove('active'));
    button.classList.add('active');
    const wanted = button.dataset.filter;
    const kind = button.dataset.kind;
    document.querySelectorAll('tbody tr').forEach(row => {{
      row.hidden = kind === 'all' ? false :
        kind === 'gate' ? row.dataset.gate !== wanted : row.dataset.decision !== wanted;
    }});
  }});
}}
</script>
</body></html>"""


def write_reports(run: RunResult) -> dict[str, Path]:
    output = Path(run.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    html_path = output / "report.html"
    csv_path = output / "frames.csv"

    # Validate both serializations before replacing any prior report.  JSON is
    # written last and acts as the machine-readable completion marker.
    json_text = (
        json.dumps(run.serializable(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    html_text = render_html(run)
    _atomic_text(html_path, html_text)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=csv_path.name + ".", suffix=".tmp", dir=output
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "path",
                    "decision",
                    "confidence",
                    "gate_disposition",
                    "gate_evidence_count",
                    "gate_evidence_codes",
                    "group_id",
                    "filter",
                    "frame_role",
                    "exposure_seconds",
                    "star_count",
                    "matched_stars",
                    "registration_rms_px",
                    "transparency_ratio",
                    "extra_extinction_mag",
                    "star_completeness",
                    "detected_source_ratio",
                    "spatial_transparency_mad_mag",
                    "spatial_transparency_p90_mag",
                    "spatial_dimming_p90_mag",
                    "spatial_brightening_p90_mag",
                    "largest_missing_region",
                    "overlap_fraction",
                    "nina_hfr_pixels",
                    "median_fwhm_preview_pixels",
                    "p90_fwhm_preview_pixels",
                    "median_ellipticity",
                    "p90_ellipticity",
                    "image_median",
                    "image_mad",
                    "median_fwhm_native_pixels",
                    "p90_fwhm_native_pixels",
                    "median_axis_ratio",
                    "median_eccentricity",
                    "elongated_fraction",
                    "orientation_coherence",
                    "valid_morphology_star_count",
                    "nightly_extinction_residual",
                    "background_z",
                    "noise_z",
                    "cloud_score",
                    "occlusion_score",
                    "shape_score",
                    "source_sha256",
                    "reasons",
                    "warnings",
                ]
            )
            for frame in run.frames:
                writer.writerow(
                    [
                        _csv_safe(value)
                        for value in [
                            frame.path,
                            frame.decision.value,
                            frame.confidence.value,
                            (
                                frame.quality_gate.disposition.value
                                if frame.quality_gate is not None
                                else None
                            ),
                            (
                                len(frame.quality_gate.evidence)
                                if frame.quality_gate is not None
                                else 0
                            ),
                            (
                                ";".join(
                                    item.code for item in frame.quality_gate.evidence
                                )
                                if frame.quality_gate is not None
                                else ""
                            ),
                            frame.group_id,
                            frame.metadata.filter_name,
                            frame.metadata.role.value,
                            frame.metadata.exposure_seconds,
                            frame.star_count,
                            frame.registration.matched_stars,
                            frame.registration.rms_pixels,
                            frame.features.transparency_ratio,
                            frame.features.extra_extinction_mag,
                            frame.features.star_completeness,
                            frame.features.detected_source_ratio,
                            frame.features.spatial_transparency_mad_mag,
                            frame.features.spatial_transparency_p90_mag,
                            frame.features.spatial_dimming_p90_mag,
                            frame.features.spatial_brightening_p90_mag,
                            frame.features.largest_missing_region,
                            frame.features.overlap_fraction,
                            frame.features.nina_hfr_pixels,
                            frame.features.median_fwhm_preview_pixels,
                            frame.features.p90_fwhm_preview_pixels,
                            frame.features.median_ellipticity,
                            frame.features.p90_ellipticity,
                            frame.features.image_median,
                            frame.features.image_mad,
                            frame.features.median_fwhm_native_pixels,
                            frame.features.p90_fwhm_native_pixels,
                            frame.features.median_axis_ratio,
                            frame.features.median_eccentricity,
                            frame.features.elongated_fraction,
                            frame.features.orientation_coherence,
                            frame.features.valid_morphology_star_count,
                            frame.features.nightly_extinction_residual,
                            frame.features.background_z,
                            frame.features.noise_z,
                            frame.features.cloud_score,
                            frame.features.occlusion_score,
                            frame.features.shape_score,
                            frame.identity.sha256 if frame.identity is not None else None,
                            ";".join(frame.reasons),
                            ";".join(frame.warnings),
                        ]
                    ]
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, csv_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    _atomic_text(json_path, json_text)
    return {"json": json_path, "csv": csv_path, "html": html_path}
