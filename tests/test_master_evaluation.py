"""Negative evidence must survive the master evaluator's summary gates."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("evaluate_masters", Path(__file__).resolve().parents[1] / "tools/validation/evaluate_masters.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def report():
    return {
        "filter": "L",
        "metrics": [{"family": family, "metric": name, "status": "PASS", "verdict": True,
                     "pi": 1., "ours": 1., "d": 0., "tau": .02, "ci": [0., 0.]}
                    for family, name in [("psf", "FWHM"), ("background", "flatness"), ("artefacts", "trails"),
                                         ("photometry", "linearity"), ("noise", "q_faint"), ("noise", "q_LSB")]],
        "info": {"unmeasured_standard_gates": [], "valid_alignment": True, "alignment": {"matched": 500, "rms": .03},
                 "photometric_model": {"a": 1., "b": 0., "lnRatioMadn": 0., "stars": 500},
                 "noise": {"G": {"8": [1.1, 1.08, 1.12], "4": [1.08, 1.06, 1.1]},
                           "sigma1_ours_Punits": 1., "sigma1_pi": 1.,
                           "rho_pi": {"x1": 0., "y1": 0.}, "rho_ours": {"x1": 0., "y1": 0.}},
                 "background": {"sigma_ref": 1.}, "timing_seconds": 0., "depth": {"dm": .1, "ci": [.08, .12]}},
    }


@pytest.mark.parametrize("defect, expected", [("alignment", "INCONCLUSIVE"), ("faint", "WORSE"), ("lsb_missing", "INCONCLUSIVE")])
def test_scientific_gates_cannot_be_bypassed(defect, expected):
    value = report()
    assert evaluator.filter_verdict(value)[0] == "BETTER"
    if defect == "alignment":
        value["info"]["valid_alignment"] = False
        value["info"]["alignment"] = {"matched": 30, "rms": .8}
    elif defect == "faint":
        value["metrics"][-2].update(status="FAIL", ours=.8, d=.2, ci=[.18, .22])
    else:
        value["metrics"].pop()
    assert evaluator.filter_verdict(value)[0] == expected


def test_candidate_cross_channel_failure_changes_run_verdict(tmp_path):
    cross = {"label": "ours", "pairs": [{"pair": "R-G", "matched": 500, "median": [.5, 0.], "rms": .3, "worstZone": .5}]}
    assert "**Run verdict: WORSE**" in evaluator.summarize([report()], [cross], tmp_path)
    reference = copy.deepcopy(cross)
    reference["label"] = "pi"
    assert "**Run verdict: BETTER**" in evaluator.summarize([report()], [reference], tmp_path)
    cross["pairs"][0].update(matched=0, worstZone=float("nan"))
    assert "**Run verdict: INCONCLUSIVE**" in evaluator.summarize([report()], [cross], tmp_path)


@pytest.mark.parametrize("lo,hi,expected", [(.96,.99,(.01,.04)),(.98,1.02,(0.,.02)),(1.01,1.04,(.01,.04))])
def test_absolute_confidence_interval_keeps_the_nearest_endpoint(lo, hi, expected):
    assert evaluator.absolute_interval(lo, hi) == pytest.approx(expected)


def test_unmeasured_standard_gates_cannot_certify_equivalence():
    value = report()
    value['info']['noise']['G']['8'] = [1., .99, 1.01]
    value['info']['unmeasured_standard_gates'] = ['q_LSB']
    assert evaluator.filter_verdict(value)[0] == 'INCONCLUSIVE'


@pytest.fixture
def star_field():
    noise = np.random.default_rng(42).normal(size=(512, 512))
    rows, cols = np.indices(noise.shape)
    for row, col in [(55, 65), (96, 240), (168, 144), (256, 202)]:
        noise += 60 * np.exp(-((rows - row)**2 + (cols - col)**2) / 8)
    return noise.astype(np.float32)


@pytest.mark.parametrize("offset, blank", [(3.9, False), (6., True), (9.5, False), (12.49, False), (12.6, True)])
@pytest.mark.parametrize("defect", ["source", "invalid"])
def test_blank_aperture_checks_both_footprints_but_not_the_gap(star_field, offset, blank, defect):
    image = evaluator.Analysis("test", star_field)
    if defect == "source":
        image.star_mask[350, 370] = True
    else:
        image.valid[350, 370] = False
    positions = np.array([[370 - offset, 350.]])
    assert bool(np.isfinite(evaluator._blank_flux(image, positions)[0])) == blank


def test_identical_sky_gives_exact_paired_depth_after_alignment(star_field):
    reference = evaluator.Analysis("reference", star_field)
    candidate = evaluator.Analysis("candidate", np.roll(star_field, (7, -9), axis=(0, 1)))
    candidate.sub = np.roll(reference.sub, (7, -9), axis=(0, 1)) * 2
    reference.valid[110:140, 180:190] = False
    candidate.star_mask[200:220, 96:110] = True
    model = np.array([[1., 0., 9.], [0., 1., -7.]])
    depth = evaluator._paired_depth(candidate, reference, model, .5, np.random.default_rng(0))
    assert depth['apertures'] == 3000 and depth['reason'] is None
    assert depth['flim5_pi'] == pytest.approx(depth['flim5_ours_Punits'], abs=1e-10)
    assert depth['dm'] == pytest.approx(0., abs=1e-12)
    assert depth['ci'] == pytest.approx([0., 0.], abs=1e-12)


@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_depth_unavailable_without_common_blank_sky(star_field, side, tmp_path):
    reference = evaluator.Analysis("reference", star_field)
    candidate = evaluator.Analysis("candidate", star_field)
    (candidate if side == "candidate" else reference).valid[:] = False
    depth = evaluator._paired_depth(candidate, reference, np.eye(2, 3), 1., np.random.default_rng(0))
    assert depth['apertures'] == 0 and np.isnan(depth['dm'])
    assert np.isnan(depth['ci']).all()
    value = report()
    value['info']['depth'] = depth
    value['metrics'].append(dict(value['metrics'][-1], metric='depth Δm', status='N/A'))
    summary = evaluator.summarize([value], [], tmp_path)
    assert "Depth unavailable: insufficient paired blank apertures (0/3000" in summary
    assert "**Run verdict: INCONCLUSIVE**" in summary


def test_flat_sky_cannot_produce_a_depth_verdict(star_field):
    image = evaluator.Analysis("flat", star_field)
    image.sub.fill(0.)
    depth = evaluator._paired_depth(image, image, np.eye(2, 3), 1., np.random.default_rng(0))
    assert depth['apertures'] == 3000 and np.isnan(depth['dm'])
    assert np.isnan(depth['ci']).all() and depth['reason']
