"""Negative evidence must survive the master evaluator's summary gates."""
import copy
import importlib.util
from pathlib import Path

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
