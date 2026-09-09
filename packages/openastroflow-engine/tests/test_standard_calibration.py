from dataclasses import replace
import hashlib
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from conftest import write_frame
from openastroflow_engine.calibration import CalibrationError, read_frame_info
from openastroflow_engine.calibration_preflight import inspect_calibration
from openastroflow_engine.inventory import inventory_project
from openastroflow_engine.pixel_pipeline import MasterMetadataOverride, run_portable_pipeline, _apply_master_metadata_overrides
from openastroflow_engine.recipe import Recipe, RecipeError
from openastroflow_engine.runtime import build_e2e_request
from test_pixel_pipeline import _dataset, _parameters, _write_frame, _sha

STANDARD = 'mono-standard-v1'


def _standard_recipe(**calibration):
    return Recipe.from_dict({'calibration': {'workflow': STANDARD, **calibration}})


def _drop(path, keys):
    with fits.open(path, mode='update') as hdus:
        for key in keys:
            hdus[0].header.remove(key, ignore_missing=True, remove_all=True)


def _unknown_masters(root):
    light = write_frame(root/'light.fits', 'Light')
    masters = [write_frame(root/'master_bias.fits', 'Master Bias', exposure=0.001), write_frame(root/'master_dark.fits', 'Master Dark'), write_frame(root/'master_flat.fits', 'Master Flat', exposure=2)]
    _drop(light, ('BAYERPAT',))
    for master in masters:
        _drop(master, ('INSTRUME','GAIN','OFFSET','READOUTM','BAYERPAT','CCD-TEMP'))
    return light, masters


def test_standard_preflight_preserves_unknowns_and_strict_still_blocks(tmp_path):
    light, masters = _unknown_masters(tmp_path)
    standard = inspect_calibration([str(tmp_path)], _standard_recipe())
    assert standard['calibrationReady'], standard['issues']
    assert 'DARK_TEMPERATURE_UNRECORDED' in {i['code'] for i in standard['issues']}
    assert read_frame_info(masters[0]).gain is None
    request = build_e2e_request(inventory_project([tmp_path]), _standard_recipe(), tmp_path/'output')
    assert request.pipeline_parameters.calibration_workflow == STANDARD
    assert request.pipeline_parameters.master_metadata_overrides == ()
    assert request.pipeline_parameters.serializable()['calibrationPolicy']['undeclaredMasterDarkBiasIncluded'] is True
    strict = inspect_calibration([str(tmp_path)])
    assert not strict['calibrationReady']
    assert 'CFA_CONFIRMATION_REQUIRED' in {i['code'] for i in strict['issues']}


@pytest.mark.parametrize('keyword,value', [('GAIN', 9), ('OFFSET', 99), ('READOUTM', 'OTHER'), ('INSTRUME','OTHER'), ('XBINNING',2), ('FILTER','G'), ('BAYERPAT','RGGB'), ('EXPTIME', 90), ('CCD-TEMP', 20)])
def test_standard_retains_known_master_conflicts(tmp_path, keyword, value):
    light, masters = _unknown_masters(tmp_path)
    fits.setval(light, 'CCD-TEMP', value=-10)
    target = masters[2] if keyword == 'FILTER' else masters[1]
    fits.setval(target, keyword, value=value)
    assert not inspect_calibration([str(tmp_path)], _standard_recipe())['calibrationReady']


def test_unknown_bias_does_not_hide_conflicting_lights(tmp_path):
    light, _ = _unknown_masters(tmp_path)
    second = write_frame(tmp_path/'second.fits', 'Light')
    fits.setval(second, 'GAIN', value=999)
    report = inspect_calibration([str(tmp_path)], _standard_recipe())
    assert not report['calibrationReady']
    assert 'CALIBRATION_PROFILE_UNSUPPORTED' in {i['code'] for i in report['issues']}


def test_sparse_overrides_keep_real_zero_and_unknown_does_not_erase_metadata(tmp_path):
    master = write_frame(tmp_path/'bias.fits', 'Master Bias', exposure=0.001)
    override = {'sourceSha256': _sha(master), 'biasIncluded': False, 'camera': 'UNKNOWN', 'gain': 0, 'offset': None}
    recipe = _standard_recipe(masterMetadataOverrides=[override])
    parsed = recipe.calibration.master_metadata_overrides[0]
    assert parsed.binning_x is None
    assert Recipe.from_dict(recipe.serializable()) == recipe
    actual = MasterMetadataOverride(source_sha256=_sha(master), camera='UNKNOWN', gain=0, bias_included=False)
    actual.validate()
    info = read_frame_info(master)
    updated, = _apply_master_metadata_overrides(({master: info},), (actual,), {})
    assert updated[master].camera == info.camera
    assert updated[master].offset == info.offset
    assert updated[master].gain == 0
    with pytest.raises(CalibrationError, match='exactly one'):
        _apply_master_metadata_overrides(({master: info},), (replace(actual, source_sha256='sha256:'+'0'*64),), {})
    with pytest.raises(RecipeError):
        _standard_recipe(workflow='automatic')


@pytest.mark.parametrize('mode', ['default','true','false','header-false','no-bias'])
def test_standard_master_dark_bias_semantics_are_numerically_correct(tmp_path, mode):
    _, _, _, lights, signal, response = _dataset(tmp_path/'raw')
    bias = _write_frame(tmp_path/'masters'/'bias.fits', 'Master Bias', np.full(signal.shape, 100), exposure=0.001)
    dark = _write_frame(tmp_path/'masters'/'dark.fits', 'Master Dark', np.full(signal.shape, 20 if mode in {'false','header-false'} else 120))
    flat = _write_frame(tmp_path/'masters'/'flat.fits', 'Master Flat', 5000*response, exposure=2)
    for path in (bias, dark, flat):
        _drop(path, ('INSTRUME','GAIN','OFFSET','READOUTM','BAYERPAT','CCD-TEMP'))
    for path in lights:
        _drop(path, ('BAYERPAT',))
    if mode == 'header-false':
        fits.setval(dark, 'OAFBIAS', value='SUBTRACTED')
    overrides = (MasterMetadataOverride(source_sha256=_sha(dark), bias_included=mode=='true'),) if mode in {'true','false'} else ()
    result = run_portable_pipeline(master_bias_file=None if mode=='no-bias' else bias, master_dark_files=[dark], master_flat_files=[flat], light_files=lights, output_directory=tmp_path/'result', parameters=replace(_parameters(), calibration_workflow=STANDARD, master_metadata_overrides=overrides))
    np.testing.assert_allclose(fits.getdata(result.master_light_paths[0]), signal, atol=3e-3)
    receipt=json.loads(Path(result.receipt_path).read_text())
    assert receipt['parameters']['calibrationWorkflow'] == STANDARD
    assert receipt['statistics']['calibration']['masterDark:30']['biasIncluded'] is (mode not in {'false','header-false'})


def test_no_bias_is_blocked_when_dark_has_bias_removed(tmp_path):
    light, masters = _unknown_masters(tmp_path)
    masters[0].unlink()
    report = inspect_calibration([str(tmp_path)], _standard_recipe(bias='OPTIONAL'))
    assert report['calibrationReady'], report['issues']
    assert 'BIAS_MATCH_MISSING' not in {i['code'] for i in report['issues']}
    fits.setval(masters[1], 'OAFBIAS', value='SUBTRACTED')
    report = inspect_calibration([str(tmp_path)], _standard_recipe(bias='OPTIONAL'))
    assert not report['calibrationReady']
    assert 'BIAS_REQUIRED_FOR_CALIBRATION' in {i['code'] for i in report['issues']}
