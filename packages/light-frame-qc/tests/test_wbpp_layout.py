from __future__ import annotations

import numpy as np

from lightframeqc.metadata import (
    infer_filter_from_path,
    infer_frame_role_from_path,
    parse_frame_role,
    processing_markers,
    session_keywords_from_path,
    session_keywords_of,
)
from lightframeqc.models import FrameRole
from lightframeqc.readers import _xisf_header_dict


def test_plural_and_flat_dark_folders_name_the_role_when_headers_do_not() -> None:
    assert infer_frame_role_from_path("/data/Lights/img_0001.fits") is FrameRole.LIGHT
    assert infer_frame_role_from_path("/data/Flats/img_0001.fits") is FrameRole.RAW_FLAT
    assert infer_frame_role_from_path("/data/Darks/img_0001.fits") is FrameRole.DARK
    assert infer_frame_role_from_path("/data/Biases/img_0001.fits") is FrameRole.BIAS
    assert infer_frame_role_from_path("/data/DarkFlats/img_0001.fits") is FrameRole.DARK
    assert infer_frame_role_from_path("/data/FlatDarks/img_0001.fits") is FrameRole.DARK
    assert infer_frame_role_from_path("/data/masterFlatDark_2s.xisf") is FrameRole.MASTER_DARK
    assert infer_frame_role_from_path("/data/masterFlat_FILTER-L.xisf") is FrameRole.MASTER_FLAT
    # N.I.N.A. writes IMAGETYP DARKFLAT; exact-exposure matching keeps it on the Flats.
    assert parse_frame_role("DARKFLAT") is FrameRole.DARK
    assert parse_frame_role("Flat Dark") is FrameRole.DARK


def test_filter_comes_from_a_wbpp_keyword_folder_only_when_named_as_such() -> None:
    assert infer_filter_from_path("/data/FILTER_Ha/img_0001.fits") == "HA"
    assert infer_filter_from_path("/w/Light_BIN-1_EXPOSURE-300.00s_FILTER-OIII_mono/a.fits") == "OIII"
    assert infer_filter_from_path("/data/L/img_0001.fits") == "UNKNOWN"


def test_session_keywords_follow_wbpp_grouping_names() -> None:
    assert session_keywords_from_path("/data/supernova/DATE_0322/light.fits") == "DATE=0322"
    # WBPP's own output folders carry the same keyword and value.
    assert session_keywords_from_path("/w/calibrated/Light_BIN-1_FILTER-R_mono_DATE-0322/a_c.xisf") == "DATE=0322"
    assert session_keywords_from_path("/data/NIGHT-2/SESSION_B/light.fits") == "NIGHT=2;SESSION=B"
    assert session_keywords_from_path("/data/UPDATE_1/light.fits") is None
    # A shared ancestor is not a keyword of the frames compared below it.
    assert session_keywords_of(["/trips/one_night_trip/DATE_A/f.fits", "/trips/one_night_trip/DATE_B/f.fits"]) == ["DATE=A", "DATE=B"]


def test_processing_markers_from_xisf_properties_and_the_wbpp_layout() -> None:
    header = _xisf_header_dict(
        {
            "XISFProperties": {
                "PCL:Calibration:CosmeticCorrection:HighCounts": {"value": 3},
                "PCL:AlignmentMatrix": {"value": np.eye(3)},
            }
        }
    )
    assert header["XISF:PROCESSING"] == "CALIBRATED,REGISTERED"
    assert processing_markers(header, "/anywhere/frame.xisf") == ["CALIBRATED", "REGISTERED"]
    assert processing_markers({}, "/w/calibrated/Light_x/a_c.fits") == ["CALIBRATED"]
    assert processing_markers({}, "/w/registered/Light_x/a_c_r.fits") == ["CALIBRATED", "REGISTERED"]
    # A raw file that merely ends in _c, outside WBPP's folders, is untouched.
    assert processing_markers({}, "/data/night/a_c.fits") == []
