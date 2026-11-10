from __future__ import annotations

import numpy as np

from lightframeqc.metadata import (
    grouping_keywords_from_path,
    infer_bayer_pattern_from_path,
    infer_binning_from_path,
    infer_exposure_from_path,
    infer_filter_from_path,
    infer_frame_role_from_path,
    infer_target_from_folders,
    infer_target_from_path,
    normalize_filter,
    parse_frame_role,
    path_metadata,
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


def test_grouping_keywords_follow_wbpp_syntax_anywhere_in_the_path() -> None:
    assert grouping_keywords_from_path("/data/NIGHT_1/Flats/f.fits") == (("NIGHT", "1"),)
    # "_", "-" or a space; names in any case, values as written.
    assert grouping_keywords_from_path("/data/night 2/L_SESSION-12Feb_001.fits") == (
        ("NIGHT", "2"),
        ("SESSION", "12Feb"),
    )
    # The innermost value wins; an underscore ends a value.
    assert grouping_keywords_from_path("/d/NIGHT_1/sub/NIGHT_3/l.fits") == (("NIGHT", "3"),)
    assert grouping_keywords_from_path("/d/DATE_2024-03-22_moon/l.fits") == (("DATE", "2024-03-22"),)
    assert grouping_keywords_from_path("/m/PANEL_2/Lights/l.fits") == (("PANEL", "2"),)
    # A word that merely contains a keyword, or a N.I.N.A. target folder, is none.
    assert grouping_keywords_from_path("/d/UPDATE_1/NGC 6822_2026-08-17/LIGHT/l.fits") == ()


def test_smart_naming_supplies_filter_binning_exposure_and_pattern() -> None:
    metadata = path_metadata("/w/Light_BIN-2_EXPOSURE-300.00s_FILTER-Ha_BAYERPAT-RGGB/NIGHT_1/a.fits")
    assert metadata.filter_name == "HA"
    assert metadata.binning == 2
    assert metadata.exposure_seconds == 300.0
    assert metadata.cfa_pattern == "RGGB"
    assert metadata.keywords == (("NIGHT", "1"),)
    assert infer_exposure_from_path("/data/lights_60s/img_0001.fits") == 60.0
    assert infer_binning_from_path("/data/Binning_1x1/img.fits") == 1
    assert infer_binning_from_path("/data/cabin_2/img.fits") is None
    assert infer_bayer_pattern_from_path("/data/img.fits") is None


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


def test_one_filter_identity_names_every_alias_spelling() -> None:
    for spelling, expected in (
        ("L", "L"), ("Lum", "L"), ("Luminance", "L"), ("lum", "L"),
        ("Red", "R"), ("Green", "G"), ("Blue", "B"),
        ("Ha", "HA"), ("H-alpha", "HA"), ("H_Alpha", "HA"), ("Halpha", "HA"), ("H alpha", "HA"), ("Hα", "HA"),
        ("O3", "OIII"), ("OIII", "OIII"), ("O-III", "OIII"), ("Oxygen III", "OIII"),
        ("S2", "SII"), ("SII", "SII"), ("Sulfur II", "SII"),
        ("NoFilter", "NOFILTER"), ("No Filter", "NOFILTER"),
        # Other names are themselves, in upper case, and never an alias.
        ("L-Pro", "L-PRO"), ("Clear", "CLEAR"), ("Ha 3nm", "HA 3NM"), ("UV/IR", "UV/IR"),
    ):
        assert normalize_filter(spelling) == expected, spelling
        assert normalize_filter(normalize_filter(spelling)) == expected, spelling
    assert normalize_filter(None) == normalize_filter("  ") == "UNKNOWN"


def test_the_documented_folder_spellings_name_their_role() -> None:
    assert infer_frame_role_from_path("/data/FlatField/x_001.fits") is FrameRole.RAW_FLAT
    assert infer_frame_role_from_path("/data/Science/x_001.fits") is FrameRole.LIGHT
    assert infer_frame_role_from_path("/data/M31/2026-03-22/x_001.fits") is FrameRole.UNKNOWN


def test_a_file_name_states_a_target_only_without_a_date_or_time() -> None:
    assert infer_target_from_path("/d/M31_300s_0001.fits") == "M31"
    assert infer_target_from_path("/d/2026-03-22_21-10-11_L_-10.00_300.00s_0001.fits") == "UNKNOWN"
    assert infer_target_from_path("/d/img_0001.fits") == "UNKNOWN"


def test_the_folders_below_the_import_name_a_light_without_object() -> None:
    def named(path: str, filter_name: str = "L") -> str | None:
        return infer_target_from_folders(path, root="/data", filter_name=filter_name)

    assert named("/data/M31/2026-03-22/LIGHT/L/x.fits") == "M31"
    assert named("/data/SDSS J1004+4112_2026-11-02/LIGHT/L/x.fits") == "SDSS J1004+4112"
    assert named("/data/玫瑰星云_2026-03-22/LIGHT/x.fits") == "玫瑰星云"
    assert named("/data/NGC7000/Ha 3nm/x.fits", filter_name="HA 3NM") == "NGC7000"
    # Frame-type, filter, date, exposure and grouping-keyword folders name nothing,
    # and no folder above the import is read.
    assert named("/data/2026-03-22/LIGHT/x.fits") is None
    assert named("/data/NIGHT_1/Lights_300s/R/x.fits", filter_name="R") is None
    assert infer_target_from_folders("/volume/owner/LIGHT/x.fits", root="/volume/owner") is None
