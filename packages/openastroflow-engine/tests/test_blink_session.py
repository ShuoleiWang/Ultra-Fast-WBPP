"""``blink-measure``: session directory, manifest contract, previews, errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from openastroflow_engine.blink_previews import (
    ChannelStretch,
    block_mean,
    compose_to_reference,
    stretch_to_8bit,
    warp_to_reference,
)
from openastroflow_engine.blink_session import (
    BlinkRequest,
    BlinkSessionError,
    run_blink_session,
)
from openastroflow_engine.cli import main
from conftest import write_frame
from test_e2e import synthetic_project  # noqa: F401  (fixture re-export)

JPEG_SIGNATURE = b"\xff\xd8\xff"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def isolated_qc_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENASTROFLOW_QC_CACHE_DIR", str(tmp_path / "analysis-cache"))


def _request(lights: list[Path], session: Path, **extra) -> dict:
    return {
        "schemaVersion": 1,
        "lightPaths": [str(path) for path in lights],
        "sessionDirectory": str(session),
        "workers": 1,
        **extra,
    }


def _check_manifest_contract(manifest: dict, session: Path, count: int) -> None:
    """The invariants the desktop loader enforces (contract-changes B-3)."""

    assert manifest["kind"] == "blink-manifest-v1" and manifest["schemaVersion"] == 1
    assert Path(manifest["sessionDirectory"]).resolve() == session.resolve()
    assert manifest["sessionId"] == session.name
    for key in ("inventorySha256", "gatePolicyDigest", "flagsPolicyDigest"):
        assert manifest[key].startswith("sha256:") and len(manifest[key]) == 71
    frames = manifest["frames"]
    assert len(frames) == count
    assert sorted(frame["index"] for frame in frames) == list(range(count))
    channels = {channel["channelId"]: channel for channel in manifest["channels"]}
    for frame in frames:
        assert frame["sourceSha256"].startswith("sha256:")
        assert frame["channelId"] in channels
        assert frame["defaultDecision"] in {"KEEP", "DROP"}
        has_exclude = any(flag["severity"] == "EXCLUDE" for flag in frame["flags"])
        assert (frame["defaultDecision"] == "DROP") == has_exclude
        assert all(flag["severity"] in {"EXCLUDE", "ATTENTION"} for flag in frame["flags"])
        for key in ("filmstrip", "zoom"):
            relative = frame["previews"][key]
            if relative is None:
                continue
            assert ".." not in relative and not relative.startswith("/")
            path = session / relative
            assert path.is_file() and not path.is_symlink()
            head = path.read_bytes()[:8]
            assert head.startswith(JPEG_SIGNATURE) or head.startswith(PNG_SIGNATURE)
    drop = sum(frame["defaultDecision"] == "DROP" for frame in frames)
    attention = sum(frame["defaultDecision"] == "KEEP" and bool(frame["flags"]) for frame in frames)
    assert manifest["counts"] == {
        "frames": count,
        "exclude": drop,
        "attention": attention,
        "clean": count - drop - attention,
    }
    for channel_id, channel in channels.items():
        members = [frame for frame in frames if frame["channelId"] == channel_id]
        assert channel["frameCount"] == len(members)
        reference = channel["reference"]
        assert reference is not None
        marked = [frame for frame in members if frame["reference"]]
        assert len(marked) == 1
        assert marked[0]["index"] == reference["index"]
        assert marked[0]["sourceSha256"] == reference["sourceSha256"]


def test_request_validation_codes(tmp_path: Path) -> None:
    light = write_frame(tmp_path / "in" / "light.fit", "Light")
    for raw in (
        [],
        {"schemaVersion": 2, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s")},
        {"schemaVersion": 1, "lightPaths": [], "sessionDirectory": str(tmp_path / "s")},
        {"schemaVersion": 1, "lightPaths": [str(light)]},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "extra": 1},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "workers": 0},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "previews": {"filmstripFormat": "gif"}},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "previews": {"filmstripScale": 2, "zoomScale": 4}},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "masterFlats": [{"path": "x"}]},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "masterFlats": [{"filter": "L", "path": "x"}, {"filter": "L", "path": "y"}]},
        {"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s"), "masterDarks": [{"path": "x", "exposureSeconds": -1}]},
    ):
        with pytest.raises(BlinkSessionError) as excinfo:
            BlinkRequest.from_mapping(raw)
        assert excinfo.value.code == "BLINK_REQUEST_INVALID"
    request = BlinkRequest.from_mapping(
        _request([light], tmp_path / "s", masterFlats=[{"filter": "l", "path": "flat"}], masterDarks=[{"path": "dark", "exposureSeconds": 120}], masterBias="bias")
    )
    assert request.master_flats[0].filter_name == "L"
    assert request.master_darks[0].exposure_seconds == 120.0 and request.master_bias == "bias"
    assert request.previews.filmstrip_scale == 8 and request.previews.jpeg_quality == 85
    # The desktop may omit ``workers``: the hardware default applies.
    assert BlinkRequest.from_mapping({"schemaVersion": 1, "lightPaths": [str(light)], "sessionDirectory": str(tmp_path / "s")}).workers is None


def test_session_is_create_only_and_refuses_non_lights(tmp_path: Path) -> None:
    light = write_frame(tmp_path / "in" / "盾牌座 light.fit", "Light")
    flat = write_frame(tmp_path / "in" / "flat.fit", "Flat", exposure=2.0)
    with pytest.raises(BlinkSessionError) as excinfo:
        run_blink_session(BlinkRequest.from_mapping(_request([light, flat], tmp_path / "s1")))
    assert excinfo.value.code == "BLINK_INPUT_NOT_LIGHT"
    assert not (tmp_path / "s1").exists()
    existing = tmp_path / "s2"
    existing.mkdir()
    with pytest.raises(BlinkSessionError) as excinfo:
        run_blink_session(BlinkRequest.from_mapping(_request([light], existing)))
    assert excinfo.value.code == "BLINK_SESSION_EXISTS"
    with pytest.raises(BlinkSessionError) as excinfo:
        run_blink_session(BlinkRequest.from_mapping(_request([light], light.parent / "session")))
    assert excinfo.value.code == "BLINK_REQUEST_INVALID"
    with pytest.raises(BlinkSessionError) as excinfo:
        run_blink_session(BlinkRequest.from_mapping(_request([tmp_path / "in" / "missing.fit"], tmp_path / "s3")))
    assert excinfo.value.code == "BLINK_REQUEST_INVALID"


def test_starless_lights_produce_a_complete_unregistered_session(tmp_path: Path) -> None:
    lights = [write_frame(tmp_path / "in" / f"light {index} °.fit", "Light") for index in range(3)]
    before = [path.read_bytes() for path in lights]
    session = tmp_path / "sessions" / "abc-20260922"
    manifest = run_blink_session(BlinkRequest.from_mapping(_request(lights, session)))
    _check_manifest_contract(manifest, session, 3)
    assert [path.read_bytes() for path in lights] == before
    on_disk = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert not (session / "linear").exists()
    assert manifest["counts"]["frames"] == 3
    # Star-less synthetic frames register nowhere: pre-dropped, unregistered
    # previews, but every frame still has both preview files.
    for frame in manifest["frames"]:
        assert "BLINK_UNREGISTRABLE" in {flag["code"] for flag in frame["flags"]}
        assert frame["defaultDecision"] == "DROP"
        assert frame["previews"]["filmstrip"] is not None and frame["previews"]["zoom"] is not None
        assert 0 < frame["previews"]["filmstripBytes"] <= 200 * 1024
        assert 0 < frame["previews"]["zoomBytes"] <= 2 * 1024 * 1024
        assert frame["name"] == Path(frame["path"]).name
        assert frame["gate"]["disposition"] in {"REVIEW", "HARD_FAIL"}
    (channel,) = manifest["channels"]
    assert channel["reference"]["candidacy"] == "unscored"
    reference = next(frame for frame in manifest["frames"] if frame["reference"])
    assert reference["normalization"]["registered"] is True
    assert reference["transformToReference"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    others = [frame for frame in manifest["frames"] if not frame["reference"]]
    assert all(frame["normalization"]["registered"] is False and frame["transformToReference"] is None for frame in others)
    assert set(manifest["timings"]) >= {"measurementSeconds", "analysisSeconds", "gateSeconds", "flagsSeconds", "previewSeconds"}
    assert manifest["previewOptions"]["filmstripFormat"] == "jpeg"
    assert channel["stretch"]["white"] > channel["stretch"]["black"]
    assert channel["previewGeometry"]["zoom"] == [16, 16]
    assert channel["previewGeometry"]["filmstrip"] == [8, 8]


def test_registered_star_field_session_and_cli(
    tmp_path: Path, synthetic_project: dict[str, tuple[Path, ...]], capsys: pytest.CaptureFixture[str]
) -> None:
    lights = list(synthetic_project["lights"][:8])
    session = tmp_path / "sessions" / "stars"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request(
                lights,
                session,
                workers=2,
                previews={"filmstripScale": 2, "zoomScale": 1, "filmstripFormat": "png", "jpegQuality": 90},
                # Raw calibration frames stand in for masters: the gradient
                # needs a flat and a pedestal source of the Lights' exposure.
                masterFlats=[{"filter": "R", "path": str(synthetic_project["flats"][0])}],
                masterDarks=[{"path": str(synthetic_project["darks"][0]), "exposureSeconds": 60.0}],
                masterBias=str(synthetic_project["biases"][0]),
            )
        ),
        encoding="utf-8",
    )
    assert main(["blink-measure", "--request-json", str(request_path), "--compact"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    _check_manifest_contract(manifest, session, 8)
    on_disk = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert (on_disk["kind"], on_disk["sessionId"], on_disk["inventorySha256"]) == (
        manifest["kind"], manifest["sessionId"], manifest["inventorySha256"],
    )
    (channel,) = manifest["channels"]
    assert channel["reference"]["candidacy"] == "clean"
    assert channel["statistics"]["cleanCount"] == 8
    assert channel["previewGeometry"]["zoom"] == [128, 128]
    assert channel["previewGeometry"]["filmstrip"] == [64, 64]
    ranks = sorted(frame["score"]["rank"] for frame in manifest["frames"])
    assert ranks == list(range(1, 9))
    for frame in manifest["frames"]:
        assert frame["defaultDecision"] == "KEEP" and frame["flags"] == []
        assert frame["normalization"]["registered"] is True
        assert frame["previews"]["coverage"] > 0.95
        assert frame["previews"]["filmstrip"].endswith(".png")
        assert 0.9 < frame["normalization"]["fluxScale"] < 1.1
        assert frame["metrics"]["gradientRatio"] is not None
        matrix = np.array(frame["transformToReference"])
        assert matrix.shape == (2, 3) and abs(matrix[0, 0] - 1.0) < 1e-3 and abs(matrix[0, 2]) < 2.0
    assert manifest["calibration"]["gradientFrames"] == 8
    assert manifest["calibration"]["notes"] == []
    assert channel["previewGeometry"]["calibration"] == {"flat": "flat_R_00.fits", "pedestal": "dark_00.fits"}
    assert all(frame["normalization"]["calibrated"] for frame in manifest["frames"])
    # The zoom previews of two dithered frames align after registration
    # (compared at 4x4 block scale so the pixel noise does not decide; a
    # 2-pixel misalignment drops this correlation to about 0.7).
    frames = sorted(manifest["frames"], key=lambda frame: frame["index"])
    reference = next(frame for frame in frames if frame["reference"])
    other = next(frame for frame in frames if not frame["reference"])
    a = np.asarray(Image.open(session / reference["previews"]["zoom"]), dtype=np.float32)
    b = np.asarray(Image.open(session / other["previews"]["zoom"]), dtype=np.float32)
    assert a.shape == b.shape == (128, 128)
    assert np.corrcoef(block_mean(a, 4).ravel(), block_mean(b, 4).ravel())[0, 1] > 0.9
    shifted = np.roll(b, 2, axis=1)
    assert np.corrcoef(block_mean(a, 4).ravel(), block_mean(shifted, 4).ravel())[0, 1] < 0.85
    # A second session into the same directory is refused with the stable code.
    assert main(["blink-measure", "--request-json", str(request_path)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "BLINK_SESSION_EXISTS"


def test_preview_primitives() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    small = block_mean(image, 2)
    assert small.shape == (2, 2) and small[0, 0] == pytest.approx(2.5)
    with_nan = image.copy()
    with_nan[:2, :2] = np.nan
    assert np.isnan(block_mean(with_nan, 2)[0, 0])
    # A pure translation by (+3, +2) preview pixels: the warped frame's pixel
    # (y, x) shows the frame's (y - 2, x - 3) and the border is NaN.
    frame = np.arange(64, dtype=np.float32).reshape(8, 8)
    warped = warp_to_reference(frame, ((1.0, 0.0, 3.0), (0.0, 1.0, 2.0)), (8, 8))
    assert warped[5, 6] == pytest.approx(frame[3, 3])
    assert np.isnan(warped[0, 0]) and np.isnan(warped[5, 1])
    identity = compose_to_reference(np.eye(3), np.eye(3))
    assert identity == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    shifted = compose_to_reference([[1, 0, 5], [0, 1, 1], [0, 0, 1]], [[1, 0, 2], [0, 1, 1], [0, 0, 1]])
    assert shifted == ((1.0, 0.0, 3.0), (0.0, 1.0, 0.0))
    assert compose_to_reference(None, np.eye(3)) is None
    stretch = ChannelStretch(black=0.0, white=100.0, softness=4.0, sky_reference=25.0, sigma_reference=10.0)
    pixels = stretch_to_8bit(np.array([[-10.0, 0.0, 50.0, 100.0, np.nan]], dtype=np.float32), stretch)
    assert pixels.dtype == np.uint8
    assert pixels[0, 0] == 0 and pixels[0, 1] == 0 and pixels[0, 3] == 255 and pixels[0, 4] == 0
    assert 128 < pixels[0, 2] < 255


def test_diagnostic_session_exports_complementary_views_without_changing_admission(synthetic_project: dict, tmp_path: Path) -> None:
    request = _request(
        synthetic_project["lights"][:8], tmp_path / "diagnostic-session",
        previews={"displayAlgorithm": "blink-complementary-display-v2", "filmstripFormat": "png"},
        masterFlats=[{"filter": "R", "path": str(synthetic_project["flats"][0])}],
        masterDarks=[{"path": str(synthetic_project["darks"][0]), "exposureSeconds": 60}],
    )
    manifest = run_blink_session(BlinkRequest.from_mapping(request))
    _check_manifest_contract(manifest, tmp_path / "diagnostic-session", 8)
    assert manifest["previewOptions"]["displayAlgorithm"] == "blink-complementary-display-v2"
    assert manifest["referenceRule"] == "calibrated-local-noise-psf-v2"
    assert not (tmp_path / "diagnostic-session" / "linear").exists()
    for frame in manifest["frames"]:
        assert frame["defaultDecision"] == "KEEP"
        assert frame["diagnostics"]["calibration"] == "calibrated"
        assert frame["diagnostics"]["backgroundStatus"] == "ready"
        for relative in frame["previews"]["diagnostic"].values():
            if relative:
                assert (tmp_path / "diagnostic-session" / relative).is_file()
        assert frame["diagnostics"]["nativeStatus"] == "ready"
    reference = next(f for f in manifest["frames"] if f["reference"])
    assert reference["diagnostics"]["relativeSignal"] == pytest.approx(1)
    assert reference["diagnostics"]["matchedSignalNoise"] == pytest.approx(1)
    assert reference["diagnostics"]["backgroundSpan"] == pytest.approx(0)
    # A Flat alone must not masquerade as complete preview calibration.
    request.pop("masterDarks")
    request["sessionDirectory"] = str(tmp_path / "flat-only-diagnostic")
    uncalibrated = run_blink_session(BlinkRequest.from_mapping(request))
    assert all(f["diagnostics"]["calibration"] == "uncalibrated" for f in uncalibrated["frames"])
    assert all(f["normalization"]["calibrated"] is False for f in uncalibrated["frames"])
