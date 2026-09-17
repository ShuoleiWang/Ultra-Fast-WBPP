from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import xml.etree.ElementTree as ET

from astropy.io import fits
import numpy as np
import pytest
from lightframeqc.xisf import XISF

from lightframeqc.adjudication import (
    AdjudicationAction,
    AdjudicationError,
    AdjudicationRecord,
    parse_adjudication,
)
from lightframeqc.identity import compute_file_identity
from lightframeqc.models import (
    Confidence,
    Decision,
    FrameFeatures,
    FrameResult,
    FrameRole,
    GateDisposition,
    QualityGateResult,
    RegistrationMetrics,
)
from lightframeqc.prepare import (
    MANIFEST_NAME,
    PrepareApplyError,
    PreparePlanError,
    apply_prepare_plan,
    build_prepare_plan,
    load_prepare_plan,
)
from lightframeqc.quality_gate import GatePolicy
from lightframeqc.readers import probe_frame_metadata


def _write_frame(
    path: Path,
    *,
    role: str,
    target: str,
    filter_name: str,
    seed: int,
    camera: str = "QHY268M",
    gain: int | None = 0,
    offset: int | None = 30,
    width: int = 32,
    height: int = 24,
    binning: int = 1,
    cfa: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(
        0, 65535, size=(height, width), dtype=np.uint16
    )
    hdu = fits.PrimaryHDU(pixels)
    hdu.header["IMAGETYP"] = role
    hdu.header["OBJECT"] = target
    hdu.header["FILTER"] = filter_name
    hdu.header["EXPTIME"] = 300.0 if "Light" in role else 0.75
    hdu.header["INSTRUME"] = camera
    hdu.header["XBINNING"] = binning
    hdu.header["YBINNING"] = binning
    if gain is not None:
        hdu.header["GAIN"] = gain
    if offset is not None:
        hdu.header["OFFSET"] = offset
    if cfa is not None:
        hdu.header["BAYERPAT"] = cfa
    hdu.writeto(path)
    # Fixed nanosecond mtimes make copy2 and deterministic-plan assertions
    # independent of test execution speed.
    timestamp = 1_780_000_000_123_456_789 + seed
    os.utime(path, ns=(timestamp, timestamp))


def _result(path: Path, decision: Decision) -> FrameResult:
    metadata = probe_frame_metadata(path)
    result = FrameResult(
        path=str(path.resolve()),
        group_id="group-test",
        reference_path=str(path.resolve()),
        decision=decision,
        confidence=(
            Confidence.HIGH if decision is Decision.KEEP else Confidence.MEDIUM
        ),
        reasons=[],
        warnings=[],
        registration=RegistrationMetrics(ok=True, matched_stars=40, match_fraction=0.9),
        features=FrameFeatures(),
        metadata=metadata,
        star_count=100,
    )
    # FrameResult remains backward-compatible, so the preparation layer also
    # supports the identity as an attached analysis artifact.
    result.identity = compute_file_identity(path)
    disposition = (
        GateDisposition.PASS
        if decision is Decision.KEEP
        else GateDisposition.REVIEW
        if decision is Decision.REVIEW
        else GateDisposition.HARD_FAIL
    )
    policy = GatePolicy()
    evidence = []
    if disposition is GateDisposition.REVIEW:
        from lightframeqc.models import EvidenceFamily, EvidenceSeverity, QualityEvidence

        evidence = [
            QualityEvidence(
                code="TEST.REVIEW",
                family=EvidenceFamily.PROVENANCE,
                severity=EvidenceSeverity.REVIEW,
                message="test review",
            )
        ]
    elif disposition is GateDisposition.HARD_FAIL:
        from lightframeqc.models import EvidenceFamily, EvidenceSeverity, QualityEvidence

        evidence = [
            QualityEvidence(
                code="TEST.HARD_FAIL",
                family=EvidenceFamily.PROVENANCE,
                severity=EvidenceSeverity.HARD_FAIL,
                message="test hard fail",
            )
        ]
    result.quality_gate = QualityGateResult(
        disposition=disposition,
        evidence=evidence,
        summary="prepare test gate",
        version=f"{policy.version}@{policy.canonical_digest()}",
        policy_digest=policy.canonical_digest(),
        policy=policy.serializable(),
    )
    return result


def _flat(tmp_path: Path, filter_name: str = "R", *, seed: int = 900) -> Path:
    path = tmp_path / "flats" / f"masterFlat_{filter_name}.fits"
    _write_frame(
        path,
        role="Master Flat",
        target="FlatWizard",
        filter_name=filter_name,
        seed=seed,
    )
    return path


def test_prepare_requires_qc_source_identity(tmp_path: Path) -> None:
    light = tmp_path / "identity-missing.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=899,
    )
    result = _result(light, Decision.KEEP)
    result.identity = None

    with pytest.raises(PreparePlanError, match="QC_SOURCE_IDENTITY_MISSING"):
        build_prepare_plan([result], [_flat(tmp_path)], tmp_path / "prepared")


def test_prepare_requires_supported_quality_gate_pass(tmp_path: Path) -> None:
    light = tmp_path / "gate-missing.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=895,
    )
    result = _result(light, Decision.KEEP)
    result.quality_gate = None

    with pytest.raises(PreparePlanError, match="QUALITY_GATE_MISSING"):
        build_prepare_plan([result], [_flat(tmp_path)], tmp_path / "prepared")


def test_prepare_plan_round_trip_loader_is_strict(tmp_path: Path) -> None:
    light = tmp_path / "round-trip.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=894,
    )
    plan = build_prepare_plan(
        [_result(light, Decision.KEEP)],
        [_flat(tmp_path)],
        tmp_path / "prepared",
    )
    path = tmp_path / "prepare-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    assert load_prepare_plan(path) == plan
    path.write_text('{"planId":"x","planId":"y"}', encoding="utf-8")
    with pytest.raises(PrepareApplyError, match="duplicate manifest JSON key"):
        load_prepare_plan(path)


def test_resigned_plan_cannot_forge_review_approval_or_drop_matched_flat(
    tmp_path: Path,
) -> None:
    light = tmp_path / "review.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=893,
    )
    flat = _flat(tmp_path, seed=892)
    review_plan = build_prepare_plan(
        [_result(light, Decision.REVIEW)], [flat], tmp_path / "review-out"
    )
    entry = next(item for item in review_plan["entries"] if item["role"] == "LIGHT")
    entry["action"] = "COPY_LIGHT"
    entry["destinationRelativePath"] = "forged/LIGHT/R/review.fits"
    entry["destinationSha256"] = entry["sourceIdentity"]["sha256"]
    entry["adjudication"] = {"action": "APPROVE"}
    review_plan["summary"]["actionCounts"] = {"COPY_LIGHT": 1}
    forged = _resign_plan(review_plan)
    with pytest.raises(PrepareApplyError, match="invalid adjudication"):
        apply_prepare_plan(forged, tmp_path / "review-out")

    pass_plan = build_prepare_plan(
        [_result(light, Decision.KEEP)], [flat], tmp_path / "pass-out"
    )
    pass_plan["entries"] = [
        item for item in pass_plan["entries"] if item["role"] == "LIGHT"
    ]
    pass_plan["summary"]["actionCounts"] = {"COPY_LIGHT": 1}
    pass_plan["summary"]["uniqueMasterFlatContentCount"] = 0
    forged_no_flat = _resign_plan(pass_plan)
    with pytest.raises(PrepareApplyError, match="missing matched master flat"):
        apply_prepare_plan(forged_no_flat, tmp_path / "pass-out")


def test_prepare_rejects_path_only_light_role(tmp_path: Path) -> None:
    light = tmp_path / "LIGHT" / "path-hint-only.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=898,
    )
    with fits.open(light, mode="update") as hdul:
        del hdul[0].header["IMAGETYP"]
        hdul.flush()
    result = _result(light, Decision.KEEP)
    assert result.metadata.role is FrameRole.LIGHT
    assert result.metadata.role_evidence == ["PATH=LIGHT"]

    with pytest.raises(PreparePlanError, match="NON_AUTHORITATIVE_FRAME_ROLE"):
        build_prepare_plan([result], [_flat(tmp_path)], tmp_path / "prepared")


def test_prepare_rejects_unknown_cfa_or_binning_profile(tmp_path: Path) -> None:
    flat = _flat(tmp_path)
    unknown_cfa = tmp_path / "unknown-cfa.fits"
    _write_frame(
        unknown_cfa,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        camera="Generic Camera",
        seed=897,
    )
    with pytest.raises(PreparePlanError, match="UNKNOWN_CFA_PROFILE"):
        build_prepare_plan(
            [_result(unknown_cfa, Decision.KEEP)],
            [flat],
            tmp_path / "unknown-cfa-out",
        )

    unknown_binning = tmp_path / "unknown-binning.fits"
    _write_frame(
        unknown_binning,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=896,
    )
    with fits.open(unknown_binning, mode="update") as hdul:
        del hdul[0].header["XBINNING"]
        del hdul[0].header["YBINNING"]
        hdul.flush()
    with pytest.raises(PreparePlanError, match="UNKNOWN_BINNING_PROFILE"):
        build_prepare_plan(
            [_result(unknown_binning, Decision.KEEP)],
            [flat],
            tmp_path / "unknown-binning-out",
        )


def _xisf_flat(tmp_path: Path, filter_name: str = "R") -> Path:
    path = tmp_path / "flats" / f"masterFlat_{filter_name}.xisf"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "imageType": "MasterFlat",
        "FITSKeywords": {
            "IMAGETYP": [{"value": "'Master Flat'", "comment": ""}],
            "FILTER": [{"value": f"'{filter_name}'", "comment": ""}],
            "INSTRUME": [{"value": "'QHY268M'", "comment": ""}],
            "XBINNING": [{"value": "1", "comment": ""}],
            "YBINNING": [{"value": "1", "comment": ""}],
            "GAIN": [{"value": "0", "comment": ""}],
            "OFFSET": [{"value": "30", "comment": ""}],
            "BAYERPAT": [{"value": "'NONE'", "comment": ""}],
        },
    }
    XISF.write(
        str(path),
        np.linspace(0.8, 1.2, 32 * 24, dtype=np.float32).reshape(24, 32, 1),
        image_metadata=metadata,
        codec=None,
    )
    timestamp = 1_780_000_000_987_654_321
    os.utime(path, ns=(timestamp, timestamp))
    return path


def _add_xisf_rejection_images(path: Path) -> None:
    raw = path.read_bytes()
    header_length = struct.unpack("<I", raw[8:12])[0]
    root = ET.fromstring(raw[16 : 16 + header_length])
    image = next(element for element in root.iter() if element.tag.endswith("Image"))
    parent = next(
        element for element in root.iter() if image in list(element)
    )
    for suffix, image_type in (("low", "RejectionMapLow"), ("high", "RejectionMapHigh")):
        clone = ET.fromstring(ET.tostring(image, encoding="utf-8"))
        clone.set("id", "rejection-" + suffix)
        clone.set("imageType", image_type)
        parent.append(clone)
    header = ET.tostring(root, encoding="utf-8")
    path.write_bytes(
        raw[:8]
        + struct.pack("<I", len(header))
        + raw[12:16]
        + header
        + raw[16 + header_length :]
    )


def _copy_entries(plan: dict) -> list[dict]:
    return [entry for entry in plan["entries"] if entry["action"].startswith("COPY_")]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resign_plan(plan: dict) -> dict:
    value = deepcopy(plan)
    value.pop("planId", None)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value["planId"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return value


def test_build_plan_is_order_independent_and_separates_targets_and_filters(
    tmp_path: Path,
) -> None:
    lights: list[FrameResult] = []
    flats = [_flat(tmp_path, "R", seed=900), _flat(tmp_path, "G", seed=901)]
    for target_index, target in enumerate(("NGC 7000", "M31")):
        for filter_index, filter_name in enumerate(("R", "G")):
            for frame_index in range(2):
                path = (
                    tmp_path
                    / "lights"
                    / target
                    / filter_name
                    / f"light-{frame_index}.fits"
                )
                _write_frame(
                    path,
                    role="Light Frame",
                    target=target,
                    filter_name=filter_name,
                    seed=100 * target_index + 10 * filter_index + frame_index,
                )
                lights.append(_result(path, Decision.KEEP))

    destination = tmp_path / "prepared"
    first = build_prepare_plan(lights, flats, destination)
    second = build_prepare_plan(reversed(lights), reversed(flats), destination)

    assert first == second
    assert first["planId"].startswith("sha256:")
    copy_entries = _copy_entries(first)
    assert sum(entry["action"] == "COPY_LIGHT" for entry in copy_entries) == 8
    assert sum(entry["action"] == "COPY_MASTER_FLAT" for entry in copy_entries) == 4
    light_paths = [
        PurePosixPath(entry["destinationRelativePath"])
        for entry in copy_entries
        if entry["action"] == "COPY_LIGHT"
    ]
    assert len({path.parts[0] for path in light_paths}) == 2
    assert {path.parts[1] for path in light_paths} == {"LIGHT"}
    assert all(".." not in path.parts for path in light_paths)


def test_only_keep_copies_without_adjudication_and_hash_bound_overrides_apply(
    tmp_path: Path,
) -> None:
    decisions = (
        Decision.KEEP,
        Decision.REVIEW,
        Decision.REJECT_CLOUD,
        Decision.UNASSESSABLE,
    )
    results: list[FrameResult] = []
    for index, decision in enumerate(decisions):
        path = tmp_path / "lights" / f"frame-{index}.fits"
        _write_frame(
            path,
            role="Light Frame",
            target="NGC 7000",
            filter_name="R",
            seed=index + 1,
        )
        results.append(_result(path, decision))
    flat = _flat(tmp_path)

    automatic = build_prepare_plan(results, [flat], tmp_path / "automatic")
    assert sum(entry["action"] == "COPY_LIGHT" for entry in automatic["entries"]) == 1
    assert sum(entry["action"] == "MANIFEST_ONLY" for entry in automatic["entries"]) == 3

    keep_identity = compute_file_identity(results[0].path)
    review_identity = compute_file_identity(results[1].path)
    adjudication = [
        AdjudicationRecord(
            path=results[0].path,
            sha256=keep_identity.sha256,
            action=AdjudicationAction.REJECT,
            reason="manual cloud",
        ),
        AdjudicationRecord(
            path=results[1].path,
            sha256=review_identity.sha256,
            action=AdjudicationAction.APPROVE,
            reason="false positive",
        ),
    ]
    overridden = build_prepare_plan(
        results, [flat], tmp_path / "overridden", adjudication
    )
    copied = [entry for entry in overridden["entries"] if entry["action"] == "COPY_LIGHT"]
    assert [entry["sourcePath"] for entry in copied] == [results[1].path]
    excluded = {
        entry["sourcePath"]: entry
        for entry in overridden["entries"]
        if entry["action"] == "MANIFEST_ONLY"
    }
    assert excluded[results[0].path]["adjudication"]["action"] == "REJECT"
    assert copied[0]["adjudication"]["action"] == "APPROVE"


def test_stale_or_unknown_adjudication_fails_closed(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=1,
    )
    result = _result(light, Decision.KEEP)
    flat = _flat(tmp_path)

    with pytest.raises(PreparePlanError, match="STALE_ADJUDICATION"):
        build_prepare_plan(
            [result],
            [flat],
            tmp_path / "out",
            [{"path": str(light), "sha256": "0" * 64, "action": "REJECT"}],
        )
    with pytest.raises(PreparePlanError, match="UNKNOWN_ADJUDICATION_TARGET"):
        build_prepare_plan(
            [result],
            [flat],
            tmp_path / "out",
            [
                {
                    "path": str(tmp_path / "other.fits"),
                    "sha256": "1" * 64,
                    "action": "REJECT",
                }
            ],
        )


def test_single_review_can_be_approved_but_automatic_reject_cannot(
    tmp_path: Path,
) -> None:
    light = tmp_path / "review.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=51,
    )
    flat = _flat(tmp_path, seed=52)
    identity = compute_file_identity(light)
    approval = AdjudicationRecord(
        path=str(light),
        sha256=identity.sha256,
        action=AdjudicationAction.APPROVE,
        reason="manual clear preview",
    )

    approved = build_prepare_plan(
        [_result(light, Decision.REVIEW)],
        [flat],
        tmp_path / "approved",
        [approval],
    )
    assert sum(entry["action"] == "COPY_LIGHT" for entry in approved["entries"]) == 1

    with pytest.raises(PreparePlanError, match="UNSAFE_ADJUDICATION_APPROVAL"):
        build_prepare_plan(
            [_result(light, Decision.REJECT_CLOUD)],
            [flat],
            tmp_path / "unsafe",
            [approval],
        )


def test_adjudication_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    document = tmp_path / "duplicate-adjudication.json"
    document.write_text(
        '{"schemaVersion":1,"records":[{'
        '"path":"/tmp/frame.fits",'
        '"sha256":"' + "0" * 64 + '",'
        '"action":"REJECT","action":"APPROVE"}]}',
        encoding="utf-8",
    )

    with pytest.raises(AdjudicationError, match="duplicate adjudication JSON key"):
        parse_adjudication(document)


def test_missing_conflicting_and_ambiguous_master_flat_fail(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=1,
    )
    result = _result(light, Decision.KEEP)

    with pytest.raises(PreparePlanError, match="MISSING_MASTER_FLAT"):
        build_prepare_plan([result], [], tmp_path / "missing")

    wrong_camera = tmp_path / "wrong" / "masterFlat_R.fits"
    _write_frame(
        wrong_camera,
        role="Master Flat",
        target="FlatWizard",
        filter_name="R",
        camera="OTHER",
        cfa="NONE",
        seed=20,
    )
    with pytest.raises(PreparePlanError, match="MISSING_MASTER_FLAT"):
        build_prepare_plan([result], [wrong_camera], tmp_path / "wrong-out")

    first = _flat(tmp_path, seed=30)
    second = tmp_path / "other" / "masterFlat_R.fits"
    _write_frame(
        second,
        role="Master Flat",
        target="FlatWizard",
        filter_name="R",
        seed=31,
    )
    with pytest.raises(PreparePlanError, match="AMBIGUOUS_MASTER_FLAT"):
        build_prepare_plan([result], [first, second], tmp_path / "ambiguous")


def test_byte_identical_flat_alias_and_light_duplicate_are_deduplicated(
    tmp_path: Path,
) -> None:
    first_light = tmp_path / "night-a" / "one.fits"
    second_light = tmp_path / "night-b" / "two.fits"
    _write_frame(
        first_light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=4,
    )
    second_light.parent.mkdir(parents=True)
    shutil.copy2(first_light, second_light)

    first_flat = _flat(tmp_path, seed=50)
    second_flat = tmp_path / "flat-alias" / "same-master.fits"
    second_flat.parent.mkdir(parents=True)
    shutil.copy2(first_flat, second_flat)

    plan = build_prepare_plan(
        [_result(first_light, Decision.KEEP), _result(second_light, Decision.KEEP)],
        [first_flat, second_flat],
        tmp_path / "prepared",
    )

    assert sum(entry["action"] == "COPY_LIGHT" for entry in plan["entries"]) == 1
    duplicates = [entry for entry in plan["entries"] if entry["action"] == "DUPLICATE_LIGHT"]
    assert len(duplicates) == 1
    assert duplicates[0]["duplicateOf"]
    flats = [entry for entry in plan["entries"] if entry["action"] == "COPY_MASTER_FLAT"]
    assert len(flats) == 1
    canonical = min(str(first_flat.resolve()), str(second_flat.resolve()))
    alias = max(str(first_flat.resolve()), str(second_flat.resolve()))
    assert flats[0]["sourcePath"] == canonical
    assert flats[0]["sourceAliases"] == [alias]


def test_same_destination_basename_with_different_content_fails(tmp_path: Path) -> None:
    results = []
    for index, directory in enumerate(("night-a", "night-b")):
        light = tmp_path / directory / "same-name.fits"
        _write_frame(
            light,
            role="Light Frame",
            target="NGC 7000",
            filter_name="R",
            seed=60 + index,
        )
        results.append(_result(light, Decision.KEEP))

    with pytest.raises(PreparePlanError, match="DESTINATION_COLLISION"):
        build_prepare_plan(results, [_flat(tmp_path)], tmp_path / "prepared")


def test_windows_reserved_filter_is_hash_slugged_on_every_host(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="CON",
        seed=67,
    )
    plan = build_prepare_plan(
        [_result(light, Decision.KEEP)],
        [_flat(tmp_path, "CON", seed=68)],
        tmp_path / "prepared",
    )
    entry = next(item for item in plan["entries"] if item["action"] == "COPY_LIGHT")
    relative = PurePosixPath(entry["destinationRelativePath"])
    assert entry["filterSlug"] != "CON"
    assert relative.parts[2] == entry["filterSlug"]


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot create a CON source path")
def test_windows_reserved_source_filename_is_portably_encoded(tmp_path: Path) -> None:
    light = tmp_path / "CON.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=69,
    )
    plan = build_prepare_plan(
        [_result(light, Decision.KEEP)],
        [_flat(tmp_path, seed=691)],
        tmp_path / "prepared",
    )
    entry = next(item for item in plan["entries"] if item["action"] == "COPY_LIGHT")
    filename = PurePosixPath(entry["destinationRelativePath"]).name
    assert filename != "CON.fits"
    assert filename.endswith(".fits")


def test_resigned_plan_rejects_windows_trailing_dot_component(tmp_path: Path) -> None:
    light = tmp_path / "portable.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=692,
    )
    plan = build_prepare_plan(
        [_result(light, Decision.KEEP)],
        [_flat(tmp_path, seed=693)],
        tmp_path / "prepared",
    )
    entry = next(item for item in plan["entries"] if item["action"] == "COPY_LIGHT")
    entry["destinationRelativePath"] = "target/LIGHT/R./portable.fits"
    forged = _resign_plan(plan)
    with pytest.raises(PreparePlanError, match="UNSAFE_DESTINATION_PATH"):
        apply_prepare_plan(forged, tmp_path / "prepared")


def test_unknown_target_and_raw_flat_fail_without_creating_destination(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown.fits"
    _write_frame(
        unknown,
        role="Light Frame",
        target="UNKNOWN",
        filter_name="R",
        seed=70,
    )
    destination = tmp_path / "prepared"
    master_flat = _flat(tmp_path)
    with pytest.raises(PreparePlanError, match="UNKNOWN_PATH_COMPONENT"):
        build_prepare_plan([_result(unknown, Decision.KEEP)], [master_flat], destination)
    assert not destination.exists()

    raw_flat = tmp_path / "raw-flat.fits"
    _write_frame(
        raw_flat,
        role="Flat Frame",
        target="FlatWizard",
        filter_name="R",
        seed=71,
    )
    good = tmp_path / "good.fits"
    _write_frame(
        good,
        role="Light Frame",
        target="../../NGC 7000",
        filter_name="R",
        seed=72,
    )
    # Malicious-looking metadata is encoded into safe, single components.
    safe_plan = build_prepare_plan(
        [_result(good, Decision.KEEP)], [master_flat], destination
    )
    relative = next(
        entry["destinationRelativePath"]
        for entry in safe_plan["entries"]
        if entry["action"] == "COPY_LIGHT"
    )
    assert ".." not in PurePosixPath(relative).parts
    assert not PurePosixPath(relative).is_absolute()

    with pytest.raises(PreparePlanError, match="RAW_FLAT_UNSUPPORTED"):
        build_prepare_plan([_result(good, Decision.KEEP)], [raw_flat], destination)


def test_apply_is_atomic_byte_exact_copy2_and_idempotent(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=80,
    )
    flat = _flat(tmp_path, seed=81)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)

    applied = apply_prepare_plan(plan, destination)
    assert applied["status"] == "APPLIED"
    assert (destination / MANIFEST_NAME).is_file()
    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["planId"] == plan["planId"]

    for entry in _copy_entries(plan):
        copied = destination.joinpath(*PurePosixPath(entry["destinationRelativePath"]).parts)
        source = Path(entry["sourcePath"])
        assert copied.is_file() and not copied.is_symlink()
        assert os.stat(copied).st_ino != os.stat(source).st_ino
        assert _sha256(copied) == _sha256(source) == entry["sourceIdentity"]["sha256"]
        assert copied.stat().st_mtime_ns == source.stat().st_mtime_ns

    before = {
        path.relative_to(destination).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in destination.rglob("*")
        if path.is_file()
    }
    repeated = apply_prepare_plan(plan, destination)
    after = {
        path.relative_to(destination).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert repeated["status"] == "ALREADY_COMPLETE"
    assert after == before


def test_streaming_apply_minimizes_multimegabyte_content_reads_and_reuses_flat_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New plans read copied bytes once each and hash each target only once."""

    results = []
    source_paths: set[str] = set()
    for filename, target, seed, decision in (
        ("target-a.fits", "Target A", 180, Decision.KEEP),
        ("target-b.fits", "Target B", 181, Decision.KEEP),
        ("excluded.fits", "Target A", 182, Decision.REVIEW),
    ):
        path = tmp_path / filename
        _write_frame(
            path,
            role="Light Frame",
            target=target,
            filter_name="R",
            seed=seed,
            width=2048,
            height=1024,
        )
        source_paths.add(str(path.resolve()))
        results.append(_result(path, decision))

    flat = tmp_path / "flats" / "masterFlat_R.fits"
    _write_frame(
        flat,
        role="Master Flat",
        target="FlatWizard",
        filter_name="R",
        seed=183,
        width=2048,
        height=1024,
    )
    source_paths.add(str(flat.resolve()))
    destination = tmp_path / "prepared"
    plan = build_prepare_plan(results, [flat], destination)

    from lightframeqc import prepare as prepare_module

    assert plan["policy"]["applyVerification"] == prepare_module.APPLY_VERIFICATION_POLICY
    copy_entries = _copy_entries(plan)
    assert len(copy_entries) == 4  # two lights and one flat per target
    assert sum(entry["action"] == "MANIFEST_ONLY" for entry in plan["entries"]) == 1

    real_identity = prepare_module._compute_file_identity
    real_stream_copy = prepare_module._stream_copy
    identity_reads: list[tuple[str, int]] = []
    stream_reads: list[tuple[int, bool]] = []

    def count_identity(path: str | os.PathLike[str]):
        resolved = str(Path(path).resolve(strict=True))
        identity_reads.append((resolved, Path(path).stat().st_size))
        return real_identity(path)

    def count_stream(source, target, digest):
        copied = real_stream_copy(source, target, digest)
        stream_reads.append((copied, digest is not None))
        return copied

    monkeypatch.setattr(prepare_module, "_compute_file_identity", count_identity)
    monkeypatch.setattr(prepare_module, "_stream_copy", count_stream)

    result = apply_prepare_plan(plan, destination)

    assert result["status"] == "APPLIED"
    assert not any(path in source_paths for path, _ in identity_reads)
    assert len(identity_reads) == len(copy_entries)
    assert len(stream_reads) == len(copy_entries)
    # Two distinct lights plus the shared flat need a source digest.  The
    # second flat copy reuses the already identity-bound source validation.
    assert [hashed for _, hashed in stream_reads].count(True) == 3
    assert [hashed for _, hashed in stream_reads].count(False) == 1
    expected_bytes = sum(entry["sourceIdentity"]["sizeBytes"] for entry in copy_entries)
    assert sum(size for _, size in identity_reads) == expected_bytes
    assert sum(size for size, _ in stream_reads) == expected_bytes

    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["policy"]["applyVerification"]["manifestOnlySource"] == (
        "stat-only-no-current-content-hash"
    )


def test_legacy_schema_one_plan_keeps_all_source_content_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    light = tmp_path / "light.fits"
    excluded = tmp_path / "excluded.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=184,
    )
    _write_frame(
        excluded,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=185,
    )
    flat = _flat(tmp_path, seed=186)
    destination = tmp_path / "legacy"
    plan = build_prepare_plan(
        [_result(light, Decision.KEEP), _result(excluded, Decision.REVIEW)],
        [flat],
        destination,
    )
    plan["policy"].pop("applyVerification")
    legacy_plan = _resign_plan(plan)

    from lightframeqc import prepare as prepare_module

    real_identity = prepare_module._compute_file_identity
    source_paths = {str(path.resolve()) for path in (light, excluded, flat)}
    hashed_sources: list[str] = []

    def count_identity(path: str | os.PathLike[str]):
        resolved = str(Path(path).resolve(strict=True))
        if resolved in source_paths:
            hashed_sources.append(resolved)
        return real_identity(path)

    monkeypatch.setattr(prepare_module, "_compute_file_identity", count_identity)
    apply_prepare_plan(legacy_plan, destination)

    assert set(hashed_sources) == source_paths
    assert len(hashed_sources) == len(source_paths)


def test_xisf_master_flat_matches_and_preserves_its_extension(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=85,
        cfa="NONE",
    )
    flat = _xisf_flat(tmp_path)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)

    flat_entry = next(
        entry for entry in plan["entries"] if entry["action"] == "COPY_MASTER_FLAT"
    )
    assert flat_entry["destinationRelativePath"].endswith(".xisf")
    apply_prepare_plan(plan, destination)
    copied = destination.joinpath(
        *PurePosixPath(flat_entry["destinationRelativePath"]).parts
    )
    assert copied.suffix == ".xisf"
    assert _sha256(copied) == _sha256(flat)


def test_multi_image_wbpp_xisf_master_flat_is_copied_as_whole_container(
    tmp_path: Path,
) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=86,
        cfa="NONE",
    )
    flat = _xisf_flat(tmp_path)
    _add_xisf_rejection_images(flat)
    before = _sha256(flat)
    assert probe_frame_metadata(flat).image_count == 3
    destination = tmp_path / "prepared"

    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)
    apply_prepare_plan(plan, destination)
    copied = next(destination.glob("*/FLAT/R/*.xisf"))

    assert _sha256(copied) == before
    assert probe_frame_metadata(copied).image_count == 3


def test_source_change_or_injected_copy_failure_leaves_no_final_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=90,
    )
    flat = _flat(tmp_path, seed=91)
    changed_destination = tmp_path / "changed"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], changed_destination)
    with light.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(PrepareApplyError, match="SOURCE_CHANGED"):
        apply_prepare_plan(plan, changed_destination)
    assert not changed_destination.exists()
    assert not list(tmp_path.glob(".changed.staging-*"))
    assert not (tmp_path / ".changed.prepare.lock").exists()

    stable_light = tmp_path / "stable.fits"
    _write_frame(
        stable_light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=92,
    )
    failed_destination = tmp_path / "failed"
    failed_plan = build_prepare_plan(
        [_result(stable_light, Decision.KEEP)], [flat], failed_destination
    )
    from lightframeqc import prepare as prepare_module

    real_stream_copy = prepare_module._stream_copy
    calls = 0

    def fail_second_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return real_stream_copy(*args, **kwargs)

    monkeypatch.setattr(prepare_module, "_stream_copy", fail_second_copy)
    with pytest.raises(PrepareApplyError, match="PREPARE_TRANSACTION_FAILED"):
        apply_prepare_plan(failed_plan, failed_destination)
    assert not failed_destination.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
    assert not (tmp_path / ".failed.prepare.lock").exists()


def test_existing_tree_from_other_plan_or_with_drift_is_never_overwritten(
    tmp_path: Path,
) -> None:
    first_light = tmp_path / "first.fits"
    _write_frame(
        first_light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=100,
    )
    flat = _flat(tmp_path, seed=101)
    destination = tmp_path / "prepared"
    first_plan = build_prepare_plan(
        [_result(first_light, Decision.KEEP)], [flat], destination
    )
    apply_prepare_plan(first_plan, destination)

    second_light = tmp_path / "second.fits"
    _write_frame(
        second_light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=102,
    )
    second_plan = build_prepare_plan(
        [_result(second_light, Decision.KEEP)], [flat], destination
    )
    with pytest.raises(PrepareApplyError, match="DESTINATION_PLAN_CONFLICT"):
        apply_prepare_plan(second_plan, destination)

    copied = next(
        destination.joinpath(*PurePosixPath(entry["destinationRelativePath"]).parts)
        for entry in first_plan["entries"]
        if entry["action"] == "COPY_LIGHT"
    )
    with copied.open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(PrepareApplyError, match="DESTINATION_DRIFT"):
        apply_prepare_plan(first_plan, destination)


def test_existing_tree_rejects_source_hardlink_alias(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=109,
    )
    flat = _flat(tmp_path, seed=108)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)
    apply_prepare_plan(plan, destination)
    copied = next(
        destination.joinpath(*PurePosixPath(entry["destinationRelativePath"]).parts)
        for entry in plan["entries"]
        if entry["action"] == "COPY_LIGHT"
    )
    copied.unlink()
    os.link(light, copied)

    with pytest.raises(PrepareApplyError, match="hardlink"):
        apply_prepare_plan(plan, destination)


def test_concurrent_lock_is_not_removed_by_a_losing_apply(tmp_path: Path) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=110,
    )
    flat = _flat(tmp_path, seed=111)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)
    lock = tmp_path / ".prepared.prepare.lock"
    lock.write_text("other-process\n", encoding="utf-8")

    with pytest.raises(PrepareApplyError, match="DESTINATION_LOCKED"):
        apply_prepare_plan(plan, destination)

    assert lock.read_text(encoding="utf-8") == "other-process\n"
    assert not destination.exists()


def test_atomic_publish_does_not_replace_external_racing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=120,
    )
    flat = _flat(tmp_path, seed=121)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)
    from lightframeqc import prepare as prepare_module

    real_rename = prepare_module._rename_no_replace

    def inject_race(source: Path, target: Path) -> None:
        target.mkdir()
        (target / "external-owner.txt").write_text("do not replace", encoding="utf-8")
        real_rename(source, target)

    monkeypatch.setattr(prepare_module, "_rename_no_replace", inject_race)
    with pytest.raises(PrepareApplyError, match="DESTINATION_PLAN_CONFLICT"):
        apply_prepare_plan(plan, destination)

    assert (destination / "external-owner.txt").read_text(encoding="utf-8") == "do not replace"
    assert not (destination / MANIFEST_NAME).exists()
    assert not list(tmp_path.glob(".prepared.staging-*"))


def test_post_commit_failure_is_reported_as_commit_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=122,
    )
    flat = _flat(tmp_path, seed=123)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)
    from lightframeqc import prepare as prepare_module

    real_verify = prepare_module._verify_published_tree
    calls = 0

    def fail_final_verify(path: Path, value: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = real_verify(path, value, **kwargs)
        if calls == 2:
            raise OSError("injected post-commit failure")
        return result

    monkeypatch.setattr(prepare_module, "_verify_published_tree", fail_final_verify)
    with pytest.raises(PrepareApplyError, match="COMMIT_UNCERTAIN"):
        apply_prepare_plan(plan, destination)

    assert destination.is_dir()
    assert (destination / MANIFEST_NAME).is_file()
    assert calls == 2


def test_post_publish_stat_binding_detects_changed_target_without_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    light = tmp_path / "light.fits"
    _write_frame(
        light,
        role="Light Frame",
        target="NGC 7000",
        filter_name="R",
        seed=124,
    )
    flat = _flat(tmp_path, seed=125)
    destination = tmp_path / "prepared"
    plan = build_prepare_plan([_result(light, Decision.KEEP)], [flat], destination)
    from lightframeqc import prepare as prepare_module

    real_rename = prepare_module._rename_no_replace

    def publish_then_change(source: Path, target: Path) -> None:
        real_rename(source, target)
        copied_light = next(target.glob("*/LIGHT/R/*.fits"))
        with copied_light.open("ab") as stream:
            stream.write(b"post-publish drift")

    monkeypatch.setattr(prepare_module, "_rename_no_replace", publish_then_change)
    with pytest.raises(PrepareApplyError, match="COMMIT_UNCERTAIN"):
        apply_prepare_plan(plan, destination)

    assert destination.is_dir()
    assert (destination / MANIFEST_NAME).is_file()
