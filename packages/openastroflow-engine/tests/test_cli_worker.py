from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.backends import (
    BackendDescriptor,
    BackendRegistry,
    DeviceKind,
    StageKind,
)
from openastroflow_engine.cli import main
from openastroflow_engine.controller import controller_plan_envelope
from openastroflow_engine.e2e import (
    E2EResult,
    E2EState,
    ProgressEvent,
    ProgressStage,
)
from openastroflow_engine.inventory import inventory_project
from openastroflow_engine.hardware import detect_hardware
from openastroflow_engine.planning import RuntimeStageBackend
from openastroflow_engine.performance_profile import GIB, select_execution_tuning
from openastroflow_engine.protocol_v1 import ProtocolCursor, WorkerEnvelope, decode_ndjson_line
from openastroflow_engine.solver import SolutionKind, SolverResult, SolverStatus
from openastroflow_engine.worker import (
    _progress_payload,
    _supported_hardware_profiles,
    _validate_stage_evidence,
    run_worker,
    worker_capabilities,
)
from openastroflow_engine.project_e2e import ProjectE2EResult, ProjectProgressEvent, SciencePanel
from openastroflow_engine.runtime import pixel_backend_for_hardware_profile


def test_project_progress_keeps_strict_v1_payload_and_reserves_root_success() -> None:
    plan = SimpleNamespace(request_id="request-1", recipe=SimpleNamespace(stages=[]))
    event = ProjectProgressEvent(ProgressStage.PUBLISH, "completed", 1, 1, "panel complete",
        overall_fraction=0.23, scope="panel", panel=SciencePanel("Cartwheel", "cartwheel", "B", "b", ("light",)),
        panel_index=1, panel_count=4)
    payload = _progress_payload(plan, "run-1", event)
    assert payload == {"requestId": "request-1", "runId": "run-1", "state": "finalizing", "fraction": 0.23,
        "message": "panel 1/4 Cartwheel/B: panel complete"}
    final_phase = _progress_payload(plan, "run-1", replace(event, overall_fraction=1, scope="project", panel=None))
    assert final_phase["fraction"] == 0.99
    assert final_phase["state"] != "succeeded"
    failure = _progress_payload(plan, "run-1", replace(event, stage=ProgressStage.FAILED))
    assert failure["state"] == "failed"


class ReadySolver:
    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="astrometry-net",
            stage=StageKind.SOLVER,
            display_name="Synthetic astrometry.net",
            version="test",
            available=True,
            execution_ready=True,
            devices=(DeviceKind.CPU,),
            capabilities=("catalog-correspondence-quality-v1",),
        )

    def validate_options(self, options: dict[str, object]) -> tuple[str, ...]:
        return ()

    def solve(self, request: object) -> SolverResult:
        return SolverResult(
            backend_id="astrometry-net",
            status=SolverStatus.FAILED,
            solution_kind=SolutionKind.NONE,
            backend_confirmed=False,
            error="the synthetic E2E runner owns this test boundary",
        )


def _registry() -> BackendRegistry:
    stages = (
        ("light-frame-qc", StageKind.QUALITY_GATE),
        ("native-calibration", StageKind.CALIBRATION),
        ("native-registration", StageKind.REGISTRATION),
        ("native-integration", StageKind.INTEGRATION),
    )
    backends = [
        RuntimeStageBackend(
            BackendDescriptor(
                backend_id=backend_id,
                stage=stage,
                display_name=backend_id,
                version="test",
                available=True,
                execution_ready=True,
                devices=(DeviceKind.CPU,),
            )
        )
        for backend_id, stage in stages
    ]
    return BackendRegistry([*backends, ReadySolver()])


def _inventory(nina_project: Path):
    return inventory_project(
        [
            nina_project / "LIGHT",
            nina_project / "FLAT",
            nina_project / "DARK",
            nina_project / "BIAS",
        ],
        name="M16 synthetic",
    )


def _controller_handshake(session_id: str = "session-test") -> dict[str, object]:
    return {
        "protocolVersion": 1,
        "sessionId": session_id,
        "sequence": 0,
        "sentAtUnixMs": 1,
        "type": "handshake",
        "payload": {
            "role": "controller",
            "implementation": "test-controller",
            "implementationVersion": "1",
            "supportedProtocolVersions": [1],
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_e2e(request, *, solver_backends, progress) -> E2EResult:
    assert solver_backends and solver_backends[0].descriptor.backend_id == "astrometry-net"
    assert request.pipeline_parameters.ordinary_integration_backend == "m3-pro-tuned"
    output = Path(request.output_directory)
    product = output / "products" / "R" / "master_light_R_wcs.fits"
    product.parent.mkdir(parents=True)
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 8.5
    header["CRPIX2"] = 8.5
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 20.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    header["RADESYS"] = "ICRS"
    fits.writeto(product, np.ones((16, 16), dtype=np.float32), header)

    evidence = {
        "qc/manifest.json": {
            "stage": "quality-control",
            "counts": {"PASS": 1, "REVIEW": 0, "HARD_FAIL": 0},
        },
        "receipts/registration-calibration.json": {
            "stage": "registration-calibration-masters",
            "artifacts": [],
        },
        "receipts/registration.json": {
            "stage": "registration",
            "transforms": [],
        },
        "receipts/pixel-pipeline.json": {
            "pipelineVersion": "portable-pixel-pipeline-v1",
            "state": "UNSOLVED_WORKING",
            "outputs": [],
        },
        "coverage/coverage.json": {"mode": "ordinary", "filters": {"R": {}}},
    }
    for relative, payload in evidence.items():
        _write_json(output / relative, payload)
    quality = {
        "matchedStars": 30,
        "rmsPixels": 0.5,
        "rmsArcsec": 0.5,
        "parity": "NEGATIVE",
        "catalogIdentity": "1" * 64,
        "indexIdentities": ["astrometry.net:index:4200:healpix:1:hpnside:1"],
        "correspondenceSha256": "2" * 64,
        "catalogManaged": True,
        "installedSetIdentity": "3" * 64,
        "catalogManifestSha256": "4" * 64,
        "indexArtifacts": [
            {
                "indexId": "4200",
                "relativeName": "index-4200.fits",
                "sizeBytes": 4096,
                "sha256": "5" * 64,
                "manifestSha256": "4" * 64,
                "installedSetIdentity": "3" * 64,
            }
        ],
    }
    receipt = {
        "qualityControl": {"manifest": "qc/manifest.json"},
        "registration": {"receipt": "receipts/registration.json"},
        "integration": {
            "pixelPipelineReceipt": "receipts/pixel-pipeline.json",
            "coverage": "coverage/coverage.json",
        },
        "astrometry": {
            "filters": {
                "R": {
                    "status": "SOLVED",
                    "output": "products/R/master_light_R_wcs.fits",
                    "attempts": [
                        {
                            "accepted": True,
                            "result": {"astrometricQuality": quality},
                        }
                    ],
                }
            }
        },
    }
    _write_json(output / "receipt.json", receipt)
    progress(
        ProgressEvent(
            ProgressStage.INTEGRATION,
            "completed",
            1,
            1,
            "synthetic integration complete",
        )
    )
    return E2EResult(
        success=True,
        code="E2E_SUCCEEDED",
        state=E2EState.SOLVED,
        output_directory=str(output),
        evidence_directory=None,
        receipt_path=str(output / "receipt.json"),
        product_paths=(str(product),),
        preview_paths=(),
        passed_light_paths=request.light_files,
        excluded_light_paths=(),
    )


def _synthetic_project_e2e(request, *, solver_backends, progress) -> ProjectE2EResult:
    assert len(request.inventory.assets) > 0
    assert solver_backends
    output = Path(request.output_directory)
    product = output / "products" / "mono" / "master_light_R_solved.fits"
    product.parent.mkdir(parents=True)
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 8.5
    header["CRPIX2"] = 8.5
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 20.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    header["RADESYS"] = "ICRS"
    header["FILTER"] = "R"
    header["OAFSTATE"] = "SOLVED"
    header["OAFWCS"] = "SOLVED"
    fits.writeto(product, np.ones((16, 16), dtype=np.float32), header)
    quality = {
        "referenceFrame": "ICRS",
        "projection": "TAN",
        "centerRaDegrees": 150.0,
        "centerDecDegrees": 20.0,
        "pixelScaleArcsec": 1.0,
        "rotationDegrees": 180.0,
        "rmsPixels": 0.5,
        "rmsArcsec": 0.5,
        "matchedStars": 30,
        "parity": "NEGATIVE",
        "catalogIdentity": "1" * 64,
        "indexIdentities": ["astrometry.net:index:4200:healpix:1:hpnside:1"],
        "correspondenceSha256": "2" * 64,
        "wcsSha256": "6" * 64,
        "catalogManaged": True,
        "installedSetIdentity": "3" * 64,
        "catalogManifestSha256": "4" * 64,
        "indexArtifacts": [
            {
                "indexId": "4200",
                "relativeName": "index-4200.fits",
                "sizeBytes": 4096,
                "sha256": "5" * 64,
                "manifestSha256": "4" * 64,
                "installedSetIdentity": "3" * 64,
            }
        ],
    }
    receipt = {
        "pipelineVersion": "openastroflow-project-e2e-v1",
        "layout": {"panels": []},
        "finalProducts": {
            "guiArtifacts": [
                {
                    "kind": "SOLVED_MONO_FITS",
                    "filter": "R",
                    "relativePath": product.relative_to(output).as_posix(),
                    "astrometry": quality,
                }
            ]
        },
    }
    _write_json(output / "receipt.json", receipt)
    progress(ProgressEvent(ProgressStage.COMPLETE, "completed", 1, 1, "project complete"))
    return ProjectE2EResult(
        success=True,
        code="PROJECT_MONO_SUCCEEDED",
        state=E2EState.SOLVED,
        output_directory=str(output),
        evidence_directory=None,
        receipt_path=str(output / "receipt.json"),
        product_paths=(str(product),),
        preview_paths=(),
        passed_light_paths=(),
        excluded_light_paths=(),
        mono_filters=("R",),
    )


def test_cli_doctor_distinguishes_executors_from_catalog_coverage(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"]["inventoryReady"] is True
    assert payload["status"]["pixelExecutionReady"] is True
    assert payload["status"]["catalogCoverageVerified"] is False
    assert payload["status"]["endToEndReady"] is False
    assert {backend["stage"] for backend in payload["backends"]} >= {
        "CALIBRATION",
        "DRIZZLE",
        "SOLVER",
    }


def test_cli_inventory_plan_and_canonical_controller_plan(
    nina_project: Path, capsys
) -> None:
    inputs = [
        str(nina_project / role) for role in ("LIGHT", "FLAT", "DARK", "BIAS")
    ]
    assert main(["inventory", *inputs, "--compact"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["counts"]["LIGHT"] == 1
    assert len(inventory["inputManifestSha256"]) == 64

    assert main(["plan", *inputs, "--compact"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["contractValid"] is True
    assert plan["claims"]["pixelExecutionImplemented"] is True

    assert main(
        [
            "controller-plan",
            *inputs,
            "--session-id",
            "session-test",
            "--hardware-profile",
            "m3-pro-tuned",
            "--compact",
        ]
    ) == 0
    envelope = WorkerEnvelope.from_mapping(json.loads(capsys.readouterr().out))
    assert envelope.message_type == "plan"
    assert envelope.payload["recipe"]["stages"][0]["kind"] == "quality-control"
    assert envelope.payload["recipe"]["solver"]["catalog"] == "astrometry-net-offline"


def test_canonical_handshake_plan_execute_emits_real_stage_receipts_and_final_master(
    nina_project: Path, tmp_path: Path
) -> None:
    inventory = _inventory(nina_project)
    plan = controller_plan_envelope(
        inventory,
        session_id="session-test",
        requested_hardware_profile="m3-pro-tuned",
        created_at_unix_ms=1,
    ).to_dict()
    execute = {
        "protocolVersion": 1,
        "sessionId": "session-test",
        "sequence": 2,
        "sentAtUnixMs": 3,
        "type": "execute",
        "payload": {
            "requestId": plan["payload"]["requestId"],
            "planId": plan["payload"]["planId"],
            "runId": "run-test",
            "outputParentHostPath": str(tmp_path),
            "outputDirectoryName": "result",
        },
    }
    input_stream = StringIO(
        "".join(
            json.dumps(item) + "\n"
            for item in (_controller_handshake(), plan, execute)
        )
    )
    output_stream = StringIO()

    assert run_worker(
        input_stream,
        output_stream,
        registry_factory=_registry,
        e2e_runner=_synthetic_e2e,
        hardware_detector=lambda: detect_hardware(
            system="Darwin",
            machine="arm64",
            cpu_brand="Apple M3 Pro",
        ),
        metal_probe=lambda _hardware: True,
    ) == 0
    cursor = ProtocolCursor()
    responses = [
        cursor.accept(decode_ndjson_line(line))
        for line in output_stream.getvalue().splitlines()
    ]
    assert responses[0].message_type == "handshake"
    assert any(response.message_type == "progress" for response in responses)
    assert responses[-1].message_type == "progress", output_stream.getvalue()
    assert responses[-1].payload["state"] == "succeeded"
    artifacts = [response.payload for response in responses if response.message_type == "artifact"]
    assert {payload["stage"]["kind"] for payload in artifacts} == {
        "quality-control",
        "calibration",
        "registration",
        "integration",
        "astrometric-solve",
    }
    final = next(
        payload["artifact"]
        for payload in artifacts
        if payload["artifact"]["designation"] == "final-master"
    )
    assert final["relativePath"] == "products/R/master_light_R_wcs.fits"
    assert final["attributes"]["filter"] == "R"
    assert "rgb" not in final["artifactId"]
    assert final["astrometry"]["matchedStars"] == 30
    assert len(final["astrometry"]["wcsSha256"]) == 64


@pytest.mark.parametrize("failed", [False, True])
def test_worker_automatically_selects_project_runner_for_multiple_targets(
    nina_project: Path, tmp_path: Path, failed: bool
) -> None:
    lights = sorted((nina_project / "LIGHT").glob("*.fits"))
    if len(lights) == 1:
        clone = lights[0].with_name("M16_120s_R_002.fits")
        shutil.copyfile(lights[0], clone)
        lights.append(clone)
    assert len(lights) >= 2
    for path in lights[len(lights) // 2 :]:
        fits.setval(path, "OBJECT", value="M16-PANEL-2")
    inventory = _inventory(nina_project)
    plan = controller_plan_envelope(
        inventory,
        session_id="session-project",
        requested_hardware_profile="generic-arm64-cpu",
        created_at_unix_ms=1,
    ).to_dict()
    execute = {
        "protocolVersion": 1,
        "sessionId": "session-project",
        "sequence": 2,
        "sentAtUnixMs": 3,
        "type": "execute",
        "payload": {
            "requestId": plan["payload"]["requestId"],
            "planId": plan["payload"]["planId"],
            "runId": "run-project-test",
            "outputParentHostPath": str(tmp_path),
            "outputDirectoryName": "project-result",
        },
    }
    input_stream = StringIO(
        "".join(
            json.dumps(item) + "\n"
            for item in (_controller_handshake("session-project"), plan, execute)
        )
    )
    output_stream = StringIO()

    def wrong_single_runner(*_args, **_kwargs):
        raise AssertionError("single-target runner must not be selected")

    def project_runner(*args, **kwargs):
        result = _synthetic_project_e2e(*args, **kwargs)
        if failed:
            return replace(
                result,
                success=False,
                code="QC_INSUFFICIENT_LIGHTS",
                message="B: quality gate admitted 1 Light frame; at least 2 are required",
            )
        return result

    assert run_worker(
        input_stream,
        output_stream,
        registry_factory=_registry,
        e2e_runner=wrong_single_runner,
        project_runner=project_runner,
        hardware_detector=lambda: detect_hardware(
            system="Darwin", machine="arm64", cpu_brand="Apple M2"
        ),
        metal_probe=lambda _hardware: False,
    ) == 0
    responses = [
        decode_ndjson_line(line) for line in output_stream.getvalue().splitlines()
    ]
    if failed:
        assert responses[-1].message_type == "error", output_stream.getvalue()
        assert responses[-1].payload["code"] == "qc-insufficient-lights"
        assert responses[-1].payload["message"] == (
            "B: quality gate admitted 1 Light frame; at least 2 are required"
        )
        return
    assert responses[-1].message_type == "progress", output_stream.getvalue()
    finals = [item for item in responses if item.message_type == "artifact"]
    assert len(finals) == 1
    assert finals[0].payload["artifact"]["kind"] == "final-master"
    assert finals[0].payload["artifact"]["astrometry"]["catalogManaged"] is True


def test_old_simplified_worker_protocol_is_removed() -> None:
    input_stream = StringIO('{"id":"h","type":"handshake","protocolVersion":1}\n')
    output_stream = StringIO()
    assert run_worker(input_stream, output_stream, registry_factory=_registry) == 2
    assert output_stream.getvalue() == ""


def test_linux_x86_worker_truthfully_advertises_portable_cpu() -> None:
    hardware = detect_hardware(system="Linux", machine="x86_64", cpu_brand="x86_64")
    capabilities = worker_capabilities(
        _registry(), hardware, metal_available=False
    )
    assert capabilities["hardwareProfiles"] == ["portable-cpu"]
    assert "cpu-execution" in capabilities["features"]
    assert "metal-execution" not in capabilities["features"]


def test_windows_worker_keeps_the_windows_cpu_profile() -> None:
    hardware = detect_hardware(
        system="Windows", machine="AMD64", cpu_brand="AMD Ryzen"
    )
    capabilities = worker_capabilities(_registry(), hardware, metal_available=False)
    assert capabilities["hardwareProfiles"] == ["windows-cpu"]
    assert "cpu-execution" in capabilities["features"]
    assert "metal-execution" not in capabilities["features"]


def test_wire_profiles_bind_the_authorized_pixel_backend() -> None:
    assert pixel_backend_for_hardware_profile("portable-cpu") == "portable-cpu"
    assert pixel_backend_for_hardware_profile("generic-arm64-cpu") == "portable-cpu"
    assert pixel_backend_for_hardware_profile("windows-cpu") == "portable-cpu"
    assert (
        pixel_backend_for_hardware_profile("generic-apple-metal")
        == "generic-apple-metal"
    )
    assert pixel_backend_for_hardware_profile("m3-pro-tuned") == "m3-pro-tuned"
    with pytest.raises(Exception) as error:
        pixel_backend_for_hardware_profile("future-accelerator")
    assert getattr(error.value, "code", None) == "HARDWARE_PROFILE_UNMAPPABLE"


def test_apple_metal_profiles_require_a_real_executor_probe() -> None:
    hardware = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=12, physical_memory_bytes=36 * GIB
    )
    assert _supported_hardware_profiles(
        hardware, metal_available=False, tuning=tuning
    ) == (
        "generic-arm64-cpu",
    )
    unavailable = worker_capabilities(
        _registry(), hardware, metal_available=False, tuning=tuning
    )
    assert "metal-execution" not in unavailable["features"]
    assert "m3-pro-tuned" not in unavailable["hardwareProfiles"]

    available = worker_capabilities(
        _registry(), hardware, metal_available=True, tuning=tuning
    )
    assert "generic-apple-metal" in available["hardwareProfiles"]
    assert "m3-pro-tuned" in available["hardwareProfiles"]
    assert "metal-execution" in available["features"]


def test_low_memory_m3_pro_does_not_advertise_unusable_tuned_profile() -> None:
    hardware = detect_hardware(
        system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
    )
    tuning = select_execution_tuning(
        hardware, logical_cores=12, physical_memory_bytes=18 * GIB
    )

    capabilities = worker_capabilities(
        _registry(), hardware, metal_available=True, tuning=tuning
    )

    assert tuning.profile_id == "apple-silicon-generic-v1"
    assert capabilities["hardwareProfiles"] == [
        "generic-arm64-cpu",
        "generic-apple-metal",
    ]
    assert "metal-execution" in capabilities["features"]
    assert "m3-pro-tuning" not in capabilities["features"]


def test_metal_probe_failure_preserves_a_canonical_cpu_handshake() -> None:
    input_stream = StringIO(json.dumps(_controller_handshake()) + "\n")
    output_stream = StringIO()

    def broken_probe(_hardware):
        raise OSError("native Metal library is absent")

    assert run_worker(
        input_stream,
        output_stream,
        registry_factory=_registry,
        hardware_detector=lambda: detect_hardware(
            system="Darwin", machine="arm64", cpu_brand="Apple M3 Pro"
        ),
        metal_probe=broken_probe,
    ) == 0
    response = decode_ndjson_line(output_stream.getvalue())
    capabilities = response.payload["capabilities"]
    assert capabilities["hardwareProfiles"] == ["generic-arm64-cpu"]
    assert "metal-execution" not in capabilities["features"]
    assert "m3-pro-tuning" not in capabilities["features"]


def test_controller_plan_does_not_emit_schema_valid_but_unexecutable_drizzle(
    nina_project: Path,
) -> None:
    with pytest.raises(Exception) as error:
        controller_plan_envelope(
            _inventory(nina_project),
            mode="drizzle",
            drizzle_scale=4,
            requested_hardware_profile="m3-pro-tuned",
        )
    assert getattr(error.value, "code", None) == "DRIZZLE_SCALE_UNSUPPORTED"


def test_worker_will_not_promote_a_synthetic_stage_placeholder(tmp_path: Path) -> None:
    placeholder = tmp_path / "receipt.json"
    placeholder.write_text('{"status":"succeeded"}', encoding="utf-8")
    with pytest.raises(Exception, match="connected executor receipt"):
        _validate_stage_evidence("integration", placeholder)
