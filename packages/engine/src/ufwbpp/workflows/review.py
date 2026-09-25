"""Human review applied to a run: bound approval selections and explicit Light selections."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from enum import StrEnum
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from lightframeqc.models import FrameResult, GateDisposition

from ..blink.session import BlinkEvidence
from ..integrity import canonical_json_document
from .contracts import E2EError, ReviewApproval, E2ERequest, SELECTION_POLICY
from .sources import _capture_source, _SourceIdentity


def _approval_config_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise E2EError("APPROVAL_CONTEXT_INVALID", f"{name} is non-finite")
        return value
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return _approval_config_value(asdict(value), name)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise E2EError(
                "APPROVAL_CONTEXT_INVALID", f"{name} contains a non-string key"
            )
        return {
            key: _approval_config_value(item, f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _approval_config_value(item, f"{name}[]") for item in value
        ]
    raise E2EError(
        "APPROVAL_CONTEXT_INVALID",
        f"{name} has unsupported type {type(value).__name__}",
    )


def _review_approval_request_digest(
    request: E2ERequest, identities: Sequence[_SourceIdentity]
) -> str:
    """Bind manual admission to every science-affecting request/input field."""

    payload = {
        "schemaVersion": 1,
        "sources": [
            identity.local_serializable()
            for identity in sorted(identities, key=lambda item: (item.role, item.path))
        ],
        "qualityControl": {
            "config": request.qc_config.serializable(),
            "gatePolicy": request.gate_policy.serializable(),
            "gatePolicyDigest": request.gate_policy.canonical_digest(),
            # The selection policy changes admission, so an unattended policy
            # is bound into the approval digest; the legacy digest is unchanged.
            **(
                {"selection": request.selection.serializable()}
                if request.selection.unattended
                else {}
            ),
        },
        "pipeline": request.pipeline_parameters.serializable(),
        "recipeDigest": request.recipe_digest,
        "workers": request.workers,
        "integrationMode": request.integration_mode.value,
        "drizzle": request.drizzle.serializable(),
        "solverQuality": {
            "raHintDegrees": request.ra_hint_degrees,
            "decHintDegrees": request.dec_hint_degrees,
            "fieldOfViewDegrees": request.field_of_view_degrees,
            "searchRadiusDegrees": request.search_radius_degrees,
            "minimumMatches": request.min_matches,
            "maximumRmsArcsec": request.max_rms_arcsec,
        },
        "registration": {
            "detection": _approval_config_value(
                request.registration_detection, "registrationDetection"
            ),
            "config": _approval_config_value(
                request.registration_config, "registrationConfig"
            ),
        },
    }
    return "sha256:" + hashlib.sha256(canonical_json_document(payload)).hexdigest()


def bind_review_approval_selections(
    request: E2ERequest, selections: Sequence[Mapping[str, str]]
) -> E2ERequest:
    """Turn GUI review choices into full approvals bound to the current request.

    The GUI supplies only the exact preflight Light digest and gate-policy
    digest.  This trusted runtime recomputes every source identity and the full
    science-request digest immediately before execution.  The normal E2E gate
    still requires that each selected source is uniquely present and remains
    REVIEW; PASS and HARD_FAIL can never be promoted.
    """

    if not selections:
        return request
    pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    policy_digest = request.gate_policy.canonical_digest()
    sources = (
        ("LIGHT", request.light_files),
        ("FLAT", request.flat_files),
        ("DARK", request.dark_files),
        ("BIAS", request.bias_files),
        ("MASTER_BIAS", request.master_bias_files),
        ("MASTER_DARK", request.master_dark_files),
        ("MASTER_FLAT", request.master_flat_files),
    )
    identities = tuple(
        _capture_source(Path(value).expanduser().resolve(strict=True), role)
        for role, values in sources
        for value in values
    )
    request_digest = _review_approval_request_digest(request, identities)
    approvals: list[ReviewApproval] = []
    seen: set[str] = set()
    for index, selection in enumerate(selections):
        if set(selection) != {"sourceSha256", "gatePolicyDigest"}:
            raise E2EError(
                "REVIEW_SELECTION_INVALID",
                f"review selection {index} has unsupported fields",
            )
        source_sha256 = selection.get("sourceSha256")
        selected_policy = selection.get("gatePolicyDigest")
        if (
            not isinstance(source_sha256, str)
            or pattern.fullmatch(source_sha256) is None
            or source_sha256 in seen
            or not isinstance(selected_policy, str)
            or pattern.fullmatch(selected_policy) is None
        ):
            raise E2EError(
                "REVIEW_SELECTION_INVALID",
                "review selections require unique lowercase source and policy SHA-256 values",
            )
        if selected_policy != policy_digest:
            raise E2EError(
                "REVIEW_APPROVAL_POLICY_DRIFT",
                "preflight approval policy differs from the current execution policy",
            )
        seen.add(source_sha256)
        approvals.append(
            ReviewApproval(
                source_sha256=source_sha256,
                gate_policy_digest=selected_policy,
                request_digest=request_digest,
            )
        )
    return replace(request, review_approvals=tuple(approvals))


def _apply_review_approvals(
    *,
    request: E2ERequest,
    identities: Sequence[_SourceIdentity],
    results: Sequence[FrameResult],
) -> tuple[set[str], str, list[dict[str, Any]]]:
    request_digest = _review_approval_request_digest(request, identities)
    policy_digest = request.gate_policy.canonical_digest()
    approvals: dict[str, ReviewApproval] = {}
    for approval in request.review_approvals:
        if approval.source_sha256 in approvals:
            raise E2EError(
                "REVIEW_APPROVAL_DUPLICATE",
                "a source SHA-256 appears in more than one REVIEW approval",
            )
        if approval.gate_policy_digest != policy_digest:
            raise E2EError(
                "REVIEW_APPROVAL_POLICY_DRIFT",
                "approval gate-policy digest differs from the executed policy",
            )
        if approval.request_digest != request_digest:
            raise E2EError(
                "REVIEW_APPROVAL_REQUEST_DRIFT",
                "approval request digest differs from the executed inputs or recipe",
            )
        approvals[approval.source_sha256] = approval

    light_identities = {
        identity.path: identity
        for identity in identities
        if identity.role == "LIGHT"
    }
    light_paths_by_sha: dict[str, list[str]] = {}
    for identity in light_identities.values():
        light_paths_by_sha.setdefault(identity.sha256, []).append(identity.path)
    ambiguous = sorted(
        digest
        for digest in approvals
        if len(light_paths_by_sha.get(digest, ())) != 1
    )
    if ambiguous:
        raise E2EError(
            "REVIEW_APPROVAL_SOURCE_AMBIGUOUS",
            "approval SHA-256 must identify exactly one current Light: "
            + ", ".join(ambiguous),
        )
    admitted: set[str] = set()
    evidence: list[dict[str, Any]] = []
    matched_approvals: set[str] = set()
    for result in results:
        canonical = str(Path(result.path).resolve(strict=True))
        identity = light_identities.get(canonical)
        if identity is None or result.quality_gate is None:
            continue
        approval = approvals.get(identity.sha256)
        if approval is None:
            continue
        if result.quality_gate.disposition is not GateDisposition.REVIEW:
            raise E2EError(
                "REVIEW_APPROVAL_SOURCE_NOT_REVIEW",
                "manual approval may admit REVIEW only, never PASS or HARD_FAIL",
                path=canonical,
            )
        if not result.registration.ok:
            # No transform in the quality pass: the registration would fail on
            # this frame and take the whole run with it.  Refuse now instead.
            raise E2EError(
                "REVIEW_APPROVAL_UNREGISTRABLE",
                "an approved REVIEW frame could not be registered in the quality pass",
                path=canonical,
            )
        matched_approvals.add(identity.sha256)
        admitted.add(canonical)
        evidence.append(
            {
                **approval.serializable(),
                "path": canonical,
                "gateDisposition": GateDisposition.REVIEW.value,
                "admitted": True,
            }
        )
    unmatched = sorted(set(approvals) - matched_approvals)
    if unmatched:
        raise E2EError(
            "REVIEW_APPROVAL_SOURCE_UNKNOWN",
            "approval SHA-256 does not uniquely identify a current REVIEW Light: "
            + ", ".join(unmatched),
        )
    return admitted, request_digest, evidence


def _apply_explicit_selection(
    *,
    request: E2ERequest,
    identities: Sequence[_SourceIdentity],
    results: Sequence[FrameResult],
    evidence: BlinkEvidence,
) -> tuple[set[str], dict[str, Any]]:
    """Admit exactly the Lights the selection keeps; record every override.

    The gate and the blink flags have run; they inform the record but never
    decide.  A KEEP on a frame without a registration transform fails closed
    (the run's registration would fail on it); a KEEP on a HARD_FAIL or on
    an EXCLUDE-flagged frame is honoured and counted as an override.
    """

    selection = request.explicit_selection
    assert selection is not None
    light_identities = {
        identity.path: identity for identity in identities if identity.role == "LIGHT"
    }
    paths_by_sha: dict[str, list[str]] = {}
    for identity in light_identities.values():
        paths_by_sha.setdefault(identity.sha256, []).append(identity.path)
    # Source identities carry the same ``sha256:`` form as the selection.
    decisions = dict(selection.by_source)
    ambiguous = sorted(digest for digest in decisions if len(paths_by_sha.get(digest, ())) > 1)
    if ambiguous:
        raise E2EError(
            "SELECTION_SOURCE_AMBIGUOUS",
            "a selection digest names more than one current Light: " + ", ".join(ambiguous),
        )
    unknown = sorted(digest for digest in decisions if digest not in paths_by_sha)
    if unknown:
        raise E2EError(
            "SELECTION_SOURCE_UNKNOWN",
            "a selection digest names no current Light: " + ", ".join(unknown),
        )
    flags_by_path = evidence.flags_by_path
    admitted: set[str] = set()
    frames: list[dict[str, Any]] = []
    counts = {
        "keep": 0,
        "drop": 0,
        "overriddenExcludeFlags": 0,
        "overriddenGateHardFail": 0,
        "undecided": 0,
    }
    undecided_paths: list[str] = []
    for result in sorted(results, key=lambda item: item.path):
        canonical = str(Path(result.path).resolve(strict=True))
        identity = light_identities.get(canonical)
        if identity is None:
            continue
        decision = decisions.get(identity.sha256)
        undecided = decision is None
        verdict = selection.undecided if undecided else decision.decision
        if undecided:
            if verdict == "ERROR":
                undecided_paths.append(canonical)
                continue
            counts["undecided"] += 1
        record = flags_by_path.get(result.path)
        flag_codes = list(record.codes) if record is not None else []
        default_decision = record.default_decision if record is not None else None
        gate = result.quality_gate
        gate_disposition = gate.disposition.value if gate is not None else None
        overrode: list[str] = []
        if verdict == "KEEP":
            if not result.registration.ok:
                raise E2EError(
                    "SELECTION_UNREGISTRABLE",
                    "a kept Light could not be registered in the quality pass",
                    path=canonical,
                )
            if record is not None and record.exclude:
                counts["overriddenExcludeFlags"] += 1
                overrode.append("EXCLUDE_FLAGS")
            if gate is not None and gate.disposition is GateDisposition.HARD_FAIL:
                counts["overriddenGateHardFail"] += 1
                overrode.append("GATE_HARD_FAIL")
            counts["keep"] += 1
            admitted.add(canonical)
        else:
            counts["drop"] += 1
        frames.append(
            {
                "sourceSha256": identity.sha256,
                "path": canonical,
                "decision": verdict,
                "undecided": undecided,
                "defaultDecision": default_decision,
                "flags": flag_codes,
                "gateDisposition": gate_disposition,
                "overrode": overrode,
                **({"note": decision.note} if decision is not None and decision.note else {}),
            }
        )
    if undecided_paths:
        raise E2EError(
            "SELECTION_INCOMPLETE",
            f"{len(undecided_paths)} Light(s) have no decision and the selection says undecided=ERROR",
            path=undecided_paths[0],
        )
    block = {
        "policy": SELECTION_POLICY,
        "selectionDigest": selection.digest,
        "origin": dict(selection.origin) if selection.origin is not None else None,
        "undecided": selection.undecided,
        "flagsPolicyDigest": evidence.flags_policy.canonical_digest(),
        "counts": counts,
        "frames": frames,
    }
    return admitted, block
