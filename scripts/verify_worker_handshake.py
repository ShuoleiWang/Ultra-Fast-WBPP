#!/usr/bin/env python3
"""Validate the CI worker handshake against the actual hosted-runner profile."""

from __future__ import annotations

import argparse
import sys

from openastroflow_engine.protocol_v1 import decode_ndjson_line


def _normalized(value: str) -> str:
    return value.strip().casefold().replace("_", "").replace("-", "")


def verify(raw: bytes, *, runner_os: str, runner_arch: str) -> tuple[str, ...]:
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"worker emitted {len(lines)} records; expected one handshake")
    envelope = decode_ndjson_line(lines[0])
    if envelope.message_type != "handshake" or envelope.sequence != 0:
        raise ValueError("worker did not emit the canonical sequence-0 handshake")
    if envelope.payload["role"] != "worker":
        raise ValueError("handshake peer is not a worker")
    capabilities = envelope.payload["capabilities"]
    profiles = tuple(capabilities["hardwareProfiles"])
    profile_set = set(profiles)
    features = set(capabilities["features"])
    os_name = _normalized(runner_os)
    architecture = _normalized(runner_arch)

    if os_name == "linux" and architecture in {"x64", "x8664", "amd64"}:
        if profile_set != {"portable-cpu"}:
            raise ValueError(f"Linux x86-64 must advertise only portable-cpu: {profiles}")
    elif os_name == "windows" and architecture in {"x64", "x8664", "amd64"}:
        if profile_set != {"windows-cpu"}:
            raise ValueError(f"Windows x64 must advertise only windows-cpu: {profiles}")
    elif os_name in {"macos", "darwin"} and architecture in {"arm64", "aarch64"}:
        if "generic-arm64-cpu" not in profile_set:
            raise ValueError("Apple arm64 worker omitted generic-arm64-cpu")
        if profile_set & {"portable-cpu", "windows-cpu"}:
            raise ValueError(f"Apple arm64 advertised a foreign CPU profile: {profiles}")
    else:
        raise ValueError(f"CI profile verifier has no contract for {runner_os}/{runner_arch}")

    metal_profiles = profile_set & {"generic-apple-metal", "m3-pro-tuned"}
    if bool(metal_profiles) != ("metal-execution" in features):
        raise ValueError("Metal profile/feature advertisement is inconsistent")
    if "m3-pro-tuned" in profile_set and "m3-pro-tuning" not in features:
        raise ValueError("m3-pro-tuned lacks m3-pro-tuning")
    if "cpu-execution" not in features:
        raise ValueError("worker omitted the CPU execution feature")
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-os", required=True)
    parser.add_argument("--runner-arch", required=True)
    args = parser.parse_args()
    try:
        profiles = verify(
            sys.stdin.buffer.read(),
            runner_os=args.runner_os,
            runner_arch=args.runner_arch,
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"worker handshake verification failed: {error}", file=sys.stderr)
        return 1
    print("verified worker profiles: " + ", ".join(profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
