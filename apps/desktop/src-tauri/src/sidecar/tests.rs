use super::*;

#[test]
fn release_discovery_contract_never_accepts_environment_or_adjacent_overrides() {
    let candidates = development_candidate_paths(
        false,
        Some(PathBuf::from("/tmp/injected-worker")),
        Some(PathBuf::from("/tmp/OpenAstroFlow")),
    );
    assert!(candidates.is_empty());
}

/// Linux refuses to execute a file that is open for writing (`ETXTBSY`);
/// the launcher waits for the writer to finish instead of failing at once.
#[cfg(target_os = "linux")]
#[test]
fn sidecar_launch_waits_for_a_text_busy_executable() {
    use std::os::unix::fs::OpenOptionsExt;

    let root =
        std::env::temp_dir().join(new_public_identifier("text-busy-test").expect("temporary id"));
    std::fs::create_dir(&root).expect("create test directory");
    let script = root.join("busy-sidecar");
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(&script)
        .expect("create fake sidecar");
    writer
        .write_all(b"#!/bin/sh\nexit 0\n")
        .expect("write fake sidecar");
    writer.sync_all().expect("sync fake sidecar");
    // The writer closes only after the launcher has started retrying.
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        drop(writer);
    });
    let mut command = Command::new(&script);
    let output = sidecar_output(&mut command).expect("launch after the writer closed");
    assert!(output.status.success());
    release.join().expect("writer thread");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn application_shutdown_terminates_every_pipeline_process_tree() {
    let registry = PipelineRegistry::default();
    let mut command = platform::test_support::sleeping_command();
    let child = Arc::new(Mutex::new(
        ManagedChild::spawn(&mut command).expect("pipeline child"),
    ));
    registry
        .jobs
        .lock()
        .unwrap()
        .insert("pipeline-shutdown-test".to_owned(), child.clone());

    terminate_all(&registry).expect("terminate pipeline children");
    let status = child.lock().unwrap().wait().expect("reap pipeline child");
    assert!(!status.success());
    assert!(registry.shutting_down.load(Ordering::Acquire));
    terminate_all(&registry).expect("shutdown is idempotent");
}

#[cfg(unix)]
#[test]
fn bundled_runtime_manifest_binds_every_file_and_executable_mode() {
    use std::os::unix::fs::PermissionsExt;

    let resource_root = std::env::temp_dir()
        .join(new_identifier("openastroflow-runtime-manifest").expect("temporary identifier"));
    let runtime_root = resource_root.join(target_suffixed_name().trim_end_matches(".exe"));
    let internal = runtime_root.join("_internal");
    fs::create_dir_all(&internal).expect("runtime fixture directories");
    let entry_point = runtime_root.join(target_suffixed_name());
    fs::write(&entry_point, b"#!/bin/sh\nexit 0\n").expect("runtime fixture executable");
    let mut permissions = fs::metadata(&entry_point).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&entry_point, permissions).unwrap();
    let data = internal.join("runtime.dat");
    fs::write(&data, b"python-runtime").expect("runtime fixture data");
    let entries = serde_json::json!([
        {"path":"_internal","type":"directory"},
        {"executable":false,"path":"_internal/runtime.dat","sha256":sha256_file(&data).unwrap(),"sizeBytes":14,"type":"file"},
        {"executable":true,"path":target_suffixed_name(),"sha256":sha256_file(&entry_point).unwrap(),"sizeBytes":17,"type":"file"}
    ]);
    let canonical = serde_json::to_vec(entries.as_array().unwrap()).unwrap();
    let mut tree = Sha256::new();
    tree.update(canonical);
    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "kind": "openastroflow-worker-sidecar",
        "targetTriple": target_triple_name(),
        "runtime": {
            "directoryName": target_suffixed_name().trim_end_matches(".exe"),
            "entryPoint": target_suffixed_name(),
            "treeSha256": format!("{:x}", tree.finalize()),
            "sizeBytes": 31,
            "fileCount": 2,
            "entryCount": 3,
            "entries": entries
        },
        "protocol": {},
        "versions": {},
        "collections": []
    });
    fs::write(
        resource_root.join(target_manifest_name()),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .expect("runtime fixture manifest");

    let verified = verify_bundled_runtime(&resource_root).expect("verified runtime tree");
    assert_eq!(verified.path, entry_point);
    fs::write(&data, b"tampered").expect("tamper runtime fixture");
    assert!(verify_bundled_runtime(&resource_root).is_err());
    let _ = fs::remove_dir_all(resource_root);
}

fn profile_capabilities(profiles: &[HardwareProfile]) -> BackendCapabilities {
    let hardware_profiles = profiles.iter().copied().collect();
    let mut features = std::collections::BTreeSet::from([BackendFeature::CpuExecution]);
    if profiles.iter().any(|profile| {
        matches!(
            profile,
            HardwareProfile::GenericAppleMetal | HardwareProfile::M3ProTuned
        )
    }) {
        features.insert(BackendFeature::MetalExecution);
    }
    if profiles.contains(&HardwareProfile::M3ProTuned) {
        features.insert(BackendFeature::M3ProTuning);
    }
    BackendCapabilities {
        schema_version: 1,
        backend_id: "profile-test".to_owned(),
        backend_version: "1".to_owned(),
        worker_build: "test".to_owned(),
        hardware_profiles,
        stages: std::collections::BTreeSet::from([StageKind::Integration]),
        features,
        maximum_parallel_stages: 1,
        input_extensions: std::collections::BTreeSet::new(),
        output_extensions: std::collections::BTreeSet::new(),
    }
}

fn simulated_host(
    platform: &'static str,
    architecture: &'static str,
    chip: &str,
) -> platform::PlatformProfile {
    platform::PlatformProfile {
        platform,
        architecture,
        chip: chip.to_owned(),
        cpu_backend: "test-cpu",
        gpu_backend: "test-gpu",
        optimization_tier: "PORTABLE",
    }
}

#[test]
fn profile_selection_never_maps_linux_x86_to_arm64() {
    let capabilities = profile_capabilities(&[
        HardwareProfile::PortableCpu,
        HardwareProfile::GenericArm64Cpu,
    ]);
    let host = simulated_host("linux", "x86_64", "x86_64");
    assert_eq!(
        select_profile_for_host(&capabilities, &host),
        Ok(HardwareProfile::PortableCpu)
    );
}

#[test]
fn profile_selection_keeps_windows_on_its_explicit_cpu_contract() {
    let host = simulated_host("windows", "x86_64", "x86_64");
    let capabilities =
        profile_capabilities(&[HardwareProfile::PortableCpu, HardwareProfile::WindowsCpu]);
    assert_eq!(
        select_profile_for_host(&capabilities, &host),
        Ok(HardwareProfile::WindowsCpu)
    );
    let portable_only = profile_capabilities(&[HardwareProfile::PortableCpu]);
    assert!(select_profile_for_host(&portable_only, &host).is_err());
}

#[test]
fn profile_selection_uses_only_worker_advertised_metal() {
    let host = simulated_host("macos", "aarch64", "Apple M3 Pro");
    let cpu_only = profile_capabilities(&[
        HardwareProfile::PortableCpu,
        HardwareProfile::GenericArm64Cpu,
    ]);
    assert_eq!(
        select_profile_for_host(&cpu_only, &host),
        Ok(HardwareProfile::GenericArm64Cpu)
    );
    let probed = profile_capabilities(&[
        HardwareProfile::PortableCpu,
        HardwareProfile::GenericArm64Cpu,
        HardwareProfile::GenericAppleMetal,
        HardwareProfile::M3ProTuned,
    ]);
    assert_eq!(
        select_profile_for_host(&probed, &host),
        Ok(HardwareProfile::M3ProTuned)
    );
}

#[test]
fn windows_release_validation_is_x86_64_only() {
    assert!(platform_scientific_release_validated(&simulated_host(
        "windows",
        "x86_64",
        "AMD Ryzen 7 5800H"
    )));
    assert!(platform_scientific_release_validated(&simulated_host(
        "macos",
        "aarch64",
        "Apple M3 Pro"
    )));
    let arm = simulated_host("windows", "aarch64", "Snapdragon X Elite");
    assert!(!platform_scientific_release_validated(&arm));
    let reason = platform_unavailable_reason(&arm);
    assert!(reason.contains("aarch64"));
    assert!(reason.contains("x86-64 only"));
}

#[test]
fn worker_environment_forces_utf8_stdio() {
    let command = EngineExecutable {
        path: PathBuf::from("openastroflow-worker"),
    }
    .command("worker");
    let environment = command
        .get_envs()
        .filter_map(|(key, value)| Some((key.to_str()?, value?.to_str()?)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(environment.get("PYTHONUTF8"), Some(&"1"));
    assert_eq!(environment.get("PYTHONIOENCODING"), Some(&"utf-8"));
    assert_eq!(
        command.get_args().collect::<Vec<_>>(),
        vec![std::ffi::OsStr::new("worker")]
    );
}

#[test]
fn lossy_lines_survive_invalid_utf8_and_keep_later_lines() {
    let stream: Vec<u8> = b"first\r\nbad \xff byte\n{\"type\":\"progress\"}\nno newline".to_vec();
    let lines = LossyLines::new(std::io::Cursor::new(stream)).collect::<Vec<_>>();
    assert_eq!(
        lines,
        vec![
            "first".to_owned(),
            "bad \u{fffd} byte".to_owned(),
            "{\"type\":\"progress\"}".to_owned(),
            "no newline".to_owned(),
        ]
    );
    assert!(LossyLines::new(std::io::Cursor::new(Vec::new()))
        .next()
        .is_none());
}

#[cfg(unix)]
fn fake_sidecar() -> (PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir()
        .join(new_identifier("openastroflow-fake-sidecar").expect("temporary identifier"));
    std::fs::create_dir(&root).expect("create fake sidecar directory");
    let script = root.join("fake-openastroflow-engine");
    let source = r###"#!/usr/bin/env python3
import argparse, base64, hashlib, json, os, pathlib, platform, re, sys

CPU_PROFILE = "generic-arm64-cpu" if platform.machine().lower() in {"arm64", "aarch64"} else "portable-cpu"

CAPS = {
  "schemaVersion": 1, "backendId": "fake-sidecar", "backendVersion": "0.1.0",
  "workerBuild": "test", "hardwareProfiles": [CPU_PROFILE],
  "stages": ["quality-control", "calibration", "registration", "integration", "drizzle", "astrometric-solve"],
  "features": ["cpu-execution", "deterministic-receipts", "offline-astrometric-solver", "drizzle", "fits"],
  "maximumParallelStages": 1, "inputExtensions": ["fits"], "outputExtensions": ["fits"]
}

def envelope(session, sequence, kind, payload):
  return {"protocolVersion": 1, "sessionId": session, "sequence": sequence,
          "sentAtUnixMs": 10 + sequence, "type": kind, "payload": payload}

def recipe(mode):
  stages = [
    {"stageId":"quality-control","kind":"quality-control","enabled":True,"dependsOn":[],"parameters":{}},
    {"stageId":"calibrate","kind":"calibration","enabled":True,"dependsOn":["quality-control"],"parameters":{}},
    {"stageId":"register","kind":"registration","enabled":True,"dependsOn":["calibrate"],"parameters":{}},
    {"stageId":"integrate","kind":"integration","enabled":True,"dependsOn":["register"],"parameters":{}},
  ]
  dependency = "integrate"
  if mode == "drizzle":
    stages.append({"stageId":"drizzle","kind":"drizzle","enabled":True,"dependsOn":["integrate"],"parameters":{}})
    dependency = "drizzle"
  stages.append({"stageId":"solve","kind":"astrometric-solve","enabled":True,"dependsOn":[dependency],"parameters":{}})
  return {"schemaVersion":1,"recipeId":"fake-e2e","displayName":"Fake E2E","stages":stages,
          "solver":{"result":"required","catalog":"astrometry-net-offline","projection":"TAN","minimumMatches":12,"maximumRmsArcsec":2.0},
          "drizzle":{"result":"required" if mode == "drizzle" else "disabled","scale":2.0,"dropShrink":0.9,"kernel":"square"},
          "parameters":{}}

def controller_plan(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+")
  parser.add_argument("--mode", default="ordinary")
  parser.add_argument("--session-id", required=True)
  parser.add_argument("--sequence", type=int, required=True)
  parser.add_argument("--request-id", required=True)
  parser.add_argument("--plan-id", required=True)
  parser.add_argument("--hardware-profile", required=True)
  parser.add_argument("--compact", action="store_true")
  args = parser.parse_args(argv)
  sources = []
  for index, path in enumerate(args.inputs):
    name = pathlib.Path(path).name.lower()
    role = "flat" if "flat" in name else "bias" if "bias" in name else "dark" if "dark" in name else "light"
    sources.append({"sourceId":f"source-{index}","role":role,"hostPath":path,"recursive":False})
  payload = {"requestId":args.request_id,"planId":args.plan_id,
    "project":{"schemaVersion":1,"projectId":"fake-project","displayName":"Unicode 盾牌座","createdAtUnixMs":1,"sources":sources,"labels":{}},
    "recipe":recipe(args.mode),"requestedHardwareProfile":args.hardware_profile,"inputManifestSha256":"0"*64}
  print(json.dumps(envelope(args.session_id,args.sequence,"plan",payload),ensure_ascii=False,separators=(",",":")),flush=True)

def inventory():
  print(json.dumps({"name":"Unicode 盾牌座","assets":[
    {"path":"/数据/盾牌座/亮场 01.fit","role":"LIGHT","status":"READY","cfaPattern":"NONE","roleEvidence":["header:IMAGETYP"]},
    {"path":"/数据/校准/平场 R.fit","role":"FLAT","status":"READY","cfaPattern":"NONE","roleEvidence":["header:IMAGETYP"]}],"issues":[]},ensure_ascii=False),flush=True)

def quality_check(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("--request-json", required=True)
  parser.add_argument("--compact", action="store_true")
  args = parser.parse_args(argv)
  request = json.loads(pathlib.Path(args.request_json).read_text())
  frames = [{"path":path,"sourceSha256":"sha256:"+hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
             "disposition":"REVIEW","decision":"REVIEW","confidence":"HIGH","starCount":42,
             "summary":"fake real gate","evidence":[{"code":"GATE_FAKE","family":"PROVENANCE","severity":"REVIEW","message":"review fixture"}]}
            for path in request["lightPaths"]]
  print(json.dumps({"schemaVersion":1,"gatePolicyDigest":"sha256:"+"7"*64,"workers":1,
                    "counts":{"PASS":0,"REVIEW":len(frames),"HARD_FAIL":0},"frames":frames},separators=(",",":")),flush=True)

FILMSTRIP_JPEG = base64.b64decode(
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/wAALCAAIAAwBAREA"
  "/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRol"
  "JicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
  "4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AMPzGdVym53+Z2ZGCgjpknGTkAeg5/FqXMsMakKhL5Y7sDGCR0yOwFf/2Q==")
ZOOM_PNG = base64.b64decode(
  "iVBORw0KGgoAAAANSUhEUgAAAAwAAAAICAAAAADoj0EtAAAAd0lEQVR4AQKMMVqB6R3nD/5/r39KvmBh+f7+l+Cr9wI/+Z8Ks/zj/8P0ipGX+x87/2cW3mc/uCQ+sD5nFeV7z8Im"
  "cP834/v//1j+32FnYbyj8PcZv+i7f3/4vzH9EWd+KsZ6W+jrf24ups/vGOWeMDC+kPz+/z0A4Qsw9hBiytcAAAAASUVORK5CYII=")

def blink_fail(code, message):
  print(json.dumps({"ok":False,"error":{"code":code,"message":message}}), file=sys.stderr, flush=True)
  raise SystemExit(2)

def blink_measure(argv):
  # A schema-exact blink-manifest-v1 fixture: two channels (L, R) with nights,
  # the combined EXCLUDE rule, an ATTENTION flag, one reference per channel and
  # real JPEG/PNG previews written create-only into the session directory.
  parser = argparse.ArgumentParser()
  parser.add_argument("--request-json", required=True)
  parser.add_argument("--compact", action="store_true")
  args = parser.parse_args(argv)
  request = json.loads(pathlib.Path(args.request_json).read_text(encoding="utf-8"))
  if request.get("schemaVersion") != 1 or set(request) - {"schemaVersion","lightPaths","sessionDirectory","workers","previews","masterFlats","masterDarks","masterBias"}:
    blink_fail("BLINK_REQUEST_INVALID", "unsupported request fields")
  session = pathlib.Path(request["sessionDirectory"])
  try:
    session.mkdir(parents=False, exist_ok=False)
  except FileExistsError:
    blink_fail("BLINK_SESSION_EXISTS", "session directory exists")
  (session / "filmstrip").mkdir(); (session / "zoom").mkdir()
  frames = []; channels = {}
  ordered = sorted(request["lightPaths"], key=lambda p: (("_R_" in pathlib.Path(p).name), p))
  for index, path in enumerate(ordered):
    name = pathlib.Path(path).name
    filt = "R" if "_R_" in name else "L"
    night = re.search(r"(\d{4}-\d{2}-\d{2})", name); night = night.group(1) if night else "2026-08-17"
    flags = []
    if "moon" in name:
      flags = [{"code":"BLINK_SKY_BRIGHT","severity":"EXCLUDE","value":2.43,"threshold":1.6,"combined":True,"message":"Sky 2.43x the clean-sky level and 55 % of its stars"},
               {"code":"BLINK_SOURCES_LOW","severity":"ATTENTION","value":0.55,"threshold":0.6,"combined":True,"message":"55 % of the channel's best star count"}]
    elif "haze" in name:
      flags = [{"code":"BLINK_EXTINCTION","severity":"ATTENTION","value":0.67,"threshold":0.5,"combined":False,"message":"0.67 mag extra extinction"}]
    safe = re.sub(r"[^A-Za-z0-9]+", "-", pathlib.Path(name).stem)[:40]
    filmstrip = f"filmstrip/{index:04d}-{filt}-{safe}.jpg"; zoom = f"zoom/{index:04d}-{filt}-{safe}.png"
    (session / filmstrip).write_bytes(FILMSTRIP_JPEG + (b"\0" * 210 * 1024 if "oversize" in name else b""))
    (session / zoom).write_bytes(ZOOM_PNG)
    digest = "sha256:" + hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    channel = channels.setdefault(filt, {"channelId":f"group-{filt.lower()}0000000000","target":"NGC 6822","filter":filt,"frameCount":0,
      "reference":None,"statistics":{"skyClean":1090.0,"cleanCount":3,"sourcesBest":5680,"fwhmBest":4.07},
      "stretch":{"black":872.0,"white":1420.0,"softness":4.0,"skyReference":986.0,"sigmaReference":43.3},
      "previewGeometry":{"filmstrip":[12,8],"zoom":[12,8],"sourceShape":[4176,6252]},"nights":{}})
    channel["frameCount"] += 1
    excluded = any(f["severity"] == "EXCLUDE" for f in flags)
    reference = channel["reference"] is None and not flags
    if reference:
      channel["reference"] = {"index":index,"sourceSha256":digest,"rule":"psf-signal-weight-proxy-v1"}
    summary = channel["nights"].setdefault(night, {"night":night,"frameCount":0,"medianSky":2360 if excluded else 1000,"skyRatio":2.17 if excluded else 1.0,
      "medianSourceRatio":0.53 if excluded else 0.95,"medianExtinction":0.28,"exclude":0,"attention":0,"defaultDropNight":excluded})
    summary["frameCount"] += 1; summary["exclude"] += int(excluded); summary["attention"] += int(bool(flags) and not excluded)
    frames.append({"index":index,"channelId":channel["channelId"],"filter":filt,"target":"NGC 6822","night":night,"path":path,"name":name,
      "sourceSha256":digest,"observedAt":night+"T22:57:01","airmass":1.31,"reference":reference,"defaultDecision":"DROP" if excluded else "KEEP",
      "flags":flags,"notes":[] if flags else ["GATE_INSUFFICIENT_COHORT"],"gate":{"disposition":"PASS","codes":[]},
      "metrics":{"sky":2647.3 if excluded else 986.0,"skyRatio":2.43 if excluded else 1.0,"starCount":2947,"sourceRatio":0.55,"extinctionMag":0.23,
        "transparency":0.84,"fwhmNative":4.47,"fwhmRatio":1.1,"ellipticity":0.09,"eccentricity":0.38,"registrationRms":0.21,"matchedStars":1900,
        "overlap":1.0,"backgroundShape":0.13,"gradientRatio":None},
      "score":{"log10":-5.0,"z":-3.1,"rank":index+1},
      "previews":{"filmstrip":filmstrip,"zoom":zoom,"coverage":0.99},
      "transformToReference":[[1.0,0.0,1.27],[0.0,1.0,-0.81]],"normalization":{"skyOffset":2647.3,"fluxScale":1.19,"registered":True}})
  for channel in channels.values():
    if channel["reference"] is None:
      first = next(f for f in frames if f["channelId"] == channel["channelId"]); first["reference"] = True
      channel["reference"] = {"index":first["index"],"sourceSha256":first["sourceSha256"],"rule":"psf-signal-weight-proxy-v1"}
    channel["nights"] = list(channel["nights"].values())
  exclude = sum(f["defaultDecision"] == "DROP" for f in frames); attention = sum(f["defaultDecision"] == "KEEP" and bool(f["flags"]) for f in frames)
  manifest = {"schemaVersion":1,"kind":"blink-manifest-v1","sessionId":session.name,"sessionDirectory":str(session.resolve()),
    "createdAt":"2026-09-22T00:00:00","engineVersion":"fake","gatePolicyDigest":"sha256:"+"7"*64,"flagsPolicyDigest":"sha256:"+"8"*64,
    "flagsPolicy":{"version":"blink-flags-v1","skyBrightAttention":1.6},
    "inventorySha256":"sha256:"+hashlib.sha256("".join(f["sourceSha256"] for f in frames).encode()).hexdigest(),
    "timings":{"measurementSeconds":0.1,"analysisSeconds":0.1,"gateSeconds":0.1,"flagsSeconds":0.01,"previewSeconds":0.1},
    "counts":{"frames":len(frames),"exclude":exclude,"attention":attention,"clean":len(frames)-exclude-attention},
    "channels":list(channels.values()),"frames":frames,
    "requestEcho":{"workers":request.get("workers"),"masterFlats":request.get("masterFlats"),"masterDarks":request.get("masterDarks"),"masterBias":request.get("masterBias"),"previews":request["previews"]}}
  (session / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
  print(json.dumps(manifest, ensure_ascii=False, separators=(",",":") if args.compact else None), flush=True)

def artifact_payload(request_id, run_id, stage_id, stage_kind, artifact):
  stage={"schemaVersion":1,"stageId":stage_id,"kind":stage_kind,"status":"succeeded","startedAtUnixMs":1,
         "finishedAtUnixMs":2,"artifactIds":[artifact["artifactId"]],"metrics":{}}
  return {"requestId":request_id,"runId":run_id,"stage":stage,"artifact":artifact}

def worker():
  hello=json.loads(sys.stdin.readline())
  session=hello["sessionId"]
  print(json.dumps(envelope(session,0,"handshake",{"role":"worker","implementation":"fake-sidecar","implementationVersion":"0.1.0","supportedProtocolVersions":[1],"capabilities":CAPS}),separators=(",",":")),flush=True)
  plan_line=sys.stdin.readline()
  if not plan_line: return
  plan=json.loads(plan_line)
  execute=json.loads(sys.stdin.readline())
  request_id=execute["payload"]["requestId"]; run_id=execute["payload"]["runId"]
  sequence=1
  for stage_id,kind in [("quality-control","quality-control"),("calibrate","calibration"),("register","registration"),("integrate","integration")]:
    artifact={"schemaVersion":1,"artifactId":"evidence-"+stage_id,"producedByStageId":stage_id,"kind":"run-log","designation":"diagnostic",
              "relativePath":"evidence/"+stage_id+".json","mediaType":"application/json","sha256":"1"*64,"sizeBytes":1,"createdAtUnixMs":2,"attributes":{}}
    print(json.dumps(envelope(session,sequence,"artifact",artifact_payload(request_id,run_id,stage_id,kind,artifact)),separators=(",",":")),flush=True); sequence+=1
  output=pathlib.Path(execute["payload"]["outputParentHostPath"])/execute["payload"]["outputDirectoryName"]
  master=output/"master"/"Unicode 盾牌座_master.fits"; master.parent.mkdir(parents=True)
  master.write_bytes(b"FAKE-FITS-FOR-CONTROLLER-TEST")
  digest=hashlib.sha256(master.read_bytes()).hexdigest()
  final={"schemaVersion":1,"artifactId":"final-master","producedByStageId":"solve","kind":"final-master","designation":"final-master",
         "relativePath":"master/Unicode 盾牌座_master.fits","mediaType":"image/fits","sha256":digest,"sizeBytes":master.stat().st_size,"createdAtUnixMs":3,
         "astrometry":{"referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,"pixelScaleArcsec":1.4,
           "rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,"parity":"POSITIVE","catalogIdentity":"2"*64,
           "indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],"correspondenceSha256":"3"*64,
           "catalogManaged":True,"installedSetIdentity":"5"*64,"catalogManifestSha256":"6"*64,
           "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7"*64,"manifestSha256":"6"*64,"installedSetIdentity":"5"*64}],
           "wcsSha256":"4"*64},"attributes":{}}
  print(json.dumps(envelope(session,sequence,"artifact",artifact_payload(request_id,run_id,"solve","astrometric-solve",final)),ensure_ascii=False,separators=(",",":")),flush=True)
  print("fake diagnostic stays on stderr",file=sys.stderr,flush=True)

if sys.argv[1] == "controller-plan": controller_plan(sys.argv[2:])
elif sys.argv[1] == "inventory": inventory()
elif sys.argv[1] == "quality-check": quality_check(sys.argv[2:])
elif sys.argv[1] == "blink-measure": blink_measure(sys.argv[2:])
elif sys.argv[1] == "worker": worker()
else: raise SystemExit(2)
"###;
    std::fs::write(&script, source).expect("write fake sidecar");
    let mut permissions = std::fs::metadata(&script)
        .expect("fake sidecar metadata")
        .permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&script, permissions).expect("make fake sidecar executable");
    (root, script)
}

#[test]
fn calibration_preflight_rejects_false_ready_reports() {
    let ready = serde_json::json!({
        "schemaVersion": 1, "status": "READY", "calibrationReady": true,
        "groups": [{"groupId": "group-1", "target": "M31", "filter": "B",
            "lightCount": 8, "observedDates": ["2026-09-04", "2026-09-05"],
            "status": "READY", "matches": {
                "FLAT": {"rawCount": 20, "masterCount": 0},
                "DARK": {"rawCount": 0, "masterCount": 1},
                "BIAS": {"rawCount": 0, "masterCount": 1}}}],
        "issues": []
    });
    let good: CalibrationInspection = serde_json::from_value(ready.clone()).unwrap();
    validate_calibration_inspection(&good).expect("matching metadata should be ready");
    let mut contradictory = ready.clone();
    contradictory["issues"] = serde_json::json!([{
        "code": "CALIBRATION_MISSING", "severity": "ERROR", "message": "No matching flat",
        "paths": [], "lightGroups": ["group-1"]
    }]);
    let bad: CalibrationInspection = serde_json::from_value(contradictory).unwrap();
    assert!(validate_calibration_inspection(&bad).is_err());
    let mut empty = ready;
    empty["groups"] = serde_json::json!([]);
    let bad: CalibrationInspection = serde_json::from_value(empty).unwrap();
    assert!(validate_calibration_inspection(&bad).is_err());
}

#[cfg(unix)]
#[test]
fn calibration_inspection_transports_recipe_and_returns_blockers() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(new_identifier("wbpp-calibration-bridge-test").unwrap());
    fs::create_dir(&root).unwrap();
    let input = root.join("亮场.fit");
    fs::write(&input, b"read-only transport fixture").unwrap();
    let worker = root.join("worker.py");
    fs::write(
        &worker,
        r###"#!/usr/bin/env python3
import json, pathlib, sys
assert sys.argv[1:3] == ['calibration-check', '--request-json']
p = pathlib.Path(sys.argv[3]); data = json.loads(p.read_text())
assert data['schemaVersion'] == 1 and len(data['paths']) == 1
assert data['recipe']['calibration']['bias'] == 'REQUIRED'
(pathlib.Path(__file__).parent/'request-path.txt').write_text(str(p))
print(json.dumps({'schemaVersion':1, 'status':'BLOCKED', 'calibrationReady':False,
 'groups':[{'groupId':'b','target':'M31','filter':'B','lightCount':1,'observedDates':[],
 'status':'BLOCKED','matches':{k:{'rawCount':0,'masterCount':0} for k in ['FLAT','DARK','BIAS']}}],
 'issues':[{'code':'CALIBRATION_MISSING','severity':'ERROR','message':'No matching flat',
 'paths':data['paths'],'lightGroups':['b']}]}))
"###,
    )
    .unwrap();
    fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
    let result = inspect_calibration_with(
        EngineExecutable { path: worker },
        InspectCalibrationRequest {
            paths: vec![input.to_string_lossy().into_owned()],
            recipe: serde_json::json!({"calibration":{"bias":"REQUIRED"}}),
        },
    )
    .expect("a scientific blocker is a report, not a transport failure");
    assert!(!result.calibration_ready);
    assert_eq!(result.issues[0].code, "CALIBRATION_MISSING");
    let private_request = fs::read_to_string(root.join("request-path.txt")).unwrap();
    assert!(!Path::new(&private_request).exists());
    assert_eq!(fs::read(&input).unwrap(), b"read-only transport fixture");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_missing_calibration_roles_before_spawning() {
    let error = unique_input_paths(&[RunSource {
        role: "LIGHT".to_owned(),
        paths: vec!["/tmp/light.fit".to_owned()],
    }])
    .expect_err("missing flats and bias must fail");
    assert!(error.contains("FLAT"));
}

#[test]
fn unicode_paths_are_preserved_and_deduplicated() {
    let path = "/tmp/盾牌座/亮场 01.fit".to_owned();
    let sources = [
        RunSource {
            role: "LIGHT".to_owned(),
            paths: vec![path.clone(), path.clone()],
        },
        RunSource {
            role: "FLAT".to_owned(),
            paths: vec!["/tmp/平场.fit".to_owned()],
        },
        RunSource {
            role: "BIAS".to_owned(),
            paths: vec!["/tmp/偏置.fit".to_owned()],
        },
    ];
    let result = unique_input_paths(&sources).expect("valid sources");
    assert_eq!(result.iter().filter(|item| *item == &path).count(), 1);
}

#[test]
fn deferred_cfa_confirmation_hashes_unicode_files_without_mutating_them() {
    let root = std::env::temp_dir()
        .join(new_identifier("openastroflow-hash-source").expect("temporary identifier"));
    std::fs::create_dir(&root).expect("temporary directory");
    let path = root.join("盾牌座 亮场.fit");
    std::fs::write(&path, b"read-only-source-fixture").expect("source fixture");
    let before = std::fs::read(&path).unwrap();
    let result = hash_sources(HashSourcesRequest {
        paths: vec![path.to_string_lossy().into_owned()],
    })
    .expect("content hash");
    assert!(result.entries[0].source_sha256.starts_with("sha256:"));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(hash_sources(HashSourcesRequest {
        paths: vec![
            path.to_string_lossy().into_owned(),
            path.to_string_lossy().into_owned(),
        ],
    })
    .is_err());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn recipe_mapping_is_explicit() {
    assert_eq!(recipe_cli_id("balanced"), Ok("balanced"));
    assert_eq!(recipe_cli_id("drizzle-2x"), Ok("drizzle-2x"));
    assert!(recipe_cli_id("future-recipe").is_err());
}

#[cfg(unix)]
#[test]
fn calibration_only_import_accepts_batch_without_lights_but_rejects_bad_frames() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(new_identifier("calibration-only-import").unwrap());
    let other_directory = root.join("other calibration directory");
    fs::create_dir_all(&other_directory).unwrap();
    let script = root.join("inventory-worker");
    fs::write(
        &script,
        r###"#!/usr/bin/env python3
import pathlib, sys
assert sys.argv[1] == 'inventory'
print((pathlib.Path(__file__).parent / 'inventory.json').read_text())
"###,
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
    let executable = EngineExecutable { path: script };
    let no_lights = serde_json::json!({
        "code":"NO_LIGHTS", "severity":"ERROR",
        "message":"the selected inputs contain no unprocessed Light frames"
    });
    let mut last = serde_json::Value::Null;
    let mut input = String::new();
    for role in [
        "MASTER_FLAT",
        "MASTER_DARK",
        "MASTER_BIAS",
        "FLAT",
        "DARK",
        "BIAS",
    ] {
        let path = other_directory.join(format!("{role}.fits"));
        fs::write(&path, b"read-only calibration fixture").unwrap();
        input = path.to_string_lossy().into_owned();
        last = serde_json::json!({"name":"new batch", "assets":[{
                "path":input, "role":role, "status":"READY", "width":4,
                "height":4,"channels":1,"filter":"B","camera":"test camera",
                "roleEvidence":["FITS:IMAGETYP"]}], "issues":[no_lights.clone()]});
        fs::write(
            root.join("inventory.json"),
            serde_json::to_vec(&last).unwrap(),
        )
        .unwrap();
        let imported = inspect_paths_with(
            executable.clone(),
            InspectRequest {
                paths: vec![input.clone()],
                role_hint: None,
            },
        )
        .expect("a calibration-only addition must not require a Light in that batch");
        assert_eq!(imported.total_files, 1);
        assert_eq!(imported.sources[0].role, role);
        assert_eq!(imported.sources[0].paths, vec![input.clone()]);
        assert_eq!(
            imported.assets[0].source_sha256.is_some(),
            role.starts_with("MASTER_")
        );
        assert_eq!(fs::read(&path).unwrap(), b"read-only calibration fixture");
    }
    for code in ["ROLE_CONFLICT", "SOURCE_STAT_FAILED", "UNSUPPORTED_FORMAT"] {
        let mut invalid = last.clone();
        invalid["issues"] = serde_json::json!([no_lights.clone(), {
            "code":code,"severity":"ERROR","message":"invalid calibration source"
        }]);
        fs::write(
            root.join("inventory.json"),
            serde_json::to_vec(&invalid).unwrap(),
        )
        .unwrap();
        let error = inspect_paths_with(
            executable.clone(),
            InspectRequest {
                paths: vec![input.clone()],
                role_hint: None,
            },
        )
        .expect_err("real per-frame errors must still block import");
        assert!(error.contains(code));
    }
    last["assets"][0]["status"] = serde_json::json!("ERROR");
    fs::write(
        root.join("inventory.json"),
        serde_json::to_vec(&last).unwrap(),
    )
    .unwrap();
    assert!(inspect_paths_with(
        executable,
        InspectRequest {
            paths: vec![input],
            role_hint: None,
        }
    )
    .unwrap_err()
    .contains("frame is not ready"));
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn fake_sidecar_handshake_and_inventory_preserve_unicode_roles() {
    let (root, script) = fake_sidecar();
    let executable = EngineExecutable { path: script };
    let probe = probe_runtime_with(executable.clone()).expect("canonical handshake");
    assert_eq!(probe.capabilities.backend_id, "fake-sidecar");
    let inventory = inspect_paths_with(
        executable,
        InspectRequest {
            paths: vec!["/选择的目录".to_owned()],
            role_hint: None,
        },
    )
    .expect("real fake-sidecar inventory");
    assert_eq!(inventory.total_files, 2);
    assert!(inventory.sources.iter().any(|source| source
        .paths
        .contains(&"/数据/盾牌座/亮场 01.fit".to_owned())));
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn fake_sidecar_quality_inspection_is_content_bound_and_real() {
    let (root, script) = fake_sidecar();
    let light = root.join("盾牌座 真实亮场.fit");
    std::fs::write(&light, b"quality-light-bytes").expect("write quality fixture");
    let inspection = inspect_quality_with(
        EngineExecutable { path: script },
        InspectQualityRequest {
            paths: vec![light.to_string_lossy().into_owned()],
        },
    )
    .expect("quality inspection");
    assert_eq!(inspection.counts["REVIEW"], 1);
    assert_eq!(inspection.frames[0].disposition, "REVIEW");
    assert!(inspection.frames[0]
        .source_sha256
        .as_deref()
        .is_some_and(|value| value.starts_with("sha256:")));
    assert_eq!(std::fs::read(&light).unwrap(), b"quality-light-bytes");
    let _ = std::fs::remove_dir_all(root);
}

/// Four Lights for the fake `blink-measure`: an L reference night, a
/// moonlit L frame (combined EXCLUDE rule), a hazy R frame (ATTENTION)
/// and a clean R frame whose filmstrip exceeds the inline bound.
#[cfg(unix)]
fn blink_lights(root: &Path) -> Vec<String> {
    let lights = root.join("Lights 盾牌座");
    std::fs::create_dir_all(&lights).expect("lights directory");
    [
        "NGC 6822_300.00s_L_2026-08-17_22-57-01_+8.00°C.fits",
        "NGC 6822_300.00s_L_2026-08-20_21-15-27_moon.fits",
        "NGC 6822_300.00s_R_2026-09-06_haze.fits",
        "NGC 6822_300.00s_R_2026-09-08_oversize 盾牌座.fits",
    ]
    .iter()
    .map(|name| {
        let path = lights.join(name);
        std::fs::write(&path, format!("light bytes of {name}")).expect("write light");
        path.to_string_lossy().into_owned()
    })
    .collect()
}

#[cfg(unix)]
#[test]
fn fake_sidecar_blink_measure_is_validated_and_previews_are_bounded() {
    let (root, script) = fake_sidecar();
    let paths = blink_lights(&root);
    let flat = root.join("masterFlat_L.xisf");
    std::fs::write(&flat, b"flat bytes").expect("write flat");
    let second_r_flat = root.join("masterFlat_R_2.xisf");
    std::fs::write(&second_r_flat, b"flat bytes").expect("write flat");
    let sessions = root.join("blink-sessions");
    let master_flat = |filter: &str, path: &Path| BlinkMasterFlat {
        filter: filter.to_owned(),
        path: path.to_string_lossy().into_owned(),
    };
    let manifest = blink_measure_with(
        EngineExecutable { path: script },
        BlinkMeasureRequest {
            master_darks: vec![BlinkMasterDark {
                path: flat.to_string_lossy().into_owned(),
                exposure_seconds: Some(300.0),
            }],
            master_bias: Some(second_r_flat.to_string_lossy().into_owned()),
            paths: paths.clone(),
            // Two flats for R: that filter is left out, L travels.
            master_flats: vec![
                master_flat("L", &flat),
                master_flat("R", &flat),
                master_flat("R", &second_r_flat),
            ],
            workers: Some(3),
        },
        &sessions,
    )
    .expect("blink measurement");
    assert_eq!(
        manifest.extra["requestEcho"]["masterDarks"],
        serde_json::json!([{"path": flat, "exposureSeconds": 300.0}])
    );
    assert_eq!(
        manifest.extra["requestEcho"]["masterBias"],
        serde_json::json!(second_r_flat)
    );
    assert_eq!(manifest.kind, BLINK_MANIFEST_KIND);
    assert_eq!(
        manifest.counts,
        BlinkCounts {
            frames: 4,
            exclude: 1,
            attention: 1,
            clean: 2
        }
    );
    assert_eq!(manifest.channels.len(), 2);
    // The request transported the optional fields and the preview policy.
    assert_eq!(manifest.extra["requestEcho"]["workers"], 3);
    assert_eq!(
        manifest.extra["requestEcho"]["masterFlats"],
        serde_json::json!([{"filter": "L", "path": flat.canonicalize().unwrap()}])
    );
    assert_eq!(
        manifest.extra["requestEcho"]["previews"]["filmstripFormat"],
        "jpeg"
    );
    assert!(manifest.extra.contains_key("timings"));
    // The session lives under the sessions root with the desktop's name.
    let session = Path::new(&manifest.session_directory);
    assert_eq!(
        session.parent(),
        Some(sessions.canonicalize().unwrap().as_path())
    );
    assert!(crate::project::blink_session_name_parts(
        session.file_name().unwrap().to_str().unwrap()
    )
    .is_some());
    let manifest_file = session.join("manifest.json");
    assert!(manifest_file.is_file());
    assert_eq!(
        manifest.manifest_sha256.as_deref().unwrap(),
        format!("sha256:{}", sha256_file(&manifest_file).unwrap())
    );
    let frame = |needle: &str| {
        manifest
            .frames
            .iter()
            .find(|frame| frame.path.contains(needle))
            .expect("frame")
    };
    let reference = frame("22-57-01");
    assert!(reference.reference && reference.flags.is_empty());
    let l_channel = manifest
        .channels
        .iter()
        .find(|channel| channel.filter == "L")
        .unwrap();
    assert_eq!(l_channel.reference.index, reference.index);
    assert_eq!(l_channel.frame_count, 2);
    let moon = frame("moon");
    assert_eq!(moon.default_decision, "DROP");
    assert_eq!(moon.flags[0].code, "BLINK_SKY_BRIGHT");
    assert_eq!(moon.flags[0].extra["combined"], true);
    let haze = frame("haze");
    assert_eq!(haze.default_decision, "KEEP");
    assert_eq!(haze.flags[0].severity, "ATTENTION");
    for item in [reference, moon, haze] {
        let url = item.previews.filmstrip_data_url.as_deref().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,/9j/"));
        assert_eq!(item.extra["metrics"]["matchedStars"], 1900);
    }
    // The oversize filmstrip stays on disk for the on-demand path.
    let oversize = frame("oversize");
    assert!(oversize.previews.filmstrip_data_url.is_none());
    let zoom = oversize.previews.zoom.as_deref().unwrap();
    let loaded =
        crate::project::load_blink_preview_with(&sessions, &manifest.session_directory, zoom)
            .expect("zoom preview");
    assert!(loaded.starts_with("data:image/png;base64,iVBOR"));
    let filmstrip = oversize.previews.filmstrip.as_deref().unwrap();
    assert!(crate::project::load_blink_preview_with(
        &sessions,
        &manifest.session_directory,
        filmstrip
    )
    .is_ok());
    // Every source file is untouched and the serialised manifest keeps
    // the pass-through fields next to the typed ones.
    for path in &paths {
        assert!(std::fs::read_to_string(path)
            .unwrap()
            .starts_with("light bytes"));
    }
    let encoded = serde_json::to_value(&manifest).unwrap();
    assert_eq!(encoded["frames"][0]["score"]["rank"], 1);
    assert_eq!(encoded["channels"][0]["nights"][0]["night"], "2026-08-17");
    assert!(encoded["frames"][0]["previews"]["filmstripDataUrl"].is_string());
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn fake_sidecar_blink_sessions_are_pruned_to_the_newest_three() {
    let (root, script) = fake_sidecar();
    let paths = blink_lights(&root);
    let sessions = root.join("blink-sessions");
    std::fs::create_dir_all(sessions.join("not-ours")).unwrap();
    std::fs::write(sessions.join("not-ours/keep.txt"), b"foreign").unwrap();
    let mut directories = Vec::new();
    for _ in 0..4 {
        let manifest = blink_measure_with(
            EngineExecutable {
                path: script.clone(),
            },
            BlinkMeasureRequest {
                master_darks: vec![],
                master_bias: None,
                paths: paths.clone(),
                master_flats: vec![],
                workers: None,
            },
            &sessions,
        )
        .expect("blink measurement");
        directories.push(PathBuf::from(manifest.session_directory));
    }
    assert!(!directories[0].exists(), "oldest session removed");
    for directory in &directories[1..] {
        assert!(directory.join("manifest.json").is_file());
    }
    assert!(sessions.join("not-ours/keep.txt").is_file());
    // A refused request leaves no session behind: it names a Light twice,
    // which the request check rejects before spawning.
    let mut duplicated = paths.clone();
    duplicated.push(paths[0].clone());
    let error = blink_measure_with(
        EngineExecutable { path: script },
        BlinkMeasureRequest {
            master_darks: vec![],
            master_bias: None,
            paths: duplicated,
            master_flats: vec![],
            workers: Some(0),
        },
        &sessions,
    )
    .unwrap_err();
    assert!(error.contains("unique regular Light files"));
    assert_eq!(std::fs::read_dir(&sessions).unwrap().count(), 4);
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn fake_sidecar_blink_failure_removes_its_session() {
    use std::os::unix::fs::PermissionsExt;

    let (root, script) = fake_sidecar();
    let paths = blink_lights(&root);
    let sessions = root.join("blink-sessions");
    // A sidecar whose manifest rebinds an input: the fake script is
    // wrapped so that its stdout names a different Light.
    let wrapper = root.join("rebinding-sidecar");
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\n\"{}\" \"$@\" | sed 's/22-57-01/22-57-02/'\n",
            script.display()
        ),
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&wrapper).unwrap().permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&wrapper, permissions).unwrap();
    let error = blink_measure_with(
        EngineExecutable { path: wrapper },
        BlinkMeasureRequest {
            master_darks: vec![],
            master_bias: None,
            paths,
            master_flats: vec![],
            workers: None,
        },
        &sessions,
    )
    .unwrap_err();
    assert!(error.contains("invalid contract"), "{error}");
    assert_eq!(std::fs::read_dir(&sessions).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(root);
}

/// The real engine on real Lights: `OAF_TEST_ENGINE` names an
/// `ultra-fast-wbpp` executable, `OAF_TEST_BLINK_LIGHTS` a text file with
/// one Light path per line.  Run with `--ignored --nocapture` to see the
/// session summary.
#[test]
#[ignore = "requires OAF_TEST_ENGINE and OAF_TEST_BLINK_LIGHTS pointing to a real engine and Lights"]
fn real_engine_blink_measure_session_is_accepted() {
    let engine = PathBuf::from(std::env::var("OAF_TEST_ENGINE").expect("engine path"));
    let paths =
        std::fs::read_to_string(std::env::var("OAF_TEST_BLINK_LIGHTS").expect("light list path"))
            .expect("light list")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
    let sessions = std::env::temp_dir()
        .join(new_identifier("real-blink-sessions").expect("temporary identifier"));
    let started = std::time::Instant::now();
    let manifest = blink_measure_with(
        EngineExecutable { path: engine },
        BlinkMeasureRequest {
            master_darks: vec![],
            master_bias: None,
            paths: paths.clone(),
            master_flats: vec![],
            workers: None,
        },
        &sessions,
    )
    .expect("real blink-measure session");
    let elapsed = started.elapsed();
    assert_eq!(manifest.frames.len(), paths.len());
    let inline = manifest
        .frames
        .iter()
        .filter(|frame| frame.previews.filmstrip_data_url.is_some())
        .count();
    for frame in &manifest.frames {
        for relative in [&frame.previews.filmstrip, &frame.previews.zoom]
            .into_iter()
            .flatten()
        {
            crate::project::load_blink_preview_with(
                &sessions,
                &manifest.session_directory,
                relative,
            )
            .expect("preview loads on demand");
        }
    }
    eprintln!(
            "blink-measure: {} frames in {:.1?}, {} channels, counts {:?}, {inline} inline filmstrip previews, manifest {}",
            manifest.frames.len(),
            elapsed,
            manifest.channels.len(),
            manifest.counts,
            manifest.manifest_sha256.as_deref().unwrap_or("-"),
        );
    for channel in &manifest.channels {
        let reference = manifest
            .frames
            .iter()
            .find(|frame| frame.index == channel.reference.index)
            .expect("validated reference");
        eprintln!(
            "  {} {}: {} frames, reference {}",
            channel.target,
            channel.filter,
            channel.frame_count,
            reference
                .extra
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(&reference.path)
        );
    }
    for frame in &manifest.frames {
        eprintln!(
            "  {:>4} {:<5} {} {:?}",
            frame.index,
            frame.default_decision,
            frame
                .extra
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(&frame.path),
            frame
                .flags
                .iter()
                .map(|flag| format!("{}:{}", flag.code, flag.severity))
                .collect::<Vec<_>>()
        );
    }
    let _ = std::fs::remove_dir_all(sessions);
}

/// A schema-exact manifest over `lights` with real preview files, for the
/// validation cases below.
fn sample_blink_manifest(session: &Path, lights: &[PathBuf]) -> serde_json::Value {
    std::fs::create_dir_all(session.join("filmstrip")).unwrap();
    std::fs::create_dir_all(session.join("zoom")).unwrap();
    let jpeg: Vec<u8> = [0xff, 0xd8, 0xff, 0xe0]
        .into_iter()
        .chain([7_u8; 64])
        .collect();
    let png: Vec<u8> = crate::project::PNG_SIGNATURE
        .iter()
        .copied()
        .chain([1_u8; 32])
        .collect();
    let frames = lights
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let filmstrip = format!("filmstrip/{index:04}-L.jpg");
                let zoom = format!("zoom/{index:04}-L.png");
                std::fs::write(session.join(&filmstrip), &jpeg).unwrap();
                std::fs::write(session.join(&zoom), &png).unwrap();
                let excluded = index == 1;
                serde_json::json!({
                    "index": index, "channelId": "group-l", "filter": "L", "night": "2026-08-17",
                    "path": path, "name": path.file_name().unwrap().to_str().unwrap(),
                    "sourceSha256": format!("sha256:{}", format!("{index}").repeat(64)),
                    "reference": index == 0, "defaultDecision": if excluded { "DROP" } else { "KEEP" },
                    "flags": if excluded {
                        serde_json::json!([{"code": "BLINK_SKY_BRIGHT", "severity": "EXCLUDE", "value": 2.4, "threshold": 1.6}])
                    } else if index == 2 {
                        serde_json::json!([{"code": "BLINK_EXTINCTION", "severity": "ATTENTION", "value": 0.6, "threshold": 0.5}])
                    } else {
                        serde_json::json!([])
                    },
                    "previews": {"filmstrip": filmstrip, "zoom": zoom, "coverage": 1.0},
                })
            })
            .collect::<Vec<_>>();
    serde_json::json!({
        "schemaVersion": 1, "kind": "blink-manifest-v1", "sessionId": "session-1",
        "sessionDirectory": session, "inventorySha256": format!("sha256:{}", "a".repeat(64)),
        "gatePolicyDigest": format!("sha256:{}", "b".repeat(64)),
        "flagsPolicyDigest": format!("sha256:{}", "c".repeat(64)),
        "counts": {"frames": lights.len(), "exclude": 1, "attention": 1, "clean": lights.len() - 2},
        "channels": [{"channelId": "group-l", "target": "NGC 6822", "filter": "L", "frameCount": lights.len(),
            "reference": {"index": 0, "sourceSha256": format!("sha256:{}", "0".repeat(64)), "rule": "psf-signal-weight-proxy-v1"},
            "nights": []}],
        "frames": frames,
    })
}

#[test]
fn blink_manifest_validation_rejects_rebound_inputs_and_inconsistent_records() {
    let root = std::env::temp_dir()
        .join(new_identifier("blink-manifest-validation").expect("temporary identifier"));
    let session = root.join("session");
    std::fs::create_dir_all(&session).unwrap();
    let session = session.canonicalize().unwrap();
    let lights = (0..3)
        .map(|index| {
            let path = root.join(format!("light {index} 盾牌座.fits"));
            std::fs::write(&path, format!("light {index}")).unwrap();
            path.canonicalize().unwrap()
        })
        .collect::<Vec<_>>();
    let expected = lights.iter().cloned().collect::<BTreeSet<_>>();
    let valid = sample_blink_manifest(&session, &lights);
    let check = |value: serde_json::Value| -> Result<(), String> {
        let manifest: BlinkManifest = serde_json::from_value(value).map_err(|e| e.to_string())?;
        validate_blink_manifest(&manifest, &expected, &session)
    };
    check(valid.clone()).expect("valid manifest");
    let mutated = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut value = valid.clone();
        edit(&mut value);
        check(value)
    };
    type Edit = fn(&mut serde_json::Value);
    let cases: &[(&str, Edit)] = &[
        ("kind", |v| v["kind"] = "quality-manifest".into()),
        ("schema", |v| v["schemaVersion"] = 2.into()),
        ("session", |v| {
            v["sessionDirectory"] = v["sessionDirectory"]
                .as_str()
                .unwrap()
                .trim_end_matches("session")
                .into()
        }),
        ("digest case", |v| {
            v["flagsPolicyDigest"] = format!("sha256:{}", "C".repeat(64)).into()
        }),
        ("rebound path", |v| {
            v["frames"][2]["path"] = v["frames"][0]["path"].clone()
        }),
        ("dropped frame", |v| {
            v["frames"].as_array_mut().unwrap().pop();
            v["counts"]["frames"] = 2.into();
            v["counts"]["clean"] = 0.into();
            v["channels"][0]["frameCount"] = 2.into();
        }),
        ("duplicate index", |v| v["frames"][2]["index"] = 0.into()),
        ("frame digest", |v| {
            v["frames"][1]["sourceSha256"] = "sha256:short".into()
        }),
        ("unknown channel", |v| {
            v["frames"][1]["channelId"] = "group-r".into()
        }),
        ("severity", |v| {
            v["frames"][1]["flags"][0]["severity"] = "WARN".into()
        }),
        ("flag code", |v| {
            v["frames"][1]["flags"][0]["code"] = "sky bright".into()
        }),
        ("drop without exclude", |v| {
            v["frames"][0]["defaultDecision"] = "DROP".into()
        }),
        ("keep with exclude", |v| {
            v["frames"][1]["defaultDecision"] = "KEEP".into()
        }),
        ("decision enum", |v| {
            v["frames"][0]["defaultDecision"] = "MAYBE".into()
        }),
        ("counts", |v| {
            v["counts"]["exclude"] = 2.into();
            v["counts"]["clean"] = 0.into();
        }),
        ("escaping preview", |v| {
            v["frames"][0]["previews"]["zoom"] = "../session/zoom/0000-L.png".into()
        }),
        ("absolute preview", |v| {
            v["frames"][0]["previews"]["filmstrip"] =
                v["sessionDirectory"].as_str().unwrap().to_owned().into()
        }),
        ("missing preview", |v| {
            v["frames"][0]["previews"]["filmstrip"] = "filmstrip/9999-L.jpg".into()
        }),
        ("preview extension", |v| {
            v["frames"][0]["previews"]["filmstrip"] = "filmstrip/0000-L.txt".into()
        }),
        ("reference index", |v| {
            v["channels"][0]["reference"]["index"] = 2.into()
        }),
        ("reference digest", |v| {
            v["channels"][0]["reference"]["sourceSha256"] =
                format!("sha256:{}", "9".repeat(64)).into()
        }),
        ("two references", |v| {
            v["frames"][2]["reference"] = true.into()
        }),
        ("channel count", |v| {
            v["channels"][0]["frameCount"] = 2.into()
        }),
        ("duplicate channel", |v| {
            let channel = v["channels"][0].clone();
            v["channels"].as_array_mut().unwrap().push(channel);
        }),
    ];
    for &(name, edit) in cases {
        assert!(mutated(&edit).is_err(), "{name} must be rejected");
    }
    // Null previews are allowed (a frame the renderer skipped is shown
    // without an image); unknown fields pass through.
    mutated(&|v| {
        v["frames"][0]["previews"]["filmstrip"] = serde_json::Value::Null;
        v["frames"][0]["previews"]["zoom"] = serde_json::Value::Null;
        v["frames"][0]["metrics"] = serde_json::json!({"sky": 986.0});
    })
    .expect("null previews and extra fields");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn blink_filmstrip_previews_respect_the_transport_budget() {
    let root = std::env::temp_dir()
        .join(new_identifier("blink-preview-budget").expect("temporary identifier"));
    let session = root.join("session");
    std::fs::create_dir_all(&session).unwrap();
    let session = session.canonicalize().unwrap();
    let lights = (0..3)
        .map(|index| {
            let path = root.join(format!("light{index}.fits"));
            std::fs::write(&path, b"light").unwrap();
            path.canonicalize().unwrap()
        })
        .collect::<Vec<_>>();
    let mut manifest: BlinkManifest =
        serde_json::from_value(sample_blink_manifest(&session, &lights)).unwrap();
    // The second filmstrip is a PNG under a .jpg name: not carried.
    std::fs::write(
        session.join("filmstrip/0001-L.jpg"),
        crate::project::PNG_SIGNATURE,
    )
    .unwrap();
    let one_preview = "data:image/jpeg;base64,".len() + 68_usize.div_ceil(3) * 4;
    attach_blink_previews(&mut manifest, &session, one_preview * 2 - 1);
    let urls = manifest
        .frames
        .iter()
        .map(|frame| frame.previews.filmstrip_data_url.is_some())
        .collect::<Vec<_>>();
    assert_eq!(urls, vec![true, false, false]);
    assert_eq!(
        manifest.frames[0]
            .previews
            .filmstrip_data_url
            .as_deref()
            .unwrap()
            .len(),
        one_preview
    );
    let mut generous: BlinkManifest =
        serde_json::from_value(sample_blink_manifest(&session, &lights)).unwrap();
    attach_blink_previews(&mut generous, &session, MAX_BLINK_TRANSPORT_BYTES);
    assert!(generous
        .frames
        .iter()
        .all(|frame| frame.previews.filmstrip_data_url.is_some()));
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn fake_sidecar_error_is_reported_without_parsing_stderr_as_protocol() {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir()
        .join(new_identifier("openastroflow-error-sidecar").expect("temporary identifier"));
    std::fs::create_dir(&root).expect("create fake sidecar directory");
    let script = root.join("error-sidecar");
    std::fs::write(
        &script,
        "#!/bin/sh\necho 'diagnostic only, not NDJSON' >&2\nexit 7\n",
    )
    .expect("write error sidecar");
    let mut permissions = std::fs::metadata(&script).unwrap().permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&script, permissions).unwrap();
    let error = inspect_paths_with(
        EngineExecutable { path: script },
        InspectRequest {
            paths: vec!["/tmp/input".to_owned()],
            role_hint: None,
        },
    )
    .expect_err("failing sidecar must fail inventory");
    assert!(error.contains("diagnostic only, not NDJSON"));
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn fake_sidecar_plan_execute_reaches_ready_gate_and_verified_artifact() {
    use std::sync::mpsc;
    use tauri::Listener;

    let (root, script) = fake_sidecar();
    let output_parent = root.join("Unicode 输出父目录");
    std::fs::create_dir(&output_parent).expect("output parent");
    let executable = EngineExecutable { path: script };
    let probe = probe_runtime_with(executable).expect("canonical handshake");
    let app = tauri::test::mock_app();
    let handle = app.handle().clone();
    let (sender, receiver) = mpsc::channel();
    handle.listen(COMPLETE_EVENT, move |event| {
        let _ = sender.send(event.payload().to_owned());
    });
    let registry = Arc::new(PipelineRegistry::default());
    let receipt = start_pipeline_with_probe(
        handle,
        registry,
        RunRequest {
            sources: vec![
                RunSource {
                    role: "LIGHT".to_owned(),
                    paths: vec!["/输入/盾牌座_light.fit".to_owned()],
                },
                RunSource {
                    role: "FLAT".to_owned(),
                    paths: vec!["/输入/校准_flat.fit".to_owned()],
                },
                RunSource {
                    role: "BIAS".to_owned(),
                    paths: vec!["/输入/校准_bias.fit".to_owned()],
                },
            ],
            recipe_id: "balanced".to_owned(),
            output_parent_directory: output_parent.to_string_lossy().into_owned(),
        },
        probe,
    )
    .expect("start fake canonical worker");
    assert_eq!(receipt.execution_mode, "native");
    let payload = receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("ready completion event");
    let value: serde_json::Value = serde_json::from_str(&payload).expect("completion JSON");
    assert_eq!(value["gate"]["decision"], "ready");
    assert_eq!(
        value["artifacts"][0]["receipt"]["astrometry"]["matchedStars"],
        73
    );
    assert!(Path::new(value["artifacts"][0]["path"].as_str().unwrap()).is_file());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn cancellation_terminates_the_worker_process_tree() {
    let mut command = platform::test_support::sleeping_command();
    let child = ManagedChild::spawn(&mut command).expect("spawn cancellable process");
    let id = "cancel-test".to_owned();
    let registry = PipelineRegistry::default();
    registry
        .jobs
        .lock()
        .unwrap()
        .insert(id.clone(), Arc::new(Mutex::new(child)));
    cancel_pipeline(&registry, &id).expect("kill worker process group");
    let status = registry.jobs.lock().unwrap()[&id]
        .lock()
        .unwrap()
        .wait()
        .expect("wait cancelled process");
    assert!(!status.success());
}
