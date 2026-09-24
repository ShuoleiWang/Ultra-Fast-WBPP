use super::*;

#[test]
fn output_directory_label_keeps_designations_and_readable_names() {
    assert_eq!(output_directory_label("NGC 7331"), "NGC7331");
    assert_eq!(output_directory_label("  M 31  "), "M31");
    assert_eq!(output_directory_label("Sh2-155 / Cave"), "Sh2-155-Cave");
    assert_eq!(output_directory_label("盾牌座 马赛克"), "盾牌座-马赛克");
    assert_eq!(
        output_directory_label("Ultra-Fast WBPP project"),
        "Ultra-Fast-WBPP-project"
    );
    assert_eq!(output_directory_label("../..//"), "wbpp");
    assert_eq!(output_directory_label(""), "wbpp");
    assert_eq!(output_directory_label(&"x".repeat(80)).chars().count(), 48);
}

#[test]
fn unique_output_directory_adds_a_counter_on_collision() {
    let root = std::env::temp_dir().join(new_public_identifier("output-name-test").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let now = chrono::Local::now();
    let first = unique_output_directory(&root, "NGC 7331", now).unwrap();
    let expected = format!("NGC7331_{}", now.format("%Y-%m-%d_%H%M"));
    assert_eq!(first.file_name().unwrap().to_string_lossy(), expected);
    std::fs::create_dir_all(&first).unwrap();
    let second = unique_output_directory(&root, "NGC 7331", now).unwrap();
    assert_eq!(
        second.file_name().unwrap().to_string_lossy(),
        format!("{expected}_2")
    );
    std::fs::create_dir_all(second.with_extension("unsolved")).unwrap();
    let third = unique_output_directory(&root, "NGC 7331", now).unwrap();
    assert_eq!(
        third.file_name().unwrap().to_string_lossy(),
        format!("{expected}_3")
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn selection_request(root: &Path, light: &Path, selection: serde_json::Value) -> ProjectRunRequest {
    serde_json::from_value(serde_json::json!({
        "sources": [{"sourceId": "light-1", "role": "LIGHT", "paths": [light], "recursive": false}],
        "projectName": "NGC 6822", "runLabel": "NGC 6822",
        "recipe": {"balanced": true, "drizzleEnabled": false,
                   "solverRequired": true, "calibrationWorkflow": "mono-standard-v1"},
        "masterMetadataOverrides": [], "rawFrameMetadataOverrides": [], "reviewSelections": [],
        "selection": selection,
        "outputParentDirectory": root,
    }))
    .expect("request JSON")
}

fn sample_selection() -> serde_json::Value {
    serde_json::json!({
        "schemaVersion": 1, "kind": "ultra-fast-wbpp-selection", "policy": "explicit-v1",
        "origin": {"sessionId": "abc-20260922-101010", "blinkManifestSha256": format!("sha256:{}", "d".repeat(64)),
                   "flagsPolicyDigest": format!("sha256:{}", "e".repeat(64)), "createdAt": "2026-09-22T10:10:10"},
        "undecided": "ERROR",
        "decisions": [
            {"sourceSha256": format!("sha256:{}", "1".repeat(64)), "decision": "KEEP", "defaultDecision": "DROP",
             "flags": ["BLINK_SKY_BRIGHT", "BLINK_SOURCES_LOW"], "note": "user: kept, faint gradient acceptable"},
            {"sourceSha256": format!("sha256:{}", "2".repeat(64)), "decision": "DROP"},
            {"sourceSha256": format!("sha256:{}", "3".repeat(64)), "decision": "KEEP", "flags": []}
        ]
    })
}

#[test]
fn blink_selection_travels_top_level_and_is_validated() {
    let root = std::env::temp_dir().join(new_public_identifier("selection-wire").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let light = root.join("light 盾牌座.fits");
    std::fs::write(&light, b"light input").unwrap();
    let output = root.join("new-output");
    let request = selection_request(&root, &light, sample_selection());
    let value = project_request_json(&request, &output).unwrap();
    assert_eq!(value["selection"], sample_selection());
    assert_eq!(value["reviewSelections"], serde_json::json!([]));
    assert_eq!(value["recipe"]["reviewApprovals"], serde_json::json!([]));
    // Without a blink session the key is absent, not null: the engine's
    // request loader treats the key as the switch to explicit-v1.
    let mut legacy = selection_request(&root, &light, serde_json::Value::Null);
    legacy.selection = None;
    assert!(project_request_json(&legacy, &output)
        .unwrap()
        .get("selection")
        .is_none());
    // Legacy REVIEW approvals and a blink selection are two policies.
    let mut conflict = selection_request(&root, &light, sample_selection());
    conflict.review_selections.push(UiReviewSelection {
        source_sha256: format!("sha256:{}", "1".repeat(64)),
        gate_policy_digest: format!("sha256:{}", "f".repeat(64)),
    });
    assert!(project_request_json(&conflict, &output)
        .unwrap_err()
        .starts_with("SELECTION_POLICY_CONFLICT"));
    let rejected = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut selection = sample_selection();
        edit(&mut selection);
        let request = selection_request(&root, &light, selection);
        project_request_json(&request, &output).unwrap_err()
    };
    type Edit = fn(&mut serde_json::Value);
    let cases: &[(&str, Edit)] = &[
        ("kind", |v| v["kind"] = "selection".into()),
        ("schema", |v| v["schemaVersion"] = 2.into()),
        ("policy", |v| v["policy"] = "unattended-v1".into()),
        ("undecided", |v| v["undecided"] = "SKIP".into()),
        ("decision", |v| {
            v["decisions"][1]["decision"] = "MAYBE".into()
        }),
        ("default decision", |v| {
            v["decisions"][1]["defaultDecision"] = "REVIEW".into()
        }),
        ("duplicate digest", |v| {
            v["decisions"][1]["sourceSha256"] = v["decisions"][0]["sourceSha256"].clone()
        }),
        ("uppercase digest", |v| {
            v["decisions"][1]["sourceSha256"] = format!("sha256:{}", "A".repeat(64)).into()
        }),
        ("bare digest", |v| {
            v["decisions"][1]["sourceSha256"] = "2".repeat(64).into()
        }),
        ("flag code", |v| {
            v["decisions"][0]["flags"][0] = "sky bright".into()
        }),
        ("note length", |v| {
            v["decisions"][0]["note"] = "x".repeat(1001).into()
        }),
        ("origin digest", |v| {
            v["origin"]["blinkManifestSha256"] = "sha256:nope".into()
        }),
        ("empty", |v| v["decisions"] = serde_json::json!([])),
        ("too many", |v| {
            v["decisions"] = (0..10_001)
                    .map(|index| serde_json::json!({"sourceSha256": format!("sha256:{index:064x}"), "decision": "KEEP"}))
                    .collect();
        }),
    ];
    for &(name, edit) in cases {
        let error = rejected(&edit);
        assert!(error.starts_with("SELECTION_INVALID"), "{name}: {error}");
    }
    // Unknown fields are refused at the boundary, before validation.
    let mut unknown = sample_selection();
    unknown["decisions"][0]["reason"] = "why".into();
    assert!(serde_json::from_value::<UiSelection>(unknown).is_err());
    let mut unknown_origin = sample_selection();
    unknown_origin["origin"]["user"] = "me".into();
    assert!(serde_json::from_value::<UiSelection>(unknown_origin).is_err());
    // A hand-written file without origin, notes or flags is valid.
    let minimal = serde_json::json!({
        "schemaVersion": 1, "kind": "ultra-fast-wbpp-selection", "policy": "explicit-v1", "undecided": "DROP",
        "decisions": [{"sourceSha256": format!("sha256:{}", "1".repeat(64)), "decision": "KEEP"}]
    });
    let request = selection_request(&root, &light, minimal.clone());
    assert_eq!(
        project_request_json(&request, &output).unwrap()["selection"],
        minimal
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn screening_summary_carries_user_selection_reasons_and_flags() {
    let root = std::env::temp_dir().join(new_public_identifier("screening-reasons").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    let receipt = serde_json::json!({"execution": {"screening": {
        "admitted": 62, "excluded": 35, "counts": {"PASS": 83, "REVIEW": 13, "HARD_FAIL": 1},
        "frames": [
            {"path": "source/src-1/NGC 6822_300.00s_L_moon.fits", "disposition": "PASS", "admitted": false,
             "summary": "dropped by you", "evidence": [], "starCount": 2947,
             "reason": "USER_DROP", "flags": ["BLINK_SKY_BRIGHT", "BLINK_SOURCES_LOW", "not a code"]},
            {"path": "source/src-2/NGC 6822_300.00s_G_cloud.fits", "disposition": "REVIEW", "admitted": true,
             "summary": "restored by you", "evidence": ["cloud"], "reason": "USER_KEEP_OVERRIDE", "flags": ["BLINK_SOURCES_LOW"]},
            {"path": "source/src-3/legacy.fits", "disposition": "HARD_FAIL", "admitted": false, "summary": "trail",
             "reason": null}
        ]
    }}});
    let screening = screening_summary(&root, &receipt).unwrap().unwrap();
    assert_eq!(screening.frames[0].reason.as_deref(), Some("USER_DROP"));
    assert_eq!(
        screening.frames[0].flags,
        vec!["BLINK_SKY_BRIGHT", "BLINK_SOURCES_LOW"]
    );
    assert!(!screening.frames[0].admitted);
    assert_eq!(
        screening.frames[1].reason.as_deref(),
        Some("USER_KEEP_OVERRIDE")
    );
    assert!(screening.frames[2].reason.is_none() && screening.frames[2].flags.is_empty());
    let encoded = serde_json::to_value(&screening).unwrap();
    assert_eq!(encoded["frames"][0]["reason"], "USER_DROP");
    assert!(encoded["frames"][2].get("reason").is_none());
    assert!(encoded["frames"][2].get("flags").is_none());
    let unknown = serde_json::json!({"execution": {"screening": {"admitted": 1, "excluded": 0, "counts": {},
            "frames": [{"path": "x", "disposition": "PASS", "reason": "GATE"}]}}});
    assert!(screening_summary(&root, &unknown).is_err());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn blink_preview_loader_stays_inside_its_own_session() {
    let root = std::env::temp_dir().join(new_public_identifier("blink-preview-loader").unwrap());
    let sessions = root.join("blink-sessions");
    let session = sessions.join(format!("{}-20260922-101010", "a".repeat(16)));
    std::fs::create_dir_all(session.join("zoom")).unwrap();
    std::fs::create_dir_all(session.join("filmstrip")).unwrap();
    let png: Vec<u8> = PNG_SIGNATURE.iter().copied().chain([1_u8; 32]).collect();
    let jpeg: Vec<u8> = JPEG_SIGNATURE.iter().copied().chain([2_u8; 32]).collect();
    std::fs::write(session.join("zoom/0001-L.png"), &png).unwrap();
    std::fs::write(session.join("filmstrip/0001-L.jpg"), &jpeg).unwrap();
    std::fs::write(session.join("zoom/0002-L.png"), &jpeg).unwrap();
    std::fs::write(session.join("zoom/0003-L.jpg"), &jpeg).unwrap();
    std::fs::write(session.join("zoom/notes.txt"), b"text").unwrap();
    let mut large = png.clone();
    large.resize(2 * 1024 * 1024 + 1, 0);
    std::fs::write(session.join("zoom/large.png"), &large).unwrap();
    std::fs::write(root.join("outside.png"), &png).unwrap();
    let session_text = session.to_string_lossy().into_owned();
    let load =
        |directory: &str, relative: &str| load_blink_preview_with(&sessions, directory, relative);
    assert_eq!(
        load(&session_text, "zoom/0001-L.png").unwrap(),
        format!("data:image/png;base64,{}", base64_encode(&png))
    );
    assert_eq!(
        load(&session_text, "filmstrip/0001-L.jpg").unwrap(),
        format!("data:image/jpeg;base64,{}", base64_encode(&jpeg))
    );
    // The extension declares the media type; the bytes must agree.
    assert!(load(&session_text, "zoom/0002-L.png").is_err());
    assert!(load(&session_text, "zoom/0003-L.jpg").is_ok());
    for relative in [
        "zoom/notes.txt",
        "zoom/large.png",
        "../outside.png",
        "zoom/../../outside.png",
        "zoom/../zoom/0001-L.png",
        "",
        "zoom",
        "zoom/missing.png",
    ] {
        assert!(load(&session_text, relative).is_err(), "{relative}");
    }
    assert!(load(root.join("outside.png").to_str().unwrap(), "").is_err());
    assert!(load(
        &session_text,
        session.join("zoom/0001-L.png").to_str().unwrap()
    )
    .is_err());
    // Only the desktop's own session directories, directly under the root.
    assert!(load(root.to_str().unwrap(), "outside.png").is_err());
    assert!(load(sessions.to_str().unwrap(), "outside.png").is_err());
    let foreign = sessions.join("not-a-session");
    std::fs::create_dir_all(foreign.join("zoom")).unwrap();
    std::fs::write(foreign.join("zoom/0001-L.png"), &png).unwrap();
    assert!(load(foreign.to_str().unwrap(), "zoom/0001-L.png").is_err());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(root.join("outside.png"), session.join("zoom/link.png"))
            .unwrap();
        assert!(load(&session_text, "zoom/link.png").is_err());
        std::os::unix::fs::symlink(session.join("zoom"), session.join("linked")).unwrap();
        assert!(load(&session_text, "linked/0001-L.png").is_err());
    }
    for (name, valid) in [
        ("0123456789abcdef-20260922-101010", true),
        ("0123456789abcdef-20260922-101010-2", true),
        ("0123456789ABCDEF-20260922-101010", false),
        ("0123456789abcdef-20260922", false),
        ("0123456789abcdef-20260922-101010-", false),
        ("0123456789abcdef-20260922-101010-2-3", false),
        ("not-a-session", false),
    ] {
        assert_eq!(blink_session_name_parts(name).is_some(), valid, "{name}");
    }
    assert!(
        blink_session_name_parts("0123456789abcdef-20260922-101010")
            < blink_session_name_parts("0123456789abcdef-20260922-101010-2")
    );
    assert!(
        blink_session_name_parts("0123456789abcdef-20260922-101010-2")
            < blink_session_name_parts("0123456789abcdef-20260922-101011")
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn base64_encodes_the_reference_vectors() {
    for (input, expected) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(base64_encode(input.as_bytes()), expected);
    }
    assert_eq!(base64_encode(&[0xff, 0xee, 0xdd, 0x00]), "/+7dAA==");
}

#[test]
fn screening_summary_loads_bounded_previews_and_rejects_malformed_records() {
    let root = std::env::temp_dir().join(new_public_identifier("screening-test").unwrap());
    let review = root.join("details/runs/NGC7331/qc/review");
    std::fs::create_dir_all(&review).unwrap();
    // Previews resolve against the canonical output root, as in production.
    let root = root.canonicalize().unwrap();
    let png: Vec<u8> = PNG_SIGNATURE.iter().copied().chain([1_u8; 32]).collect();
    std::fs::write(review.join("0001-cloudy.png"), &png).unwrap();
    std::fs::write(review.join("0002-not-a-png.png"), b"plain text").unwrap();
    let receipt = serde_json::json!({"execution": {"screening": {
        "admitted": 61, "excluded": 2,
        "counts": {"PASS": 61, "REVIEW": 1, "HARD_FAIL": 1},
        "frames": [
            {"path": "source/src-1/NGC 7331_300.00s_L_cloudy.fits", "disposition": "HARD_FAIL", "admitted": false,
             "summary": "clouds", "evidence": ["star count collapsed", "background rose"], "starCount": 12,
             "reviewPreview": "details/runs/NGC7331/qc/review/0001-cloudy.png", "target": "NGC 7331"},
            {"path": "source/src-2/trail.fits", "disposition": "REVIEW", "admitted": true,
             "summary": "trail", "evidence": [], "starCount": null,
             "reviewPreview": "details/runs/NGC7331/qc/review/0002-not-a-png.png"},
            {"path": "source/src-3/missing.fits", "disposition": "REVIEW", "admitted": false,
             "summary": "", "reviewPreview": "../escaped.png"}
        ]
    }}});
    let screening = screening_summary(&root, &receipt).unwrap().unwrap();
    assert_eq!((screening.admitted, screening.excluded), (61, 2));
    assert_eq!(screening.counts["REVIEW"], 1);
    assert_eq!(screening.frames.len(), 3);
    let cloudy = &screening.frames[0];
    assert_eq!(cloudy.name, "NGC 7331_300.00s_L_cloudy.fits");
    assert_eq!(cloudy.target.as_deref(), Some("NGC 7331"));
    assert_eq!(cloudy.star_count, Some(12));
    assert_eq!(cloudy.evidence.len(), 2);
    let preview = cloudy.preview_data_url.as_deref().unwrap();
    assert_eq!(
        preview,
        format!("data:image/png;base64,{}", base64_encode(&png))
    );
    // Not a PNG, and a path escaping the output: no preview, frame kept.
    assert!(screening.frames[1].preview_data_url.is_none() && screening.frames[1].admitted);
    assert!(screening.frames[2].preview_data_url.is_none());
    assert!(
        screening_summary(&root, &serde_json::json!({"execution": {}}))
            .unwrap()
            .is_none()
    );
    let bad = serde_json::json!({"execution": {"screening": {"admitted": 1, "excluded": 0, "counts": {},
            "frames": [{"path": "x", "disposition": "MAYBE"}]}}});
    assert!(screening_summary(&root, &bad).is_err());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
#[ignore = "requires UFWBPP_TEST_PROJECT_RECEIPT pointing to retained real project output"]
fn retained_real_project_passes_native_final_gate_read_only() {
    let receipt_path =
        PathBuf::from(std::env::var("UFWBPP_TEST_PROJECT_RECEIPT").expect("retained receipt path"));
    let root = receipt_path.parent().unwrap();
    let result = serde_json::json!({"success":true, "state":"SOLVED", "outputDirectory":root, "receiptPath":receipt_path});
    let completion = validate_completion(root, &result)
        .expect("real project must pass native final artifact validation");
    let artifacts = completion.artifacts;
    assert_eq!(artifacts.len(), 12); // 11 products plus the outer receipt.
    let screening = completion
        .screening
        .expect("a current receipt records the run's screening");
    assert_eq!(
        screening.admitted + screening.excluded,
        screening.counts.values().sum::<u64>()
    );
    assert!(screening.frames.iter().all(|frame| frame
        .preview_data_url
        .as_deref()
        .is_some_and(|url| url.starts_with("data:image/png;base64,"))));
    println!(
        "Screening: {} admitted, {} excluded, {} frames with previews.",
        screening.admitted,
        screening.excluded,
        screening.frames.len()
    );
    assert_eq!(
        artifacts
            .iter()
            .filter(|item| item.kind == "SOLVED_MONO_FITS")
            .count(),
        4
    );
    assert_eq!(
        artifacts
            .iter()
            .filter(|item| item.kind == "LINEAR_RGB_FITS")
            .count(),
        1
    );
    assert!(artifacts
        .iter()
        .all(|item| checked_sha256(&item.receipt.sha256)));
    println!("Validated 11 retained real project products plus receipt; all GUI SHA-256 values normalized to bare hex.");
}

#[test]
fn final_gate_accepts_known_hash_forms_and_rejects_malformed_or_changed_content() {
    let root = std::env::temp_dir().join(new_public_identifier("project-hash-test").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let artifact_path = root.join("product.fits");
    std::fs::write(&artifact_path, b"synthetic solved product").unwrap();
    let digest = sha256_file(&artifact_path).unwrap();
    let receipt_path = root.join("receipt.json");
    let result = serde_json::json!({"success":true,"state":"SOLVED","outputDirectory":root,"receiptPath":receipt_path});
    let mut receipt = serde_json::json!({"success":true,"state":"SOLVED","finalProducts":{
        "resultGate":{"status":"PASS","allMonoProductsSolved":true,"managedCatalogEvidenceRequired":true,"sourceIdentityVerifiedAtCommit":true,"mosaicCoverageOverlapSeamPassed":true},
        "guiArtifacts":[{"kind":"SOLVED_MONO_FITS","path":"product.fits","sha256":digest,"sizeBytes":artifact_path.metadata().unwrap().len(),"finalGate":{"status":"PASS"},"astrometry":{
            "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,
            "pixelScaleArcsec":1.4,"rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,
            "parity":"POSITIVE","catalogIdentity":"2".repeat(64),"indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],
            "correspondenceSha256":"3".repeat(64),"catalogManaged":true,"installedSetIdentity":"5".repeat(64),"catalogManifestSha256":"6".repeat(64),
            "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7".repeat(64),"manifestSha256":"6".repeat(64),"installedSetIdentity":"5".repeat(64)}],
            "wcsSha256":"4".repeat(64),"imageShape":[4176,6248],"state":"SOLVED"
        }}]
    }});
    let validate = |receipt: &serde_json::Value| {
        std::fs::write(&receipt_path, serde_json::to_vec(receipt).unwrap()).unwrap();
        validate_completion(&root, &result).map(|completion| completion.artifacts)
    };
    for value in [digest.clone(), format!("sha256:{digest}")] {
        receipt["finalProducts"]["guiArtifacts"][0]["sha256"] = value.into();
        let artifacts = validate(&receipt).unwrap();
        assert_eq!(artifacts[0].receipt.sha256, digest);
    }
    for value in [
        format!("SHA256:{digest}"),
        format!("sha256:sha256:{digest}"),
        format!("sha256:{digest} "),
        format!("sha256:{}", "g".repeat(64)),
        format!("sha256:{}", "a".repeat(63)),
    ] {
        receipt["finalProducts"]["guiArtifacts"][0]["sha256"] = value.into();
        assert!(validate(&receipt)
            .unwrap_err()
            .contains("SHA-256 identity is malformed"));
    }
    receipt["finalProducts"]["guiArtifacts"][0]["sha256"] = format!("sha256:{digest}").into();
    std::fs::write(&artifact_path, b"Synthetic solved product").unwrap(); // Same size, different hash.
    assert!(validate(&receipt)
        .unwrap_err()
        .contains("content identity changed"));
    std::fs::write(&artifact_path, b"synthetic solved product").unwrap();
    receipt["finalProducts"]["guiArtifacts"][0]["sizeBytes"] = 1.into();
    assert!(validate(&receipt)
        .unwrap_err()
        .contains("content identity changed"));
    receipt["finalProducts"]["guiArtifacts"][0]["sizeBytes"] =
        artifact_path.metadata().unwrap().len().into();
    receipt["finalProducts"]["guiArtifacts"][0]["astrometry"]["catalogManaged"] = false.into();
    assert!(validate(&receipt).is_err());
    receipt["finalProducts"]["guiArtifacts"][0]["astrometry"]["catalogManaged"] = true.into();
    receipt["finalProducts"]["guiArtifacts"][0]["path"] = "../escaped.fits".into();
    assert!(validate(&receipt)
        .unwrap_err()
        .contains("unsafe relative path"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn progress_stream_survives_an_invalid_byte_and_keeps_later_events() {
    use std::sync::mpsc;
    use tauri::Listener;

    let app = tauri::test::mock_app();
    let handle = app.handle().clone();
    let (sender, receiver) = mpsc::channel();
    handle.listen(PROGRESS_EVENT, move |event| {
        let _ = sender.send(event.payload().to_owned());
    });
    let mut stream = Vec::new();
    stream.extend_from_slice(br#"{"type":"progress","event":{"stage":"quality-control","status":"RUNNING","current":1,"total":4}}"#);
    stream.extend_from_slice(b"\nASTAP console \xff\xfe garbage\r\n");
    stream.extend_from_slice(br#"{"type":"progress","event":{"stage":"registration","status":"RUNNING","current":2,"total":4}}"#);
    stream.extend_from_slice(b"\nTraceback tail \xc3\xa9\n");
    let diagnostics = stream_progress(handle, "job".to_owned(), std::io::Cursor::new(stream))
        .join()
        .expect("progress thread");
    let first: serde_json::Value = serde_json::from_str(
        &receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("first progress event"),
    )
    .unwrap();
    assert_eq!(first["stageId"], "quality-control");
    let second: serde_json::Value = serde_json::from_str(
        &receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("progress after the invalid byte"),
    )
    .unwrap();
    assert_eq!(second["stageId"], "register");
    assert_eq!(second["fraction"], 0.5);
    assert!(receiver.try_recv().is_err());
    assert_eq!(
        diagnostics,
        "ASTAP console \u{fffd}\u{fffd} garbage\nTraceback tail \u{e9}\n"
    );
}

#[test]
fn artifact_previews_are_carried_as_bounded_data_urls() {
    let root = std::env::temp_dir().join(new_public_identifier("artifact-preview-test").unwrap());
    std::fs::create_dir_all(root.join("previews")).unwrap();
    std::fs::create_dir_all(root.join("products/color")).unwrap();
    let root = root.canonicalize().unwrap();
    let product = root.join("products/product_L.fits");
    std::fs::write(&product, b"synthetic solved product").unwrap();
    let small_png: Vec<u8> = PNG_SIGNATURE.iter().copied().chain([7_u8; 64]).collect();
    let large_png: Vec<u8> = PNG_SIGNATURE
        .iter()
        .copied()
        .chain(std::iter::repeat_n(9_u8, MAX_MONO_PREVIEW_BYTES as usize))
        .collect();
    let mono_preview = root.join("previews/product_L.png");
    let oversized_preview = root.join("previews/product_R.png");
    let rgb_preview = root.join("products/color/preview-16bit.png");
    std::fs::write(&mono_preview, &small_png).unwrap();
    std::fs::write(&oversized_preview, &large_png).unwrap();
    std::fs::write(&rgb_preview, &small_png).unwrap();
    let record = |kind: &str, path: &Path, extra: serde_json::Value| {
        let mut value = serde_json::json!({
            "kind": kind,
            "path": path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/"),
            "sha256": sha256_file(path).unwrap(),
            "sizeBytes": path.metadata().unwrap().len(),
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        value
    };
    let astrometry = serde_json::json!({
        "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,
        "pixelScaleArcsec":1.4,"rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,
        "parity":"POSITIVE","catalogIdentity":"2".repeat(64),"indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],
        "correspondenceSha256":"3".repeat(64),"catalogManaged":true,"installedSetIdentity":"5".repeat(64),"catalogManifestSha256":"6".repeat(64),
        "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7".repeat(64),"manifestSha256":"6".repeat(64),"installedSetIdentity":"5".repeat(64)}],
        "wcsSha256":"4".repeat(64)
    });
    let receipt = serde_json::json!({"success":true,"state":"SOLVED","finalProducts":{
        "resultGate":{"status":"PASS","allMonoProductsSolved":true,"managedCatalogEvidenceRequired":true,"sourceIdentityVerifiedAtCommit":true,"mosaicCoverageOverlapSeamPassed":true},
        "guiArtifacts":[
            record("SOLVED_MONO_FITS", &product, serde_json::json!({"filter":"L","finalGate":{"status":"PASS"},"astrometry":astrometry})),
            record("MONO_PREVIEW_PNG", &mono_preview, serde_json::json!({"filter":"L"})),
            record("MONO_PREVIEW_PNG", &oversized_preview, serde_json::json!({"filter":"R"})),
            record("RGB_PREVIEW_PNG_16", &rgb_preview, serde_json::json!({})),
        ]
    }});
    let receipt_path = root.join("receipt.json");
    std::fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    let result = serde_json::json!({"success":true,"state":"SOLVED","outputDirectory":root,"receiptPath":receipt_path});
    let artifacts = validate_completion(&root, &result).unwrap().artifacts;
    let by_kind = |kind: &str, filter: Option<&str>| {
        artifacts
            .iter()
            .find(|item| item.kind == kind && item.filter.as_deref() == filter)
            .unwrap()
    };
    let expected = format!("data:image/png;base64,{}", base64_encode(&small_png));
    assert_eq!(
        by_kind("MONO_PREVIEW_PNG", Some("L"))
            .preview_data_url
            .as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(
        by_kind("RGB_PREVIEW_PNG_16", None)
            .preview_data_url
            .as_deref(),
        Some(expected.as_str())
    );
    // Over the per-file bound: the file stays on disk only.
    assert!(by_kind("MONO_PREVIEW_PNG", Some("R"))
        .preview_data_url
        .is_none());
    assert!(by_kind("SOLVED_MONO_FITS", Some("L"))
        .preview_data_url
        .is_none());
    assert!(by_kind("RECEIPT", None).preview_data_url.is_none());
    let serialized = serde_json::to_value(&artifacts).unwrap();
    assert!(serialized[0].get("previewDataUrl").is_none());
    assert_eq!(serialized[1]["previewDataUrl"], expected);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn project_progress_preserves_panel_context_and_reserves_completion() {
    let value = serde_json::json!({"stage":"complete", "status":"completed", "current":9, "total":2,
            "overallFraction":1.2, "scope":"panel", "panelId":"cartwheel__b", "panelTarget":"Cartwheel",
            "panelFilter":"B", "panelIndex":1, "panelCount":4});
    let event = normalize_progress("job", &value);
    assert_eq!(event.stage_id.as_deref(), Some("publish"));
    assert_eq!(event.fraction, 1.0);
    assert_eq!(event.overall_fraction, Some(0.99));
    assert_eq!(event.completed_units, Some(2));
    let serialized = serde_json::to_value(event).unwrap();
    assert_eq!(serialized["scope"], "panel");
    assert_eq!(serialized["panelFilter"], "B");
    assert_eq!(serialized["panelIndex"], 1);
    assert_eq!(serialized["panelCount"], 4);
}

#[test]
fn progress_normalization_preserves_legacy_counts_and_failure_state() {
    let legacy = normalize_progress(
        "job",
        &serde_json::json!({"stage":"quality-control", "status":"running", "current":1, "total":2}),
    );
    assert_eq!(legacy.fraction, 0.5);
    assert_eq!(legacy.state, "running");
    assert!(legacy.overall_fraction.is_none());
    assert!(serde_json::to_value(legacy)
        .unwrap()
        .get("panelId")
        .is_none());
    let failed = normalize_progress(
        "job",
        &serde_json::json!({"stage":"failed", "status":"completed", "overallFraction":-1.0}),
    );
    assert_eq!(failed.state, "failed");
    assert_eq!(failed.overall_fraction, Some(0.0));
    assert_eq!(failed.fraction, 0.0);
    for stage in ["alignment", "color", "verify", "publish"] {
        assert_eq!(stage_id(stage).as_deref(), Some(stage));
    }
}

#[test]
fn execution_failure_prefers_structured_reason_and_keeps_fallbacks() {
    let result = serde_json::json!({
        "code": "QC_INSUFFICIENT_LIGHTS",
        "message": "B: quality gate admitted 1 Light frame; at least 2 are required"
    });
    assert_eq!(
        execution_failure_detail(&result, "unrelated diagnostic"),
        result["message"].as_str().unwrap()
    );
    assert_eq!(
        execution_failure_detail(&serde_json::json!({"message": "  "}), " legacy detail "),
        "legacy detail"
    );
    assert_eq!(
        execution_failure_detail(&serde_json::json!({}), ""),
        "project execution failed closed"
    );
    assert_eq!(
        execution_failure_detail(
            &serde_json::json!({"message": "x".repeat(MAX_DIAGNOSTIC_BYTES + 1)}),
            ""
        )
        .len(),
        MAX_DIAGNOSTIC_BYTES
    );
}

#[cfg(unix)]
#[test]
fn malformed_response_keeps_exception_tail_and_never_echoes_stdout() {
    use std::os::unix::process::ExitStatusExt;
    let mut diagnostics = String::new();
    append_diagnostic(&mut diagnostics, &"earlier warning 隐私".repeat(2048));
    append_diagnostic(&mut diagnostics, "Traceback (most recent call last):");
    append_diagnostic(
        &mut diagnostics,
        "TypeError: final product header is invalid",
    );
    assert!(diagnostics.len() <= MAX_DIAGNOSTIC_BYTES);
    let (code, detail) = decode_project_response(
        b"PRIVATE_RAW_SOURCE_CONTENT and a partial response",
        Some(ExitStatus::from_raw(0)),
        &diagnostics,
    )
    .unwrap_err();
    assert_eq!(code, "PROJECT_RESPONSE_INVALID");
    assert!(detail.contains("invalid JSON result"));
    assert!(detail.ends_with("TypeError: final product header is invalid"));
    assert!(!detail.contains("PRIVATE_RAW_SOURCE_CONTENT"));
    assert!(detail.len() <= MAX_DIAGNOSTIC_BYTES);

    let (code, detail) =
        decode_project_response(b"", Some(ExitStatus::from_raw(1 << 8)), &diagnostics).unwrap_err();
    assert_eq!(code, "PROJECT_EXECUTION_FAILED");
    assert!(detail.contains("no JSON result"));
    assert!(detail.contains("exit status: 1"));
    assert!(detail.ends_with("TypeError: final product header is invalid"));
    assert!(detail.len() <= MAX_DIAGNOSTIC_BYTES);
    assert!(decode_project_response(b"{}", None, "wait failed").is_err());
    assert!(decode_project_response(
        b"{\"success\":true}",
        Some(ExitStatus::from_raw(1 << 8)),
        "worker crashed"
    )
    .is_err());
}

#[cfg(unix)]
#[test]
fn crashed_project_streams_stderr_to_gui_without_a_completion_event() {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::mpsc;
    use tauri::Listener;

    let root = std::env::temp_dir().join(new_public_identifier("project-crash-test").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let light = root.join("fixture.fit");
    std::fs::write(&light, b"synthetic source fixture").unwrap();
    let script = root.join("fake-project-sidecar");
    std::fs::write(
        &script,
        r###"#!/usr/bin/env python3
import sys
print("earlier warnings " * 1024, file=sys.stderr)
print("Traceback (most recent call last):", file=sys.stderr)
print("TypeError: final product header is invalid", file=sys.stderr)
sys.exit(1)
"###,
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
    let app = tauri::test::mock_app();
    let handle = app.handle().clone();
    let (error_sender, error_receiver) = mpsc::channel();
    let (complete_sender, complete_receiver) = mpsc::channel();
    handle.listen(ERROR_EVENT, move |event| {
        let _ = error_sender.send(event.payload().to_owned());
    });
    handle.listen(COMPLETE_EVENT, move |event| {
        let _ = complete_sender.send(event.payload().to_owned());
    });
    let registry = Arc::new(ProjectRegistry::default());
    let run = start_with(
        handle,
        registry.clone(),
        ProjectRunRequest {
            sources: vec![UiRunSource {
                source_id: "light-1".to_owned(),
                role: "LIGHT".to_owned(),
                paths: vec![light.to_string_lossy().into_owned()],
                recursive: false,
            }],
            project_name: "synthetic crash".to_owned(),
            run_label: String::new(),
            recipe: UiRecipeOptions {
                balanced: true,
                drizzle_enabled: false,
                drizzle_scale: default_drizzle_scale(),
                drizzle_drop_shrink: default_drizzle_drop_shrink(),
                drizzle_kernel: default_drizzle_kernel(),
                solver_required: true,
                calibration_workflow: default_calibration_workflow(),
            },
            master_metadata_overrides: vec![],
            raw_frame_metadata_overrides: vec![],
            review_selections: vec![],
            selection: None,
            blink_review: None,
            output_parent_directory: root.to_string_lossy().into_owned(),
        },
        crate::sidecar::EngineExecutable { path: script },
    )
    .unwrap();
    let error: serde_json::Value = serde_json::from_str(
        &error_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(error["jobId"], run.job_id);
    assert_eq!(error["code"], "PROJECT_EXECUTION_FAILED");
    let message = error["message"].as_str().unwrap();
    assert!(message.contains("no JSON result"));
    assert!(message.ends_with("TypeError: final product header is invalid"));
    assert!(message.len() <= MAX_DIAGNOSTIC_BYTES);
    assert!(complete_receiver.try_recv().is_err());
    assert!(registry.jobs.lock().unwrap().is_empty());
    assert!(!Path::new(&run.output_directory).exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn unsafe_roles_and_ids_are_rejected_before_spawning() {
    assert!(checked_role("MASTER_DARK"));
    assert!(!checked_role("MASTER_LIGHT"));
    assert!(checked_id("light-0001"));
    assert!(!checked_id("--request-json"));
}

fn master_override_with_units(
    numeric_domain: Option<&str>,
    normalized_unit_scale: Option<f64>,
) -> UiMasterOverride {
    UiMasterOverride {
        source_sha256: format!("sha256:{}", "a".repeat(64)),
        camera: Some("QHY268M".to_owned()),
        gain: Some(0.0),
        offset: Some(30.0),
        binning: Some([1, 1]),
        filter: Some("NONE".to_owned()),
        cfa_pattern: Some("NONE".to_owned()),
        readout_mode: Some("HIGH GAIN 2CMS".to_owned()),
        temperature_celsius: Some(-10.0),
        exposure_seconds: Some(300.0),
        bias_included: Some(true),
        numeric_domain: numeric_domain.map(str::to_owned),
        normalized_unit_scale,
    }
}

#[test]
fn standard_master_workflow_reaches_worker_without_fabricated_metadata() {
    let root = std::env::temp_dir().join(format!(
        "standard-master-wire-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let light = root.join("light.fits");
    let dark = root.join("masterDark.xisf");
    std::fs::write(&light, b"light input").unwrap();
    std::fs::write(&dark, b"master input").unwrap();
    let mut request = ProjectRunRequest {
        sources: vec![
            UiRunSource {
                source_id: "light-1".into(),
                role: "LIGHT".into(),
                paths: vec![light.to_string_lossy().into_owned()],
                recursive: false,
            },
            UiRunSource {
                source_id: "dark-1".into(),
                role: "MASTER_DARK".into(),
                paths: vec![dark.to_string_lossy().into_owned()],
                recursive: false,
            },
        ],
        project_name: "Standard masters".into(),
        run_label: String::new(),
        recipe: UiRecipeOptions {
            balanced: true,
            drizzle_enabled: false,
            drizzle_scale: default_drizzle_scale(),
            drizzle_drop_shrink: default_drizzle_drop_shrink(),
            drizzle_kernel: default_drizzle_kernel(),
            solver_required: true,
            calibration_workflow: "mono-standard-v1".into(),
        },
        master_metadata_overrides: vec![],
        raw_frame_metadata_overrides: vec![],
        review_selections: vec![],
        selection: None,
        blink_review: None,
        output_parent_directory: root.to_string_lossy().into_owned(),
    };
    let output = root.join("new-output");
    let value = project_request_json(&request, &output).unwrap();
    assert_eq!(
        value["recipe"]["calibration"]["workflow"],
        "mono-standard-v1"
    );
    assert_eq!(value["recipe"]["calibration"]["bias"], "OPTIONAL");
    assert_eq!(value["recipe"]["solver"]["backend"], "auto");
    assert_eq!(value["recipe"]["solver"]["policy"], "REQUIRED");
    assert_eq!(
        value["recipe"]["calibration"]["masterMetadataOverrides"],
        serde_json::json!([])
    );
    assert_eq!(
        value["recipe"]["rawFrameMetadataOverrides"],
        serde_json::json!([])
    );
    assert!(!output.exists());
    request.recipe.calibration_workflow = "strict-v1".into();
    assert!(project_request_json(&request, &output)
        .unwrap_err()
        .contains("MasterDark requires"));
    request.recipe.calibration_workflow = "unknown".into();
    assert!(project_request_json(&request, &output)
        .unwrap_err()
        .contains("unsupported calibration workflow"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn drizzle_options_travel_with_the_recipe_and_are_range_checked() {
    let root = std::env::temp_dir().join(format!(
        "drizzle-recipe-wire-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let light = root.join("light.fits");
    std::fs::write(&light, b"light input").unwrap();
    let recipe: UiRecipeOptions = serde_json::from_value(serde_json::json!({
        "balanced": true, "drizzleEnabled": true, "drizzleScale": 3,
        "drizzleDropShrink": 0.7, "drizzleKernel": "gaussian",
        "solverRequired": true,
        "calibrationWorkflow": "strict-v1"
    }))
    .unwrap();
    let mut request = ProjectRunRequest {
        sources: vec![UiRunSource {
            source_id: "light-1".into(),
            role: "LIGHT".into(),
            paths: vec![light.to_string_lossy().into_owned()],
            recursive: false,
        }],
        project_name: "Drizzle".into(),
        run_label: String::new(),
        recipe,
        master_metadata_overrides: vec![],
        raw_frame_metadata_overrides: vec![],
        review_selections: vec![],
        selection: None,
        blink_review: None,
        output_parent_directory: root.to_string_lossy().into_owned(),
    };
    let output = root.join("new-output");
    let value = project_request_json(&request, &output).unwrap();
    assert_eq!(value["recipe"]["drizzle"]["enabled"], true);
    assert_eq!(value["recipe"]["drizzle"]["scale"], 3);
    assert_eq!(value["recipe"]["drizzle"]["dropShrink"], 0.7);
    assert_eq!(value["recipe"]["drizzle"]["kernel"], "gaussian");
    // Older front ends that omit the geometry keep the 2x square defaults.
    let legacy: UiRecipeOptions = serde_json::from_value(serde_json::json!({
        "balanced": true, "drizzleEnabled": true,
        "solverRequired": true
    }))
    .unwrap();
    assert_eq!(legacy.drizzle_scale, 2);
    assert_eq!(legacy.drizzle_drop_shrink, 0.9);
    assert_eq!(legacy.drizzle_kernel, "square");
    request.recipe.drizzle_scale = 5;
    assert!(project_request_json(&request, &output)
        .unwrap_err()
        .contains("drizzle scale"));
    request.recipe.drizzle_scale = 2;
    request.recipe.drizzle_drop_shrink = 0.0;
    assert!(project_request_json(&request, &output)
        .unwrap_err()
        .contains("drop shrink"));
    request.recipe.drizzle_drop_shrink = 0.9;
    request.recipe.drizzle_kernel = "lanczos".into();
    assert!(project_request_json(&request, &output)
        .unwrap_err()
        .contains("drizzle kernel"));
    // Disabled drizzle never blocks a run on stale geometry values.
    request.recipe.drizzle_enabled = false;
    assert!(project_request_json(&request, &output).is_ok());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn master_overrides_preserve_unknowns_and_explicit_zero() {
    let override_: UiMasterOverride = serde_json::from_value(serde_json::json!({
        "sourceSha256": format!("sha256:{}", "a".repeat(64)),
        "biasIncluded": false, "offset": 0.0
    }))
    .unwrap();
    validate_master_override(&override_).unwrap();
    let encoded = serde_json::to_value(override_).unwrap();
    assert_eq!(encoded["biasIncluded"], false);
    assert_eq!(encoded["offset"], 0.0);
    assert!(encoded.get("temperatureCelsius").is_none());
    assert!(encoded.get("gain").is_none());
    assert!(encoded.get("camera").is_none());
    let unknown: UiMasterOverride = serde_json::from_value(serde_json::json!({
        "sourceSha256": format!("sha256:{}", "a".repeat(64)), "camera":"UNKNOWN"
    }))
    .unwrap();
    assert!(validate_master_override(&unknown).is_err());
}

#[test]
fn additive_master_numeric_units_are_optional_but_atomic_and_bounded() {
    assert!(validate_master_override(&master_override_with_units(None, None)).is_ok());
    assert!(validate_master_override(&master_override_with_units(
        Some("NORMALIZED_UNIT"),
        Some(1.0),
    ))
    .is_ok());
    assert!(validate_master_override(&master_override_with_units(
        Some("SENSOR_CODE"),
        Some(65535.0),
    ))
    .is_ok());
    assert!(
        validate_master_override(&master_override_with_units(Some("NORMALIZED_UNIT"), None,))
            .is_err()
    );
    assert!(
        validate_master_override(&master_override_with_units(Some("ELECTRONS"), Some(1.0),))
            .is_err()
    );
}

#[test]
fn application_shutdown_terminates_every_project_process_tree() {
    let registry = ProjectRegistry::default();
    let mut command = platform::test_support::sleeping_command();
    let child = Arc::new(Mutex::new(
        ManagedChild::spawn(&mut command).expect("project child"),
    ));
    registry.jobs.lock().unwrap().insert(
        "project-shutdown-test".to_owned(),
        ProjectJob {
            child: child.clone(),
            request_path: std::env::temp_dir().join(format!(
                "{}.json",
                new_public_identifier("project-shutdown-request").unwrap()
            )),
        },
    );

    let request_path = registry
        .jobs
        .lock()
        .unwrap()
        .get("project-shutdown-test")
        .unwrap()
        .request_path
        .clone();
    std::fs::write(&request_path, b"private request").expect("private request fixture");

    terminate_all(&registry).expect("terminate project children");
    let status = child.lock().unwrap().wait().expect("reap project child");
    assert!(!status.success());
    assert!(!request_path.exists());
    assert!(registry.shutting_down.load(Ordering::Acquire));
    terminate_all(&registry).expect("shutdown is idempotent");
}

#[test]
fn managed_astrometry_schema_rejects_unbound_catalogs() {
    let value = serde_json::json!({
        "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":1.0,"centerDecDegrees":2.0,
        "pixelScaleArcsec":1.0,"rotationDegrees":0.0,"rmsPixels":0.2,"rmsArcsec":0.3,"matchedStars":30,
        "parity":"POSITIVE","catalogIdentity":"1".repeat(64),"indexIdentities":["astrometry.net:index:4108:healpix:1:hpnside:1"],
        "correspondenceSha256":"2".repeat(64),"catalogManaged":false,"installedSetIdentity":"3".repeat(64),
        "catalogManifestSha256":"4".repeat(64),"indexArtifacts":[],"wcsSha256":"5".repeat(64)
    });
    let parsed: AstrometricSolutionReceipt = serde_json::from_value(value).unwrap();
    assert!(parsed.validate().is_err());
}

#[cfg(unix)]
#[test]
fn fake_run_project_streams_progress_and_revalidates_the_outer_receipt() {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::mpsc;
    use tauri::Listener;

    let root = std::env::temp_dir()
        .join(new_public_identifier("project-controller-test").expect("temporary identifier"));
    let input = root.join("Unicode 输入");
    let output_parent = root.join("Unicode 输出");
    std::fs::create_dir_all(&input).expect("input directory");
    std::fs::create_dir_all(&output_parent).expect("output directory");
    let light = input.join("盾牌座 light.fit");
    std::fs::write(&light, b"source-frame").expect("source fixture");
    let script = root.join("fake-project-sidecar");
    let source = r###"#!/usr/bin/env python3
import hashlib, json, pathlib, sys
assert sys.argv[1] == "run-project" and sys.argv[2] == "--request-json"
request_path = pathlib.Path(sys.argv[3])
request = json.loads(request_path.read_text())
assert request["sources"][0]["expectedRole"] == "LIGHT"
assert "盾牌座" in request["sources"][0]["hostPath"]
assert request["recipe"]["rawFrameMetadataOverrides"][0]["cfaPattern"] == "NONE"
print(json.dumps({"type":"progress","event":{"stage":"quality-control","status":"RUNNING","current":1,"total":2,"message":"checked one"}}), file=sys.stderr, flush=True)
output = pathlib.Path(request["outputDirectory"])
products = output / "products"
products.mkdir(parents=True)
artifact = products / "盾牌座_R_mosaic.fits"
artifact.write_bytes(b"verified-solved-product")
digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
astro = {
  "referenceFrame":"ICRS","projection":"TAN","centerRaDegrees":281.0,"centerDecDegrees":-6.0,
  "pixelScaleArcsec":1.4,"rotationDegrees":0.0,"rmsPixels":0.3,"rmsArcsec":0.42,"matchedStars":73,
  "parity":"POSITIVE","catalogIdentity":"2"*64,"indexIdentities":["astrometry.net:index:4108:healpix:123:hpnside:4"],
  "correspondenceSha256":"3"*64,"catalogManaged":True,"installedSetIdentity":"5"*64,"catalogManifestSha256":"6"*64,
  "indexArtifacts":[{"indexId":"4108","relativeName":"index-4108.fits","sizeBytes":94550400,"sha256":"7"*64,"manifestSha256":"6"*64,"installedSetIdentity":"5"*64}],
  "wcsSha256":"4"*64,"imageShape":[4176,6248],"state":"SOLVED"
}
record = {"path":"products/盾牌座_R_mosaic.fits","relativePath":"products/盾牌座_R_mosaic.fits","kind":"SOLVED_MONO_FITS","sha256":digest,"sizeBytes":artifact.stat().st_size,"filter":"R","astrometry":astro,"finalGate":{"status":"PASS"}}
receipt = {"schemaVersion":1,"success":True,"state":"SOLVED","finalProducts":{"guiArtifacts":[record],"resultGate":{"status":"PASS","allMonoProductsSolved":True,"managedCatalogEvidenceRequired":True,"sourceIdentityVerifiedAtCommit":True,"mosaicCoverageOverlapSeamPassed":True,"rgbState":"MONO_ONLY_CHANNELS_MISSING"}}}
receipt_path = output / "receipt.json"
receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
print(json.dumps({"success":True,"code":"PROJECT_MONO_SUCCEEDED","state":"SOLVED","outputDirectory":str(output),"evidenceDirectory":None,"receiptPath":str(receipt_path),"productPaths":[str(artifact)],"previewPaths":[],"passedLightPaths":[],"excludedLightPaths":[],"monoFilters":["R"],"colorProductPath":None}), flush=True)
"###;
    std::fs::write(&script, source).expect("write fake sidecar");
    let mut permissions = std::fs::metadata(&script).unwrap().permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&script, permissions).unwrap();

    let app = tauri::test::mock_app();
    let handle = app.handle().clone();
    let (progress_sender, progress_receiver) = mpsc::channel();
    let (complete_sender, complete_receiver) = mpsc::channel();
    handle.listen(PROGRESS_EVENT, move |event| {
        let _ = progress_sender.send(event.payload().to_owned());
    });
    handle.listen(COMPLETE_EVENT, move |event| {
        let _ = complete_sender.send(event.payload().to_owned());
    });
    let receipt = start_with(
        handle,
        Arc::new(ProjectRegistry::default()),
        ProjectRunRequest {
            sources: vec![UiRunSource {
                source_id: "light-0001".to_owned(),
                role: "LIGHT".to_owned(),
                paths: vec![light.to_string_lossy().into_owned()],
                recursive: false,
            }],
            project_name: "盾牌座 马赛克".to_owned(),
            run_label: "盾牌座 马赛克".to_owned(),
            recipe: UiRecipeOptions {
                balanced: true,
                drizzle_enabled: false,
                drizzle_scale: default_drizzle_scale(),
                drizzle_drop_shrink: default_drizzle_drop_shrink(),
                drizzle_kernel: default_drizzle_kernel(),
                solver_required: true,
                calibration_workflow: default_calibration_workflow(),
            },
            master_metadata_overrides: vec![],
            raw_frame_metadata_overrides: vec![UiRawFrameOverride {
                source_sha256: format!("sha256:{}", sha256_file(&light).unwrap()),
                cfa_pattern: "NONE".to_owned(),
            }],
            review_selections: vec![],
            selection: None,
            blink_review: None,
            output_parent_directory: output_parent.to_string_lossy().into_owned(),
        },
        crate::sidecar::EngineExecutable { path: script },
    )
    .expect("launch fake project");
    assert!(receipt.accepted);
    let progress: serde_json::Value = serde_json::from_str(
        &progress_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("project progress"),
    )
    .expect("progress JSON");
    assert_eq!(progress["stageId"], "quality-control");
    assert_eq!(progress["fraction"], 0.5);
    let completion: serde_json::Value = serde_json::from_str(
        &complete_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("project completion"),
    )
    .expect("completion JSON");
    assert_eq!(completion["gate"]["decision"], "ready");
    assert_eq!(completion["artifacts"][0]["kind"], "SOLVED_MONO_FITS");
    assert_eq!(
        completion["artifacts"][0]["receipt"]["astrometry"]["catalogManaged"],
        true
    );
    assert_eq!(completion["artifacts"][1]["kind"], "RECEIPT");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn desktop_requires_complete_review_of_the_unchanged_measured_light_set() {
    let root = std::env::temp_dir().join(new_public_identifier("blink-review-test").unwrap());
    let session = root.join("0123456789abcdef-20260922-101010");
    std::fs::create_dir_all(&session).unwrap();
    let lights: Vec<_> = (1..=3)
        .map(|index| {
            let path = root.join(format!("light-{index}.fits"));
            std::fs::write(&path, format!("light {index}")).unwrap();
            path
        })
        .collect();
    let manifest = serde_json::json!({
        "kind": "blink-manifest-v1", "sessionId": "abc-20260922-101010",
        "flagsPolicyDigest": format!("sha256:{}", "e".repeat(64)),
        "channels": [{"channelId": "L"}, {"channelId": "R"}],
        "frames": lights.iter().enumerate().map(|(index, path)| serde_json::json!({
            "sourceSha256": format!("sha256:{}", (index + 1).to_string().repeat(64)),
            "path": path, "channelId": if index < 2 { "L" } else { "R" },
            "previews": {"error": if index == 1 { Some("unreadable") } else { None }},
        })).collect::<Vec<_>>(),
    });
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(session.join("manifest.json"), &bytes).unwrap();
    let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
    let mut selection = sample_selection();
    selection["origin"]["blinkManifestSha256"] = serde_json::json!(digest);
    let mut request = selection_request(&root, &lights[0], selection);
    request.sources[0].paths = lights
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_REQUIRED"));
    request.blink_review = Some(BlinkReviewProof {
        session_directory: session.to_string_lossy().into_owned(),
        manifest_sha256: digest,
        reviewed_source_sha256s: (1..=3)
            .map(|i| format!("sha256:{}", i.to_string().repeat(64)))
            .collect(),
        confirmed_channel_ids: vec!["L".into(), "R".into()],
    });
    validate_blink_review(&request, &root).unwrap();
    let proof = request.blink_review.clone().unwrap();
    request
        .blink_review
        .as_mut()
        .unwrap()
        .reviewed_source_sha256s
        .pop();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_INCOMPLETE"));
    request.blink_review = Some(proof.clone());
    request
        .blink_review
        .as_mut()
        .unwrap()
        .confirmed_channel_ids
        .pop();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_INCOMPLETE"));
    request.blink_review = Some(proof);
    request.selection.as_mut().unwrap().decisions[1].decision = "KEEP".into();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_UNAVAILABLE"));
    request.selection.as_mut().unwrap().decisions[1].decision = "DROP".into();
    request.sources[0].paths.pop();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_STALE"));
    request.sources[0]
        .paths
        .push(lights[2].to_string_lossy().into_owned());
    request.selection.as_mut().unwrap().decisions.pop();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_INCOMPLETE"));
    std::fs::write(session.join("manifest.json"), b"{}").unwrap();
    assert!(validate_blink_review(&request, &root)
        .unwrap_err()
        .starts_with("BLINK_REVIEW_STALE"));
    std::fs::remove_dir_all(root).unwrap();
}
