mod common;

use std::io::{self, Read};
use std::path::{Path, PathBuf};

use openastroflow_app_core::{
    NewDirectoryPublication, PlatformFs, PublicationAuthorization, PublicationError,
    PublicationFile, ResultRequirement, SafeFileName, SafeRelativePath, StageKind, StageStatus,
    StdPlatformFs, Validate,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixture(root: &TempDir) -> (PathBuf, PathBuf, NewDirectoryPublication) {
    let staging = root.path().join("staging");
    let destination_parent = root.path().join("published");
    std::fs::create_dir_all(staging.join("input")).expect("staging dirs");
    std::fs::create_dir(&destination_parent).expect("destination parent");
    std::fs::write(staging.join("input/frame.fits"), b"pixels").expect("source");
    let publication = NewDirectoryPublication {
        schema_version: 1,
        publication_id: "publication-1".to_owned(),
        destination_name: SafeFileName::new("run-1").expect("safe name"),
        files: vec![PublicationFile {
            source_relative_path: SafeRelativePath::new("input/frame.fits").expect("safe path"),
            destination_relative_path: SafeRelativePath::new("master/final.fits")
                .expect("safe path"),
            sha256: digest(b"pixels"),
            size_bytes: 6,
        }],
    };
    (staging, destination_parent, publication)
}

fn authorization(publication: &NewDirectoryPublication) -> PublicationAuthorization {
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let mut integrate = common::stage("integrate", StageKind::Integration);
    integrate.artifact_ids.clear();
    let stages = vec![
        integrate,
        common::stage("solve", StageKind::AstrometricSolve),
    ];
    let mut artifact = common::final_master();
    artifact
        .relative_path
        .clone_from(&publication.files[0].destination_relative_path);
    artifact.sha256.clone_from(&publication.files[0].sha256);
    artifact.size_bytes = publication.files[0].size_bytes;
    publication
        .authorize(&recipe, &stages, &[artifact])
        .expect("authorize fixture")
}

#[test]
fn portable_paths_reject_traversal_and_windows_hazards() {
    for value in [
        "../secret",
        "a/../../secret",
        "/absolute",
        "C:\\data\\file.fit",
        "CON",
        "aux.txt",
        "dir/com1.fit",
        "trailing. ",
        "a//b",
    ] {
        assert!(SafeRelativePath::new(value).is_err(), "accepted {value:?}");
    }
    SafeRelativePath::new("targets/目标一/master.fits").expect("portable Unicode path");
}

#[test]
fn publication_is_new_directory_only_and_writes_completion_last() {
    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, publication) = fixture(&root);
    let authorization = authorization(&publication);
    let receipt = publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect("publish");
    let destination = destination_parent.join("run-1");
    assert_eq!(
        std::fs::read(destination.join("master/final.fits")).unwrap(),
        b"pixels"
    );
    assert!(
        destination
            .join(".openastroflow-publication.json")
            .is_file()
    );
    assert!(destination.join(".openastroflow-complete.json").is_file());
    assert_eq!(receipt.file_count, 1);
    assert_eq!(receipt.total_bytes, 6);
}

#[test]
fn unicode_publication_paths_preserve_identity_and_never_replace() {
    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, mut publication) = fixture(&root);
    publication.destination_name = SafeFileName::new("盾牌座-输出").expect("Unicode name");
    publication.files[0].destination_relative_path =
        SafeRelativePath::new("目标/主亮场.fits").expect("Unicode relative path");
    let authorization = authorization(&publication);
    publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect("publish Unicode path");
    let destination = destination_parent.join("盾牌座-输出");
    assert_eq!(
        std::fs::read(destination.join("目标").join("主亮场.fits")).unwrap(),
        b"pixels"
    );

    let error = publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            124,
        )
        .expect_err("second publication must not replace Unicode destination");
    assert!(matches!(error, PublicationError::DestinationExists(_)));
    assert!(destination.join(".openastroflow-complete.json").is_file());
}

#[test]
fn existing_destination_is_never_modified() {
    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, publication) = fixture(&root);
    let authorization = authorization(&publication);
    let destination = destination_parent.join("run-1");
    std::fs::create_dir(&destination).expect("existing destination");
    std::fs::write(destination.join("sentinel"), b"keep").expect("sentinel");
    let error = publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect_err("must not replace");
    assert!(matches!(error, PublicationError::DestinationExists(_)));
    assert_eq!(
        std::fs::read(destination.join("sentinel")).unwrap(),
        b"keep"
    );
}

#[test]
fn authorization_is_bound_to_the_exact_publication_plan() {
    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, mut publication) = fixture(&root);
    let authorization = authorization(&publication);
    publication.files[0].destination_relative_path =
        SafeRelativePath::new("master/changed.fits").expect("safe path");
    let error = publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect_err("changed plan must not publish");
    assert!(matches!(error, PublicationError::AuthorizationMismatch));
    assert!(!destination_parent.join("run-1").exists());
}

#[test]
fn blocked_result_gate_cannot_authorize_publication() {
    let root = TempDir::new().expect("tempdir");
    let (_, _, publication) = fixture(&root);
    let recipe = common::recipe(ResultRequirement::Required, ResultRequirement::Disabled);
    let mut integrate = common::stage("integrate", StageKind::Integration);
    integrate.artifact_ids.clear();
    let stages = vec![
        integrate,
        common::stage("solve", StageKind::AstrometricSolve),
    ];
    let mut artifact = common::final_master();
    artifact
        .relative_path
        .clone_from(&publication.files[0].destination_relative_path);
    artifact.sha256.clone_from(&publication.files[0].sha256);
    artifact.size_bytes = publication.files[0].size_bytes;
    artifact.astrometry = None;
    let error = publication
        .authorize(&recipe, &stages, &[artifact])
        .expect_err("unsolved final master must not authorize");
    assert!(matches!(error, PublicationError::ResultGateBlocked(_)));
}

#[test]
fn best_effort_solver_may_be_skipped_without_blocking_publication() {
    let root = TempDir::new().expect("tempdir");
    let (_, _, publication) = fixture(&root);
    let recipe = common::recipe(ResultRequirement::BestEffort, ResultRequirement::Disabled);
    let integrate = common::stage("integrate", StageKind::Integration);
    let mut solve = common::stage("solve", StageKind::AstrometricSolve);
    solve.status = StageStatus::Skipped;
    solve.artifact_ids.clear();
    let mut artifact = common::final_master();
    artifact.produced_by_stage_id = "integrate".to_owned();
    artifact.astrometry = None;
    artifact
        .relative_path
        .clone_from(&publication.files[0].destination_relative_path);
    artifact.sha256.clone_from(&publication.files[0].sha256);
    artifact.size_bytes = publication.files[0].size_bytes;
    publication
        .authorize(&recipe, &[integrate, solve], &[artifact])
        .expect("best-effort solver skip should authorize");
}

#[test]
fn identity_mismatch_fails_before_destination_claim() {
    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, mut publication) = fixture(&root);
    publication.files[0].sha256 = "0".repeat(64);
    let authorization = authorization(&publication);
    let error = publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect_err("bad identity");
    assert!(matches!(
        error,
        PublicationError::SourceIdentityMismatch { .. }
    ));
    assert!(!destination_parent.join("run-1").exists());
}

#[test]
fn file_directory_prefix_collisions_are_rejected() {
    let plan = NewDirectoryPublication {
        schema_version: 1,
        publication_id: "publication-1".to_owned(),
        destination_name: SafeFileName::new("run-1").unwrap(),
        files: vec![
            PublicationFile {
                source_relative_path: SafeRelativePath::new("a").unwrap(),
                destination_relative_path: SafeRelativePath::new("master").unwrap(),
                sha256: "a".repeat(64),
                size_bytes: 1,
            },
            PublicationFile {
                source_relative_path: SafeRelativePath::new("b").unwrap(),
                destination_relative_path: SafeRelativePath::new("master/final.fits").unwrap(),
                sha256: "b".repeat(64),
                size_bytes: 1,
            },
        ],
    };
    assert!(plan.validate().is_err());
}

#[cfg(unix)]
#[test]
fn source_symlinks_are_rejected_even_when_they_stay_under_staging() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, mut publication) = fixture(&root);
    symlink("frame.fits", staging.join("input/link.fits")).expect("symlink");
    publication.files[0].source_relative_path =
        SafeRelativePath::new("input/link.fits").expect("safe path");
    let authorization = authorization(&publication);
    let error = publication
        .publish(
            &authorization,
            &StdPlatformFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect_err("symlink must fail");
    assert!(matches!(error, PublicationError::SourceSymlink(_)));
}

#[derive(Clone, Copy)]
struct FailCopyFs;

impl PlatformFs for FailCopyFs {
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        StdPlatformFs.canonicalize(path)
    }

    fn metadata_no_follow(
        &self,
        path: &Path,
    ) -> io::Result<Option<openastroflow_app_core::fs::FsMetadata>> {
        StdPlatformFs.metadata_no_follow(path)
    }

    fn open_read(&self, path: &Path) -> io::Result<Box<dyn Read + Send>> {
        StdPlatformFs.open_read(path)
    }

    fn create_directory_new(&self, path: &Path) -> io::Result<()> {
        StdPlatformFs.create_directory_new(path)
    }

    fn copy_file_new(&self, _source: &Path, _destination: &Path) -> io::Result<u64> {
        Err(io::Error::other("injected copy failure"))
    }

    fn write_file_new(&self, destination: &Path, contents: &[u8]) -> io::Result<()> {
        StdPlatformFs.write_file_new(destination, contents)
    }
}

#[test]
fn failed_publication_is_auditable_but_never_complete() {
    let root = TempDir::new().expect("tempdir");
    let (staging, destination_parent, publication) = fixture(&root);
    let authorization = authorization(&publication);
    let error = publication
        .publish(
            &authorization,
            &FailCopyFs,
            &staging,
            &destination_parent,
            123,
        )
        .expect_err("injected failure");
    assert!(matches!(error, PublicationError::Incomplete { .. }));
    let destination = destination_parent.join("run-1");
    assert!(
        destination
            .join(".openastroflow-publication.json")
            .is_file()
    );
    assert!(!destination.join(".openastroflow-complete.json").exists());
}
