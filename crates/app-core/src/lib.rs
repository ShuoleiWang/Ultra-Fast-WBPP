//! Product control-plane contracts shared by the Ultra-Fast WBPP GUI and workers.
//!
//! This crate deliberately contains no image-processing implementation and no
//! operating-system UI code. It defines the stable wire format, persisted
//! receipts, backend/profile negotiation, final-result gates, and conservative
//! filesystem publication semantics used at those boundaries.

#![forbid(unsafe_code)]

pub mod backend;
pub mod fs;
pub mod gate;
pub mod model;
pub mod protocol;
pub mod validation;

pub use backend::{
    Architecture, BackendCapabilities, BackendFeature, CapabilityError, HardwareProfile,
    HostPlatform, OperatingSystem,
};
pub use fs::{
    NewDirectoryPublication, PlatformFs, PublicationAuthorization, PublicationError,
    PublicationFile, PublicationReceipt, SafeFileName, SafeRelativePath, StdPlatformFs,
};
pub use gate::{GateCheck, GateDecision, RequiredResultGate, ResultGateReport};
pub use model::*;
pub use protocol::{
    HandshakeMessage, WORKER_PROTOCOL_VERSION, WorkerEnvelope, WorkerMessage, decode_ndjson_line,
    encode_ndjson_line,
};
pub use validation::{Validate, ValidationError};
