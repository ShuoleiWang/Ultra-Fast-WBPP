use thiserror::Error;

/// A stable validation error suitable for GUI display and protocol error details.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{path}: {message}")]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl ValidationError {
    #[must_use]
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

/// Runtime validation for invariants JSON Schema alone cannot express.
pub trait Validate {
    /// Validate semantic invariants not completely expressible in JSON Schema.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] at the first invalid field.
    fn validate(&self) -> Result<(), ValidationError>;
}

pub(crate) fn validate_identifier(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.is_empty() || value.len() > 128 {
        return Err(ValidationError::new(
            path,
            "must contain between 1 and 128 ASCII characters",
        ));
    }
    let mut characters = value.chars();
    let first = characters.next().expect("non-empty checked above");
    if !first.is_ascii_alphanumeric() {
        return Err(ValidationError::new(
            path,
            "must begin with an ASCII letter or digit",
        ));
    }
    if !characters
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'))
    {
        return Err(ValidationError::new(
            path,
            "may contain only ASCII letters, digits, dot, underscore, and hyphen",
        ));
    }
    Ok(())
}

pub(crate) fn validate_sha256(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ValidationError::new(
            path,
            "must be 64 lowercase hexadecimal SHA-256 characters",
        ));
    }
    Ok(())
}
