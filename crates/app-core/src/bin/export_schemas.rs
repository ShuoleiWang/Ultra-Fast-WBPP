use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use openastroflow_app_core::{
    ArtifactReceipt, BackendCapabilities, NewDirectoryPublication, Project, ProjectReceipt, Recipe,
    RecipeReceipt, ResultGateReport, RunReceipt, StageReceipt, WorkerEnvelope,
};
use schemars::{JsonSchema, schema_for};

fn write_schema<T: JsonSchema>(directory: &Path, name: &str) -> Result<(), Box<dyn Error>> {
    let schema = schema_for!(T);
    let mut value = serde_json::to_value(schema)?;
    {
        let object = value
            .as_object_mut()
            .ok_or("schema root is not an object")?;
        object.insert(
            "$id".to_owned(),
            serde_json::Value::String(format!("https://openastroflow.org/schema/v1/{name}")),
        );
        object.insert(
            "x-openastroflow-schema-version".to_owned(),
            serde_json::Value::from(1),
        );
        if name == "worker-envelope-v1.schema.json" {
            // `WorkerEnvelope` flattens a tagged enum. Schemars emits the four
            // common properties at the root and `type`/`payload` in `oneOf`.
            // Root additionalProperties=false therefore rejects every valid
            // message because JSON Schema evaluates it before the oneOf branch.
            // `propertyNames` preserves the strict six-key wire contract without
            // that cross-subschema false negative.
            object.remove("additionalProperties");
            object.insert(
                "propertyNames".to_owned(),
                serde_json::json!({
                    "enum": [
                        "protocolVersion",
                        "sessionId",
                        "sequence",
                        "sentAtUnixMs",
                        "type",
                        "payload"
                    ]
                }),
            );
        }
    }
    if let Some(catalog_managed) = value
        .pointer_mut("/definitions/AstrometricSolutionReceipt/properties/catalogManaged")
        .and_then(serde_json::Value::as_object_mut)
    {
        // A durable/final astrometry receipt is not the diagnostic solver
        // result model. It must claim the one value accepted by app-core's
        // runtime validator as well as merely having a boolean type.
        catalog_managed.insert("const".to_owned(), serde_json::Value::Bool(true));
    }
    let bytes = serde_json::to_vec_pretty(&value)?;
    let path = directory.join(name);
    fs::write(path, [bytes.as_slice(), b"\n"].concat())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("protocol/schema"), PathBuf::from);
    fs::create_dir_all(&output)?;
    write_schema::<WorkerEnvelope>(&output, "worker-envelope-v1.schema.json")?;
    write_schema::<Project>(&output, "project-v1.schema.json")?;
    write_schema::<Recipe>(&output, "recipe-v1.schema.json")?;
    write_schema::<ProjectReceipt>(&output, "project-receipt-v1.schema.json")?;
    write_schema::<RecipeReceipt>(&output, "recipe-receipt-v1.schema.json")?;
    write_schema::<StageReceipt>(&output, "stage-receipt-v1.schema.json")?;
    write_schema::<ArtifactReceipt>(&output, "artifact-receipt-v1.schema.json")?;
    write_schema::<RunReceipt>(&output, "run-receipt-v1.schema.json")?;
    write_schema::<BackendCapabilities>(&output, "backend-capabilities-v1.schema.json")?;
    write_schema::<NewDirectoryPublication>(&output, "publication-v1.schema.json")?;
    write_schema::<ResultGateReport>(&output, "result-gate-v1.schema.json")?;
    Ok(())
}
