use std::path::PathBuf;
use ort::Error;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// Standardised way to load the model during self-play iterations
pub(crate) fn load_model(path: PathBuf) -> Result<Session, Error> {
    ort::set_api(ort_tract::api());

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?;

    Ok(model)
}
