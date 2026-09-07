use std::fs::File;
use std::io::Read;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    /// Name of the training run, used in saved models and data organisation
    pub name: String,
    pub model: ModelConfig,
    pub eval: EvalConfig,
    pub self_play: SelfPlayConfig,
    pub training: TrainingConfig
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LrSchedule {
    Cosine {
        /// Initial learning rate
        lr_start: f32,
        /// Final learning rate
        lr_end: f32,
    },
    Constant {
        /// Constant learning rate (not recommended)
        lr: f32,
    },
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Number of epochs to train the model for per iteration
    pub epochs: u32,
    /// Model batch size used in training
    pub batch_size: u32,
    /// Number of conv blocks to include in the model
    pub res_blocks: u32,
    /// Learning rate schedule: 'cosine' (default) or 'constant'
    pub lr_schedule: LrSchedule,
    /// Offset to store the model at, useful when resuming a training run
    pub offset: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct EvalConfig {
    /// Number of games to play when evaluating the model against random and against the previous iteration
    pub games: u32,
    /// Number of simulations to use per move during eval
    pub sims: u32,
    /// Skip the eval to reduce training time, good if you are confident in params and just need to train
    pub skip: bool,
    /// Enable model gating (only promote models that beat current best)
    pub gating: bool,
    /// Win rate threshold for model promotion
    pub gating_threshold: f64,
    /// Minimum win rate against random to allow promotion
    pub min_random_win_rate: f64,
}

#[derive(Debug, Deserialize)]
pub struct SelfPlayConfig {
    /// Number of self-play games per iteration
    pub games: u32,
    /// Number of MCTS simulations per self-play game
    pub sims: u32,
    /// Offset to store the training data, useful when resuming a training run
    pub offset: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    /// Number of iterations to run training for
    pub iterations: u32,
    /// Number of past datasets to use per training iteration (size of sliding window)
    pub window: u32,
    /// Number of GPUs to use for training. Uses torchrun for >1, plain python for 1.
    pub num_gpus: u32,
}

#[derive(Debug)]
pub enum ConfigError {
    FileError,
    DecodeError
}

impl Config {
    pub fn try_from_file(path: &str) -> Result<Self, ConfigError> {
        let mut file = File::open(path).map_err(|_| ConfigError::FileError)?;
        let mut file_string = String::new();
        file.read_to_string(&mut file_string).map_err(|_| ConfigError::FileError)?;

        let config: Config = toml::from_str(&file_string).map_err(|_| ConfigError::DecodeError)?;

        Ok(config)
    }
}
