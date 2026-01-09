use crate::neural_net::load_model;
use crate::self_play::{generate_self_play_data, Sample};
use crate::write_data::write_samples;
use clap::Parser;
use ort::session::Session;
use std::path::PathBuf;
use tracing::{info, warn};

mod mcts;
mod neural_net;
mod self_play;
mod write_data;

/// Self-play data generator for Othello
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to generate
    #[arg(short, long, default_value_t = 1)]
    games: usize,

    /// Number of MCTS simulations per move
    #[arg(short, long, default_value_t = 200)]
    sims: u32,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    out: PathBuf,

    /// Optional ONNX model path (for NN-guided MCTS)
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Optional previous model path (for evaluation)
    #[arg(long)]
    prev_model: Option<PathBuf>,

    /// Starting index for file naming
    #[arg(long, default_value_t = 0)]
    offset: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    std::fs::create_dir_all(&args.out)?;

    // Generate self-play data
    let samples: Vec<Sample> =
        generate_self_play_data(args.games, args.sims, args.model).expect("Error generating self-play data");

    // Write dataset
    let filename = args.out.join(format!(
        "selfplay_{:05}_{:05}.bin",
        args.offset,
        args.offset + args.games
    ));

    write_samples(filename.to_str().unwrap(), &samples);

    info!(
        "Wrote {} samples from {} games to {:?}",
        samples.len(),
        args.games,
        filename
    );

    Ok(())
}
