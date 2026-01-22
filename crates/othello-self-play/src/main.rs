use crate::async_self_play::{generate_self_play_data, Sample};
use crate::write_data::write_samples;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod neural_net;
mod write_data;
mod distr;
mod symmetry;
mod eval_queue;
mod async_mcts;
mod async_self_play;

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
    model: PathBuf,

    /// Optional previous model path (for evaluation)
    #[arg(long)]
    prev_model: Option<PathBuf>,

    /// Starting index for file naming
    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// File name prefix
    #[arg(long, short)]
    prefix: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let indicatif_layer = IndicatifLayer::new();
    let args = Args::parse();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .init();

    std::fs::create_dir_all(&args.out)?;

    let prefix = args.prefix.unwrap_or_default();

    let num_parallel_games = 8usize; // or 4, 8, etc.
    let mcts_threads_per_game = 2usize;            // small

    // NEW async self-play
    let samples: Vec<Sample> = generate_self_play_data(
        &prefix,
        args.games,
        args.sims,
        args.model,
        num_parallel_games,
        mcts_threads_per_game
    )?;

    info!("Generated {} samples", samples.len());

    let filename = args.out.join(format!(
        "{}{}selfplay_{:05}_{:05}.bin",
        prefix,
        if prefix.is_empty() { "" } else { "_" },
        args.offset,
        args.offset + args.games
    ));

    write_samples(&filename, &samples);

    info!(
        "Wrote {} samples from {} games to {:?}",
        samples.len(),
        args.games,
        filename
    );

    Ok(())
}
