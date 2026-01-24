use crate::async_self_play::{Sample, generate_self_play_data};
use crate::evaluate::evaluate_models;
use crate::write_data::write_samples;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::info;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod async_mcts;
mod async_self_play;
mod distr;
mod eval_queue;
mod evaluate;
mod neural_net;
mod symmetry;
mod write_data;

/// Self-play data generator for Othello
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
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
    model: Option<PathBuf>,

    /// Starting index for file naming
    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// File name prefix
    #[arg(long, short)]
    prefix: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate self-play training data
    Selfplay {
        /// Number of games to generate
        #[arg(short, long, default_value_t = 1)]
        games: usize,

        /// Number of MCTS simulations per move
        #[arg(short, long, default_value_t = 200)]
        sims: u32,

        /// Output directory
        #[arg(short, long, default_value = "data")]
        out: PathBuf,

        /// ONNX model path
        #[arg(short, long)]
        model: PathBuf,

        /// Starting index for file naming
        #[arg(long, default_value_t = 0)]
        offset: usize,

        /// File name prefix
        #[arg(long, short)]
        prefix: Option<String>,

        /// Training iteration (0-2 uses reduced noise for better early training)
        #[arg(long, short, default_value_t = 0)]
        iteration: u32,

        /// Disable reduced Dirichlet noise for early iterations (always use eps=0.25)
        #[arg(long, default_value_t = false)]
        no_early_noise_reduction: bool,
    },
    /// Evaluate two models head-to-head
    Eval {
        /// Path to the new model
        #[arg(long)]
        new_model: PathBuf,

        /// Path to the old/baseline model
        #[arg(long)]
        old_model: PathBuf,

        /// Number of evaluation games
        #[arg(short, long, default_value_t = 50)]
        games: u32,

        /// Simulations per move (can be lower than training)
        #[arg(short, long, default_value_t = 100)]
        sims: u32,
    },
}

fn main() -> anyhow::Result<()> {
    let indicatif_layer = IndicatifLayer::new();
    let args = Args::parse();

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .init();

    match args.command {
        Some(Command::Eval {
            new_model,
            old_model,
            games,
            sims,
        }) => {
            let result = evaluate_models(new_model, old_model, games, sims)?;
            println!("{}", result);
            // Exit with code based on win rate (0 = new model wins convincingly)
            if result.win_rate() >= 0.55 {
                std::process::exit(0);
            } else {
                std::process::exit(1);
            }
        }
        Some(Command::Selfplay {
            games,
            sims,
            out,
            model,
            offset,
            prefix,
            iteration,
            no_early_noise_reduction,
        }) => run_selfplay(games, sims, out, model, offset, prefix, iteration, no_early_noise_reduction),
        None => {
            // Backwards compatibility: run self-play with top-level args
            let model = args.model.expect("--model is required for self-play");
            run_selfplay(
                args.games,
                args.sims,
                args.out,
                model,
                args.offset,
                args.prefix,
                0, // default iteration for backwards compat
                false, // use early noise reduction by default
            )
        }
    }
}

fn run_selfplay(
    games: usize,
    sims: u32,
    out: PathBuf,
    model: PathBuf,
    offset: usize,
    prefix: Option<String>,
    iteration: u32,
    no_early_noise_reduction: bool,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(&out)?;

    let prefix = prefix.unwrap_or_default();

    let num_parallel_games = num_cpus::get_physical() + 2;
    let mcts_threads_per_game = 2usize;

    // If flag is set, always use standard noise; otherwise reduce for early iterations
    let dirichlet_eps = if no_early_noise_reduction {
        0.25
    } else if iteration <= 2 {
        0.15
    } else {
        0.25
    };

    info!(
        "Starting self-play: iteration={}, games={}, sims={}, dirichlet_eps={}",
        iteration,
        games,
        sims,
        dirichlet_eps
    );

    let samples: Vec<Sample> = generate_self_play_data(
        &prefix,
        games,
        sims,
        model,
        num_parallel_games,
        mcts_threads_per_game,
        iteration,
        no_early_noise_reduction,
    )?;

    info!("Generated {} samples", samples.len());

    let filename = out.join(format!(
        "{}{}selfplay_{:05}_{:05}.bin",
        prefix,
        if prefix.is_empty() { "" } else { "_" },
        offset,
        offset + games
    ));

    write_samples(&filename, &samples);

    info!(
        "Wrote {} samples from {} games to {:?}",
        samples.len(),
        games,
        filename
    );

    Ok(())
}
