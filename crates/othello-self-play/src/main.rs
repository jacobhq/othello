use crate::mcts::{play_match, self_play};
use crate::neural_net::load_model;
use clap::Parser;
use ort::session::Session;
use othello::othello_game::{Color, OthelloGame};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

mod mcts;
mod neural_net;

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

fn save_dataset_json(samples: &[(OthelloGame, Vec<f32>, f32, Color)], path: &str) -> Result<(), std::io::Error> {
    let mut file = File::create(path)?;

    file.write_all(b"[")?;

    for (i, (game, policy, value, player)) in samples.iter().enumerate() {
        let state = game.encode(*player);
        let json_obj = format!(
            "{{\"state\": {:?}, \"policy\": {:?}, \"value\": {}}}",
            state, policy, value
        );

        file.write_all(json_obj.as_bytes())?;

        if i < samples.len() - 1 {
            file.write_all(b",\n")?;
        }
    }

    file.write_all(b"]")?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut model: Option<&mut Session> = if let Some(ref path) = args.model {
        println!("Loading model from {:?}", path);
        Some(&mut load_model(path.to_str().unwrap())?)
    } else {
        println!("No model provided, using random rollouts.");
        None
    };

    std::fs::create_dir_all(&args.out)?;

    for g in 0..args.games {
        let samples = self_play(args.sims, model.as_deref_mut());
        let filename = args
            .out
            .join(format!("selfplay_data_{:05}.json", g + args.offset));

        if let Err(e) = save_dataset_json(&samples, filename.to_str().unwrap()) {
            eprintln!("Error saving game {}: {:?}", g + args.offset, e);
        } else {
            println!("Saved game {} to {:?}", g + args.offset, filename);
        }
    }


    // === Evaluation step ===
    if let Some(ref mut new_model) = model {
        println!("Evaluating new model vs random...");
        let mut score = 0.0;
        let eval_games = 20;
        for i in 0..eval_games {
            score += play_match(args.sims, Some(new_model), None, i % 2 == 0);
        }
        println!(
            "Win rate vs random: {:.1}%",
            (score / eval_games as f32 + 1.0) / 2.0 * 100.0
        );

        if let Some(prev_path) = &args.prev_model {
            println!("Loading previous model from {:?}", prev_path);
            let mut prev_model = load_model(prev_path.to_str().unwrap())?;
            let mut score_vs_old = 0.0;
            for i in 0..eval_games {
                score_vs_old +=
                    play_match(args.sims, Some(new_model), Some(&mut prev_model), i % 2 == 0);
            }
            println!(
                "Win rate vs previous: {:.1}%",
                (score_vs_old / eval_games as f32 + 1.0) / 2.0 * 100.0
            );
        }
    }

    Ok(())
}
