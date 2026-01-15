use std::process::Command;
use clap::Parser;

/// CLI tool responsible for orchestration of Python training and Rust self-play loop
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Prefix used in file names of dataset and model files
    #[arg(short, long)]
    prefix: String,
    /// Number of times to run self-play and training
    #[arg(short, long)]
    iterations: u32,

    /// Number of self play games to run per iteration
    #[arg(long)]
    self_play_games: u32,
    /// Number of simulations per move per game
    #[arg(long)]
    self_play_sims: Option<u32>,
    /// Number to start the self play games at
    #[arg(long)]
    self_play_offset: Option<u32>,

    /// Number of training epochs to train the model for
    #[arg(long)]
    model_epochs: Option<u32>,
    /// Number of examples used per parameter update
    #[arg(long)]
    model_batch_size: Option<u32>,
    /// Number of convolutional blocks used
    #[arg(long)]
    model_res_blocks: Option<u32>,
    /// Rate that the model learns
    #[arg(long)]
    model_lr: Option<f32>,
    /// Number to start models at
    #[arg(long)]
    model_offset: Option<u32>,
}

fn main() {
    #[cfg(target_os = "windows")]
    let python_path = "../../packages/othello-training/.venv/Scripts/python.exe";

    #[cfg(not(target_os = "windows"))]
    let python_path = "../../packages/othello-training/.venv/bin/python3";

    // Parse the arguments
    let args = Args::parse();

    // Calculate the offsets for self play and training respectively
    let sp_offset0 = args.self_play_offset.unwrap_or(0);
    let model_offset0 = args.model_offset.unwrap_or(0);

    // Iterate through the specified number of iterations
    for i in 0..args.iterations {
        // Calculate the specific offsets that will be used for this specific iteration
        let base_offset = sp_offset0 + i * args.self_play_games;
        let model_idx = model_offset0 + i;

        // Format the dataset path
        let dataset_path = format!(
            "../othello-self-play/data/{}_selfplay_{:05}_{:05}.bin",
            &args.prefix,
            base_offset,
            base_offset + args.self_play_games
        );

        println!("=== Iteration {} ===", i);

        // Run the self-play process, passing the necessary arguments
        let mut self_play = Command::new("../othello-self-play/target/release/othello-self-play");
        self_play
            .env("RUST_LOG", "debug,ort=warn")
            .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
            .arg("--out").arg("../othello-self-play/data")
            .arg("--offset").arg(base_offset.to_string())
            .arg("--games").arg(args.self_play_games.to_string())
            .arg("--prefix").arg(&args.prefix);

        if let Some(s) = args.self_play_sims {
            self_play.arg("--sims").arg(s.to_string());
        }

        // On the first iteration, no model is used, only random rollouts
        if i > 0 {
            let model_in = format!(
                "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
                &args.prefix,
                model_idx,
                args.model_epochs.unwrap_or(10)
            );
            self_play.arg("--model").arg(&model_in);
        }

        // Run self-play, assert that it did not fail
        assert!(self_play.status().expect("self-play failed").success());

        // Generate the prefix that we will use to store the model (includes path)
        let model_out_prefix = format!(
            "../../packages/othello-training/models/{}_{}",
            args.prefix,
            model_idx + 1
        );

        // Run the training process, using python from uv
        let mut train = Command::new(python_path);
        train
            .arg("../../packages/othello-training/main.py")
            .arg("--data").arg(&dataset_path)
            .arg("--out-prefix").arg(&model_out_prefix);

        // If the user gave us optional args, add them
        if let Some(e) = args.model_epochs {
            train.arg("--epochs").arg(e.to_string());
        }
        if let Some(b) = args.model_batch_size {
            train.arg("--batch-size").arg(b.to_string());
        }
        if let Some(lr) = args.model_lr {
            train.arg("--lr").arg(lr.to_string());
        }
        if let Some(rb) = args.model_res_blocks {
            train.arg("--res-blocks").arg(rb.to_string());
        }

        // Assert that training didn't crash
        assert!(train.status().expect("training failed").success());
    }

    // Print args so that we can reproduce the training run if needed
    println!("{:?}", args);
}
