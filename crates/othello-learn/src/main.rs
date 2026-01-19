use std::process::Command;
use clap::Parser;

/// CLI tool orchestrating Rust self-play and Python training loop
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)] prefix: String,
    #[arg(short, long)] iterations: u32,
    #[arg(long)] self_play_games: u32,
    #[arg(long)] self_play_sims: Option<u32>,
    #[arg(long)] self_play_offset: Option<u32>,
    #[arg(long)] model_epochs: Option<u32>,
    #[arg(long)] model_batch_size: Option<u32>,
    #[arg(long)] model_res_blocks: Option<u32>,
    #[arg(long)] model_lr: Option<f32>,
    #[arg(long)] model_offset: Option<u32>,
}

fn main() {
    #[cfg(target_os = "windows")]
    let python_path = "../../packages/othello-training/.venv/Scripts/python.exe";
    #[cfg(not(target_os = "windows"))]
    let python_path = "../../packages/othello-training/.venv/bin/python3";

    let args = Args::parse();

    let sp_offset0 = args.self_play_offset.unwrap_or(0);
    let model_offset0 = args.model_offset.unwrap_or(0);

    for i in 0..args.iterations {
        let base_offset = sp_offset0 + i * args.self_play_games;
        let model_idx = model_offset0 + i;

        println!("=== Iteration {} ===", i);

        // Generate dummy model for iteration 0
        if i == 0 {
            println!("Generating dummy ONNX model for iteration 0...");

            let mut dummy_cmd = Command::new(python_path);
            dummy_cmd
                .arg("../../packages/othello-training/main.py")
                .arg("--out-prefix")
                .arg(format!("../../packages/othello-training/models/{}_{}", &args.prefix, model_idx))
                .arg("--init-model");

            assert!(
                dummy_cmd.status().expect("Failed to generate dummy model").success(),
                "Dummy model generation failed"
            );
        }

        // Rust self-play
        let mut self_play = Command::new("../othello-self-play/target/release/othello-self-play");

        self_play
            .env("RUST_LOG", "debug,ort=warn")
            .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
            .arg("--out").arg("../othello-self-play/data")
            .arg("--offset").arg(base_offset.to_string())
            .arg("--games").arg(args.self_play_games.to_string())
            .arg("--prefix").arg(&args.prefix);

        if let Some(sims) = args.self_play_sims {
            self_play.arg("--sims").arg(sims.to_string());
        }

        // Always pass model (dummy for iteration 0, trained otherwise)
        let model_in = format!(
            "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
            &args.prefix,
            model_idx,
            if i == 0 { 0 } else { args.model_epochs.unwrap_or(10) }
        );
        self_play.arg("--model").arg(&model_in);

        assert!(self_play.status().expect("self-play failed").success());

        // Python training
        let dataset_path = format!(
            "../othello-self-play/data/{}_selfplay_{:05}_{:05}.bin",
            &args.prefix,
            base_offset,
            base_offset + args.self_play_games
        );

        let model_out_prefix = format!(
            "../../packages/othello-training/models/{}_{}",
            args.prefix,
            model_idx + 1
        );

        let mut train = Command::new(python_path);
        train
            .arg("../../packages/othello-training/main.py")
            .arg("--data")
            .arg(&dataset_path)
            .arg("--out-prefix")
            .arg(&model_out_prefix);

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

        assert!(train.status().expect("training failed").success());
    }

    println!("{:?}", args);
}
