use clap::Parser;
use std::process::Command;

/// CLI tool orchestrating Rust self-play and Python training loop
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    prefix: String,
    #[arg(short, long)]
    iterations: u32,
    #[arg(long)]
    self_play_games: u32,
    #[arg(long)]
    self_play_sims: Option<u32>,
    #[arg(long)]
    self_play_offset: Option<u32>,
    #[arg(long)]
    model_epochs: Option<u32>,
    #[arg(long)]
    model_batch_size: Option<u32>,
    #[arg(long)]
    model_res_blocks: Option<u32>,
    #[arg(long)]
    model_lr: Option<f32>,
    #[arg(long)]
    model_offset: Option<u32>,
    #[arg(long, default_value_t = 3)]
    window: u32,
    #[arg(long, default_value_t = 50)]
    eval_games: u32,
    #[arg(long, default_value_t = 100)]
    eval_sims: u32,
    #[arg(long, default_value_t = false)]
    skip_eval: bool,
}

fn main() {
    #[cfg(target_os = "windows")]
    let python_path = "../../packages/othello-training/.venv/Scripts/python.exe";
    #[cfg(not(target_os = "windows"))]
    let python_path = "../../packages/othello-training/.venv/bin/python3";

    let args = Args::parse();

    let sp_offset0 = args.self_play_offset.unwrap_or(0);
    let model_offset0 = args.model_offset.unwrap_or(0);

    // Data directory for all self-play output
    let data_dir = "../othello-self-play/data";

    // Path to the random/initial model (epoch 0) for baseline comparison
    let baseline_model = format!(
        "../../packages/othello-training/models/{}_{}_othello_net_epoch_000.onnx",
        &args.prefix, model_offset0
    );

    // Track evaluation results
    let mut eval_results: Vec<String> = Vec::new();

    for i in 0..args.iterations {
        let base_offset = sp_offset0 + i * args.self_play_games;
        let model_idx = model_offset0 + i;

        println!("=== Iteration {} ===", i);

        // Generate dummy model for iteration 0
        if i == 0 {
            println!("Generating initial ONNX model for iteration 0...");

            let mut init_cmd = Command::new(python_path);
            init_cmd
                .arg("../../packages/othello-training/main.py")
                .arg("--out-prefix")
                .arg(format!(
                    "../../packages/othello-training/models/{}_{}",
                    &args.prefix, model_idx
                ));

            if let Some(rb) = args.model_res_blocks {
                init_cmd.arg("--res-blocks").arg(rb.to_string());
            }

            init_cmd.arg("--init-model");

            assert!(
                init_cmd
                    .status()
                    .expect("Failed to generate dummy model")
                    .success(),
                "Dummy model generation failed"
            );
        }

        // Rust self-play
        let mut self_play = Command::new("../othello-self-play/target/release/othello-self-play");

        self_play
            .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
            .arg("--out")
            .arg(data_dir)
            .arg("--offset")
            .arg(base_offset.to_string())
            .arg("--games")
            .arg(args.self_play_games.to_string())
            .arg("--prefix")
            .arg(&args.prefix);

        if let Some(sims) = args.self_play_sims {
            self_play.arg("--sims").arg(sims.to_string());
        }

        // Always pass model (dummy for iteration 0, trained otherwise)
        let model_in = format!(
            "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
            &args.prefix,
            model_idx,
            if i == 0 {
                0
            } else {
                args.model_epochs.unwrap_or(2)
            }
        );
        self_play.arg(&model_in);

        assert!(self_play.status().expect("self-play failed").success());

        // Python training with sliding window
        println!("\nTraining on last {} data files", args.window);

        let model_out_prefix = format!(
            "../../packages/othello-training/models/{}_{}",
            args.prefix,
            model_idx + 1
        );

        let mut train = Command::new(python_path);
        train
            .arg("../../packages/othello-training/main.py")
            .arg("--data")
            .arg(data_dir)
            .arg("--window")
            .arg(args.window.to_string())
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
        // Evaluation matches
        if !args.skip_eval {
            let new_model = format!(
                "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
                &args.prefix,
                model_idx + 1,
                args.model_epochs.unwrap_or(2)
            );

            // Eval vs previous iteration (if not first)
            if i > 0 {
                let prev_model = format!(
                    "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
                    &args.prefix,
                    model_idx,
                    args.model_epochs.unwrap_or(2)
                );

                println!("\n--- Eval: New model vs Previous iteration ---");
                let mut eval_cmd =
                    Command::new("../othello-self-play/target/release/othello-self-play");
                eval_cmd
                    .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
                    .arg("eval")
                    .arg("--new-model")
                    .arg(&new_model)
                    .arg("--old-model")
                    .arg(&prev_model)
                    .arg("--games")
                    .arg(args.eval_games.to_string())
                    .arg("--sims")
                    .arg(args.eval_sims.to_string());

                let eval_status = eval_cmd.status().expect("eval vs previous failed");
                let result_str = format!(
                    "Iter {}: vs prev - {}",
                    i,
                    if eval_status.success() {
                        "NEW WINS (>=55%)"
                    } else {
                        "no significant improvement"
                    }
                );
                println!("{}", result_str);
                eval_results.push(result_str);
            }

            // Eval vs baseline (random/epoch 0 model)
            println!("\n--- Eval: New model vs Baseline (random) ---");
            let mut baseline_cmd =
                Command::new("../othello-self-play/target/release/othello-self-play");
            baseline_cmd
                .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
                .arg("eval")
                .arg("--new-model")
                .arg(&new_model)
                .arg("--old-model")
                .arg(&baseline_model)
                .arg("--games")
                .arg(args.eval_games.to_string())
                .arg("--sims")
                .arg(args.eval_sims.to_string());

            let baseline_status = baseline_cmd.status().expect("eval vs baseline failed");
            let result_str = format!(
                "Iter {}: vs baseline - {}",
                i,
                if baseline_status.success() {
                    "BEATING RANDOM"
                } else {
                    "not beating random yet"
                }
            );
            println!("{}", result_str);
            eval_results.push(result_str);
        }
    }

    // Print summary
    println!("=== Training Complete ===");
    println!("\nConfig: {:?}", args);

    if !eval_results.is_empty() {
        println!("\n--- Evaluation Summary ---");
        for result in &eval_results {
            println!("  {}", result);
        }
    }
}
