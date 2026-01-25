use clap::Parser;
use std::process::Command;

/// CLI tool orchestrating Rust self-play and Python training loop
/// TODO (later): Args should have either `model` or `self_play` prefix to indicate where they are used.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the training run, used in saved models and data organisation.
    #[arg(short, long)]
    prefix: String,
    /// Number of iterations to run training for
    #[arg(short, long)]
    iterations: u32,
    /// Number of self-play games per iteration
    #[arg(long)]
    self_play_games: u32,
    /// Number of MCTS simulations per self-play game
    #[arg(long)]
    self_play_sims: Option<u32>,
    /// Offset to store the training data, useful when resuming a training run
    #[arg(long)]
    self_play_offset: Option<u32>,
    /// Number of epochs to train the model for per iteration
    #[arg(long)]
    model_epochs: Option<u32>,
    /// Model batch size used in training
    #[arg(long)]
    model_batch_size: Option<u32>,
    /// Number of conv blocks to include in the model
    #[arg(long)]
    model_res_blocks: Option<u32>,
    /// Initial learning rate (default: 2e-3). Used with --lr-schedule for cosine annealing.
    #[arg(long, default_value_t = 2e-3)]
    lr_start: f32,
    /// Final learning rate (default: 1e-5). Used with --lr-schedule for cosine annealing.
    #[arg(long, default_value_t = 1e-5)]
    lr_end: f32,
    /// Learning rate schedule: 'cosine' (default) or 'constant'
    #[arg(long, default_value = "cosine")]
    lr_schedule: String,
    /// Offset to store the model at, useful when resuming a training run
    #[arg(long)]
    model_offset: Option<u32>,
    /// Number of past datasets to use per training iteration (size of sliding window)
    #[arg(long, default_value_t = 3)]
    window: u32,
    /// Number of games to play when evaluating the model against random and against the previous iteration
    #[arg(long, default_value_t = 50)]
    eval_games: u32,
    /// Number of simulations to use per move during eval
    #[arg(long, default_value_t = 100)]
    eval_sims: u32,
    /// Skip the eval to reduce training time, good if you are confident in params and just need to train
    #[arg(long, default_value_t = false)]
    skip_eval: bool,
    /// Disable reduced Dirichlet noise for early iterations (always use eps=0.25)
    #[arg(long, default_value_t = false)]
    no_early_noise_reduction: bool,
    /// Skip loading checkpoint for the first iteration when resuming (start fresh but save checkpoints for subsequent iterations)
    #[arg(long, default_value_t = false)]
    skip_initial_checkpoint: bool,
}

/// Compute learning rate for a given iteration using cosine annealing
fn cosine_lr(iteration: u32, total_iterations: u32, lr_start: f32, lr_end: f32) -> f32 {
    // Cosine annealing: lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + cos(pi * t / T))
    let t = iteration as f32;
    let total = total_iterations as f32;
    let cosine_factor = (std::f32::consts::PI * t / total).cos();
    lr_end + 0.5 * (lr_start - lr_end) * (1.0 + cosine_factor)
}

fn main() {
    #[cfg(target_os = "windows")]
    let python_path = "../../packages/othello-training/.venv/Scripts/python.exe";
    #[cfg(not(target_os = "windows"))]
    let python_path = "../../packages/othello-training/.venv/bin/python3";

    let args = Args::parse();

    // Calculate offsets
    let sp_offset0 = args.self_play_offset.unwrap_or(0);
    let model_offset0 = args.model_offset.unwrap_or(0);

    // Data directory for all self-play output
    let data_dir = "../othello-self-play/data";

    // Path to the random/initial model (epoch 0) for baseline comparison
    // Always use model 0 as the baseline, even when resuming
    let baseline_model = format!(
        "../../packages/othello-training/models/{}_0_othello_net_epoch_000.onnx",
        &args.prefix
    );

    // Track evaluation results
    let mut eval_results: Vec<String> = Vec::new();

    for i in 0..args.iterations {
        let base_offset = sp_offset0 + i * args.self_play_games;
        let model_idx = model_offset0 + i;

        println!("=== Iteration {} ===", i);

        // Generate dummy model for iteration 0 (only when not resuming)
        if i == 0 && model_offset0 == 0 {
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

            // Were we able to generate the first model?
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

        // Calculate the actual iteration number (accounting for resume offset)
        let actual_iteration = model_offset0 + i;

        self_play
            .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
            .arg("selfplay")
            .arg("--out")
            .arg(data_dir)
            .arg("--offset")
            .arg(base_offset.to_string())
            .arg("--games")
            .arg(args.self_play_games.to_string())
            .arg("--prefix")
            .arg(&args.prefix)
            .arg("--iteration")
            .arg(actual_iteration.to_string());

        if args.no_early_noise_reduction {
            self_play.arg("--no-early-noise-reduction");
        }

        if let Some(sims) = args.self_play_sims {
            self_play.arg("--sims").arg(sims.to_string());
        }

        // Always pass model (dummy for iteration 0, trained otherwise)
        let model_in = format!(
            "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
            &args.prefix,
            model_idx,
            if model_idx == 0 {
                0
            } else {
                args.model_epochs.unwrap_or(2)
            }
        );
        self_play.arg("--model").arg(&model_in);

        assert!(self_play.status().expect("self-play failed").success());

        // Python training with sliding window
        println!("\nTraining on last {} data files", args.window);

        // Location to store the model
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
            .arg("--data-prefix")
            .arg(&args.prefix)
            .arg("--out-prefix")
            .arg(&model_out_prefix);

        // Load checkpoint from previous iteration (if not the first iteration)
        //
        // Skip if --skip-initial-checkpoint is set and this is the first iteration of this run. This
        // was mainly added because I was had done this code change during a training run and needed
        // it to continue.
        let skip_checkpoint = args.skip_initial_checkpoint && i == 0;
        if model_idx > 0 && !skip_checkpoint {
            let checkpoint_path = format!(
                "../../packages/othello-training/models/{}_{}_checkpoint.pt",
                &args.prefix, model_idx
            );
            train.arg("--checkpoint").arg(&checkpoint_path);
        }

        if let Some(e) = args.model_epochs {
            train.arg("--epochs").arg(e.to_string());
        }
        if let Some(b) = args.model_batch_size {
            train.arg("--batch-size").arg(b.to_string());
        }

        // Compute learning rate based on schedule
        let lr = if args.lr_schedule == "constant" {
            args.lr_start
        } else {
            // Cosine annealing over total iterations (accounting for resume offset)
            let total_iterations = args.iterations + model_offset0;
            cosine_lr(actual_iteration, total_iterations, args.lr_start, args.lr_end)
        };
        println!("Learning rate for iteration {}: {:.6}", actual_iteration, lr);
        train.arg("--lr").arg(lr.to_string());

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

    // Handle emptiness in cas eval was skipped
    if !eval_results.is_empty() {
        println!("\n--- Evaluation Summary ---");
        for result in &eval_results {
            println!("  {}", result);
        }
    }
}
