use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;
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
    #[arg(long, default_value_t = 800)]
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

    /// Enable model gating (only promote models that beat current best)
    #[arg(long, default_value_t = false)]
    enable_gating: bool,

    /// Win rate threshold for model promotion (default: 0.55 = 55%)
    #[arg(long, default_value_t = 0.55)]
    gating_threshold: f64,

    /// Minimum win rate against random to allow promotion (default: 0.55 = 55%)
    #[arg(long, default_value_t = 0.55)]
    min_random_win_rate: f64,
}

/// Evaluation result from othello-self-play eval command
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MatchResult {
    new_wins: u32,
    old_wins: u32,
    draws: u32,
    total_games: u32,
    win_rate: f64,
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

    // Evaluation results directory
    let evals_dir = PathBuf::from("evals");
    if !args.skip_eval {
        std::fs::create_dir_all(&evals_dir).expect("Failed to create evals directory");
    }

    // Path to the random/initial model (epoch 0) for baseline comparison
    // Always use model 0 as the baseline, even when resuming
    let _baseline_model = format!(
        "../../packages/othello-training/models/{}_0_othello_net_epoch_000.onnx",
        &args.prefix
    );

    // Track the "best" model for gating - initially the starting model
    // When gating is enabled, self-play uses best_model instead of latest
    let mut best_model_idx = model_offset0;

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
        // When gating is enabled, use the best model instead of the latest
        let selfplay_model_idx = if args.enable_gating && i > 0 {
            best_model_idx
        } else {
            model_idx
        };
        let model_in = format!(
            "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
            &args.prefix,
            selfplay_model_idx,
            if selfplay_model_idx == 0 {
                0
            } else {
                args.model_epochs.unwrap_or(2)
            }
        );
        if args.enable_gating && i > 0 {
            println!("Using best model (idx {}) for self-play", best_model_idx);
        }
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

            // Track whether this model passes gating checks
            let mut passes_vs_prev = true;
            let mut passes_vs_random = true;

            // Eval vs best model (when gating enabled) or previous iteration
            // Special case: iteration 1 (first trained model) should always be promoted
            // to avoid bootstrap trap of endlessly training on untrained model's data
            let is_first_trained = i == 0;
            
            if i > 0 {
                // When gating: compare against best trained model
                // But if best_model_idx is 0 (untrained), compare against previous instead
                // This prevents getting stuck in a loop with the untrained model
                let compare_model_idx = if args.enable_gating && best_model_idx > 0 {
                    best_model_idx
                } else {
                    model_idx  // Compare against previous iteration
                };
                let compare_model = format!(
                    "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
                    &args.prefix,
                    compare_model_idx,
                    if compare_model_idx == 0 { 0 } else { args.model_epochs.unwrap_or(2) }
                );

                let vs_prev_json = evals_dir.join(format!("{}_iter{:03}_vs_prev.json", &args.prefix, model_idx + 1));

                println!("\n--- Eval: New model vs {} (idx {}) ---",
                    if args.enable_gating && best_model_idx > 0 { "best model" } else { "previous" },
                    compare_model_idx);

                let mut eval_cmd =
                    Command::new("../othello-self-play/target/release/othello-self-play");
                eval_cmd
                    .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
                    .arg("eval")
                    .arg("--new-model")
                    .arg(&new_model)
                    .arg("--old-model")
                    .arg(&compare_model)
                    .arg("--games")
                    .arg(args.eval_games.to_string())
                    .arg("--sims")
                    .arg(args.eval_sims.to_string())
                    .arg("--output-json")
                    .arg(&vs_prev_json);

                let eval_status = eval_cmd.status().expect("eval vs previous failed");

                // Parse JSON result
                let vs_prev_result: Option<MatchResult> = std::fs::read_to_string(&vs_prev_json)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok());

                let (result_str, win_rate_str) = if let Some(ref result) = vs_prev_result {
                    passes_vs_prev = result.win_rate >= args.gating_threshold;
                    (
                        format!(
                            "Iter {}: vs {} - {} ({:.1}% win rate)",
                            model_idx + 1,
                            if args.enable_gating { "best" } else { "prev" },
                            if passes_vs_prev { "PASS" } else { "FAIL" },
                            result.win_rate * 100.0
                        ),
                        format!("{:.1}%", result.win_rate * 100.0),
                    )
                } else {
                    passes_vs_prev = eval_status.success();
                    (
                        format!(
                            "Iter {}: vs {} - {} (exit code)",
                            model_idx + 1,
                            if args.enable_gating { "best" } else { "prev" },
                            if passes_vs_prev { "PASS" } else { "FAIL" }
                        ),
                        "unknown".to_string(),
                    )
                };
                println!("{}", result_str);
                eval_results.push(result_str);
            }

            // Eval vs true random player
            let vs_random_json = evals_dir.join(format!("{}_iter{:03}_vs_random.json", &args.prefix, model_idx + 1));

            println!("\n--- Eval: New model vs True Random ---");
            let mut baseline_cmd =
                Command::new("../othello-self-play/target/release/othello-self-play");
            baseline_cmd
                .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
                .arg("eval-random")
                .arg("--model")
                .arg(&new_model)
                .arg("--games")
                .arg(args.eval_games.to_string())
                .arg("--sims")
                .arg(args.eval_sims.to_string())
                .arg("--output-json")
                .arg(&vs_random_json);

            let baseline_status = baseline_cmd.status().expect("eval vs random failed");

            // Parse JSON result
            let vs_random_result: Option<MatchResult> = std::fs::read_to_string(&vs_random_json)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());

            let result_str = if let Some(ref result) = vs_random_result {
                // Warn if random win rate is concerning
                if result.win_rate < 0.75 {
                    println!("⚠️  WARNING: Low win rate vs random ({:.1}%) - model may be undertrained",
                        result.win_rate * 100.0);
                }
                passes_vs_random = result.win_rate >= args.min_random_win_rate;
                format!(
                    "Iter {}: vs random - {} ({:.1}% win rate)",
                    model_idx + 1,
                    if result.win_rate >= 0.75 { "GOOD" }
                    else if result.win_rate >= args.min_random_win_rate { "WEAK" }
                    else { "FAIL" },
                    result.win_rate * 100.0
                )
            } else {
                passes_vs_random = baseline_status.success();
                format!(
                    "Iter {}: vs random - {} (exit code)",
                    model_idx + 1,
                    if passes_vs_random { "PASS" } else { "FAIL" }
                )
            };
            println!("{}", result_str);
            eval_results.push(result_str);

            // Model gating decision
            if args.enable_gating {
                // Special case: first trained model (i=0) ALWAYS gets promoted
                // to escape the untrained model's data distribution.
                // First iteration models are expected to be weak, but we need
                // to start training on data from a trained model to improve.
                if is_first_trained {
                    println!("✅ First trained model PROMOTED: idx {} (mandatory bootstrap)", model_idx + 1);
                    if !passes_vs_random {
                        println!("   ⚠️  Note: Only {:.1}% vs random - expected to improve with more iterations", 
                            vs_random_result.as_ref().map(|r| r.win_rate * 100.0).unwrap_or(0.0));
                    }
                    best_model_idx = model_idx + 1;
                } else if i > 0 {
                    if passes_vs_prev && passes_vs_random {
                        println!("✅ Model PROMOTED: new best model is idx {}", model_idx + 1);
                        best_model_idx = model_idx + 1;
                    } else {
                        println!("❌ Model NOT promoted: keeping best model idx {}", best_model_idx);
                        if !passes_vs_prev {
                            println!("   - Failed: did not beat {} by {:.0}%", 
                                if best_model_idx > 0 { "best model" } else { "previous" },
                                args.gating_threshold * 100.0);
                        }
                        if !passes_vs_random {
                            println!("   - Failed: did not beat random by {:.0}%", args.min_random_win_rate * 100.0);
                        }
                    }
                }
            }
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
