mod config;

use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;
use std::process::Command;
use tracing::{info, warn};
use crate::config::{Config, LrSchedule};

/// CLI tool orchestrating Rust self-play and Python training loop
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to config file
    #[arg(short, long)]
    config: String,
    /// Disable reduced Dirichlet noise for early iterations (always use eps=0.25)
    #[arg(long, default_value_t = false)]
    #[deprecated]
    no_early_noise_reduction: bool,
    /// Skip loading checkpoint for the first iteration when resuming (start fresh but save checkpoints for subsequent iterations)
    #[arg(long, default_value_t = false)]
    #[deprecated]
    skip_initial_checkpoint: bool,
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

    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let config = Config::try_from_file(&args.config).unwrap();

    // Calculate offsets
    let sp_offset0 = config.self_play.offset.unwrap_or(0);
    let model_offset0 = config.model.offset.unwrap_or(0);

    // Data directory for all self-play output
    let data_dir = "../othello-self-play/data";

    // Evaluation results directory
    let evals_dir = PathBuf::from("evals");
    if !config.eval.skip {
        std::fs::create_dir_all(&evals_dir).expect("Failed to create evals directory");
    }

    // Track the "best" model for gating - initially the starting model
    // When gating is enabled, self-play uses best_model instead of latest
    let mut best_model_idx = model_offset0;

    // Track evaluation results
    let mut eval_results: Vec<String> = Vec::new();

    for i in 0..config.training.iterations {
        let base_offset = sp_offset0 + i * config.self_play.games;
        let model_idx = model_offset0 + i;

        info!("Starting iteration {}", i);

        // Generate dummy model for iteration 0 (only when not resuming)
        if i == 0 && model_offset0 == 0 {
            info!("Generating initial ONNX model for iteration 0...");

            let mut init_cmd = Command::new(python_path);
            init_cmd
                .arg("../../packages/othello-training/main.py")
                .arg("--out-prefix")
                .arg(format!(
                    "../../packages/othello-training/models/{}_{}",
                    config.name, model_idx
                ));
            
            init_cmd.arg("--res-blocks").arg(config.model.res_blocks.to_string());

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
            .arg(config.self_play.games.to_string())
            .arg("--prefix")
            .arg(&config.name)
            .arg("--iteration")
            .arg(actual_iteration.to_string());

        if args.no_early_noise_reduction {
            self_play.arg("--no-early-noise-reduction");
        }
        
        self_play.arg("--sims").arg(config.self_play.sims.to_string());

        // Always pass model (dummy for iteration 0, trained otherwise)
        // When gating is enabled, use the best model instead of the latest
        let selfplay_model_idx = if config.eval.gating && i > 0 {
            best_model_idx
        } else {
            model_idx
        };
        let model_in = format!(
            "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
            &config.name,
            selfplay_model_idx,
            if selfplay_model_idx == 0 {
                0
            } else {
                config.model.epochs
            }
        );
        if config.eval.gating && i > 0 {
            info!("Using best model (idx {}) for self-play", best_model_idx);
        }
        self_play.arg("--model").arg(&model_in);

        assert!(self_play.status().expect("self-play failed").success());

        // Python training with sliding window
        info!("\nTraining on last {} data files", config.training.window);

        // Location to store the model
        let model_out_prefix = format!(
            "../../packages/othello-training/models/{}_{}",
            config.name,
            model_idx + 1
        );

        let mut train = if config.training.num_gpus > 1 {
            let mut cmd = Command::new("../../packages/othello-training/.venv/bin/torchrun");
            cmd.arg("--standalone")
                .arg(format!("--nproc_per_node={}", config.training.num_gpus));
            cmd
        } else {
            Command::new(python_path)
        };

        train.arg("../../packages/othello-training/main.py")
            .arg("--data")
            .arg(data_dir)
            .arg("--window")
            .arg(config.training.window.to_string())
            .arg("--data-prefix")
            .arg(&config.name)
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
                &config.name, model_idx
            );
            train.arg("--checkpoint").arg(&checkpoint_path);
        }

        train.arg("--epochs").arg(config.model.epochs.to_string());
        train.arg("--batch-size").arg(config.model.batch_size.to_string());

        // Compute learning rate based on schedule
        let lr = match config.model.lr_schedule {
            LrSchedule::Constant { lr } => lr,
            LrSchedule::Cosine { lr_start, lr_end } => {
                let total_iterations = config.training.iterations + model_offset0;
                cosine_lr(actual_iteration, total_iterations, lr_start, lr_end)
            }
        };
        info!("Learning rate for iteration {}: {:.6}", actual_iteration, lr);
        train.arg("--lr").arg(lr.to_string());

        train.arg("--res-blocks").arg(config.model.res_blocks.to_string());

        assert!(train.status().expect("training failed").success());

        // Evaluation matches
        if !config.eval.skip {
            let new_model = format!(
                "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
                &config.name,
                model_idx + 1,
                config.model.epochs
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
                let compare_model_idx = if config.eval.gating && best_model_idx > 0 {
                    best_model_idx
                } else {
                    model_idx  // Compare against previous iteration
                };
                let compare_model = format!(
                    "../../packages/othello-training/models/{}_{}_othello_net_epoch_{:03}.onnx",
                    &config.name,
                    compare_model_idx,
                    if compare_model_idx == 0 { 0 } else { config.model.epochs }
                );

                let vs_prev_json = evals_dir.join(format!("{}_iter{:03}_vs_prev.json", &config.name, model_idx + 1));

                info!("Eval: New model vs {} (idx {})",
                    if config.eval.gating && best_model_idx > 0 { "best model" } else { "previous" },
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
                    .arg(config.eval.games.to_string())
                    .arg("--sims")
                    .arg(config.eval.sims.to_string())
                    .arg("--output-json")
                    .arg(&vs_prev_json);

                let eval_status = eval_cmd.status().expect("eval vs previous failed");

                // Parse JSON result
                let vs_prev_result: Option<MatchResult> = std::fs::read_to_string(&vs_prev_json)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok());

                let (result_str, win_rate_str) = if let Some(ref result) = vs_prev_result {
                    passes_vs_prev = result.win_rate >= config.eval.gating_threshold;
                    (
                        format!(
                            "Iter {}: vs {} - {} ({:.1}% win rate)",
                            model_idx + 1,
                            if config.eval.gating { "best" } else { "prev" },
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
                            if config.eval.gating { "best" } else { "prev" },
                            if passes_vs_prev { "PASS" } else { "FAIL" }
                        ),
                        "unknown".to_string(),
                    )
                };
                info!("{}", result_str);
                eval_results.push(result_str);
            }

            // Eval vs true random player
            let vs_random_json = evals_dir.join(format!("{}_iter{:03}_vs_random.json", &config.name, model_idx + 1));

            info!("Eval: New model vs True Random");
            let mut baseline_cmd =
                Command::new("../othello-self-play/target/release/othello-self-play");
            baseline_cmd
                .env("LD_LIBRARY_PATH", "../othello-self-play/target/release")
                .arg("eval-random")
                .arg("--model")
                .arg(&new_model)
                .arg("--games")
                .arg(config.eval.games.to_string())
                .arg("--sims")
                .arg(config.eval.sims.to_string())
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
                    warn!("Low win rate vs random ({:.1}%) - model may be undertrained",
                        result.win_rate * 100.0);
                }
                passes_vs_random = result.win_rate >= config.eval.min_random_win_rate;
                format!(
                    "Iter {}: vs random - {} ({:.1}% win rate)",
                    model_idx + 1,
                    if result.win_rate >= 0.75 { "GOOD" }
                    else if result.win_rate >= config.eval.min_random_win_rate { "WEAK" }
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
            info!("{}", result_str);
            eval_results.push(result_str);

            // Model gating decision
            if config.eval.gating {
                // Special case: first trained model (i=0) ALWAYS gets promoted
                // to escape the untrained model's data distribution.
                // First iteration models are expected to be weak, but we need
                // to start training on data from a trained model to improve.
                if is_first_trained {
                    info!("First trained model PROMOTED: idx {} (mandatory bootstrap)", model_idx + 1);
                    best_model_idx = model_idx + 1;
                } else if i > 0 {
                    if passes_vs_prev && passes_vs_random {
                        info!("Model PROMOTED: new best model is idx {}", model_idx + 1);
                        best_model_idx = model_idx + 1;
                    } else {
                        warn!("Model NOT promoted: keeping best model idx {}", best_model_idx);
                        if !passes_vs_prev {
                            warn!("   - Failed: did not beat {} by {:.0}%",
                                if best_model_idx > 0 { "best model" } else { "previous" },
                                config.eval.gating_threshold * 100.0);
                        }
                        if !passes_vs_random {
                            warn!("   - Failed: did not beat random by {:.0}%", config.eval.min_random_win_rate * 100.0);
                        }
                    }
                }
            }
        }
    }

    // Print summary
    info!("Training Complete");

    // Handle emptiness in case eval was skipped
    if !eval_results.is_empty() {
        info!("Evaluation Summary:");
        for result in &eval_results {
            info!("  {}", result);
        }
    }
}
