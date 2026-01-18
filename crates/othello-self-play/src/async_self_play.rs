use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use crate::async_mcts::{mcts_search_parallel, new_root, select_action};
use crate::eval_queue::{EvalQueue, gpu_worker};
use crate::neural_net::load_model;
use anyhow::Result;
use othello::othello_game::{Color, OthelloGame};
use serde::Serialize;
use tracing::debug;

/// Represents a single self-play training sample
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}

/// MCTS statistics for logging
#[derive(Clone, Serialize)]
pub struct MctsStats {
    pub game_id: usize,
    pub move_idx: usize,
    pub entropy: f32,
    pub max_visit_frac: f32,
    pub q_selected: f32,
}

/// Public entry point used by main.rs
pub fn generate_self_play_data(
    prefix: &str,
    games: usize,
    sims_per_move: u32,
    model: Option<PathBuf>,
) -> Result<Vec<Sample>> {
    let model_path = model
        .as_ref()
        .expect("Async MCTS requires a neural network model")
        .to_str()
        .unwrap()
        .to_string();

    // 1. Create a single EvalQueue shared across all games
    let eval_queue = Arc::new(EvalQueue::new(4096));

    // 2. Spawn GPU worker(s)
    let gpu_workers = 1; // adjust if GPU allows
    for _ in 0..gpu_workers {
        let queue = eval_queue.clone();
        let model = load_model(&model_path)?;
        thread::spawn(move || {
            gpu_worker(queue, model, 1024);
        });
    }

    // 3. Run self-play games in parallel
    let results: Vec<(Vec<Sample>, Vec<MctsStats>)> = (0..games)
        .map(|game_idx| run_single_game(game_idx, eval_queue.clone(), sims_per_move as usize))
        .collect();

    // Flatten all the samples and apply symmetries
    let all_samples: Vec<Sample> = results
        .iter()
        .flat_map(|(s, _)| {
            s.clone()
                .into_iter()
                .flat_map(crate::symmetry::get_symmetries)
        })
        .collect();
    let all_stats: Vec<MctsStats> = results.iter().flat_map(|(_, st)| st.clone()).collect();

    // Write MCTS stats to JSONL
    {
        let file = File::create(format!("{}_mcts_stats.jsonl", prefix))?;
        let mut writer = BufWriter::new(file);

        for stat in &all_stats {
            serde_json::to_writer(&mut writer, stat)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }

    Ok(all_samples)
}

fn run_single_game(
    game_id: usize,
    eval_queue: Arc<EvalQueue>,
    sims_per_move: usize,
) -> (Vec<Sample>, Vec<MctsStats>) {
    debug!("Starting self-play game {}", game_id);

    let mut samples = Vec::new();
    let mut stats = Vec::new();
    let mut game = OthelloGame::new();
    let mut player = Color::Black;
    let mut move_count = 0;

    while !game.game_over() {
        let root = new_root(game, player);

        mcts_search_parallel(root.clone(), eval_queue.clone(), sims_per_move);

        // Extract policy and move stats
        let (policy, entropy, max_visit_frac, q_selected) =
            extract_policy_and_stats(&root, move_count < 10);

        stats.push(MctsStats {
            game_id,
            move_idx: move_count,
            entropy,
            max_visit_frac,
            q_selected,
        });

        samples.push(Sample {
            state: encode_state(&game, player),
            policy,
            value: 0.0,
        });

        let temperature = if move_count < 10 { 1.0 } else { 0.0 };
        let (row, col) = select_action(&root, temperature);

        game.play(row, col, player);
        player = player.opponent();
        move_count += 1;
    }

    let value = final_value(&game);
    for (i, sample) in samples.iter_mut().enumerate() {
        sample.value = if i % 2 == 0 { value } else { -value };
    }

    debug!("Game {} finished: final value {:.2}", game_id, value);

    (samples, stats)
}

/// Extract policy probabilities and simple stats for logging
fn extract_policy_and_stats(
    root: &crate::async_mcts::NodeRef,
    add_noise: bool,
) -> (Vec<f32>, f32, f32, f32) {
    let mut policy = vec![0.0; 64];
    let children = root.children.lock().unwrap();

    let mut total_visits: f32 = children
        .iter()
        .map(|c| c.stats.visits.load(std::sync::atomic::Ordering::Relaxed) as f32)
        .sum();

    if total_visits == 0.0 {
        for child in children.iter() {
            let (r, c) = child.action.unwrap();
            policy[r * 8 + c] = 1.0 / children.len() as f32;
        }
        return (policy, 0.0, 0.0, 0.0);
    }

    let mut visit_counts: Vec<f32> = children
        .iter()
        .map(|c| c.stats.visits.load(std::sync::atomic::Ordering::Relaxed) as f32)
        .collect();

    if add_noise {
        let epsilon = 0.25;
        let alpha = 0.3;
        let noise = crate::distr::dirichlet(alpha, visit_counts.len());
        for (v, n) in visit_counts.iter_mut().zip(noise.iter()) {
            *v = (1.0 - epsilon) * *v + epsilon * *n * total_visits;
        }
        total_visits = visit_counts.iter().sum();
    }

    for (child, &v) in children.iter().zip(visit_counts.iter()) {
        let (r, c) = child.action.unwrap();
        policy[r * 8 + c] = v / total_visits;
    }

    let entropy = -policy
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>();
    let max_visit_frac = visit_counts
        .iter()
        .copied()
        .fold(0.0f32, |a, b| a.max(b / total_visits));
    let q_selected = 0.0; // placeholder: could be mean Q of selected action if available

    (policy, entropy, max_visit_frac, q_selected)
}

fn encode_state(game: &OthelloGame, player: Color) -> [[[i32; 8]; 8]; 2] {
    let mut planes = [[[0; 8]; 8]; 2];
    for r in 0..8 {
        for c in 0..8 {
            if let Some(col) = game.get(r, c) {
                let idx = if col == player { 0 } else { 1 };
                planes[idx][r][c] = 1;
            }
        }
    }
    planes
}

fn final_value(game: &OthelloGame) -> f32 {
    let (white, black) = game.score();
    if black > white {
        1.0
    } else if white > black {
        -1.0
    } else {
        0.0
    }
}
