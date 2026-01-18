// use std::sync::Arc;
// use std::thread;
//
// use crate::eval_queue::{EvalQueue, gpu_worker};
// use crate::async_mcts::{new_root, mcts_search_parallel, select_action};
// use crate::neural_net::load_model;
// use othello::othello_game::{Color, OthelloGame};
//
// /// Represents a single self-play game
// #[derive(Clone)]
// pub struct Sample {
//     pub state: [[[i32; 8]; 8]; 2], // encoded board
//     pub policy: Vec<f32>,          // length 64
//     pub value: f32,                // final game result
// }
//
// pub fn search_position(
//     state: OthelloGame,
//     player: Color,
//     model_path: &str,
//     simulations: usize,
// ) -> (usize, usize) {
//     // 1. Create eval queue
//     let eval_queue = Arc::new(EvalQueue::new(1024));
//
//     // 2. Spawn GPU workers
//     let gpu_workers = 1; // usually 1 is best
//     for _ in 0..gpu_workers {
//         let queue = eval_queue.clone();
//         let model = load_model(model_path).expect("Failed to load model");
//
//         thread::spawn(move || {
//             gpu_worker(queue, model, 16);
//         });
//     }
//
//     // 3. Create root
//     let root = new_root(state, player);
//
//     // 4. Run MCTS
//     mcts_search_parallel(root.clone(), eval_queue, simulations);
//
//     // 5. Pick move
//     select_action(&root, 0.0)
// }


use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use anyhow::Result;

use crate::async_mcts::{mcts_search_parallel, new_root, select_action};
use crate::eval_queue::{gpu_worker, EvalQueue};
use crate::neural_net::load_model;
use othello::othello_game::{Color, OthelloGame};

/// Represents a single self-play training sample
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}

/// Public entry point used by main.rs
pub fn generate_self_play_data(
    _prefix: &str,
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

    // ------------------------------------------------------------
    // 1. Create a single EvalQueue shared across all games
    // ------------------------------------------------------------
    let eval_queue = Arc::new(EvalQueue::new(2048));

    // ------------------------------------------------------------
    // 2. Spawn GPU worker(s) ONCE
    // ------------------------------------------------------------
    let gpu_workers = 1; // 1 is optimal for CUDA
    for _ in 0..gpu_workers {
        let queue = eval_queue.clone();
        let model = load_model(&model_path)?;

        thread::spawn(move || {
            gpu_worker(queue, model, 16);
        });
    }

    // ------------------------------------------------------------
    // 3. Run self-play games sequentially
    // ------------------------------------------------------------
    let mut all_samples = Vec::new();

    for _game_idx in 0..games {
        let game_samples =
            run_single_game(eval_queue.clone(), sims_per_move as usize);

        all_samples.extend(game_samples);
    }

    Ok(all_samples)
}

fn run_single_game(
    eval_queue: Arc<EvalQueue>,
    sims_per_move: usize,
) -> Vec<Sample> {
    let mut samples = Vec::new();
    let mut game = OthelloGame::new();
    let mut player = Color::Black;
    let mut move_count = 0;

    while !game.game_over() {
        let root = new_root(game, player);

        mcts_search_parallel(root.clone(), eval_queue.clone(), sims_per_move);

        // Apply root Dirichlet noise only on first search at this position
        let policy = extract_policy(&root, true);

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

    samples
}

fn extract_policy(root: &crate::async_mcts::NodeRef, add_noise: bool) -> Vec<f32> {
    let mut policy = vec![0.0; 64];
    let children = root.children.lock().unwrap();

    let mut total_visits: f32 = children
        .iter()
        .map(|c| c.stats.visits.load(std::sync::atomic::Ordering::Relaxed) as f32)
        .sum();

    if total_visits == 0.0 {
        // fallback: uniform over legal moves
        for child in children.iter() {
            let (r, c) = child.action.unwrap();
            let idx = r * 8 + c;
            policy[idx] = 1.0 / children.len() as f32;
        }
        return policy;
    }

    let mut visit_counts: Vec<f32> = children
        .iter()
        .map(|c| c.stats.visits.load(std::sync::atomic::Ordering::Relaxed) as f32)
        .collect();

    // --- Apply Dirichlet noise if requested ---
    if add_noise {
        let epsilon = 0.25;
        let alpha = 0.3;
        let n = visit_counts.len();

        let noise = crate::distr::dirichlet(alpha, n);

        for (v, n) in visit_counts.iter_mut().zip(noise.iter()) {
            *v = (1.0 - epsilon) * *v + epsilon * *n * total_visits;
        }

        total_visits = visit_counts.iter().sum();
    }

    for (child, &v) in children.iter().zip(visit_counts.iter()) {
        let (r, c) = child.action.unwrap();
        let idx = r * 8 + c;
        policy[idx] = v / total_visits;
    }

    policy
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
