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

    while !game.game_over() {
        // 1. Run MCTS for this position
        let root = new_root(game.clone(), player);

        mcts_search_parallel(
            root.clone(),
            eval_queue.clone(),
            sims_per_move,
        );

        // 2. Extract policy (visit counts)
        let policy = extract_policy(&root);

        // 3. Record sample (value filled in later)
        samples.push(Sample {
            state: encode_state(&game, player),
            policy,
            value: 0.0,
        });

        // 4. Play move
        let (row, col) = select_action(&root, 0.0);
        game.play(row, col, player);
        player = player.opponent();
    }

    // ------------------------------------------------------------
    // 5. Assign final game result to all samples
    // ------------------------------------------------------------
    let value = final_value(&game);

    for (i, sample) in samples.iter_mut().enumerate() {
        sample.value = if i % 2 == 0 { value } else { -value };
    }

    samples
}

fn extract_policy(root: &crate::async_mcts::NodeRef) -> Vec<f32> {
    let mut policy = vec![0.0; 64];
    let children = root.children.lock().unwrap();

    let total_visits: f32 = children
        .iter()
        .map(|c| c.stats.visits.load(std::sync::atomic::Ordering::Relaxed) as f32)
        .sum();

    if total_visits > 0.0 {
        for child in children.iter() {
            let (r, c) = child.action.unwrap();
            let idx = r * 8 + c;
            let v = child.stats.visits.load(std::sync::atomic::Ordering::Relaxed) as f32;
            policy[idx] = v / total_visits;
        }
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
