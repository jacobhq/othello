//! Synchronous MCTS for single-game play, reusing the tree from othello-self-play.

use crate::neural_net::{async_nn_evaluate, NeuralNet};
use burn::prelude::Backend;
use othello::othello_game::{Color, Move, OthelloGame};
use othello_mcts::shared::game::Game;
use othello_mcts::shared::tree::Tree;
use std::collections::HashSet;
use wasm_bindgen::JsValue;

/// Encode the game state for the neural network (from player's perspective)
fn encode_state(game: &OthelloGame, player: Color) -> Vec<f32> {
    let planes = game.encode(player);

    // Flatten [[[i32;8];8];2] → Vec<f32>
    let mut out = Vec::with_capacity(2 * 8 * 8);
    for p in 0..2 {
        for r in 0..8 {
            for c in 0..8 {
                out.push(planes[p][r][c] as f32);
            }
        }
    }
    out
}

/// Filter policy to legal moves and normalise
fn filter_and_normalize(
    policy: Vec<(usize, f32)>,
    legal_moves: &[(usize, usize)],
) -> Vec<(usize, f32)> {
    let legal_set: HashSet<usize> = legal_moves.iter().map(|(r, c)| r * 8 + c).collect();

    let filtered: Vec<(usize, f32)> = policy
        .into_iter()
        .filter(|(a, _)| legal_set.contains(a))
        .collect();

    let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        filtered.into_iter().map(|(a, p)| (a, p / sum)).collect()
    } else {
        // Uniform if all zero
        let n = legal_moves.len() as f32;
        legal_moves
            .iter()
            .map(|(r, c)| (r * 8 + c, 1.0 / n))
            .collect()
    }
}

/// Run synchronous MCTS and return the best move.
/// Uses the shared Tree from othello-self-play.
pub async fn mcts_search<B: Backend>(
    model: &NeuralNet<B>,
    game: &OthelloGame,
    player: Color,
    num_simulations: u32,
) -> Option<(usize, usize)> {
    let tree = Tree::new();
    let c_puct = 1.5;

    let legal_moves = game.legal_moves(player);

    if legal_moves.is_empty() {
        // Forced pass at root
        tree.expand(tree.root(), &[(64, 1.0)]);
    } else {
        // Initial expansion of root
        let (policy, _) = async_nn_evaluate(model, game, &player).await.unwrap();
        tree.expand(tree.root(), &policy);
    }

    // Run simulations
    for _ in 0..num_simulations {
        let mut path = Vec::new();
        let mut sim_state = game.clone();
        let mut node_id = tree.root();

        // Selection & expansion
        loop {
            path.push(node_id);
            let sign = if sim_state.current_turn == Color::Black {
                1.0
            } else {
                -1.0
            };

            if sim_state.game_over() {
                // Terminal: value from Black's perspective
                let value_black = sim_state.terminal_value(Color::Black);
                tree.backprop(&path, value_black);
                break;
            }

            let current_player = sim_state.current_turn;
            let moves = sim_state.legal_moves(current_player);

            if moves.is_empty() {
                // Pass - create/find pass child (action 64)
                let pass_action = 64;
                let child = tree.get_or_create_child(node_id, pass_action, 1.0);
                // mcts_play handles turn switching
                sim_state.mcts_play(Move::Pass, current_player).ok();
                node_id = child;
                continue;
            }

            if let Some((action, child_id)) = tree.select_child(node_id, c_puct, sign) {
                let (row, col) = (action / 8, action % 8);
                sim_state
                    .mcts_play(Move::Move(row, col), current_player)
                    .unwrap();
                node_id = child_id;
            } else {
                // Leaf node: expand and evaluate
                let (policy, value) = async_nn_evaluate(model, &sim_state, &current_player)
                    .await
                    .unwrap();

                tree.expand(node_id, &policy);

                // Value is from current_player's perspective, convert to Black's
                let leaf_sign = if current_player == Color::Black {
                    1.0
                } else {
                    -1.0
                };
                let value_black = value * leaf_sign;
                tree.backprop(&path, value_black);
                break;
            }
        }
    }

    // Select most visited action using tree's child_visits method
    let visits = tree.child_visits(tree.root());
    let best = visits.iter().max_by_key(|(_, v)| v)?;

    if best.0 == 64 {
        return None; // Pass selected by MCTS
    }

    // Print top moves for debugging
    let mut sorted_visits = visits.clone();
    sorted_visits.sort_by(|a, b| b.1.cmp(&a.1));
    print!("MCTS top moves: ");
    for (action, count) in sorted_visits.iter().take(5) {
        let (r, c) = (action / 8, action % 8);
        print!("({},{}):{} ", r, c, count);
    }
    println!();

    Some((best.0 / 8, best.0 % 8))
}

/// Run MCTS using a JS evaluator function (ONNX Runtime Web).
/// Same logic as `mcts_search` but calls the JS function for leaf evaluation.
pub async fn mcts_search_js(
    eval_fn: &js_sys::Function,
    game: &OthelloGame,
    player: Color,
    num_simulations: u32,
) -> Option<(usize, usize)> {
    let tree = Tree::new();
    let c_puct = 1.5;

    let legal_moves = game.legal_moves(player);

    if legal_moves.is_empty() {
        tree.expand(tree.root(), &[(64, 1.0)]);
    } else {
        let (policy, _) = js_evaluate(eval_fn, game, &player).await.unwrap();
        tree.expand(tree.root(), &policy);
    }

    for _ in 0..num_simulations {
        let mut path = Vec::new();
        let mut sim_state = game.clone();
        let mut node_id = tree.root();

        loop {
            path.push(node_id);
            let sign = if sim_state.current_turn == Color::Black {
                1.0
            } else {
                -1.0
            };

            if sim_state.game_over() {
                let value_black = sim_state.terminal_value(Color::Black);
                tree.backprop(&path, value_black);
                break;
            }

            let current_player = sim_state.current_turn;
            let moves = sim_state.legal_moves(current_player);

            if moves.is_empty() {
                let pass_action = 64;
                let child = tree.get_or_create_child(node_id, pass_action, 1.0);
                sim_state.mcts_play(Move::Pass, current_player).ok();
                node_id = child;
                continue;
            }

            if let Some((action, child_id)) = tree.select_child(node_id, c_puct, sign) {
                let (row, col) = (action / 8, action % 8);
                sim_state
                    .mcts_play(Move::Move(row, col), current_player)
                    .unwrap();
                node_id = child_id;
            } else {
                let (policy, value) = js_evaluate(eval_fn, &sim_state, &current_player)
                    .await
                    .unwrap();

                tree.expand(node_id, &policy);

                let leaf_sign = if current_player == Color::Black {
                    1.0
                } else {
                    -1.0
                };
                let value_black = value * leaf_sign;
                tree.backprop(&path, value_black);
                break;
            }
        }
    }

    let visits = tree.child_visits(tree.root());
    let best = visits.iter().max_by_key(|(_, v)| v)?;

    if best.0 == 64 {
        return None;
    }

    let mut sorted_visits = visits.clone();
    sorted_visits.sort_by(|a, b| b.1.cmp(&a.1));
    print!("MCTS top moves: ");
    for (action, count) in sorted_visits.iter().take(5) {
        let (r, c) = (action / 8, action % 8);
        print!("({},{}):{} ", r, c, count);
    }
    println!();

    Some((best.0 / 8, best.0 % 8))
}

/// Evaluate a position by calling a JS function (ONNX Runtime Web).
/// The JS function should accept Float32Array(128) and return
/// Promise<{policy: Float32Array(64), value: number}>.
async fn js_evaluate(
    eval_fn: &js_sys::Function,
    game: &OthelloGame,
    player: &Color,
) -> Result<(Vec<(usize, f32)>, f32), JsValue> {
    // Encode board: 2 planes of 8x8
    let planes = game.encode(*player);
    let mut data = Vec::with_capacity(2 * 8 * 8);
    for p in 0..2 {
        for r in 0..8 {
            for c in 0..8 {
                data.push(planes[p][r][c] as f32);
            }
        }
    }

    let input = js_sys::Float32Array::from(data.as_slice());

    // Call JS function — returns a Promise
    let promise = eval_fn.call1(&JsValue::NULL, &input)?;
    let result = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise)).await?;

    // Parse result: {policy: Float32Array(64), value: number}
    let policy_js = js_sys::Reflect::get(&result, &JsValue::from_str("policy"))?;
    let value_js = js_sys::Reflect::get(&result, &JsValue::from_str("value"))?;

    let policy_array = js_sys::Float32Array::from(policy_js);
    let log_policy: Vec<f32> = policy_array.to_vec();
    let value = value_js.as_f64().unwrap() as f32;

    // Convert log-probabilities to probabilities
    let policy: Vec<f32> = log_policy.iter().map(|lp| lp.exp()).collect();

    // Filter to legal moves and normalise
    let legal_set: std::collections::HashSet<usize> = game
        .legal_moves(*player)
        .iter()
        .map(|(r, c)| r * 8 + c)
        .collect();

    let filtered: Vec<(usize, f32)> = policy
        .into_iter()
        .enumerate()
        .filter(|(a, _)| legal_set.contains(a))
        .collect();

    let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    let filtered_policy: Vec<(usize, f32)> = if sum > 0.0 {
        filtered.into_iter().map(|(a, p)| (a, p / sum)).collect()
    } else {
        let n = legal_set.len() as f32;
        legal_set.iter().map(|&a| (a, 1.0 / n)).collect()
    };

    Ok((filtered_policy, value))
}
