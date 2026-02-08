//! Synchronous MCTS for single-game play, reusing the tree from othello-self-play.

use crate::neural_net::{NeuralNet, nn_evaluate};
use burn::prelude::Backend;
use othello::othello_game::{Color, Move, OthelloGame};
use othello_self_play::async_mcts::{Game, Tree};
use std::collections::HashSet;

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
pub fn mcts_search<B: Backend>(
    model: &NeuralNet<B>,
    game: &OthelloGame,
    player: Color,
    num_simulations: u32,
) -> Option<(usize, usize)> {
    let legal_moves = game.legal_moves(player);
    if legal_moves.is_empty() {
        return None;
    }

    let tree = Tree::new();
    let c_puct = 1.5;

    // Initial expansion of root
    let (policy, _) = nn_evaluate(model, &game, &player).unwrap();
    let normalized = filter_and_normalize(policy, &legal_moves);
    tree.expand(tree.root(), &normalized);

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
                let (policy, value) = nn_evaluate(model, &sim_state, &current_player).unwrap();
                let normalized = filter_and_normalize(policy, &moves);

                tree.expand(node_id, &normalized);

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
