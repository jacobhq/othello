//! Synchronous MCTS for single-game play, reusing the tree from othello-self-play.

use ort::session::Session;
use ort::value::Tensor;
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

/// Evaluate position with neural network, returns (sparse_policy, value)
fn nn_evaluate(session: &mut Session, state: &[f32]) -> (Vec<(usize, f32)>, f32) {
    let input: Tensor<f32> = Tensor::from_array(([1, 2, 8, 8], state.to_vec())).unwrap();

    let outputs = session.run(ort::inputs!(input)).unwrap();

    // Extract policy (log probabilities) and convert to probs
    let log_policy: Vec<f32> = outputs[0]
        .try_extract_array::<f32>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    let policy: Vec<f32> = log_policy.iter().map(|&lp| lp.exp()).collect();

    // Extract value
    let value_array = outputs[1].try_extract_tensor::<f32>().unwrap();
    let value = value_array.1[0];

    // Return full sparse policy
    let sparse_policy: Vec<(usize, f32)> = policy.into_iter().enumerate().collect();
    (sparse_policy, value)
}

/// Filter policy to legal moves and normalize
fn filter_and_normalize(policy: Vec<(usize, f32)>, legal_moves: &[(usize, usize)]) -> Vec<(usize, f32)> {
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
        legal_moves.iter().map(|(r, c)| (r * 8 + c, 1.0 / n)).collect()
    }
}

/// Run synchronous MCTS and return the best move.
/// Uses the shared Tree from othello-self-play.
pub fn mcts_search(
    session: &mut Session,
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
    let state = encode_state(game, player);
    let (policy, _) = nn_evaluate(session, &state);
    let normalized = filter_and_normalize(policy, &legal_moves);
    tree.expand(tree.root(), &normalized);

    // Run simulations
    for _ in 0..num_simulations {
        let mut path = vec![tree.root()];
        let mut sim_state = game.clone();
        let mut current_player = player;
        let mut node_id = tree.root();

        // Selection & expansion
        loop {
            if sim_state.game_over() {
                // Terminal: backprop actual result using Game trait
                let value = sim_state.terminal_value(current_player);
                tree.backprop(&path, value);
                break;
            }

            let moves = sim_state.legal_moves(current_player);
            if moves.is_empty() {
                // Pass - create/find pass child (action 64)
                let pass_action = 64;
                let child = tree.get_or_create_child(node_id, pass_action, 1.0);
                path.push(child);
                node_id = child;
                current_player = current_player.opponent();
                // Play pass move
                sim_state.mcts_play(Move::Pass, current_player.opponent()).ok();
                continue;
            }

            if let Some((action, child_id)) = tree.select_child(node_id, c_puct) {
                path.push(child_id);
                node_id = child_id;

                let (row, col) = (action / 8, action % 8);
                sim_state.mcts_play(Move::Move(row, col), current_player).unwrap();
                current_player = current_player.opponent();
            } else {
                // Leaf node: expand and evaluate
                let encoded = encode_state(&sim_state, current_player);
                let (policy, value) = nn_evaluate(session, &encoded);
                let normalized = filter_and_normalize(policy, &moves);

                tree.expand(node_id, &normalized);

                // Value is from current_player's perspective, backprop handles sign flips
                tree.backprop(&path, value);
                break;
            }
        }
    }

    // Select most visited action using tree's child_visits method
    let visits = tree.child_visits(tree.root());
    let best = visits.iter().max_by_key(|(_, v)| v)?;

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
