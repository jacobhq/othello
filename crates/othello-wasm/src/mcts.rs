//! Synchronous MCTS for single-game play, reusing the tree from othello-self-play.

use othello::othello_game::{Color, Move, OthelloGame};
use othello_mcts::shared::tree::{NodeId, Tree};
use othello_mcts::shared::game::Game;
use wasm_bindgen::JsValue;

/// Batch size for batched MCTS (number of leaves to collect before one NN call).
const BATCH_SIZE: usize = 8;

/// Virtual loss constant applied during batched selection.
const VIRTUAL_LOSS: f32 = 1.0;

/// Information about a leaf reached during batched selection.
struct LeafInfo {
    /// The path from root to this leaf (for backprop).
    path: Vec<NodeId>,
    /// The virtual loss actions taken (node_id, sign).
    /// Used to revert virtual losses with the exact same sign.
    vl_actions: Vec<(NodeId, f32)>,
    /// The simulated game state at this leaf.
    state: OthelloGame,
    /// The current player at this leaf.
    player: Color,
    /// The leaf node id to expand.
    node_id: NodeId,
}

/// Run MCTS using a JS batch evaluator function (ONNX Runtime Web).
/// Collects multiple leaf nodes per batch using virtual losses, then
/// evaluates them all in a single ONNX call.
pub async fn mcts_search_js_batched(
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
        // Bootstrap: evaluate root with a single call (batch of 1)
        let results = js_evaluate_batch(eval_fn, &[(game.clone(), player)])
            .await
            .unwrap();
        let (policy, _) = &results[0];
        tree.expand(tree.root(), policy);
    }

    let mut sims_done: u32 = 0;

    while sims_done < num_simulations {
        // Selection phase: collect up to BATCH_SIZE leaves
        let mut leaves: Vec<LeafInfo> = Vec::with_capacity(BATCH_SIZE);
        // Terminal states resolved immediately (no NN needed)
        // Stores (path, vl_actions, value_black).
        let mut terminal_paths: Vec<(Vec<NodeId>, Vec<(NodeId, f32)>, f32)> = Vec::new();

        for _ in 0..BATCH_SIZE {
            if sims_done >= num_simulations {
                break;
            }
            sims_done += 1;

            let mut path = Vec::new();
            let mut vl_actions = Vec::new();
            let mut sim_state = game.clone();
            let mut node_id = tree.root();

            loop {
                path.push(node_id);
                // Sign for the current player at this node
                let sign = if sim_state.current_turn == Color::Black {
                    1.0
                } else {
                    -1.0
                };

                if sim_state.game_over() {
                    let value_black = sim_state.terminal_value(Color::Black);
                    terminal_paths.push((path, vl_actions, value_black));
                    break;
                }

                let current_player = sim_state.current_turn;
                let moves = sim_state.legal_moves(current_player);

                if moves.is_empty() {
                    let pass_action = 64;
                    // For pass, we just need a specific child
                    let child = tree.get_or_create_child(node_id, pass_action, 1.0);
                    
                    // Apply virtual loss to the child using current player's sign
                    tree.add_virtual_loss(child, VIRTUAL_LOSS, sign);
                    vl_actions.push((child, sign));

                    sim_state.mcts_play(Move::Pass, current_player).ok();
                    node_id = child;
                    continue;
                }

                if let Some((action, child_id)) = tree.select_child(node_id, c_puct, sign) {
                    // Apply virtual loss to the child using current player's sign
                    tree.add_virtual_loss(child_id, VIRTUAL_LOSS, sign);
                    vl_actions.push((child_id, sign));

                    let (row, col) = (action / 8, action % 8);
                    sim_state
                        .mcts_play(Move::Move(row, col), current_player)
                        .unwrap();
                    node_id = child_id;
                } else {
                    // Unexpanded leaf — queue for batch evaluation
                    leaves.push(LeafInfo {
                        path,
                        vl_actions,
                        state: sim_state,
                        player: current_player,
                        node_id,
                    });
                    break;
                }
            }
        }

        // Handle terminal states immediately
        for (path, vl_actions, value_black) in &terminal_paths {
            // Revert virtual losses
            for &(nid, sign) in vl_actions.iter() {
                tree.revert_virtual_loss(nid, VIRTUAL_LOSS, sign);
            }
            tree.backprop(path, *value_black);
        }

        if leaves.is_empty() {
            continue;
        }

        // Batch evaluation
        let inputs: Vec<(OthelloGame, Color)> =
            leaves.iter().map(|l| (l.state.clone(), l.player)).collect();

        let results = js_evaluate_batch(eval_fn, &inputs).await.unwrap();

        // Expand & backprop
        for (leaf, (policy, value)) in leaves.iter().zip(results.iter()) {
            tree.expand(leaf.node_id, policy);

            // Revert virtual losses
            for &(nid, sign) in leaf.vl_actions.iter() {
                tree.revert_virtual_loss(nid, VIRTUAL_LOSS, sign);
            }

            // Value is from current_player's perspective, convert to Black's
            let leaf_sign = if leaf.player == Color::Black {
                1.0
            } else {
                -1.0
            };
            let value_black = value * leaf_sign;
            tree.backprop(&leaf.path, value_black);
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


/// Evaluate multiple positions in a single batched call to the JS function.
/// The JS function should accept (Float32Array, batchSize: number) and return
/// Promise<{policies: Float32Array(batch*64), values: Float32Array(batch)}>.
async fn js_evaluate_batch(
    eval_fn: &js_sys::Function,
    inputs: &[(OthelloGame, Color)],
) -> Result<Vec<(Vec<(usize, f32)>, f32)>, JsValue> {
    let batch_size = inputs.len();

    // Encode all boards into a flat array
    let mut data = Vec::with_capacity(batch_size * 2 * 8 * 8);
    for (game, player) in inputs {
        let planes = game.encode(*player);
        for p in 0..2 {
            for r in 0..8 {
                for c in 0..8 {
                    data.push(planes[p][r][c] as f32);
                }
            }
        }
    }

    let input = js_sys::Float32Array::from(data.as_slice());
    let batch_js = JsValue::from_f64(batch_size as f64);

    // Call JS function — returns a Promise
    let promise = eval_fn.call2(&JsValue::NULL, &input, &batch_js)?;
    let result = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise)).await?;

    // Parse batched result: {policies: Float32Array(batch*64), values: Float32Array(batch)}
    let policies_js = js_sys::Reflect::get(&result, &JsValue::from_str("policies"))?;
    let values_js = js_sys::Reflect::get(&result, &JsValue::from_str("values"))?;

    let policies_flat: Vec<f32> = js_sys::Float32Array::from(policies_js).to_vec();
    let values_flat: Vec<f32> = js_sys::Float32Array::from(values_js).to_vec();

    // Split into per-sample results
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let (game, player) = &inputs[i];
        let log_policy = &policies_flat[i * 64..(i + 1) * 64];
        let value = values_flat[i];

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

        results.push((filtered_policy, value));
    }

    Ok(results)
}


