use crate::async_mcts::tree::{NodeId, Tree};
use crate::eval_queue::{EvalRequest, EvalResult, SearchHandle};
use othello::othello_game::{Color, Move, OthelloError, OthelloGame};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

/// Global counter for eval request ids
static NEXT_EVAL_ID: AtomicU64 = AtomicU64::new(1);

/// A trait was used which will allow us to use code from OthelloGame, but augment it with things
/// that are specific to MCTS.
pub trait Game: Clone + Send + Sync + 'static {
    /// Player to move
    fn current_player(&self) -> Color;

    /// Legal moves for the current player
    fn legal_moves(&self) -> Vec<(usize, usize)>;

    /// Apply a move for the current player
    fn play_move(&mut self, m: Move) -> Result<(), OthelloError>;

    /// Is the game over?
    fn is_terminal(&self) -> bool;

    /// Terminal value from the *root player*'s perspective
    fn terminal_value(&self, root_player: Color) -> f32;

    /// Encode state for NN from a given player's perspective
    fn encode(&self, player: Color) -> Vec<f32>;
}

impl Game for OthelloGame {
    fn current_player(&self) -> Color {
        self.current_turn
    }

    fn legal_moves(&self) -> Vec<(usize, usize)> {
        self.legal_moves(self.current_turn)
    }

    fn play_move(&mut self, m: Move) -> Result<(), OthelloError> {
        self.mcts_play(m, self.current_turn)
    }

    fn is_terminal(&self) -> bool {
        self.game_over()
    }

    /// 'If the game ended now, who would win?'
    fn terminal_value(&self, root_player: Color) -> f32 {
        // Note: score() returns (white_count, black_count)
        let (white, black) = self.score();

        let diff = match root_player {
            Color::Black => black as i32 - white as i32,
            Color::White => white as i32 - black as i32,
        };

        // Convert the score difference into a terminal value from the root player's perspective.
        //  1.0  -> root player is winning
        // -1.0  -> root player is losing
        //  0.0  -> draw (equal score)
        if diff > 0 {
            1.0
        } else if diff < 0 {
            -1.0
        } else {
            0.0
        }
    }

    /// Flatten the game into a vector
    fn encode(&self, player: Color) -> Vec<f32> {
        let planes = self.encode(player);

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
}

/// Single search worker (run this on multiple threads)
pub struct SearchWorker<G: Game> {
    pub tree: Tree,
    pub eval_queue: SearchHandle,
    pub c_puct: f32,
    pub virtual_loss: f32,

    /// Pending evals: request_id -> (path, leaf_node, player_value_sign)
    pending: HashMap<u64, PendingEval<G>>,
}

struct PendingEval<G: Game> {
    path: Vec<NodeId>,
    signs: Vec<f32>,
    leaf: NodeId,
    state: G,
}

impl<G: Game> SearchWorker<G> {
    pub fn new(tree: Tree, eval_queue: SearchHandle) -> Self {
        Self {
            tree,
            eval_queue,
            c_puct: 1.5,
            virtual_loss: 1.0,
            pending: HashMap::new(),
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Run one MCTS simulation
    pub fn simulate(&mut self, root_state: &G) {
        let mut path = Vec::new();
        let mut signs = Vec::new();
        let mut node_id = self.tree.root();
        let mut state = root_state.clone();

        loop {
            path.push(node_id);
            let sign = if state.current_player() == Color::Black { 1.0 } else { -1.0 };
            signs.push(sign);

            if state.is_terminal() {
                // Value must be from Black's perspective for global backprop
                let value_black = state.terminal_value(Color::Black);

                // Revert virtual loss from all traversed nodes (skip root)
                for i in 1..path.len() {
                    let node = path[i];
                    let sign = signs[i - 1]; // Virtual loss was added using parent's sign
                    self.tree.revert_virtual_loss(node, self.virtual_loss, sign);
                }

                self.tree.backprop(&path, value_black);
                return;
            }

            if state.legal_moves().is_empty() {
                // Tree must advance too
                let pass_action = 64;

                // Expand-on-demand: create child if missing
                let child = {
                    let node = self.tree.node(node_id);
                    let mut inner = node.inner.lock().unwrap();

                    *inner.children.entry(pass_action).or_insert_with(|| {
                        self.tree.add_child(1.0) // neutral prior
                    })
                };

                self.tree.add_virtual_loss(child, self.virtual_loss, sign);

                state.play_move(Move::Pass).unwrap();
                node_id = child;
                continue;
            }


            if let Some((action, child)) = self.tree.select_child(node_id, self.c_puct, sign) {
                self.tree.add_virtual_loss(child, self.virtual_loss, sign);

                let (row, col) = action_to_rc(action);
                state.play_move(Move::Move(row, col)).unwrap();

                node_id = child;
            } else {
                break;
            }
        }

        let leaf = node_id;
        let id = NEXT_EVAL_ID.fetch_add(1, Ordering::Relaxed);

        // Encode from leaf's current player perspective for NN
        let encoded = state.encode(state.current_player());

        self.pending.insert(id, PendingEval { path, signs, leaf, state });

        self.eval_queue.push_request(EvalRequest {
            id,
            state: encoded,
        });
    }

    /// Opportunistically consume NN eval results for this worker only
    pub fn poll_results(&mut self) {
        // Collect finished results first
        let mut finished = Vec::new();

        for &id in self.pending.keys() {
            if let Some(result) = self.eval_queue.try_take_result(id) {
                finished.push(result);
            }
        }

        // Now handle them
        for result in finished {
            self.handle_eval_result(result);
        }
    }

    fn handle_eval_result(&mut self, result: EvalResult) {
        let pending = match self.pending.remove(&result.id) {
            Some(p) => p,
            None => return, // stale / duplicate
        };

        let PendingEval { path, signs, leaf, state } = pending;

        // Remove virtual loss from every node where it was added
        for i in 1..path.len() {
            let node = path[i];
            let sign = signs[i - 1];
            self.tree.revert_virtual_loss(node, self.virtual_loss, sign);
        }

        // Filter policy to legal moves and normalize
        let legal_moves = state.legal_moves();
        let legal_actions: std::collections::HashSet<usize> = legal_moves
            .iter()
            .map(|(r, c)| r * 8 + c)
            .collect();

        let filtered_policy: Vec<(usize, f32)> = result
            .policy
            .into_iter()
            .filter(|(action, _)| legal_actions.contains(action))
            .collect();

        // Normalise the filtered policy so priors sum to 1.0
        let sum: f32 = filtered_policy.iter().map(|(_, p)| p).sum();
        let normalised_policy: Vec<(usize, f32)> = if sum > 0.0 {
            filtered_policy
                .into_iter()
                .map(|(a, p)| (a, p / sum))
                .collect()
        } else {
            // Fallback to uniform if all zeros
            warn!("Policy was all zeros, fell back to uniform");
            let n = legal_moves.len() as f32;
            legal_moves
                .iter()
                .map(|(r, c)| (r * 8 + c, 1.0 / n))
                .collect()
        };

        // Expand leaf with normalised policy
        self.tree.expand(leaf, &normalised_policy);

        // Backprop value from Black's perspective
        let leaf_sign = if state.current_player() == Color::Black { 1.0 } else { -1.0 };
        let value_black = result.value * leaf_sign;
        self.tree.backprop(&path, value_black);
    }
}

#[inline]
fn action_to_rc(action: usize) -> (usize, usize) {
    (action / 8, action % 8)
}

#[inline]
fn rc_to_action(row: usize, col: usize) -> usize {
    row * 8 + col
}
