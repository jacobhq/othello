use crate::async_mcts::tree::{NodeId, Tree};
use crate::eval_queue::{EvalRequest, EvalResult, SearchHandle};
use othello::othello_game::{Color, Move, OthelloError, OthelloGame};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for eval request ids
static NEXT_EVAL_ID: AtomicU64 = AtomicU64::new(1);

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

    fn terminal_value(&self, root_player: Color) -> f32 {
        let (black, white) = self.score();

        let diff = match root_player {
            Color::Black => black as i32 - white as i32,
            Color::White => white as i32 - black as i32,
        };

        if diff > 0 {
            1.0
        } else if diff < 0 {
            -1.0
        } else {
            0.0
        }
    }

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

    /// Run one MCTS simulation
    pub fn simulate(&mut self, root_state: &G) {
        let root_player = root_state.current_player();

        let mut path = Vec::new();
        let mut node_id = self.tree.root();
        let mut state = root_state.clone();

        loop {
            path.push(node_id);

            if state.is_terminal() {
                let value = state.terminal_value(root_player);
                self.tree.backprop(&path, value);
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

                self.tree.add_virtual_loss(child, self.virtual_loss);

                state.play_move(Move::Pass).unwrap();
                node_id = child;
                continue;
            }


            if let Some((action, child)) = self.tree.select_child(node_id, self.c_puct) {
                self.tree.add_virtual_loss(child, self.virtual_loss);

                let (row, col) = action_to_rc(action);
                state.play_move(Move::Move(row, col)).unwrap();

                node_id = child;
            } else {
                break;
            }
        }

        let leaf = node_id;
        let id = NEXT_EVAL_ID.fetch_add(1, Ordering::Relaxed);

        let encoded = state.encode(root_player);

        // FIX 1: insert pending first
        self.pending.insert(id, PendingEval { path, leaf, state });

        // FIX 2: then push request
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

        let PendingEval { path, leaf, state } = pending;

        // 🔴 Remove virtual loss from every node where it was added
        // path[0] is the root → no virtual loss there
        for &node in path.iter().skip(1) {
            self.tree.revert_virtual_loss(node, self.virtual_loss);
        }

        // Filter policy to legal moves only
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

        // Expand leaf with filtered policy
        self.tree.expand(leaf, &filtered_policy);

        // Backprop real value
        self.tree.backprop(&path, result.value);
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
