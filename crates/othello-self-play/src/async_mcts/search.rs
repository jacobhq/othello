use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use othello::othello_game::{Color, OthelloError};
use crate::eval_queue::{EvalRequest, EvalResult, SearchHandle};
use crate::async_mcts::tree::{NodeId, Tree};

/// Global counter for eval request ids
static NEXT_EVAL_ID: AtomicU64 = AtomicU64::new(1);

pub trait Game: Clone + Send + Sync + 'static {
    /// Returns legal actions as action indices
    fn legal_moves(&self, player: Color) -> Vec<(usize, usize)>;

    /// Apply action and return next state
    fn play(&mut self, row: usize, col: usize, player: Color) -> Result<(), OthelloError>;

    /// Is terminal?
    fn is_terminal(&self) -> bool;

    /// Terminal value from current player's perspective
    fn terminal_value(&self) -> f32;

    /// Encode state for NN
    fn encode(&self) -> Vec<f32>;
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

    /// Run one MCTS simulation
    pub fn simulate(&mut self, root_state: &G) {
        // 1. Selection
        let mut path = Vec::new();
        let mut node_id = self.tree.root();
        let mut state = root_state.clone();

        loop {
            path.push(node_id);

            if state.is_terminal() {
                let value = state.terminal_value();
                self.tree.backprop(&path, value);
                return;
            }

            if let Some((action, child)) =
                self.tree.select_child(node_id, self.c_puct)
            {
                // Apply virtual loss
                self.tree.add_virtual_loss(child, self.virtual_loss);

                state = state.play(action);
                node_id = child;
            } else {
                // Leaf (unexpanded)
                break;
            }
        }

        // 2. Leaf handling
        let leaf = node_id;

        // Create eval request
        let id = NEXT_EVAL_ID.fetch_add(1, Ordering::Relaxed);
        let request = EvalRequest {
            id,
            state: state.encode(),
        };

        self.eval_queue.push_request(request);

        self.pending.insert(
            id,
            PendingEval {
                path,
                leaf,
                state,
            },
        );
    }

    /// Opportunistically consume NN eval results
    pub fn poll_results(&mut self) {
        while let Some(result) = self.eval_queue.try_pop_result() {
            self.handle_eval_result(result);
        }
    }

    fn handle_eval_result(&mut self, result: EvalResult) {
        let pending = match self.pending.remove(&result.id) {
            Some(p) => p,
            None => return, // stale / duplicate
        };

        let PendingEval { path, leaf, state } = pending;

        // Expand leaf
        self.tree.expand(leaf, &result.policy);

        // Backprop value
        self.tree.backprop(&path, result.value);
    }
}
