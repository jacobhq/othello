use crate::eval_queue::EvalQueue;
use crate::neural_net::PolicyElement;
use atomic_float::AtomicF32;
use othello::othello_game::{Color, OthelloGame};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};

const VIRTUAL_LOSS: u32 = 1;
const PUCT_C: f32 = 1.414;

pub(crate) type NodeRef = Arc<Node>;

pub(crate) struct Node {
    state: OthelloGame,
    player: Color,

    parent: Option<NodeRef>,
    pub(crate) action: Option<(usize, usize)>,

    pub(crate) children: Mutex<Vec<NodeRef>>,
    pub(crate) stats: NodeStats,
    expanded: AtomicBool,
}

pub(crate) struct NodeStats {
    pub(crate) visits: AtomicU32,
    value_sum: AtomicF32,
    virtual_loss: AtomicU32,
    prior: f32,
}

pub(crate) struct PendingSimulation {
    leaf: NodeRef,
    path: Vec<NodeRef>,
    pub(crate) reply: mpsc::Receiver<(Vec<PolicyElement>, f32)>,
}

pub(crate) fn launch_simulation(root: &NodeRef, eval: &EvalQueue) -> Option<PendingSimulation> {
    let mut node = Arc::clone(root);
    let mut path = Vec::new();

    loop {
        path.push(Arc::clone(&node));

        if node.state.game_over() {
            let value = terminal_value(&node.state, node.player);

            for n in &path {
                n.stats
                    .virtual_loss
                    .fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);
            }

            complete_simulation(
                PendingSimulation {
                    leaf: node,
                    path,
                    reply: mpsc::channel().1, // dummy
                },
                Vec::new(),
                value,
            );

            return None;
        }

        // Try to expand once
        if !node.expanded.load(Ordering::Acquire)
            && node
                .expanded
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            // We "own" expansion — enqueue eval
            let (tx, rx) = mpsc::channel();
            eval.push_request_blocking(crate::eval_queue::EvalRequest {
                state: node.state,
                player: node.player,
                reply: tx,
            });

            // Apply virtual loss to path
            for n in &path {
                n.stats
                    .virtual_loss
                    .fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);
            }

            return Some(PendingSimulation {
                leaf: node,
                path,
                reply: rx,
            });
        }

        let best = {
            let children = node.children.lock().unwrap();
            if children.is_empty() {
                return None;
            }

            let parent_visits = node.stats.visits.load(Ordering::Relaxed).max(1);

            children
                .iter()
                .max_by(|a, b| {
                    let sa = &a.stats;
                    let sb = &b.stats;
                    puct_score(parent_visits, sa, PUCT_C)
                        .partial_cmp(&puct_score(parent_visits, sb, PUCT_C))
                        .unwrap()
                })
                .unwrap()
                .clone()
        }; // 👈 children lock DROPPED HERE

        best.stats
            .virtual_loss
            .fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);

        node = best;
    }

    None
}

pub(crate) fn complete_simulation(sim: PendingSimulation, policy: Vec<PolicyElement>, value: f32) {
    let leaf = &sim.leaf;

    // 1. Expand if non-terminal
    if !leaf.state.game_over() {
        let mut children = leaf.children.lock().unwrap();

        // Expand exactly once
        if children.is_empty() {
            for p in policy {
                let mut next_state = leaf.state.clone();
                next_state.play(p.0.0, p.0.1, leaf.player);

                let child = Arc::new(Node {
                    state: next_state,
                    player: leaf.player.opponent(),
                    parent: Some(Arc::clone(leaf)),
                    action: Some((p.0.0, p.0.1)),
                    children: Mutex::new(Vec::new()),
                    expanded: AtomicBool::new(false),
                    stats: NodeStats {
                        visits: AtomicU32::new(0),
                        value_sum: AtomicF32::new(0.0),
                        virtual_loss: AtomicU32::new(0),
                        prior: p.1,
                    },
                });

                children.push(child);
            }
        }
    }

    // 2. Backpropagation
    for node in sim.path.into_iter().rev() {
        // Remove virtual loss
        node.stats
            .virtual_loss
            .fetch_sub(VIRTUAL_LOSS, Ordering::Relaxed);

        node.stats.visits.fetch_add(1, Ordering::Relaxed);

        let signed_value = if node.player == leaf.player {
            value
        } else {
            -value
        };

        node.stats
            .value_sum
            .fetch_add(signed_value, Ordering::Relaxed);
    }
}

pub(crate) fn puct_score(parent_visits: u32, child: &NodeStats, c: f32) -> f32 {
    let visits = child.visits.load(Ordering::Relaxed) + child.virtual_loss.load(Ordering::Relaxed);

    let q = if visits > 0 {
        child.value_sum.load(Ordering::Relaxed) / visits as f32
    } else {
        0.0
    };

    let u = c * child.prior * (parent_visits as f32).sqrt() / (1.0 + visits as f32);

    q + u
}

pub(crate) fn terminal_value(state: &OthelloGame, player: Color) -> f32 {
    let (white, black) = state.score();
    match player {
        Color::White => {
            if white > black {
                1.0
            } else if white < black {
                -1.0
            } else {
                0.0
            }
        }
        Color::Black => {
            if black > white {
                1.0
            } else if black < white {
                -1.0
            } else {
                0.0
            }
        }
    }
}

pub(crate) fn new_root(state: OthelloGame, player: Color) -> NodeRef {
    Arc::new(Node {
        state,
        player,
        parent: None,
        action: None,
        children: Mutex::new(Vec::new()),
        expanded: AtomicBool::new(false),
        stats: NodeStats {
            visits: AtomicU32::new(0),
            value_sum: AtomicF32::new(0.0),
            virtual_loss: AtomicU32::new(0),
            prior: 1.0, // dummy
        },
    })
}
