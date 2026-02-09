//! WASM-compatible MCTS tree using RefCell instead of Mutex.
//! Single-threaded, no condvar issues.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Index into the tree arena
pub type NodeId = usize;

/// A single MCTS node
pub struct Node {
    pub(crate) inner: RefCell<NodeInner>,
}

/// Content of the MCTS node
pub struct NodeInner {
    /// Prior probability from the policy network
    prior: f32,

    /// Visit count
    visits: u32,

    /// Total value
    value_sum: f32,

    /// Children: action -> node id
    pub(crate) children: HashMap<usize, NodeId>,

    /// Has this node been expanded yet?
    expanded: bool,
}

impl Node {
    /// Creates a new node with a given prior
    fn new(prior: f32) -> Self {
        Self {
            inner: RefCell::new(NodeInner {
                prior,
                visits: 0,
                value_sum: 0.0,
                children: HashMap::new(),
                expanded: false,
            }),
        }
    }
}

/// Arena-style tree for MCTS (WASM-compatible, single-threaded)
pub struct Tree {
    nodes: Rc<RefCell<Vec<Rc<Node>>>>,
}

impl Tree {
    /// Create a new tree with a single root node
    pub fn new() -> Self {
        let root = Rc::new(Node::new(1.0));
        Self {
            nodes: Rc::new(RefCell::new(vec![root])),
        }
    }

    /// Get root node id
    pub fn root(&self) -> NodeId {
        0
    }

    /// Get a node by id
    fn node(&self, id: NodeId) -> Rc<Node> {
        let nodes = self.nodes.borrow();
        Rc::clone(&nodes[id])
    }

    /// Add a new child node, returns its NodeId
    fn add_child(&self, prior: f32) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(Rc::new(Node::new(prior)));
        id
    }

    /// Get or create a child for a specific action (useful for pass moves)
    pub fn get_or_create_child(&self, node_id: NodeId, action: usize, prior: f32) -> NodeId {
        let node = self.node(node_id);
        let mut inner = node.inner.borrow_mut();
        if let Some(&child_id) = inner.children.get(&action) {
            child_id
        } else {
            let child_id = self.add_child(prior);
            inner.children.insert(action, child_id);
            child_id
        }
    }

    /// Expand a node with NN policy output
    ///
    /// `policy` is a sparse list of (action, probability)
    pub fn expand(&self, node_id: NodeId, policy: &[(usize, f32)]) {
        let node = self.node(node_id);
        let mut inner = node.inner.borrow_mut();

        if inner.expanded {
            return;
        }

        for &(action, prob) in policy {
            let child_id = self.add_child(prob);
            inner.children.insert(action, child_id);
        }

        inner.expanded = true;
    }

    /// Select the best child using PUCT
    pub fn select_child(&self, node_id: NodeId, c_puct: f32, sign: f32) -> Option<(usize, NodeId)> {
        let node = self.node(node_id);
        let inner = node.inner.borrow();

        if inner.children.is_empty() {
            return None;
        }

        let parent_visits = inner.visits.max(1) as f32;
        let mut best_score = f32::NEG_INFINITY;
        let mut best = None;

        for (&action, &child_id) in &inner.children {
            let child = self.node(child_id);
            let child_inner = child.inner.borrow();

            let q = if child_inner.visits == 0 {
                0.0
            } else {
                child_inner.value_sum / child_inner.visits as f32
            };

            let q_perspective = q * sign;

            let u = c_puct
                * child_inner.prior
                * (parent_visits.sqrt() / (1.0 + child_inner.visits as f32));

            let score = q_perspective + u;

            if score > best_score {
                best_score = score;
                best = Some((action, child_id));
            }
        }

        best
    }

    /// Backpropagate evaluation result
    pub fn backprop(&self, path: &[NodeId], value_black: f32) {
        for &node_id in path.iter().rev() {
            let node = self.node(node_id);
            let mut inner = node.inner.borrow_mut();

            inner.visits += 1;
            inner.value_sum += value_black;
        }
    }

    /// Get visit count for a child action (used for final policy)
    pub fn child_visits(&self, node_id: NodeId) -> Vec<(usize, u32)> {
        let node = self.node(node_id);
        let inner = node.inner.borrow();

        inner
            .children
            .iter()
            .map(|(&action, &child_id)| {
                let child = self.node(child_id);
                let c = child.inner.borrow();
                (action, c.visits)
            })
            .collect()
    }
}
