use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::distr::dirichlet;

/// Index into the tree arena
pub type NodeId = usize;

/// A single MCTS node
pub struct Node {
    pub(crate) inner: Mutex<NodeInner>,
}

/// Content of the MCTS node, should be stored behind a mutex to be thread safe
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
            inner: Mutex::new(NodeInner {
                prior,
                visits: 0,
                value_sum: 0.0,
                children: HashMap::new(),
                expanded: false,
            }),
        }
    }
}

/// Arena-style tree for MCTS
#[derive(Clone)]
pub struct Tree {
    // Shared across clones so worker-local Tree copies still operate on the
    // same underlying node arena.
    nodes: Arc<Mutex<Vec<Arc<Node>>>>,
}

impl Tree {
    /// Create a new tree with a single root node
    pub fn new() -> Self {
        let root = Arc::new(Node::new(1.0));
        Self {
            nodes: Arc::new(Mutex::new(vec![root])),
        }
    }

    /// Get root node id
    pub fn root(&self) -> NodeId {
        0
    }

    /// Get a node by id
    pub fn node(&self, id: NodeId) -> Arc<Node> {
        let nodes = self.nodes.lock().unwrap();
        Arc::clone(&nodes[id])
    }

    /// Add a new child node, returns its NodeId
    pub fn add_child(&self, prior: f32) -> NodeId {
        let mut nodes = self.nodes.lock().unwrap();
        let id = nodes.len();
        nodes.push(Arc::new(Node::new(prior)));
        id
    }

    /// Get or create a child for a specific action (useful for pass moves, not used in this crate)
    #[allow(dead_code)]
    pub fn get_or_create_child(&self, node_id: NodeId, action: usize, prior: f32) -> NodeId {
        let node = self.node(node_id);
        let mut inner = node.inner.lock().unwrap();
        *inner.children.entry(action).or_insert_with(|| self.add_child(prior))
    }

    /// Expand a node with NN policy output
    ///
    /// `policy` is a sparse list of (action, probability)
    pub fn expand(
        &self,
        node_id: NodeId,
        policy: &[(usize, f32)],
    ) {
        let node = self.node(node_id);
        let mut inner = node.inner.lock().unwrap();

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
    pub fn select_child(
        &self,
        node_id: NodeId,
        c_puct: f32,
    ) -> Option<(usize, NodeId)> {
        let node = self.node(node_id);
        let inner = node.inner.lock().unwrap();

        if inner.children.is_empty() {
            return None;
        }

        // Get parent visits, max with one to avoid a ln(0), which is undefined
        let parent_visits = inner.visits.max(1) as f32;

        // Current best score has to be really low, and there is no best child yet
        let mut best_score = f32::NEG_INFINITY;
        let mut best = None;

        // Iterate through the children
        for (&action, &child_id) in &inner.children {
            // Get the child
            let child = self.node(child_id);
            let child_inner = child.inner.lock().unwrap();

            // Calculate values from formula
            let q = if child_inner.visits == 0 {
                0.0
            } else {
                child_inner.value_sum / child_inner.visits as f32
            };

            let u = c_puct
                * child_inner.prior
                * (parent_visits.sqrt() / (1.0 + child_inner.visits as f32));

            // This is what we want to argmax
            let score = q + u;

            // If this is greater than the best score, it must be a better child
            if score > best_score {
                best_score = score;
                best = Some((action, child_id));
            }
        }

        best
    }

    /// Apply virtual loss during selection
    pub fn add_virtual_loss(&self, node_id: NodeId, loss: f32) {
        let node = self.node(node_id);
        let mut inner = node.inner.lock().unwrap();
        inner.visits += 1;
        // Add to value_sum: after Q negation, this makes node less attractive
        inner.value_sum += loss;
    }

    /// Revert virtual loss
    pub fn revert_virtual_loss(&self, node_id: NodeId, loss: f32) {
        let node = self.node(node_id);
        let mut inner = node.inner.lock().unwrap();
        inner.visits -= 1;
        inner.value_sum -= loss;
    }

    /// Backpropagate evaluation result
    pub fn backprop(&self, path: &[NodeId], value: f32) {
        let mut v = value;

        for &node_id in path.iter().rev() {
            let node = self.node(node_id);
            let mut inner = node.inner.lock().unwrap();

            inner.visits += 1;
            inner.value_sum += v;

            // Value is from current player's perspective, and we can alternate because passed moves
            // are in the tree now!
            v = -v;
        }
    }

    /// Get visit count for a child action (used for final policy)
    pub fn child_visits(&self, node_id: NodeId) -> Vec<(usize, u32)> {
        let node = self.node(node_id);
        let inner = node.inner.lock().unwrap();

        // Iterate through children, convert to (action, visit) tuple
        inner
            .children
            .iter()
            .map(|(&action, &child_id)| {
                let child = self.node(child_id);
                let c = child.inner.lock().unwrap();
                (action, c.visits)
            })
            .collect()
    }

    /// Apply Dirichlet noise to a node's children priors.
    ///
    /// New prior = (1 - epsilon) * original_prior + epsilon * noise
    ///
    /// # Arguments
    /// * `node_id` - The node whose children will have noise added
    /// * `alpha` - Dirichlet concentration parameter (typically 0.3 for Othello/chess)
    /// * `epsilon` - Mixing weight for noise (typically 0.25)
    pub fn add_dirichlet_noise(&self, node_id: NodeId, alpha: f32, epsilon: f32) {
        let node = self.node(node_id);
        let inner = node.inner.lock().unwrap();

        let n_children = inner.children.len();

        // Nothing to do if no children
        if n_children == 0 {
            return;
        }

        // Sample Dirichlet noise
        let noise = dirichlet(alpha, n_children);

        // Apply noise to each child's prior
        for (i, &child_id) in inner.children.values().enumerate() {
            let child = self.node(child_id);
            let mut child_inner = child.inner.lock().unwrap();
            child_inner.prior = (1.0 - epsilon) * child_inner.prior + epsilon * noise[i];
        }
    }
}
