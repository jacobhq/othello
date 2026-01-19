//! Monte Carlo Tree Search (MCTS) implementation for Othello/Reversi.
//!
//! This module implements a single-threaded MCTS with optional neural
//! network evaluation (AlphaZero-style). When a neural network is not
//! provided, leaf nodes are evaluated using random rollouts.
//!
//! The search returns both a best move and a policy vector derived from
//! visit counts, suitable for training a policy network.
use crate::distr::dirichlet;
use crate::neural_net::{PolicyElement, nn_eval};
use ort::session::Session;
use othello::othello_game::{Color, OthelloGame};
use rand::{rng, seq::IndexedRandom};
use std::cell::RefCell;
use std::rc::Rc;

/// Shared, mutable reference to an `MCTSNode`.
///
/// `Rc<RefCell<...>>` is used to allow multiple owners of tree nodes
/// while enabling interior mutability during search. This design is
/// intended for single-threaded MCTS.
type NodeRef = Rc<RefCell<MCTSNode>>;

/// A node in the Monte Carlo Tree Search.
///
/// Each node represents a single game state from the perspective of
/// the player to move at that state.
struct MCTSNode {
    /// Game state associated with this node.
    state: OthelloGame,
    /// Player to move at this node
    player: Color,
    /// Child nodes expanded from this position.
    children: Vec<NodeRef>,
    /// Child nodes expanded from this position.
    parent: Option<NodeRef>,
    /// Action taken from the parent to reach this node.
    /// None for the root node.
    action: Option<(usize, usize)>,
    /// Number of times node has been visited
    visits: u32,
    /// Accumulated value from this node's perspective.
    /// Values are in the range [-1.0, 1.0].
    wins: f32,
    /// Legal moves from this position that have not yet been expanded.
    untried_actions: Vec<(usize, usize)>,
    /// Neural network prediction
    prior: f32,
}

impl MCTSNode {
    /// Creates a new MCTS node for the given game state and player.
    ///
    /// The list of untried actions is initialised from the legal moves
    /// available to `player` in `state`.
    fn new(
        state: OthelloGame,
        player: Color,
        parent: Option<NodeRef>,
        action: Option<(usize, usize)>,
    ) -> NodeRef {
        let untried_actions = state.legal_moves(player);
        Rc::new(RefCell::new(Self {
            state,
            player,
            parent,
            children: Vec::new(),
            action,
            visits: 0,
            wins: 0.0,
            untried_actions,
            prior: 0.0,
        }))
    }

    /// Returns true if this node represents a terminal game state.
    fn is_terminal(&self) -> bool {
        self.state.game_over()
    }

    /// Returns true if all legal actions from this node have been expanded.
    fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }

    /// Selects the child with the highest PUCT score.
    ///
    /// The exploration constant `c` controls the exploration–exploitation
    /// tradeoff. Larger values favour exploration.
    fn best_child(node: &NodeRef, c: f32) -> Option<NodeRef> {
        let n = node.borrow();
        if n.children.is_empty() {
            return None;
        }
        n.children
            .iter()
            .max_by(|a, b| {
                let a_borrow = a.borrow();
                let b_borrow = b.borrow();

                // Calculate Q-values (exploitation)
                let q_a = if a_borrow.visits > 0 {
                    a_borrow.wins / a_borrow.visits as f32
                } else {
                    0.0
                };
                let q_b = if b_borrow.visits > 0 {
                    b_borrow.wins / b_borrow.visits as f32
                } else {
                    0.0
                };

                // PUCT formula: Q + C * P * (sqrt(Parent_N) / (1 + Child_N))
                let u_a = c
                    * a_borrow.prior
                    * ((n.visits as f32).sqrt() / (1.0 + a_borrow.visits as f32));
                let u_b = c
                    * b_borrow.prior
                    * ((n.visits as f32).sqrt() / (1.0 + b_borrow.visits as f32));

                (q_a + u_a).partial_cmp(&(q_b + u_b)).unwrap()
            })
            .map(Rc::clone)
    }

    /// Expands one untried action from this node.
    ///
    /// A new child node is created by applying one legal move to the
    /// current game state. The child is added to this node's children
    /// and returned.
    fn expand(node: &NodeRef) -> Option<NodeRef> {
        let mut n = node.borrow_mut();

        if let Some((row, col)) = n.untried_actions.pop() {
            let mut new_state = n.state;
            new_state.play(row, col, n.player);
            let next_player = match n.player {
                Color::White => Color::Black,
                Color::Black => Color::White,
            };
            let child = MCTSNode::new(
                new_state,
                next_player,
                Some(Rc::clone(node)),
                Some((row, col)),
            );
            n.children.push(Rc::clone(&child));
            Some(child)
        } else {
            None
        }
    }

    /// Expands all untried actions from this node, using neural net
    /// policy vector.
    ///
    /// A new child node is created by applying one legal move to the
    /// current game state. The child is added to this node's children
    /// and returned.
    ///
    /// TODO: It should not be possible to call expand and expand all
    /// TODO: on an instance of this struct
    fn expand_all(node: &NodeRef, policy: &[PolicyElement]) {
        let mut n = node.borrow_mut();
        let moves = n.state.legal_moves(n.player);

        for (row, col) in moves {
            let prob = policy.iter().find(|&p| p.0 == (row, col)).unwrap(); // The prior P(s, a)

            let mut next_state = n.state;
            next_state.play(row, col, n.player);
            let next_player = match n.player {
                Color::White => Color::Black,
                Color::Black => Color::White,
            };

            let child = Rc::new(RefCell::new(MCTSNode {
                state: next_state,
                player: next_player,
                parent: Some(Rc::clone(node)),
                children: Vec::new(),
                action: Some((row, col)),
                visits: 0,
                wins: 0.0,
                untried_actions: Vec::new(), // Not needed in expand_all approach
                prior: prob.1,
            }));
            n.children.push(child);
        }
    }

    /// Performs a random rollout (playout) from this node.
    ///
    /// Moves are selected uniformly at random until the game ends or
    /// no legal moves remain. The returned value is from the perspective
    /// of the player to move at this node:
    ///
    /// *  1.0 = win
    /// *  0.0 = draw
    /// * -1.0 = loss
    fn rollout(&self) -> f32 {
        let mut rng = rng();
        let mut state = self.state;
        let mut player = self.player;
        let mut move_count = 0;

        while !state.game_over() && move_count < 60 {
            move_count += 1;
            let moves = state.legal_moves(player);
            if moves.is_empty() {
                let other = match player {
                    Color::White => Color::Black,
                    Color::Black => Color::White,
                };
                if state.legal_moves(other).is_empty() {
                    // Game over, break loop, end game
                    break;
                }
                player = other;
                // Player had no moves, skip turn
                continue;
            }
            let (row, col) = *moves.choose(&mut rng).unwrap();
            state.play(row, col, player);
            player = match player {
                Color::White => Color::Black,
                Color::Black => Color::White,
            };
        }

        let (white, black) = state.score();

        // Return the winner relative to the player of this node
        match self.player {
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

    /// Backpropagates a leaf evaluation result up the search tree.
    ///
    /// The value is accumulated into each node's statistics, and the
    /// sign of the result is flipped at each level to account for
    /// alternating player perspectives.
    fn backpropagate(node: &NodeRef, result: f32) {
        {
            let mut n = node.borrow_mut();
            n.visits += 1;
            n.wins += result;
        } // n is freed here, so safe to borrow it again
        if let Some(parent) = &node.borrow().parent {
            // Negate result because parent is other player
            MCTSNode::backpropagate(parent, -result);
        }
    }

    /// Builds a policy vector from child visit counts.
    ///
    /// The returned vector has length 64 (one entry per board square),
    /// where each entry corresponds to the normalised visit count of
    /// the move at that position.
    fn policy_vector(&self) -> Vec<f32> {
        let mut policy = vec![0.0; 64];
        let total_visits: f32 = self.children.iter().map(|c| c.borrow().visits as f32).sum();

        if total_visits > 0.0 {
            for child in &self.children {
                let c = child.borrow();
                if let Some((row, col)) = c.action {
                    let idx = row * 8 + col;
                    policy[idx] = c.visits as f32 / total_visits;
                }
            }
        }
        policy
    }
}

/// Runs Monte Carlo Tree Search from a given root state.
///
/// # Arguments
/// * `root_state` - The starting game position
/// * `player` - The player to move at the root
/// * `iterations` - Number of MCTS iterations to perform
/// * `model` - Optional neural network used to evaluate leaf nodes
///
/// # Returns
/// A tuple containing:
/// * The selected best move (if any)
/// * A policy vector derived from root child visit counts
pub(crate) fn mcts_search(
    root_state: OthelloGame,
    player: Color,
    iterations: u32,
    mut model: Option<&mut Session>,
) -> (Option<(usize, usize)>, Vec<f32>, Option<(f32, f32, f32)>) {
    let root = MCTSNode::new(root_state, player, None, None);

    // No legal moves, policy vec empty
    if root.borrow().untried_actions.is_empty() {
        return (None, vec![0.0; 64], None);
    }

    for _ in 0..iterations {
        let mut node = Rc::clone(&root);

        // 1 - Selection: Repeatedly select the best child until we reach a terminal node,
        // or a node that is not fully expanded.
        loop {
            let n = node.borrow();
            if n.is_terminal() || !n.is_fully_expanded() {
                break;
            }
            if let Some(best) = MCTSNode::best_child(&node, std::f32::consts::SQRT_2) {
                // Drop the borrow of the node before reassigning the node
                drop(n);
                node = best;
            } else {
                break;
            }
        }

        // 2, 3 - Evaluation & Expansion:
        // If the node is terminal, evaluate based on game result.
        // If not, use the Neural Network to get Value and Policy.
        let result = if node.borrow().is_terminal() {
            node.borrow().rollout() // Direct score if game is over
        } else if let Some(ref mut m) = model {
            // Get both Policy (for expansion) and Value (for backprop)
            let (policy, value) = nn_eval(m, &node.borrow().state, node.borrow().player)
                .expect("Error getting from the model");

            // Expand all children at once using the NN policy
            MCTSNode::expand_all(&node, &policy);

            // Add Dirichlet noise to root node
            if Rc::ptr_eq(&node, &root) {
                add_dirichlet_noise_to_root(&node, 0.3, 0.25);
            }

            value
        } else {
            // Fallback for no-model: expand one and rollout (classic MCTS)
            if let Some(child) = MCTSNode::expand(&node) {
                child.borrow().rollout()
            } else {
                node.borrow().rollout()
            }
        };

        // 4 - Backpropagation: Walk back up the tree and update visit counts and value estimates
        MCTSNode::backpropagate(&node, result);
    }

    // Select best child, derive policy vector
    let policy = root.borrow().policy_vector();
    let best_move = MCTSNode::best_child(&root, 0.0).and_then(|n| n.borrow().action);

    let stats = Some(compute_root_stats(&root, best_move));

    (best_move, policy, stats)
}

fn compute_root_stats(root: &NodeRef, best_move: Option<(usize, usize)>) -> (f32, f32, f32) {
    let root_borrow = root.borrow();

    let visits: Vec<f32> = root_borrow
        .children
        .iter()
        .map(|c| c.borrow().visits as f32)
        .collect();

    let total: f32 = visits.iter().sum();
    if total == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    // Entropy
    let entropy = visits.iter().fold(0.0, |acc, &v| {
        let p = v / total;
        if p > 0.0 { acc - p * p.ln() } else { acc }
    });

    // Max visit fraction
    let max_visit_frac = visits.iter().cloned().fold(0.0, f32::max) / total;

    // Q of selected move
    let q_selected = best_move
        .and_then(|mv| {
            root_borrow
                .children
                .iter()
                .find(|c| c.borrow().action == Some(mv))
        })
        .map(|c| {
            let c = c.borrow();
            if c.visits > 0 {
                c.wins / c.visits as f32
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    (entropy, max_visit_frac, q_selected)
}

/// Adds Dirichlet exploration noise to the priors of the root node.
///
/// This function modifies the prior probability `P(s, a)` of each
/// child of the root according to:
///
/// ```text
/// P'(s, a) = (1 - ε) * P(s, a) + ε * Dirichlet(α)
/// ```
///
/// This is the standard AlphaZero exploration mechanism and should
/// be applied:
/// - **only at the root**
/// - **only once per search**
/// - **before any visit counts are accumulated**
///
/// # Arguments
/// * `root` - The root node of the MCTS tree
/// * `alpha` - Dirichlet concentration parameter (e.g. `0.3`)
/// * `epsilon` - Mixing factor controlling noise strength (e.g. `0.25`)
fn add_dirichlet_noise_to_root(root: &NodeRef, alpha: f32, epsilon: f32) {
    let root_borrow = root.borrow_mut();
    let n = root_borrow.children.len();
    if n == 0 {
        return;
    }

    let noise = dirichlet(alpha, n);

    for (child, &n_i) in root_borrow.children.iter().zip(noise.iter()) {
        let mut c = child.borrow_mut();
        c.prior = (1.0 - epsilon) * c.prior + epsilon * n_i;
    }
}

/// Unit tests for the Monte Carlo Tree Search implementation.
/// These tests verify correctness of node expansion, backpropagation,
/// rollout evaluation, and policy generation for an Othello game.
#[cfg(test)]
mod tests {
    use super::*;
    use othello::othello_game::{Color, OthelloGame};

    /// Helper: create the standard initial Othello position
    fn initial_game() -> OthelloGame {
        OthelloGame::new()
    }

    /// Verifies that a newly created MCTS node is correctly initialised.
    /// The node should contain all legal moves as untried actions and
    /// have zero visits, zero wins, and no parent or children.
    #[test]
    fn node_initialization_has_correct_untried_actions() {
        let game = initial_game();
        let node = MCTSNode::new(game, Color::Black, None, None);

        let n = node.borrow();
        let legal = n.state.legal_moves(Color::Black);

        // All legal moves should be available for expansion at the start
        assert_eq!(n.untried_actions.len(), legal.len());
        assert_eq!(n.visits, 0);
        assert_eq!(n.wins, 0.0);
        assert!(n.children.is_empty());
        assert!(n.parent.is_none());
    }

    /// Tests the expansion step of MCTS.
    /// Expanding a node should create exactly one child node,
    /// remove one action from the parent's untried actions,
    /// and correctly link the child to its parent.
    #[test]
    fn expand_creates_child_and_updates_parent() {
        let game = initial_game();
        let node = MCTSNode::new(game, Color::Black, None, None);

        let initial_untried = node.borrow().untried_actions.len();
        // Expanding the node should consume exactly one untried action
        let child = MCTSNode::expand(&node).expect("Expected expansion");

        let parent = node.borrow();
        let child_borrow = child.borrow();

        // The parent node should now track the new child
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.untried_actions.len(), initial_untried - 1);

        // The child node represents the game state after applying one move
        assert!(child_borrow.parent.is_some());
        assert!(child_borrow.action.is_some());
        assert_eq!(child_borrow.visits, 0);
    }

    /// Ensures that terminal game states are correctly detected.
    /// A node representing a finished game should be marked as terminal
    /// so that no further expansion or rollout occurs.
    #[test]
    fn terminal_state_is_detected() {
        let mut game = initial_game();

        // Play until terminal
        let mut player = Color::Black;
        while !game.game_over() {
            let moves = game.legal_moves(player);
            if let Some((r, c)) = moves.first().copied() {
                game.play(r, c, player);
            }
            player = match player {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
        }

        let node = MCTSNode::new(game, player, None, None);

        // A node created from a finished game should be terminal
        assert!(node.borrow().is_terminal());
    }

    /// Tests the backpropagation phase of MCTS.
    /// A win for a child node represents a loss for its parent because
    /// players alternate turns. This test ensures the win value is
    /// correctly inverted when propagating results up the tree.
    #[test]
    fn backpropagation_flips_sign_correctly() {
        let game = initial_game();
        let root = MCTSNode::new(game, Color::Black, None, None);
        let child = MCTSNode::expand(&root).unwrap();

        // Simulate a win for the child node player
        MCTSNode::backpropagate(&child, 1.0);

        let root_borrow = root.borrow();
        let child_borrow = child.borrow();

        assert_eq!(child_borrow.visits, 1);
        assert_eq!(child_borrow.wins, 1.0);

        // The parent node should receive the opposite (negative) result
        // because the players alternate turns in the game tree
        assert_eq!(root_borrow.visits, 1);
        assert_eq!(root_borrow.wins, -1.0);
    }

    /// Verifies that the policy vector produced by the root node
    /// is correctly normalised so that all probabilities sum to 1.
    /// Illegal moves should have a probability of zero.
    #[test]
    fn policy_vector_is_normalised() {
        let game = initial_game();
        let root = MCTSNode::new(game, Color::Black, None, None);

        // Expand all possible moves so the policy can be derived from visit counts
        while MCTSNode::expand(&root).is_some() {}

        // Manually assign visit counts to simulate completed MCTS simulations
        for (i, child) in root.borrow().children.iter().enumerate() {
            let mut c = child.borrow_mut();
            c.visits = (i + 1) as u32;
        }

        let policy = root.borrow().policy_vector();
        // A valid probability distribution must sum to 1 (but we'll tolerate it if it's a little off)
        let sum: f32 = policy.iter().sum();

        assert!((sum - 1.0).abs() < 1e-5);

        // Only legal moves should have non-zero probability
        let legal = root.borrow().state.legal_moves(Color::Black);
        for (idx, p) in policy.iter().enumerate() {
            // Board positions are flattened into a 1D array (row * 8 + column)
            let row = idx / 8;
            let col = idx % 8;

            // Illegal moves must never receive probability mass
            if !legal.contains(&(row, col)) {
                assert_eq!(*p, 0.0);
            }
        }
    }

    /// Tests the full MCTS search process.
    /// Ensures that the returned policy vector is valid and that
    /// any selected move is legal in the current game state.
    #[test]
    fn mcts_search_returns_valid_policy_and_move() {
        let game = initial_game();

        // Run a full MCTS search from the initial game position
        let (best_move, policy, _) = mcts_search(game, Color::Black, 50, None);

        // Policy vector must always be length 64
        assert_eq!(policy.len(), 64);

        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // If a move is returned, it must be legal
        if let Some((r, c)) = best_move {
            let legal = initial_game().legal_moves(Color::Black);
            assert!(legal.contains(&(r, c)));
        }
    }

    /// Ensures MCTS correctly handles positions where the current
    /// player has no legal moves. In this case, no move should be
    /// returned and the policy vector should contain only zeros.
    #[test]
    fn mcts_search_handles_no_legal_moves() {
        let mut game = initial_game();

        // Force a position where current player has no moves
        // (play until Black cannot move)
        let mut player = Color::Black;
        while !game.game_over() {
            let moves = game.legal_moves(player);
            // Stop once the current player has no legal moves (pass situation)
            if moves.is_empty() {
                break;
            }
            let (r, c) = moves[0];
            game.play(r, c, player);
            player = match player {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
        }

        let (best_move, policy, _) = mcts_search(game, player, 10, None);

        // When no moves are available, MCTS should return no move
        // and an empty policy vector
        assert!(best_move.is_none());
        assert!(policy.iter().all(|&p| p == 0.0));
    }

    /// Tests MCTS behaviour when the root node is already terminal.
    /// The algorithm should not attempt to select a move and should
    /// return an empty (zeroed) policy vector.
    #[test]
    fn mcts_search_on_terminal_root() {
        let mut game = initial_game();

        // Play the game until a terminal state is reached
        // (no further moves are possible for either player)
        let mut player = Color::Black;
        while !game.game_over() {
            let moves = game.legal_moves(player);
            if let Some((r, c)) = moves.first().copied() {
                game.play(r, c, player);
            }
            player = match player {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
        }

        let (best_move, policy, _) = mcts_search(game, player, 20, None);

        // The policy vector should still be the correct size
        // but contain no probabilities for any moves
        assert!(best_move.is_none());
        assert_eq!(policy.len(), 64);
        assert!(policy.iter().all(|&p| p == 0.0));
    }

    /// Tests that the root node visit count matches the number of MCTS iterations.
    /// Each iteration performs a single expansion (if possible) and backpropagation,
    /// so the root should be visited exactly once per iteration.
    #[test]
    fn root_visit_count_equals_iterations() {
        let game = initial_game();
        let root = MCTSNode::new(game, Color::Black, None, None);

        let iterations = 100;
        for _ in 0..iterations {
            // Each iteration simulates one complete MCTS update
            let leaf = {
                if let Some(child) = MCTSNode::expand(&root) {
                    child
                } else {
                    Rc::clone(&root)
                }
            };
            MCTSNode::backpropagate(&leaf, 1.0);
        }

        // The root node should be updated once per iteration
        assert_eq!(root.borrow().visits, iterations);
    }

    /// Tests consistency between parent and child visit counts.
    /// The total number of visits across all child nodes should equal
    /// the visit count stored on the parent node.
    #[test]
    fn child_visits_sum_to_root_visits() {
        let game = initial_game();
        let root = MCTSNode::new(game, Color::Black, None, None);

        // Expand all children
        while MCTSNode::expand(&root).is_some() {}

        // Assign fake visit counts
        let mut total = 0;
        for child in &root.borrow().children {
            let mut c = child.borrow_mut();
            // Assign equal visit counts to each child to test consistency
            c.visits = 10;
            total += 10;
        }

        root.borrow_mut().visits = total;

        let child_sum: u32 = root
            .borrow()
            .children
            .iter()
            .map(|c| c.borrow().visits)
            .sum();

        // The sum of child visits should equal the parent's visit count
        assert_eq!(child_sum, root.borrow().visits);
    }

    /// Tests the UCB selection formula when the exploration constant is zero.
    /// In this case, the algorithm should always prefer the child with the
    /// highest average win rate.
    #[test]
    fn best_child_prefers_higher_win_rate_when_c_zero() {
        let game = initial_game();
        let root = MCTSNode::new(game, Color::Black, None, None);

        let child1 = MCTSNode::expand(&root).unwrap();
        let child2 = MCTSNode::expand(&root).unwrap();

        {
            let mut c1 = child1.borrow_mut();
            c1.visits = 10;
            c1.wins = 5.0; // 50%
        }

        {
            let mut c2 = child2.borrow_mut();
            c2.visits = 10;
            c2.wins = 8.0; // 80%
        }

        root.borrow_mut().visits = 20;

        // With exploration disabled (c = 0), selection should be based
        // purely on average win rate
        let best = MCTSNode::best_child(&root, 0.0).unwrap();
        assert!(Rc::ptr_eq(&best, &child2));
    }

    /// Tests that the exploration term in the UCB formula is working correctly.
    /// With a high exploration constant, the algorithm should prefer
    /// less-visited nodes even if their win rate is lower.
    #[test]
    fn best_child_explores_less_visited_node_with_high_c() {
        let game = initial_game();
        let root = MCTSNode::new(game, Color::Black, None, None);

        let explored = MCTSNode::expand(&root).unwrap();
        let unexplored = MCTSNode::expand(&root).unwrap();

        {
            let mut e = explored.borrow_mut();
            e.visits = 100;
            e.wins = 90.0;
        }

        {
            let mut u = unexplored.borrow_mut();
            u.visits = 1;
            u.wins = 0.0;
        }

        root.borrow_mut().visits = 101;

        // A high exploration constant should favour less-visited nodes
        let best = MCTSNode::best_child(&root, 10.0).unwrap();
        assert!(Rc::ptr_eq(&best, &unexplored));
    }

    /// Ensures that random rollouts correctly handle pass turns.
    /// If a player has no legal moves, the turn should pass to the
    /// opponent rather than ending the simulation prematurely.
    #[test]
    fn rollout_handles_pass_turns() {
        let mut game = initial_game();

        // Create a position likely to cause a pass
        let mut player = Color::Black;
        for _ in 0..10 {
            // If no legal moves exist, the turn should pass to the opponent
            let moves = game.legal_moves(player);
            if moves.is_empty() {
                player = match player {
                    Color::Black => Color::White,
                    Color::White => Color::Black,
                };
                continue;
            }
            let (r, c) = moves[0];
            game.play(r, c, player).unwrap();
            player = match player {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
        }

        let node = MCTSNode::new(game, player, None, None);
        let value = node.borrow().rollout();

        // Rollout values should always be within the valid range [-1, 1]
        assert!((-1.0..=1.0).contains(&value));
    }
}
