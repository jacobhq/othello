//! othello-self-play library exports for reuse in other crates

pub mod async_mcts;
pub mod distr;
pub mod eval_queue;

// Re-export commonly used types
pub use async_mcts::{Game, SearchWorker, Tree};
pub use eval_queue::{EvalQueue, EvalRequest, EvalResult, GpuHandle, SearchHandle};
