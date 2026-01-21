use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use std::sync::Arc;

/// Element of a policy vector: (action_index, probability)
pub type PolicyElement = (usize, f32);

/// Request from search thread to GPU worker for neural network evaluation
#[derive(Clone)]
#[derive(Debug)]
pub struct EvalRequest {
    /// Unique identifier for this evaluation request
    pub id: u64,
    /// Game state representation (flattened board state)
    pub state: Vec<f32>,
}

/// Response from GPU worker back to search threads
#[derive(Clone)]
#[derive(Debug)]
pub struct EvalResult {
    /// Matches the id from EvalRequest
    pub id: u64,
    /// Policy vector: sparse representation of (action, probability) pairs
    pub policy: Vec<PolicyElement>,
    /// Value estimate for the position [-1, 1]
    pub value: f32,
}

/// Lock-free queue system for bidirectional communication between
/// search threads and GPU worker
pub struct EvalQueue {
    /// Search threads push requests here (N producers)
    requests: Arc<SegQueue<EvalRequest>>,
    /// GPU worker pushes results here (1 producer)
    results: Arc<DashMap<u64, EvalResult>>,
}

impl EvalQueue {
    /// Create a new evaluation queue system
    pub fn new() -> Self {
        Self {
            requests: Arc::new(SegQueue::new()),
            results: Arc::new(DashMap::new()),
        }
    }

    /// Get a handle for search threads (can be cloned across threads)
    pub fn search_handle(&self) -> SearchHandle {
        SearchHandle {
            requests: Arc::clone(&self.requests),
            results: Arc::clone(&self.results),
        }
    }

    /// Get a handle for the GPU worker (single consumer/producer)
    pub fn gpu_handle(&self) -> GpuHandle {
        GpuHandle {
            requests: Arc::clone(&self.requests),
            results: Arc::clone(&self.results),
        }
    }
}

impl Default for EvalQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for search threads to interact with the queue
#[derive(Clone)]
pub struct SearchHandle {
    requests: Arc<SegQueue<EvalRequest>>,
    results: Arc<DashMap<u64, EvalResult>>,
}

impl SearchHandle {
    /// Push an evaluation request (non-blocking)
    pub fn push_request(&self, request: EvalRequest) {
        self.requests.push(request);
    }

    /// Try to retrieve and remove the evaluation result for `id`
    pub fn try_take_result(&self, id: u64) -> Option<EvalResult> {
        self.results.remove(&id).map(|(_, result)| result)
    }

    /// Check whether the evaluation result for `id` exists
    /// (useful if you want to avoid a remove attempt)
    pub fn has_result(&self, id: u64) -> bool {
        self.results.contains_key(&id)
    }

    /// Check if there are any results available without popping
    pub fn has_results(&self) -> bool {
        !self.results.is_empty()
    }
}

/// Handle for GPU worker to interact with the queue
pub struct GpuHandle {
    requests: Arc<SegQueue<EvalRequest>>,
    results: Arc<DashMap<u64, EvalResult>>,
}

impl GpuHandle {
    /// Pop a single request if available (non-blocking)
    pub fn try_pop_request(&self) -> Option<EvalRequest> {
        self.requests.pop()
    }

    /// Pop multiple requests up to max_batch_size for batching
    /// Returns empty vec if no requests available
    pub fn pop_batch(&self, max_batch_size: usize) -> Vec<EvalRequest> {
        let mut batch = Vec::with_capacity(max_batch_size);

        for _ in 0..max_batch_size {
            if let Some(req) = self.requests.pop() {
                batch.push(req);
            } else {
                break;
            }
        }

        batch
    }

    /// Push evaluation results back (non-blocking)
    pub fn push_results(&self, results: Vec<EvalResult>) {
        for result in results {
            self.results.insert(result.id, result);
        }
    }

    /// Check if there are pending requests
    pub fn has_requests(&self) -> bool {
        !self.requests.is_empty()
    }

    /// Get approximate number of pending requests (may be stale)
    pub fn pending_requests(&self) -> usize {
        // Note: This is an approximation due to concurrent access
        self.requests.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_flow() {
        let queue = EvalQueue::new();
        let search = queue.search_handle();
        let gpu = queue.gpu_handle();

        let req = EvalRequest {
            id: 1,
            state: vec![0.0; 64],
        };
        search.push_request(req);

        let batch = gpu.pop_batch(32);
        assert_eq!(batch.len(), 1);

        gpu.push_results(vec![EvalResult {
            id: 1,
            policy: vec![(0, 0.5)],
            value: 0.2,
        }]);

        let res = search.try_take_result(1).unwrap();
        assert_eq!(res.value, 0.2);
    }

    #[test]
    fn test_batching() {
        let queue = EvalQueue::new();
        let search = queue.search_handle();
        let gpu = queue.gpu_handle();

        // Push multiple requests
        for i in 0..10 {
            search.push_request(EvalRequest {
                id: i,
                state: vec![0.0; 64],
            });
        }

        // Pop batch of 5
        let batch = gpu.pop_batch(5);
        assert_eq!(batch.len(), 5);

        // Pop remaining
        let batch2 = gpu.pop_batch(10);
        assert_eq!(batch2.len(), 5);
    }
}