use crate::neural_net::{nn_eval, PolicyElement};
use ort::session::Session;
use othello::othello_game::{Color, OthelloGame};
use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Condvar, Mutex};

pub(crate) struct EvalRequest {
    pub(crate) state: OthelloGame,
    pub(crate) player: Color,
    pub(crate) reply: mpsc::Sender<(Vec<PolicyElement>, f32)>,
}

pub(crate) struct EvalQueue {
    requests: Mutex<VecDeque<EvalRequest>>,
    cv: Condvar,
    max_size: usize,
}

impl EvalQueue {
    pub(crate) fn new(max_size: usize) -> Self {
        Self {
            requests: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            max_size
        }
    }

    pub(crate) fn push_request_blocking(&self, req: EvalRequest) {
        let mut q = self.requests.lock().unwrap();

        while q.len() >= self.max_size {
            q = self.cv.wait(q).unwrap();
        }

        q.push_back(req);
        self.cv.notify_one();
    }

    fn pop_request_batch(&self, max: usize) -> Vec<EvalRequest> {
        let mut q = self.requests.lock().unwrap();

        while q.is_empty() {
            q = self.cv.wait(q).unwrap();
        }

        let n = q.len().min(max);
        let batch: Vec<_> = q.drain(..n).collect();

        // IMPORTANT: wake producers waiting for space
        self.cv.notify_all();

        batch
    }
}

pub(crate) fn gpu_worker(
    queue: Arc<EvalQueue>,
    mut model: Session,
    batch_size: usize,
) {
    loop {
        let batch = queue.pop_request_batch(batch_size);

        for req in batch {
            let (policy, value) =
                nn_eval(&mut model, &req.state, req.player)
                    .expect("NN eval failed");

            // Send result back to the requesting MCTS thread
            let _ = req.reply.send((policy, value));
        }
    }
}
