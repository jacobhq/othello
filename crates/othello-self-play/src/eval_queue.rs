use crate::neural_net::{nn_eval, nn_eval_batch, PolicyElement};
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
        // 1️⃣ Grab up to `batch_size` pending requests
        let batch = queue.pop_request_batch(batch_size);

        // 2️⃣ Collect states and players
        let states: Vec<OthelloGame> = batch.iter().map(|r| r.state).collect();
        let players: Vec<Color> = batch.iter().map(|r| r.player).collect();

        // 3️⃣ Evaluate all at once on the GPU
        let results: Vec<(Vec<PolicyElement>, f32)> =
            nn_eval_batch(&mut model, &states, &players)
                .expect("Batch NN eval failed");

        // 4️⃣ Send results back to each request
        for (req, (policy, value)) in batch.into_iter().zip(results.into_iter()) {
            let _ = req.reply.send((policy, value));
        }
    }
}
