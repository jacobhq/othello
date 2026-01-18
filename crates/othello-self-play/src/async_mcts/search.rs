use crate::async_mcts::tree::{NodeRef, PendingSimulation, complete_simulation, launch_simulation};
use crate::eval_queue::EvalQueue;
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc};

pub fn mcts_search_parallel(root: NodeRef, eval: Arc<EvalQueue>, total_simulations: usize) {
    let threads = rayon::current_num_threads();
    let sims_per_thread = total_simulations / threads.max(1);

    (0..threads).into_par_iter().for_each(|_| {
        mcts_worker(root.clone(), eval.clone(), sims_per_thread);
    });
}

fn mcts_worker(root: NodeRef, eval: Arc<EvalQueue>, max_sims: usize) {
    let mut launched = 0usize;
    let mut in_flight: Vec<PendingSimulation> = Vec::new();

    while launched < max_sims {
        // Try to launch
        if let Some(sim) = launch_simulation(&root, &eval) {
            in_flight.push(sim);
            launched += 1;
        }

        // Poll finished sims
        let mut i = 0;
        while i < in_flight.len() {
            match in_flight[i].reply.try_recv() {
                Ok((policy, value)) => {
                    let sim = in_flight.swap_remove(i);
                    complete_simulation(sim, policy, value);
                }
                Err(mpsc::TryRecvError::Empty) => {
                    i += 1;
                }
                Err(_) => {
                    i += 1;
                }
            }
        }
    }

    // Drain remaining
    for sim in in_flight {
        if let Ok((policy, value)) = sim.reply.recv() {
            complete_simulation(sim, policy, value);
        }
    }
}

pub fn select_action(root: &NodeRef, temperature: f32) -> (usize, usize) {
    let children = root.children.lock().unwrap();

    if temperature == 0.0 {
        // Argmax
        children
            .iter()
            .max_by_key(|c| c.stats.visits.load(Ordering::Relaxed))
            .and_then(|c| c.action)
            .unwrap()
    } else {
        // Softmax over visits
        use rand::distr::Distribution;
        use rand::distr::weighted::WeightedIndex;

        let weights: Vec<f32> = children
            .iter()
            .map(|c| (c.stats.visits.load(Ordering::Relaxed) as f32).powf(1.0 / temperature))
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();
        let idx = dist.sample(&mut rand::rng());

        children[idx].action.unwrap()
    }
}
