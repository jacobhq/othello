use ort::session::Session;
use othello::othello_game::{Color, OthelloGame};
use tracing::info;
use crate::mcts::mcts_search;

/// Represents a single self-play game
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}


pub fn self_play_game(
    iterations: u32,
    mut model: Option<&mut Session>,
) -> Vec<Sample> {
    let mut game = OthelloGame::new();
    let mut samples: Vec<(Sample, Color)> = Vec::new();

    while !game.game_over() {
        let player = game.current_turn;

        let encoded = game.encode(player);

        let (best_move, policy) =
            mcts_search(game, player, iterations, model.as_deref_mut());

        // Store sample with placeholder value
        samples.push((
            Sample {
                state: encoded,
                policy,
                value: 0.0,
            },
            player,
        ));

        match best_move {
            Some((row, col)) => {
                // Normal move
                let _ = game.play(row, col, player);
            }
            None => {
                // PASS TURN
                game.current_turn = match player {
                    Color::White => Color::Black,
                    Color::Black => Color::White,
                };
            }
        }
    }

    // Game ended — compute final outcome
    let (white, black) = game.score();
    let outcome = if white > black {
        Color::White
    } else if black > white {
        Color::Black
    } else {
        // Draw
        // value = 0 for all
        for (sample, _) in samples.iter_mut() {
            sample.value = 0.0;
        }
        return samples.into_iter().map(|(s, _)| s).collect();
    };

    // Assign values from each player's perspective
    for (sample, player) in samples.iter_mut() {
        sample.value = if *player == outcome { 1.0 } else { -1.0 };
    }

    samples.into_iter().map(|(s, _)| s).collect()
}

pub fn generate_self_play_data(
    games: usize,
    mcts_iters: u32,
    mut model: Option<&mut Session>,
) -> Vec<Sample> {
    let mut dataset = Vec::new();

    for g in 0..games {
        let samples = self_play_game(mcts_iters, model.as_deref_mut());
        dataset.extend(samples);

        if g % 10 == 0 {
            info!("Completed {} self-play games", g);
        }
    }

    dataset
}
