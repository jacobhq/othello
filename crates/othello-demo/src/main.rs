mod pass_and_play;
mod neural_net;

use crate::pass_and_play::pass_and_play;
use clap::{Parser, ValueEnum};
use othello::othello_game::OthelloGame;
use serde::Deserialize;
use crate::neural_net::neural_net;

/// Demo version of Othello to test new features before they are released to the website
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Select the game mode to play in
    #[arg(short, long)]
    mode: GameMode,
}

/// Represents the type of game to play
#[derive(Clone, ValueEnum, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
enum GameMode {
    PassAndPlay,
    #[default]
    NeuralNet,
}

fn main() {
    let args = Args::parse();
    let game = OthelloGame::new();

    // Call appropriate function for arg mode
    match args.mode {
        GameMode::PassAndPlay => pass_and_play(game),
        GameMode::NeuralNet => neural_net(game),
    }
}
