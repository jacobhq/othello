mod pass_and_play;
mod neural_net;

use std::path::PathBuf;
use crate::pass_and_play::pass_and_play;
use clap::{Parser, ValueEnum};
use othello::othello_game::{Color, OthelloGame};
use serde::Deserialize;
use crate::neural_net::neural_net;

/// Demo version of Othello to test new features before they are released to the website
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Select the game mode to play in
    #[arg(short, long)]
    game_mode: GameMode,
    /// Selects which color is human (only used in neural net mode)
    #[arg(short, long, value_enum, default_value_t=HumanColor::Black)]
    color: HumanColor,
    /// Location of the model relative to the current working directory (only used in neural net mode)
    #[arg(short, long, default_value = "latest.onnx")]
    model: PathBuf
}

/// Represents the type of game to play
#[derive(Clone, ValueEnum, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
enum GameMode {
    PassAndPlay,
    #[default]
    NeuralNet,
}

/// Selects which color is human
#[derive(Clone, ValueEnum, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
enum HumanColor {
    #[default]
    Black,
    White,
}

fn main() {
    let args = Args::parse();
    let game = OthelloGame::new();

    let human_color = match args.color {
        HumanColor::Black => Color::Black,
        HumanColor::White => Color::White
    };

    // Call appropriate function for arg mode
    match args.game_mode {
        GameMode::PassAndPlay => pass_and_play(game),
        GameMode::NeuralNet => neural_net(game, human_color, args.model),
    }
}
