use othello::othello_game::{OthelloError, OthelloGame};
use std::io::{stdin, stdout, Write};

pub(crate) fn pass_and_play(mut game: OthelloGame) {
    loop {
        if game.game_over() {
            println!("Game over!");
            println!("{}", game);
            let (w, b) = game.score();
            println!("Final score — ● White: {}, ○ Black: {}", w, b);
            break;
        }


        println!("It is {}'s turn.", game.current_turn);
        print!("{}", game);


        let legal = game.legal_moves(game.current_turn);
        if legal.is_empty() {
            // Trigger the internal turn swap logic by calling play with dummy coords.
            // The implementation of play handles the swap and returns NoMovesForPlayer.
            let _ = game.play(0, 0, game.current_turn);

            continue;
        }


        // Show legal moves as a hint
        print!("Legal moves: ");
        for (r, c) in &legal {
            print!("({} {}) ", r, c);
        }
        println!();


        println!("Enter your next move (in form: row col), or 'quit' to exit.");
        stdout().flush().map_err(|_| ()).unwrap();


        let mut input = String::new();
        stdin().read_line(&mut input).map_err(|_| ()).unwrap();
        let input = input.trim();


        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Quitting game.");
            break;
        }


        let mut parts = input.split_whitespace();
        let row = match parts.next().and_then(|x| x.parse::<usize>().ok()) {
            Some(v) if v < 8 => v,
            _ => {
                println!("Invalid row (must be 0..7). Try again.");
                continue;
            }
        };
        let col = match parts.next().and_then(|y| y.parse::<usize>().ok()) {
            Some(v) if v < 8 => v,
            _ => {
                println!("Invalid column (must be 0..7). Try again.");
                continue;
            }
        };


        match game.play(row, col, game.current_turn) {
            Ok(()) => {
                // move played successfully; loop continues
            }
            Err(OthelloError::NoMovesForPlayer) => {
                // `play` already switched the turn when it returned this error.
                println!("No moves for that player — turn switched to {}.", game.current_turn);
                continue;
            }
            Err(OthelloError::NotYourTurn) => {
                println!("It's not your turn. Please enter a move for {}.", game.current_turn);
                continue;
            }
            Err(OthelloError::IllegalMove) => {
                println!("Illegal move. That move isn't valid; try again.");
                continue;
            }
        }
    }
}
