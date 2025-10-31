use std::io::{Write, stdin, stdout};
use othello::othello_game::{Color, OthelloError, OthelloGame};

fn main() {
    let mut game = OthelloGame::new();

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
            // No legal moves for current player — announce and pass
            println!("{} has no legal moves and must pass.", game.current_turn);
            let next = match game.current_turn {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
            game.current_turn = next;
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