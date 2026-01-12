use ort::execution_providers::CUDAExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use ort::Error;
use othello::othello_game::{Color, OthelloError, OthelloGame};
use std::io::{stdin, stdout, Write};

/// Standardised way to load the model during self-play iterations
pub(crate) fn load_model(path: &str) -> Result<Session, Error> {
    let model = Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?;

    ort::init().with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()]).commit()?;

    Ok(model)
}

/// Evaluate the position and return the best move or an error
pub(crate) fn nn_eval(
    model: &mut Session,
    game: &OthelloGame,
    player: Color,
) -> Result<(usize, usize), Error> {
    // Encode the board state into a (1, 2, 8, 8) tensor from the current player's
    // perspective:
    //   - Channel 0: current player's stones
    //   - Channel 1: opponent's stones
    let mut input: Tensor<f32> = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 2, 8, 8)))?;

    // Build the input tensor by iterating over the board, and marking squares in correct channel
    for row in 0..8 {
        for col in 0..8 {
            if let Some(c) = game.get(row, col) {
                // Map stones to channels relative to the evaluating player
                match (player, c) {
                    (Color::White, Color::White) | (Color::Black, Color::Black) => {
                        // Safe to dangerously cast here, because i64 can represent all of 0..8
                        input[[0, 0, row as i64, col as i64]] = 1.0;
                    }
                    (Color::White, Color::Black) | (Color::Black, Color::White) => {
                        // Safe to dangerously cast here, because i64 can represent all of 0..8
                        input[[0, 1, row as i64, col as i64]] = 1.0;
                    }
                }
            }
        }
    }

    // Run neural network inference; expects policy and value outputs
    let outputs = model.run(ort::inputs!(input))?;
    let policy: Vec<f32> = outputs[0]
        .try_extract_array::<f32>()?
        .iter()
        .copied()
        .collect();
    let value_array = outputs[1].try_extract_tensor::<f32>()?;
    let _value = value_array.1[0]; // Access the first element of the [1, 1] shape

    // Filter the policy to include only legal moves
    let legal = game.legal_moves(player);
    let mut move_probs: Vec<((usize, usize), f32)> = Vec::new();
    for (row, col) in legal {
        // Convert (row, col) into a flat policy index
        let idx = row * 8 + col;
        move_probs.push(((row, col), policy[idx]));
    }

    // Select the best move, drop the probability, and convert it from an Option into a Result
    match move_probs
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(mv, _)| mv) {
        Some(x) => Ok(x),
        None => Err(Error::new("Error selecting the best move"))
    }
}

/// Human vs neural network game mode
pub(crate) fn neural_net(mut game: OthelloGame) {
    let human_color = Color::Black;
    let mut session = load_model("latest.onnx").unwrap();

    // Main game loop
    loop {
        if game.game_over() {
            println!("Game over!");
            println!("{}", game);
            let (w, b) = game.score();
            println!("Final score — ● White: {}, ○ Black: {}", w, b);
            break;
        }

        println!("It is {}'s turn.", game.current_turn);
        println!("{}", game);

        // Human turn
        let mv = if game.current_turn == human_color {
            let legal = game.legal_moves(human_color);

            // Handle turn skip
            if legal.is_empty() {
                println!("{} has no legal moves and must pass.", human_color);
                game.current_turn = match human_color {
                    Color::Black => Color::White,
                    Color::White => Color::Black,
                };
                continue;
            }

            // Show legal moves
            print!("Legal moves: ");
            for (r, c) in &legal {
                print!("({} {}) ", r, c);
            }
            println!();

            // Read move from terminal
            loop {
                print!("Enter your move (row col): ");
                stdout().flush().unwrap();
                let mut input = String::new();
                stdin().read_line(&mut input).unwrap();
                let parts: Vec<&str> = input.trim().split_whitespace().collect();

                // Parse and validate the input
                if parts.len() != 2 {
                    println!("Please enter row and column separated by a space.");
                    continue;
                }

                let row = match parts[0].parse::<usize>() {
                    Ok(r) if r < 8 => r,
                    _ => {
                        println!("Row must be 0..7");
                        continue;
                    }
                };

                let col = match parts[1].parse::<usize>() {
                    Ok(c) if c < 8 => c,
                    _ => {
                        println!("Column must be 0..7");
                        continue;
                    }
                };

                if legal.contains(&(row, col)) {
                    break (row, col);
                } else {
                    println!("That is not a legal move. Try again.");
                }
            }
        } else {
            // NN's turn
            let legal = game.legal_moves(human_color);

            // Handle turn skip
            if legal.is_empty() {
                println!("{} has no legal moves and must pass.", game.current_turn);
                game.current_turn = match game.current_turn {
                    Color::Black => Color::White,
                    Color::White => Color::Black,
                };
                continue;
            }

            match nn_eval(&mut session, &game, game.current_turn) {
                Ok(m) => {
                    println!("NN plays move: ({}, {})", m.0, m.1);
                    m
                }
                Err(_) => {
                    println!("{} has no legal moves and must pass.", game.current_turn);
                    game.current_turn = match game.current_turn {
                        Color::Black => Color::White,
                        Color::White => Color::Black,
                    };
                    continue;
                }
            }
        };

        // Play the move and handle the error
        match game.play(mv.0, mv.1, game.current_turn) {
            Ok(()) => {}
            Err(OthelloError::NoMovesForPlayer) => continue,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
