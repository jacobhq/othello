import { useEffect, useState } from "react";
import Hero from "@/components/marketing/hero";
import { WasmGame } from "../pkg";

function App() {
    const [game, setGame] = useState<WasmGame | null>(null);
    const [board, setBoard] = useState<number[][]>([]);
    const [currentPlayer, setCurrentPlayer] = useState<number>(1);

    useEffect(() => {
        // Initialize the wasm module on mount
        const g = new WasmGame();
        setGame(g);
        setBoard(g.board());
        setCurrentPlayer(g.current_player());
        console.log("Initial board:", g.board());
        console.log("Legal moves:", g.legal_moves());
        console.log("Score:", g.score());
    }, []);

    const handleClick = () => {
        if (!game) return;

        try {
            // Example: play a move at row 2, col 3 for the current player
            game.play_turn(2, 3, currentPlayer);
            const newBoard = game.board();
            const nextPlayer = game.current_player();
            setBoard(newBoard);
            setCurrentPlayer(nextPlayer);

            console.log("After move:");
            console.table(newBoard);
            console.log("Next player:", nextPlayer);
            console.log("Legal moves:", game.legal_moves());
            console.log("Score:", game.score());
        } catch (e) {
            console.error("Error:", e);
        }
    };

    return (
        <div className="p-8">
            <Hero />
            <button
                onClick={handleClick}
                className="mt-6 px-4 py-2 bg-blue-600 text-white rounded"
            >
                Play a Move
            </button>

            <pre className="mt-4 text-sm text-gray-700">
        {JSON.stringify(board, null, 2)}
      </pre>
        </div>
    );
}

export default App;
