import Counter from "@/components/game/counter.tsx";
import {WasmGame} from "@wasm/othello_wasm";
import {useEffect, useState} from "react";

export default function Board() {
    const [game, setGame] = useState<WasmGame | null>(null);
    const [board, setBoard] = useState<(0 | 1 | 2)[][]>([]);
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

    const handleClick = (i: number, j: number) => {
        try {
            if (!game) {
                return
            }

            console.log(i, j, game.current_player())
            game.play_turn(i, j, game.current_player())
            const newBoard = game.board();
            setBoard(newBoard)

        } catch (e) {
            console.error(e)
        }
    }

    return (
        <div className="grid grid-rows-8 bg-green-600 rounded-md p-2 gap-2 aspect-square max-w-3xl mx-auto">
            {Array.from({ length: 8 }, (_, i) => (
                <div className="grid grid-cols-8 gap-2" key={i}>
                    {Array.from({ length: 8 }, (_, j) => (
                        <div className="bg-green-700 rounded p-2" key={j} onClick={() => handleClick(i, j)}>
                            {board?.[i]?.[j] !== 0 && (
                                <Counter color={board?.[i]?.[j]} />
                            )}
                        </div>
                    ))}
                </div>
            ))}
        </div>
    );
}
