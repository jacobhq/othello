import Counter from "@/components/game/counter.tsx";
import Score from "@/components/game/score.tsx";
import {WasmGame} from "@wasm/othello_wasm";
import {useEffect, useState} from "react";
import { toast } from "sonner"

export default function Board() {
    const [game, setGame] = useState<WasmGame | null>(null);
    const [board, setBoard] = useState<(0 | 1 | 2)[][]>([]);
    const [currentPlayer, setCurrentPlayer] = useState<1 | 2>(1);
    const [score, setScore] = useState<[number, number]>([0, 0]);

    useEffect(() => {
        // Initialize the wasm module on mount
        const g = new WasmGame();
        setGame(g);
        setBoard(g.board());
        setCurrentPlayer(g.current_player() as 1 | 2);
        setScore([...g.score()] as [number, number]);
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
            setScore([...game.score()] as [number, number])
            setCurrentPlayer(game.current_player() as 2 | 1)

        } catch (e) {
            toast.error(e as string)
        }
    }

    return (
        <>
            <Score currentTurn={currentPlayer} whiteScore={score[0]} blackScore={score[1]} />
            <div className="grid grid-rows-8 bg-green-600 rounded-md p-1 sm:p-2 xl:p-3 gap-1 sm:gap-2 xl:gap-3 aspect-square max-w-3xl mx-auto">
                {Array.from({length: 8}, (_, i) => (
                    <div className="grid grid-cols-8 gap-1 sm:gap-2 xl:gap-3" key={i}>
                        {Array.from({length: 8}, (_, j) => (
                            <div className="bg-green-700 rounded-sm sm:rounded p-1 sm:p-2 xl:p3" key={j} onClick={() => handleClick(i, j)}>
                                {board?.[i]?.[j] !== 0 && (
                                    <Counter color={board?.[i]?.[j]}/>
                                )}
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </>
    );
}
