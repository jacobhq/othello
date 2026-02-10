import Counter from "@/components/game/counter.tsx";
import Score from "@/components/game/score.tsx";
import { WasmGame } from "@wasm/othello_wasm";
import { useEffect, useState, useRef } from "react";
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { createEvaluator } from "@/lib/onnx-inference";
import {
    Dialog,
    DialogClose,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import posthog from "posthog-js";
import { Link } from "@tanstack/react-router";
import LegalMoveDot from "@/components/game/legal-move-dot.tsx";

export default function DemoBoard() {
    const [game, setGame] = useState<WasmGame | null>(null);
    const [board, setBoard] = useState<(0 | 1 | 2)[][]>([]);
    const [legalMoves, setLegalMoves] = useState<[number, number][]>([]);
    const [currentPlayer, setCurrentPlayer] = useState<1 | 2>(1);
    const [score, setScore] = useState<[number, number]>([0, 0]);
    const [gameOver, setGameOver] = useState(false);
    const [firstMove, setFirstMove] = useState(true);
    const evaluatorRef = useRef<((input: Float32Array) => Promise<{ policy: Float32Array, value: number }>) | null>(null);

    useEffect(() => {
        initialiseGame()
    }, []);

    const initialiseGame = async () => {
        // DeviceType 3 = OnnxWeb (no burn model loaded)
        const g = new WasmGame(2, 3);

        // Load ONNX evaluator if not already loaded
        if (!evaluatorRef.current) {
            evaluatorRef.current = await createEvaluator();
        }
        g.set_evaluator(evaluatorRef.current);

        setGame(g);
        setBoard(g.board());
        setLegalMoves(g.legal_moves());
        setCurrentPlayer(g.current_player() as 1 | 2);
        setScore([...g.score()] as [number, number]);
    }

    const handleClick = (i: number, j: number) => {
        try {
            if (!game || isAiThinking || currentPlayer !== 1) {
                return
            }

            console.log(i, j, game.current_player())
            game.play_turn(i, j, game.current_player())
            const newBoard = game.board();
            setBoard(newBoard)
            setLegalMoves(game.legal_moves())
            setScore([...game.score()] as [number, number])
            setCurrentPlayer(game.current_player() as 2 | 1)
            setGameOver(game.game_over())

            if (firstMove) {
                posthog.capture("game_started", {
                    type: "demo"
                })
            }

            if (game.game_over()) {
                posthog.capture("game_terminated", {
                    type: "demo",
                    winner: score[0] > score[1] ? "white" : "black"
                })
            }

            setFirstMove(false)
        } catch (e) {
            posthog.capture("error", {
                type: "demo",
                text: e as string,
                row: i,
                col: j
            })
            toast.error(e as string)
        }
    }

    // Add this state
    const [isAiThinking, setIsAiThinking] = useState(false);

    // Add this useEffect after your existing useEffect
    useEffect(() => {
        if (game && currentPlayer === 2 && !gameOver && !isAiThinking) {
            setLegalMoves([]);
            const aiPromise = playAiMove();

            toast.promise(aiPromise, {
                position: "bottom-center",
                loading: "AI is thinking",
                success: "AI played. Your turn!",
                error: 'Error',
            })
        }
    }, [currentPlayer, game, gameOver]);

    // Add this function before handleClick
    const playAiMove = async () => {
        if (!game || isAiThinking) return;
        setIsAiThinking(true);
        try {
            await game.play_ai_move();
            setBoard(game.board());
            setLegalMoves(game.legal_moves());
            setScore([...game.score()] as [number, number]);
            setCurrentPlayer(game.current_player() as 2 | 1);
            setGameOver(game.game_over());
        } catch (e) {
            toast.error(e as string);
        } finally {
            setIsAiThinking(false);
        }
    }

    return (
        <>
            <Dialog open={gameOver} onOpenChange={(open) => {
                !open && initialiseGame()
                setGameOver(open)
            }}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle>{score[0] > score[1] ? "White" : "Black"} wins!</DialogTitle>
                        <DialogDescription>
                            If you enjoyed Othello, consider signing up for an account with us.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="mt-4">
                        <Score whiteScore={score[0]}
                            blackScore={score[1]} />
                    </div>
                    <DialogFooter>
                        <DialogClose asChild>
                            <Button variant="outline">Close</Button>
                        </DialogClose>
                        <Button asChild>
                            <Link to="/auth/signup">Create account</Link>
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
            <Score highlightedPlayer={gameOver ? undefined : currentPlayer} whiteScore={score[0]} blackScore={score[1]} />
            <div
                id="game"
                className="grid grid-rows-8 bg-green-600 rounded-md p-1 sm:p-2 xl:p-3 gap-1 sm:gap-2 xl:gap-3 aspect-square max-w-3xl mx-auto">
                {Array.from({ length: 8 }, (_, i) => (
                    <div className="grid grid-cols-8 gap-1 sm:gap-2 xl:gap-3" key={i}>
                        {Array.from({ length: 8 }, (_, j) => (
                            <div className="bg-green-700 rounded-sm sm:rounded p-1 sm:p-2 xl:p3" key={j}
                                onClick={() => handleClick(i, j)}>
                                {board?.[i]?.[j] !== 0 && (
                                    <Counter color={board?.[i]?.[j]} />
                                )}
                                {legalMoves.some(item => JSON.stringify(item) === JSON.stringify([i, j])) && <LegalMoveDot />}
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </>
    );
}
