console.log("useOthelloGame loaded");

import { useRef, useState, useEffect, useCallback } from "react";
import type { WasmGame as WasmGameType } from "@wasm/othello_wasm";
import { createBatchEvaluator } from "@/lib/onnx-inference";
import { toast } from "sonner";
import posthog from "posthog-js";

interface GameState {
    board: (0 | 1 | 2)[][];
    legalMoves: [number, number][];
    score: [number, number];
    currentPlayer: 1 | 2;
    gameOver: boolean;
    isAiThinking: boolean;
}

const INITIAL_STATE: GameState = {
    board: Array(8).fill(null).map(() => Array(8).fill(0)),
    legalMoves: [],
    score: [2, 2],
    currentPlayer: 1, // Black starts
    gameOver: false,
    isAiThinking: false,
};

export function useOthelloGame() {
    const [game, setGame] = useState<WasmGameType | null>(null);
    const [state, setState] = useState<GameState>(INITIAL_STATE);
    const [firstMove, setFirstMove] = useState(true);

    // Check if evaluator is initialised
    const evaluatorRef = useRef<((inputs: Float32Array, batchSize: number) => Promise<{ policies: Float32Array, values: Float32Array }>) | null>(null);

    const updateGameState = useCallback((g: WasmGameType, aiThinking: boolean = false) => {
        setState({
            board: g.board(),
            legalMoves: g.legal_moves(),
            score: [...g.score()] as [number, number],
            currentPlayer: g.current_player() as 1 | 2,
            gameOver: g.game_over(),
            isAiThinking: aiThinking
        });
    }, []);

    const initialiseGame = useCallback(async () => {
        try {
            // Dynamically import WASM to avoid top-level await breaking the bundle
            const { WasmGame } = await import("@wasm/othello_wasm");
            // GameType 2 = PlayerVsModel
            const g = new WasmGame(2);

            // Load ONNX evaluator if not already loaded
            if (!evaluatorRef.current) {
                evaluatorRef.current = await createBatchEvaluator();
            }
            g.set_evaluator(evaluatorRef.current);

            setGame(g);
            setFirstMove(true);
            updateGameState(g);
        } catch (e) {
            console.error("Failed to initialize game:", e);
            toast.error("Failed to start game");
        }
    }, [updateGameState]);

    // Initial load
    useEffect(() => {
        if (typeof window !== "undefined") {
            initialiseGame();
        }
    }, [initialiseGame]);

    const playAiMove = useCallback(async () => {
        if (!game || state.isAiThinking) return;

        // Optimistic UI update: AI is thinking
        setState(prev => ({ ...prev, isAiThinking: true, legalMoves: [] }));

        try {
            await game.play_ai_move();
            updateGameState(game, false);
        } catch (e) {
            toast.error(e as string);
            // Revert thinking state if error, but refresh other state to be safe
            updateGameState(game, false);
        }
    }, [game, state.isAiThinking, updateGameState]);

    const performPass = useCallback(() => {
        if (!game) return;
        try {
            // Try to pass (using dummy coordinates, relying on lib.rs implementation)
            game.play_turn(0, 0, 1);
        } catch (e) {
            if (e === "You have no moves") {
                updateGameState(game);
            } else {
                toast.error(e as string);
            }
        }
    }, [game, updateGameState]);

    // Game Loop Effect: Handles AI Turn and Human Auto-Pass
    useEffect(() => {
        if (!game || state.gameOver) return;

        // AI Turn
        if (state.currentPlayer === 2 && !state.isAiThinking) {
            const aiPromise = playAiMove();
            toast.promise(aiPromise, {
                position: "bottom-right",
                loading: "AI is thinking",
                success: "AI played. Your turn!",
                error: 'Error',
            });
        }

        // Human Auto-Pass
        else if (state.currentPlayer === 1 && state.legalMoves.length === 0) {
            const timer = setTimeout(() => {
                toast("No legal moves available. Passing turn to AI...");
                performPass();
            }, 1500);
            return () => clearTimeout(timer);
        }
    }, [game, state.currentPlayer, state.gameOver, state.isAiThinking, state.legalMoves.length, playAiMove, performPass]);

    const onHumanMove = useCallback((i: number, j: number) => {
        if (!game || state.isAiThinking || state.currentPlayer !== 1) return;

        try {
            game.play_turn(i, j, 1);
            updateGameState(game);

            if (firstMove) {
                posthog.capture("game_started", { type: "demo" });
                setFirstMove(false);
            }

            if (game.game_over()) {
                const [white, black] = game.score();
                posthog.capture("game_terminated", {
                    type: "demo",
                    winner: white > black ? "white" : "black"
                });
            }

        } catch (e) {
            posthog.capture("error", {
                type: "demo",
                text: e as string,
                row: i,
                col: j
            });
            toast.error(e as string);
        }
    }, [game, state.isAiThinking, state.currentPlayer, firstMove, updateGameState]);

    return {
        ...state,
        onHumanMove,
        resetGame: initialiseGame,
    };
}

