import Board from "@/components/game/board.tsx";
import Score from "@/components/game/score.tsx";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button.tsx";
import { useOthelloGame } from "@/hooks/use-othello-game";
import { getCookie } from "@/lib/utils.ts";
import {
  createFileRoute,
  Link,
  notFound,
  useParams,
} from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";

interface ApiGameResponse {
  bitboard_black: string;
  bitboard_white: string;
  current_turn: string;

  human_player_id: string;
  ai_model_id?: string | null;
  ai_model_url?: string | null;
}

interface ApiGame {
  bitboard_black: bigint;
  bitboard_white: bigint;
  current_turn: string;

  human_player_id: string;
  ai_model_id?: string | null;
  ai_model_url?: string | null;
}

export const Route = createFileRoute("/_dashboard/play/$gameId")({
  component: RouteComponent,
  loader: async ({ params }) => {
    const res = await fetch(
      `${import.meta.env.VITE_PUBLIC_API_URL}/api/games/${params.gameId}`,
      {
        method: "GET",
        credentials: "include",
      },
    );

    if (!res.ok) throw notFound();

    const data: ApiGameResponse = await res.json();

    return {
      bitboard_black: BigInt(data.bitboard_black),
      bitboard_white: BigInt(data.bitboard_white),
      current_turn: data.current_turn,

      human_player_id: data.human_player_id,
      ai_model_id: data.ai_model_id,
      ai_model_url: data.ai_model_url,
    } satisfies ApiGame;
  },
});

function RouteComponent() {
  const initial_state = Route.useLoaderData();
  const { gameId } = useParams({ strict: false });

  const {
    board,
    legalMoves,
    score,
    currentPlayer,
    gameOver,
    isAiThinking,
    onHumanMove,
    resetGame,
  } = useOthelloGame(
    initial_state.ai_model_id && initial_state.ai_model_url ? 2 : 1,
    initial_state.ai_model_url
      ? (initial_state.ai_model_url as string)
      : undefined,
    {
      bitboardBlack: initial_state.bitboard_black,
      bitboardWhite: initial_state.bitboard_white,
      currentTurn: initial_state.current_turn as "black" | "white",
    },
  );

  const [showLegalMoves, setShowLegalMoves] = useState<boolean>(() => {
    const localData = localStorage.getItem("show_legal_moves");
    return localData ? (JSON.parse(localData) as boolean) : true;
  });

  useEffect(() => {
    localStorage.setItem("show_legal_moves", JSON.stringify(showLegalMoves));
  }, [showLegalMoves]);

  /**
   * Backend-validated human move
   */
  const handleClick = async (i: number, j: number) => {
    try {
      const res = await fetch(
        `${import.meta.env.VITE_PUBLIC_API_URL}/api/games/${gameId}`,
        {
          method: "POST",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            "X-Csrf-Token": getCookie("csrf") ?? "",
          },
          body: JSON.stringify({
            row: i,
            col: j,
            color: currentPlayer === 1 ? "black" : "white",
          }),
        },
      );

      if (res.status !== 201) {
        toast.error("Invalid move");
        return;
      }

      // backend accepted → update wasm game
      onHumanMove(i, j);
    } catch (e) {
      toast.error(e as string);
    }
  };

  return (
    <>
      <AlertDialog open={gameOver}>
        <AlertDialogContent className="sm:max-w-[425px]">
          <AlertDialogHeader>
            <AlertDialogTitle>
              {score[0] > score[1] ? "White" : "Black"} wins!
            </AlertDialogTitle>
            <AlertDialogDescription>
              Good game! Thanks for playing Othello.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="mt-4">
            <Score whiteScore={score[0]} blackScore={score[1]} />
          </div>
          <AlertDialogFooter>
            <Button asChild>
              <Link to="/">Continue</Link>
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <header className="flex h-16 items-center justify-between px-4">
        <Button variant="ghost" disabled>
          {(currentPlayer === 1 ? "Black" : "White") + "'s turn"}
          {isAiThinking && " (AI thinking)"}
        </Button>
      </header>

      <div className="flex flex-1 items-center justify-center">
        <Board
          board={board}
          legalMoves={legalMoves}
          showLegalMoves={showLegalMoves}
          handleClick={handleClick}
        />
      </div>
    </>
  );
}
