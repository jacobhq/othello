import Counter from "@/components/game/counter.tsx";
import Score from "@/components/game/score.tsx";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Link } from "@tanstack/react-router";
import LegalMoveDot from "@/components/game/legal-move-dot.tsx";
import { useOthelloGame } from "@/hooks/use-othello-game";

export default function DemoBoard() {
  const {
    board,
    legalMoves,
    currentPlayer,
    score,
    gameOver,
    onHumanMove,
    resetGame,
  } = useOthelloGame();

  return (
    <>
      <Dialog
        open={gameOver}
        onOpenChange={(open) => {
          !open && resetGame();
        }}
      >
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>
              {score[0] > score[1] ? "White" : "Black"} wins!
            </DialogTitle>
            <DialogDescription>
              If you enjoyed Othello, consider signing up for an account with
              us.
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4">
            <Score whiteScore={score[0]} blackScore={score[1]} />
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
      <div
        id="game"
        className="grid grid-rows-8 bg-green-600 rounded-md p-1 sm:p-2 xl:p-3 gap-1 sm:gap-2 xl:gap-3 aspect-square max-w-3xl mx-auto mb-8"
      >
        {Array.from({ length: 8 }, (_, i) => (
          <div className="grid grid-cols-8 gap-1 sm:gap-2 xl:gap-3" key={i}>
            {Array.from({ length: 8 }, (_, j) => (
              <div
                className="bg-green-700 rounded-sm sm:rounded p-1 sm:p-2 xl:p3"
                key={j}
                onClick={() => onHumanMove(i, j)}
              >
                {board?.[i]?.[j] !== 0 && <Counter color={board?.[i]?.[j]} />}
                {legalMoves.some(
                  (item) => JSON.stringify(item) === JSON.stringify([i, j]),
                ) && <LegalMoveDot />}
              </div>
            ))}
          </div>
        ))}
      </div>
      <Score
        highlightedPlayer={gameOver ? undefined : currentPlayer}
        whiteScore={score[0]}
        blackScore={score[1]}
      />
    </>
  );
}
