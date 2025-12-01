import Counter from "@/components/game/counter.tsx";
import {WasmGame} from "@wasm/othello_wasm";
import {useEffect, useState} from "react";
import {toast} from "sonner";

interface BoardProps {
  disabled?: boolean
  onGameOver?: () => void
}

export default function Board({disabled}: BoardProps) {
  disabled ??= false;
  const [game, setGame] = useState<WasmGame | null>(null);
  const [board, setBoard] = useState<(0 | 1 | 2)[][]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<1 | 2>(1);
  const [score, setScore] = useState<[number, number]>([0, 0]);
  const [gameOver, setGameOver] = useState(false);

  useEffect(() => {
    // Initialise the wasm module on mount
    initialiseGame()
  }, []);

  const initialiseGame = () => {
    const g = new WasmGame();
    setGame(g);
    setBoard(g.board());
    setCurrentPlayer(g.current_player() as 1 | 2);
    setScore([...g.score()] as [number, number]);
  }

  const handleClick = (i: number, j: number) => {
    try {
      if (!game || disabled) {
        return
      }

      game.play_turn(i, j, game.current_player())
      const newBoard = game.board();
      setBoard(newBoard)
      setScore([...game.score()] as [number, number])
      setCurrentPlayer(game.current_player() as 2 | 1)
      setGameOver(game.game_over())
    } catch (e) {
      toast.error(e as string)
    }
  }

  return (
    <div className="w-full h-full max-w-[90vmin] max-h-[90vmin] aspect-square mx-auto">
      <div
        id="game"
        className="grid grid-rows-8 bg-green-600 rounded-md p-1 sm:p-2 xl:p-3 gap-1 sm:gap-2 xl:gap-3 h-full">
        {Array.from({length: 8}, (_, i) => (
          <div className="grid grid-cols-8 gap-1 sm:gap-2 xl:gap-3" key={i}>
            {Array.from({length: 8}, (_, j) => (
              <div className="bg-green-700 rounded-sm sm:rounded p-1 sm:p-2" key={j}
                   onClick={() => handleClick(i, j)}>
                {board?.[i]?.[j] !== 0 && (
                  <Counter color={board?.[i]?.[j]}/>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
