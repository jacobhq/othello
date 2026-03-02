import Counter from "@/components/game/counter.tsx";
import LegalMoveDot from "@/components/game/legal-move-dot.tsx";

interface BoardProps {
  disabled?: boolean,
  handleClick?: (i: number, j: number) => void,
  board: (0 | 1 | 2)[][],
  legalMoves?: [number, number][],
  showLegalMoves?: boolean
}

export default function Board({board, legalMoves, showLegalMoves, handleClick}: BoardProps) {
  showLegalMoves ??= true;

  return (
    <div className="w-full h-full max-w-[90vmin] max-h-[90vmin] aspect-square mx-auto">
      <div
        id="game"
        className="grid grid-rows-8 bg-green-600 rounded-md p-1 sm:p-2 xl:p-3 gap-1 sm:gap-2 xl:gap-3 h-full">
        {Array.from({length: 8}, (_, i) => (
          <div className="grid grid-cols-8 gap-1 sm:gap-2 xl:gap-3" key={i}>
            {Array.from({length: 8}, (_, j) => (
              <div className="bg-green-700 rounded-sm sm:rounded p-1 sm:p-2" key={j}
                   onClick={() => handleClick ? handleClick(i, j) : undefined}>
                {board?.[i]?.[j] !== 0 && (
                  <Counter color={board?.[i]?.[j]}/>
                )}
                {showLegalMoves && legalMoves?.some(item => JSON.stringify(item) === JSON.stringify([i,j])) && <LegalMoveDot />}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
