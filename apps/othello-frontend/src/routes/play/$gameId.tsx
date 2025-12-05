import {createFileRoute, Link, notFound} from '@tanstack/react-router'
import Board from "@/components/game/board.tsx";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList, BreadcrumbPage,
  BreadcrumbSeparator
} from "@/components/ui/breadcrumb.tsx";
import {SidebarTrigger} from "@/components/ui/sidebar.tsx";
import {Separator} from "@/components/ui/separator.tsx";
import {Button} from "@/components/ui/button.tsx";
import {Flag} from "lucide-react";
import {Popover, PopoverContent, PopoverTrigger} from "@/components/ui/popover.tsx";
import {ButtonGroup} from "@/components/ui/button-group.tsx";
import {PopoverClose} from "@radix-ui/react-popover";
import {useEffect, useState} from "react";
import {WasmGame} from "@wasm/othello_wasm";
import {toast} from "sonner";
import Score from "@/components/game/score.tsx";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle
} from "@/components/ui/alert-dialog";

interface ApiGameResponse {
  bitboard_black: string,
  bitboard_white: string,
  current_turn: string
}

interface ApiGame {
  bitboard_black: bigint,
  bitboard_white: bigint,
  current_turn: string
}

export const Route = createFileRoute('/play/$gameId')({
  component: RouteComponent,
  loader: async ({params}) => {
    try {
      const res = await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/game/${params.gameId}`, {
        method: "GET",
        credentials: "include",
      });

      if (!res.ok) {
        // No game exists
        throw notFound()
      }

      let data: ApiGameResponse = await res.json();

      return {
        bitboard_black: BigInt(data.bitboard_black),
        bitboard_white: BigInt(data.bitboard_white),
        current_turn: data.current_turn,
      } satisfies ApiGame;
    } catch (err) {
      throw notFound()
    }
  }

})

function RouteComponent() {
  const initial_state = Route.useLoaderData();
  let initial_turn = 0;

  switch (initial_state.current_turn) {
    case "black":
      initial_turn = 1;
      break;
    case "white":
      initial_turn = 2;
      break;
  }

  const [game, setGame] = useState<WasmGame | null>(null);
  const [board, setBoard] = useState<(0 | 1 | 2)[][]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<1 | 2>(1);
  const [score, setScore] = useState<[number, number]>([0, 0]);
  const [gameOver, setGameOver] = useState(false);

  useEffect(() => {
    // Initialise the wasm module on mount
    initialiseGame()
  }, []);

  const g = WasmGame.new_from_state(initial_state.bitboard_black, initial_state.bitboard_white, initial_turn);
  const initialiseGame = () => {
    setGame(g);
    setBoard(g.board());
    setCurrentPlayer(g.current_player() as 1 | 2);
    setScore([...g.score()] as [number, number]);
  }

  const handleClick = (i: number, j: number) => {
    try {
      if (!game) {
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

  return <>
    <AlertDialog open={gameOver}>
      <AlertDialogContent className="sm:max-w-[425px]">
        <AlertDialogHeader>
          <AlertDialogTitle>{score[0] > score[1] ? "White" : "Black"} wins!</AlertDialogTitle>
          <AlertDialogDescription>
            If you enjoyed Othello, consider signing up for an account with us.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="mt-4">
          <Score whiteScore={score[0]}
                 blackScore={score[1]}/>
        </div>
        <AlertDialogFooter>
          <Button asChild>
            <Link to="/play">Continue</Link>
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
    <header
      className="flex h-16 shrink-0 items-center justify-between gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
      <div className="flex flex-1 items-center gap-2 px-4">
        <SidebarTrigger className="-ml-1"/>
        <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4"/>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink asChild>
                <Link to="/">Othello</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block"/>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink asChild>
                <Link to="/play">Play</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block"/>
            <BreadcrumbItem>
              <BreadcrumbPage>Pass and Play</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="px-4">
        <Button disabled={gameOver} variant="ghost">
          <div className="flex flex-row gap-4">
            <div className="flex flex-row gap-1 items-center">
              <div className="h-4 w-4 bg-black border rounded-full"></div>
              <span className="font-semibold">{score[0]}</span>
            </div>
            <div className="flex flex-row gap-1 items-center">
              <div className="h-4 w-4 border rounded-full"></div>
              <span className="font-semibold">{score[1]}</span>
            </div>
          </div>
        </Button>
      </div>
      <div className="hidden sm:flex flex-1 justify-end items-center gap-2 px-4">
        <ButtonGroup>
          <Button disabled={gameOver}
                  variant="ghost">{currentPlayer == 1 && "Black's" || currentPlayer == 2 && "White's"} turn</Button>
          <Popover>
            <PopoverTrigger asChild>
              <Button disabled={gameOver} variant="ghost">
                <Flag/> Resign
              </Button>
            </PopoverTrigger>
            <PopoverContent>
              <div className="space-y-2">
                <h4 className="leading-none font-medium">Are you sure?</h4>
                <div className="flex flex-row gap-2">
                  <Button variant="destructive">Resign</Button>
                  <PopoverClose asChild>
                    <Button variant="secondary">Cancel</Button>
                  </PopoverClose>
                </div>
              </div>
            </PopoverContent>
          </Popover>
        </ButtonGroup>
      </div>
    </header>
    <div className="flex flex-col 2xl:flex-row gap-6 p-0 w-full h-full">
      <div className="flex items-center justify-center flex-1">
        <Board board={board} handleClick={handleClick}/>
      </div>
    </div>
  </>
}
