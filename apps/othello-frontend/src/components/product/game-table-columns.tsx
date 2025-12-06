import type {ColumnDef} from "@tanstack/react-table"
import {Button} from "@/components/ui/button.tsx";
import {Link} from "@tanstack/react-router";
import {PlayCircle} from "lucide-react";

export interface Game {
  id: string,
  timestamp: Date,
  type: "user_user" | "user_model" | "pass_and_play",
  status: "in_play" | "won" | "drew",
  current_turn: "black" | "white",
  bitboard_white: bigint,
  bitboard_black: bigint,
  player_one_color: "black" | "white",
  player_two_color: "black" | "white",
  current_user_player: 1 | 2,
  white_score: number,
  black_score: number,
}

export const columns: ColumnDef<Game>[] = [
  {
    accessorKey: "status",
    header: "Status"
  },
  {
    accessorKey: "type",
    header: "Type"
  },
  {
    accessorKey: "timestamp",
    header: "Created",
    cell: ({cell}) => {
      // @ts-expect-error: We know this type from Rust
      const value: string = cell.getValue()
      const date = new Date(value)
      // @ts-expect-error: We can subtract dates
      const seconds = Math.floor((new Date() - date) / 1000);

      let interval = seconds / 31536000;

      if (interval > 1) {
        return Math.floor(interval) + " years ago";
      }
      interval = seconds / 2592000;
      if (interval > 1) {
        return Math.floor(interval) + " months ago";
      }
      interval = seconds / 86400;
      if (interval > 1) {
        return Math.floor(interval) + " days ago";
      }
      interval = seconds / 3600;
      if (interval > 1) {
        return Math.floor(interval) + " hours ago";
      }
      interval = seconds / 60;
      if (interval > 1) {
        return Math.floor(interval) + " minutes ago";
      }
      return "Just now";
    }
  },
  {
    header: "Score",
    accessorFn: (row) => [row.black_score, row.white_score],
    cell: ({cell}) => {
      // @ts-expect-error: We know this type from Rust
      const value: [number, number] = cell.getValue()
      return <div>
        <div className="flex flex-row gap-4">
          <div className="flex flex-row gap-1 items-center">
            <div className="h-4 w-4 bg-black border rounded-full"></div>
            <span className="font-semibold">{value[0]}</span>
          </div>
          <div className="flex flex-row gap-1 items-center">
            <div className="h-4 w-4 bg-white border rounded-full"></div>
            <span className="font-semibold">{value[1]}</span>
          </div>
        </div>
      </div>
    }
  },
  {
    header: "Next move",
    accessorKey: "current_turn"
  },
  {
    header: "Play",
    cell: ({row}) => <Button variant="outline" asChild>
      <Link to="/play/$gameId" params={{ gameId: row.original.id }}>
        <PlayCircle />
        Play
      </Link>
    </Button>
  }
]