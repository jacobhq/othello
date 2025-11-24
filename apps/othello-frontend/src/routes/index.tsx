import Hero from "@/components/marketing/hero";
import Board from "@/components/game/board.tsx";
import {createFileRoute} from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: Index,
})

function Index() {

  return (
    <div className="p-8">
      <Hero />
      <div>
        <Board />
      </div>
    </div>
  );
}
