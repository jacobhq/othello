import Hero from "@/components/marketing/hero";
import DemoBoard from "@/components/game/demo-board.tsx";
import {createFileRoute} from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: Index,
})

function Index() {

  return (
    <div className="p-8">
      <Hero />
      <div>
        <DemoBoard />
      </div>
    </div>
  );
}
