import { createFileRoute } from '@tanstack/react-router'
import DemoBoard from "@/components/game/demo-board.tsx";

export const Route = createFileRoute('/ai')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div className="p-4">
    <DemoBoard />
  </div>
}
