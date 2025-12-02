import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/play/$gameId')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/play/$gameId"!</div>
}
