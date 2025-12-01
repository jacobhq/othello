import { Outlet, createFileRoute } from "@tanstack/react-router"

export const Route = createFileRoute("/play")({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>
    hi
    <Outlet />
  </div>
}
