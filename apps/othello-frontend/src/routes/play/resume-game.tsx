import {createFileRoute, Link} from '@tanstack/react-router'
import {SidebarTrigger} from "@/components/ui/sidebar.tsx";
import {Separator} from "@/components/ui/separator.tsx";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList, BreadcrumbPage,
  BreadcrumbSeparator
} from "@/components/ui/breadcrumb.tsx";
import {columns, type Game} from "@/components/product/game-table-columns.tsx";
import {DataTable} from "@/components/product/game-table.tsx";

export const Route = createFileRoute('/play/resume-game')({
  component: RouteComponent,
  loader: async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/games`, {
        method: "GET",
        credentials: "include",
      });

      if (!res.ok) {
        // No game exists
        throw new Response("Internal Server Error", {
          status: 500,
          statusText: "Internal Server Error",
        })
      }

      let data: Game[] = await res.json();

      return data
    } catch (err) {
      throw new Response("Internal Server Error", {
        status: 500,
        statusText: "Internal Server Error",
      })
    }
  }
})

function RouteComponent() {
  const data = Route.useLoaderData();

  return <>
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
              <BreadcrumbPage>Resume Game</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
    </header>
    <div className="flex flex-col gap-6 p-4 w-full h-full">
      <DataTable columns={columns} data={data} />
    </div>
  </>
}
