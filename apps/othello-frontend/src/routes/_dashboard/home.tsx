import {createFileRoute, Link} from '@tanstack/react-router'
import {SidebarTrigger} from "@/components/ui/sidebar.tsx";
import {Separator} from "@/components/ui/separator.tsx";
import {Breadcrumb, BreadcrumbItem, BreadcrumbList, BreadcrumbPage} from "@/components/ui/breadcrumb.tsx";
import {Item, ItemContent, ItemDescription, ItemGroup, ItemTitle} from "@/components/ui/item.tsx";
import {DataTable} from "@/components/product/game-table.tsx";
import {columns, type Game} from "@/components/product/game-table-columns.tsx";
import {Badge} from "@/components/ui/badge.tsx";

export const Route = createFileRoute('/_dashboard/home')({
  component: RouteComponent,
  loader: async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/games`, {
        method: "GET",
        credentials: "include",
      });

      if (!res.ok) {
        throw new Response("Internal Server Error", {
          status: 500,
          statusText: "Internal Server Error",
        })
      }

      let data: Game[] = await res.json();
      return data ? [...data].sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ) : data
    } catch (err) {
      console.log(err)
      throw new Response("Internal Server Error", {
        status: 500,
        statusText: "Internal Server Error",
      })
    }
  }
})

const items = [
  {
    title: "Take Tutorial",
    description: "Learn how to play Othello",
    to: "/tutorial",
    badge: "Recommended"
  },
  {
    title: "Play Othello",
    description: "Play Othello in many game modes",
    to: "/play"
  },
  {
    title: "Game History",
    description: "View ongoing and past games",
    to: "/history"
  }
]

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
              <BreadcrumbPage>Othello</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
    </header>
    <div className="flex p-4 pt-32 justify-center w-full h-full">
      <div className="flex flex-col container max-w-4xl gap-4">
        <h1 className="font-bold text-4xl">Welcome to Othello</h1>
        <p className="text-muted-foreground mb-4">Play Othello on the world's most popular Othello platform.</p>
        <ItemGroup className="flex flex-col xl:flex-row gap-4">
          {items.map((item) => (
            <Item key={item.title} className="container max-w-xl" variant="outline" asChild>
              <Link to={item.to}>
                <ItemContent>
                  <ItemTitle>
                    {item.title}
                    {item.badge && <Badge>{item.badge}</Badge>}
                  </ItemTitle>
                  <ItemDescription>{item.description}</ItemDescription>
                </ItemContent>
              </Link>
            </Item>
          ))}
        </ItemGroup>
        {data && <div className="flex flex-col gap-4 mt-12">
            <h1 className="font-bold text-xl">Continue Playing</h1>
            <DataTable columns={columns} data={data} n={5}/>
        </div>}
      </div>
    </div>
  </>
}