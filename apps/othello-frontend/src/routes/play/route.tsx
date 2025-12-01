import {Outlet, createFileRoute, Link, redirect} from "@tanstack/react-router"
import {SidebarInset, SidebarProvider, SidebarTrigger} from "@/components/ui/sidebar";
import {Sidebar} from "@/components/product/sidebar";
import {Separator} from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList, BreadcrumbPage,
  BreadcrumbSeparator
} from "@/components/ui/breadcrumb";
import type {User} from "@/lib/user.ts";

export const Route = createFileRoute("/play")({
  component: RouteComponent,
  pendingComponent: PlaySkeleton,
  loader: async () => {
    const res = await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/user`, {
      method: "GET",
      credentials: "include",
    });

    if (!res.ok) {
      throw redirect({
        to: "/auth/login",
      });
    }

    return await res.json() as User;
  },

  beforeLoad: async ({ location }) => {
    try {
      // Call your server endpoint that returns the public user (or 401)
      // Make sure this endpoint reads the HttpOnly cookie and responds 200 for valid session.
      const res = await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/user`, {
        method: "GET",
        credentials: "include",
      });

      if (!res.ok) {
        // Not authenticated — redirect to login
        throw redirect({
          to: "/auth/login",
          search: { redirect: location.href },
        });
      }

      // Optionally you could return the user data here and let the route receive it,
      // but for simplicity we just allow navigation to continue.
    } catch (err) {
      // Network error or other failure - treat as unauthenticated
      throw redirect({
        to: "/auth/login",
      });
    }
  },
})

function PlaySkeleton() {
  return (
    <div className="p-6 animate-pulse">
      <div className="h-6 w-1/3 bg-muted rounded mb-4" />
      <div className="h-4 w-1/2 bg-muted rounded" />
    </div>
  );
}


function RouteComponent() {
  return <SidebarProvider>
    <Sidebar/>
    <SidebarInset>
      <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
        <div className="flex items-center gap-2 px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem className="hidden md:block">
                <BreadcrumbLink asChild>
                  <Link to="/">Othello</Link>
                </BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator className="hidden md:block" />
              <BreadcrumbItem>
                <BreadcrumbPage>Play</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </div>
      </header>
      <Outlet/>
    </SidebarInset>
  </SidebarProvider>
}
