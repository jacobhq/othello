import {Outlet, createFileRoute, redirect} from "@tanstack/react-router"
import {SidebarInset, SidebarProvider} from "@/components/ui/sidebar";
import {Sidebar} from "@/components/product/sidebar";
import type {User} from "@/lib/user.ts";
import posthog from "posthog-js";

export const Route = createFileRoute("/_dashboard")({
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

  beforeLoad: async ({location}) => {
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
          search: {redirect: location.href},
        });
      }

      const user: User = await res.json()

      posthog.identify(user.id,
        {
          email: user.email,
          username: user.username,
          is_prod: import.meta.env.PROD
        }
      )

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
      <div className="h-6 w-1/3 bg-muted rounded mb-4"/>
      <div className="h-4 w-1/2 bg-muted rounded"/>
    </div>
  );
}


function RouteComponent() {
  return <SidebarProvider>
    <Sidebar/>
    <SidebarInset>
      <Outlet/>
    </SidebarInset>
  </SidebarProvider>
}
