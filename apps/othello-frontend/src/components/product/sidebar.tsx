import type * as React from "react";
import { PlayCircle, Circle, History, House } from "lucide-react";

import { SidebarNavigation } from "@/components/product/sidebar-navigation";
import { UserMenu } from "@/components/auth/user-menu";
import {
  Sidebar as SidebarComponent,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import { Link, useLoaderData } from "@tanstack/react-router";
import { Button } from "@/components/ui/button.tsx";

const sidebarLinks = [
  {
    title: "Home",
    url: "/",
    icon: House,
  },
  {
    title: "Play",
    url: "/play",
    icon: PlayCircle,
  },
  {
    title: "History",
    url: "/history",
    icon: History,
  },
];

export function Sidebar({
  ...props
}: React.ComponentProps<typeof SidebarComponent>) {
  const user = useLoaderData({ from: "/_dashboard" });

  return (
    <SidebarComponent collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" tooltip="Othello" asChild>
              <Link to="/">
                <div className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                  <Circle className="size-4" />
                </div>
                <div className="flex items-center gap-1.5 text-left text-sm leading-tight">
                  <span className="truncate font-semibold h-fit">Othello</span>
                  <Badge variant="default">Beta</Badge>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarNavigation items={sidebarLinks} />
      </SidebarContent>
      <SidebarFooter>
        <Button
          variant="outline"
          id="sidebar-feedback"
          className="hidden md:flex"
        >
          Give feedback
        </Button>
        <UserMenu user={user} />
      </SidebarFooter>
      <SidebarRail />
    </SidebarComponent>
  );
}
