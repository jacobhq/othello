import type * as React from "react"
import {PlayCircle, Bot, Star, Settings2, Circle} from "lucide-react"

import {SidebarNavigation} from "@/components/product/sidebar-navigation"
import {UserMenu} from "@/components/auth/user-menu"
import {
  Sidebar as SidebarComponent,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import {Badge} from "@/components/ui/badge";
import {Link, useLoaderData} from "@tanstack/react-router";
import {Button} from "@/components/ui/button.tsx";

const sidebarLinks = [
  {
    title: "Play",
    url: "#",
    icon: PlayCircle,
    isActive: true,
    items: [
      {
        title: "New Game",
        url: "/play",
      },
      {
        title: "Resume Game",
        url: "/play/resume-game",
      },
      {
        title: "History",
        url: "/play/history",
      },
    ],
  },
  {
    title: "Models",
    url: "#",
    icon: Bot,
    items: [
      {
        title: "Genesis",
        url: "#",
      },
      {
        title: "Explorer",
        url: "#",
      },
      {
        title: "Quantum",
        url: "#",
      },
    ],
  },
  {
    title: "Game Review",
    url: "#",
    icon: Star,
    items: [
      {
        title: "Recent Games",
        url: "#",
      },
      {
        title: "Analysis",
        url: "#",
      },
      {
        title: "Statistics",
        url: "#",
      },
    ],
  },
  {
    title: "Settings",
    url: "#",
    icon: Settings2,
    items: [
      {
        title: "General",
        url: "#",
      },
      {
        title: "Team",
        url: "#",
      },
      {
        title: "Billing",
        url: "#",
      },
      {
        title: "Limits",
        url: "#",
      },
    ],
  },
]

export function Sidebar({...props}: React.ComponentProps<typeof SidebarComponent>) {
  const user = useLoaderData({from: "/play"});

  return (
    <SidebarComponent collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" tooltip="Othello" asChild>
              <Link to="/play">
                <div
                  className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                  <Circle className="size-4"/>
                </div>
                <div className="flex items-center gap-1.5 text-left text-sm leading-tight">
                  <span className="truncate font-semibold h-fit">Othello</span>
                  <Badge variant="default">ALPHA</Badge>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarNavigation items={sidebarLinks}/>
      </SidebarContent>
      <SidebarFooter>
        <Button variant="outline" id="sidebar-feedback">Give feedback</Button>
        <UserMenu user={user}/>
      </SidebarFooter>
      <SidebarRail/>
    </SidebarComponent>
  )
}
