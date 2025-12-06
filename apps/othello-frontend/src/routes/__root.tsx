import {Outlet, createRootRoute} from "@tanstack/react-router"
import {TanStackRouterDevtools} from "@tanstack/react-router-devtools"
import {Toaster} from "@/components/ui/sonner.tsx";
import {PostHogProvider} from "posthog-js/react";
import {ThemeProvider} from "@/components/theme-provider.tsx";

const options = {
  api_host: import.meta.env.VITE_PUBLIC_POSTHOG_HOST,
  defaults: "2025-05-24",
} as const

export const Route = createRootRoute({
  component: RootComponent,
  beforeLoad: async () => await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/csrf/init`, {
    method: "GET",
    credentials: "include"
  }).catch(console.error)
})

function RootComponent() {
  return (
    <>
      <ThemeProvider defaultTheme="system" storageKey="ui-theme">
        <PostHogProvider apiKey={import.meta.env.VITE_PUBLIC_POSTHOG_KEY} options={options}>
          <Toaster/>
          <Outlet/>
        </PostHogProvider>
      </ThemeProvider>
      <TanStackRouterDevtools position="bottom-right"/>
    </>
  )
}