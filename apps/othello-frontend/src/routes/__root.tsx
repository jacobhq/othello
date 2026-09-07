import { HeadContent, Outlet, createRootRoute } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";
import { Toaster } from "@/components/ui/sonner.tsx";
import { PostHogProvider } from "posthog-js/react";
import { ThemeProvider } from "@/components/theme-provider.tsx";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator.tsx";

const options = {
  api_host: import.meta.env.VITE_PUBLIC_POSTHOG_HOST,
  defaults: "2025-05-24",
} as const;

export const Route = createRootRoute({
  component: RootComponent,
  beforeLoad: async () =>
    await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/csrf/init`, {
      method: "GET",
      credentials: "include",
    }).catch(console.error),
});

function RootComponent() {
  return (
    <>
      <HeadContent />
      <ThemeProvider defaultTheme="system" storageKey="ui-theme">
        <PostHogProvider
          apiKey={import.meta.env.VITE_PUBLIC_POSTHOG_KEY}
          options={options}
        >
          <Toaster />
          <Outlet />
          <div className="flex flex-col gap-6 md:hidden p-6 justify-center">
            <Separator />
            <Button variant="outline" size="sm" id="sidebar-feedback">
              Give feedback
            </Button>
          </div>
        </PostHogProvider>
      </ThemeProvider>
      <TanStackRouterDevtools position="bottom-right" />
    </>
  );
}
