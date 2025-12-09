import Hero from "@/components/marketing/hero";
import DemoBoard from "@/components/game/demo-board.tsx";
import {createFileRoute} from "@tanstack/react-router";
import type {User} from "@/lib/user.ts";
import posthog from "posthog-js";

export const Route = createFileRoute("/")({
  component: Index,
  loader: async () => {
    const res = await fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/user`, {
      method: "GET",
      credentials: "include",
    });

    if (!res.ok) {
      return null
    }

    const user: User = await res.json();

    posthog.identify(user.id,
      {
        email: user.email,
        username: user.username,
        is_prod: import.meta.env.PROD
      }
    )

    return user;
  },

})

function Index() {

  return (
    <div className="p-8">
      <Hero />
      <div>
        <DemoBoard />
      </div>
    </div>
  );
}
