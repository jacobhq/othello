import { Button } from "@/components/ui/button.tsx";
import { Link } from "@tanstack/react-router";
import DemoBoard from "@/components/game/demo-board";

export default function Hero() {
  return (
    <section className="py-16">
      <div className="relative z-10 mx-auto w-full max-w-2xl px-6 lg:px-0">
        <div className="relative text-center mb-12">
          <h1 className="mx-auto mt-16 max-w-xl text-balance text-5xl font-medium font-serif">
            Othello
          </h1>

          <p className="text-muted-foreground mx-auto mb-6 mt-4 text-balance text-xl font-serif">
            Become extraordinarily focused with a strategy game that is elegant,
            simple, and fun.
          </p>

          <div className="flex flex-col items-center gap-2 *:w-full sm:flex-row sm:justify-center sm:*:w-auto">
            <Button rounded="full" asChild>
              <Link to="/auth/signup">
                <span className="text-nowrap">Create Account</span>
              </Link>
            </Button>
            <Button rounded="full" asChild variant="ghost">
              <Link to="/auth/login">
                <span className="text-nowrap">Sign In</span>
              </Link>
            </Button>
          </div>
        </div>

        <DemoBoard />

        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <p className="text-muted-foreground text-center font-serif">
            &copy; 2025 Jacob Marshall
          </p>
        </div>
      </div>
    </section>
  );
}
