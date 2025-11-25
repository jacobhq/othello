import {createFileRoute, Link, useNavigate, useSearch} from '@tanstack/react-router'
import {GalleryVerticalEnd} from "lucide-react"
import {LoginForm} from "@/components/auth/login-form"
import {useEffect} from "react";
import {toast} from "sonner";

export const Route = createFileRoute('/auth/login')({
  component: LoginPage,
})

function LoginPage() {
  const search = useSearch({ from: "/auth/login" }) as { error?: string };
  const navigate = useNavigate();

  useEffect(() => {
    if (search.error === "incorrect_credentials") {
      toast.error("Incorrect username or password")
      navigate({ search: {} as any })
    }
  }, [search.error, navigate]);

  return (
    <div className="grid min-h-svh lg:grid-cols-2">
      <div className="flex flex-col gap-4 p-6 md:p-10">
        <div className="flex justify-center gap-2 md:justify-start">
          <Link to="/" className="flex items-center gap-2 font-medium">
            <div className="bg-primary text-primary-foreground flex size-6 items-center justify-center rounded-md">
              <GalleryVerticalEnd className="size-4"/>
            </div>
            Othello
          </Link>
        </div>
        <div className="flex flex-1 items-center justify-center">
          <div className="w-full max-w-xs">
            <LoginForm/>
          </div>
        </div>
      </div>
      <div className="bg-muted relative hidden lg:block">
        <img
          src="/placeholder.svg"
          alt="Image"
          className="absolute inset-0 h-full w-full object-cover dark:brightness-[0.2] dark:grayscale"
        />
      </div>
    </div>
  )
}

