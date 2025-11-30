import {createFileRoute, Link, useNavigate, useSearch} from '@tanstack/react-router'
import {GalleryVerticalEnd} from "lucide-react"
import {SignupForm} from "@/components/auth/signup-form"
import {useEffect} from "react";
import {toast} from "sonner";

export const Route = createFileRoute('/auth/signup')({
  component: SignupPage,
})

function SignupPage() {
  const search = useSearch({from: "/auth/signup"}) as { error?: string };
  const navigate = useNavigate();

  useEffect(() => {
    switch (search.error) {
      case "csrf":
        toast.error("An error occurred, try refreshing the page")
        navigate({search: {} as any})
        break;
      case "username_invalid":
        toast.error("Username is not in the required format")
        navigate({search: {} as any})
        break;
      case "email_invalid":
        toast.error("Email is not in the required format")
        navigate({search: {} as any})
        break;
      case "password_invalid":
        toast.error("Password is not in the required format")
        navigate({search: {} as any})
        break;
      case "username_taken":
        toast.error("An account with that username already exists")
        navigate({search: {} as any})
        break;
      case "email_taken":
        toast.error("An account with that email already exists")
        navigate({search: {} as any})
        break;
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
            <SignupForm/>
          </div>
        </div>
      </div>
      <div className="bg-muted relative hidden lg:block">
        <img
          src="https://images.unsplash.com/photo-1547623641-d2c56c03e2a7?q=80&w=3087&auto=format&fit=crop"
          alt="Image"
          className="absolute inset-0 h-full w-full object-cover dark:brightness-[0.6] dark:grayscale"
        />
      </div>
    </div>
  )
}