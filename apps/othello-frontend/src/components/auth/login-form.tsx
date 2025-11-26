import {cn} from "@/lib/utils"
import {Button} from "@/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import {Input} from "@/components/ui/input"
import {Link} from "@tanstack/react-router";
import {useEffect, useState} from "react";

export function LoginForm({
                            className,
                            ...props
                          }: React.ComponentProps<"form">) {
  const [csrf, setCsrf] = useState("")

  useEffect(() => {
    const readCookie = () =>
      document.cookie
        .split("; ")
        .find((x) => x.startsWith("csrf="))
        ?.split("=")[1];

    let token = readCookie();

    if (!token) {
      fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/csrf/init`, {
        method: "GET",
        credentials: "include"
      })
        .then(() => {
          token = readCookie();
          if (token) setCsrf(token);
        })
        .catch(console.error);
    } else {
      setCsrf(token);
    }
  }, []);

  return (
    <form action={`${import.meta.env.VITE_PUBLIC_API_URL}/auth/sign-in`} method="POST"
          className={cn("flex flex-col gap-6", className)} {...props}>
      <FieldGroup>
        <div className="flex flex-col items-center gap-1 text-center">
          <h1 className="text-2xl font-bold">Login to your account</h1>
          <p className="text-muted-foreground text-sm text-balance">
            Enter your email below to login to your account
          </p>
        </div>
        <Field hidden>
          <Input id="csrf" name="csrf" type="hidden" value={csrf} readOnly required/>
        </Field>
        <Field>
          <FieldLabel htmlFor="email">Email</FieldLabel>
          <Input id="email" name="email" type="email" placeholder="email@example.com" required/>
        </Field>
        <Field>
          <div className="flex items-center">
            <FieldLabel htmlFor="password">Password</FieldLabel>
            <a
              href="#"
              className="ml-auto text-sm underline-offset-4 hover:underline"
            >
              Forgot your password?
            </a>
          </div>
          <Input id="password" name="password" type="password" required/>
        </Field>
        <Field>
          <Button type="submit">Login</Button>
        </Field>
        <Field>
          <FieldDescription className="text-center">
            Don&apos;t have an account?{" "}
            <Link to="/auth/signup" className="underline underline-offset-4">
              Sign up
            </Link>
          </FieldDescription>
        </Field>
      </FieldGroup>
    </form>
  )
}
