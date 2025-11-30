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

export function SignupForm({
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
    <form action={`${import.meta.env.VITE_PUBLIC_API_URL}/auth/sign-up`} method="POST"
          className={cn("flex flex-col gap-6", className)} {...props}>
      <FieldGroup>
        <div className="flex flex-col items-center gap-1 text-center">
          <h1 className="text-2xl font-bold">Create your account</h1>
          <p className="text-muted-foreground text-sm text-balance">
            Fill in the form below to create your account
          </p>
        </div>
        <Field hidden>
          <Input id="csrf" name="csrf" type="hidden" value={csrf} readOnly required/>
        </Field>
        <Field>
          <FieldLabel htmlFor="email">Email</FieldLabel>
          <Input id="email" name="email" type="email" placeholder="email@example.com" maxLength={254} required/>
        </Field>
        <Field>
          <FieldLabel htmlFor="username">Username</FieldLabel>
          <Input id="username" name="username" type="text" placeholder="johnny123" minLength={4} maxLength={20} pattern="^[a-zA-Z0-9]+$"
                 required/>
          <FieldDescription>
            Must be unique, and consist of between 4 and 20 alphanumeric characters.
          </FieldDescription>
        </Field>
        <Field>
          <FieldLabel htmlFor="password">Password</FieldLabel>
          <Input id="password" name="password" type="password" placeholder="••••••••••••" minLength={8} maxLength={72}
                 pattern="(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,72}" required/>
          <FieldDescription>
            Must be a combination of digits, uppercase, and lowercase characters at least 8 characters long.
          </FieldDescription>
        </Field>
        <Field>
          <Button type="submit">Create Account</Button>
        </Field>
        <Field>
          <FieldDescription className="px-6 text-center">
            Already have an account? <Link to="/auth/login">Sign in</Link>
          </FieldDescription>
        </Field>
      </FieldGroup>
    </form>
  )
}
