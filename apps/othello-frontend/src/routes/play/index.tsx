import {createFileRoute} from "@tanstack/react-router"
import Board from "@/components/game/board.tsx";
import {BarChart3, BookOpen, Bot, MessageCircleWarningIcon, UserPlus, Users, Zap} from "lucide-react";
import {Button} from "@/components/ui/button.tsx";
import {Alert, AlertDescription, AlertTitle} from "@/components/ui/alert.tsx";
import {type ComponentType, useEffect, useState} from "react";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldSet,
  FieldTitle
} from "@/components/ui/field.tsx";
import {RadioGroup, RadioGroupItem} from "@/components/ui/radio-group";
import {Input} from "@/components/ui/input.tsx";

export const Route = createFileRoute("/play/")({
  component: RouteComponent,
})

interface PlayMode {
  icon: ComponentType,
  title: string,
  description: string,
  color: string,
  id: string,
  disabled?: boolean
}

const playModes: PlayMode[] = [
  {
    icon: Zap,
    title: "Play Online",
    description: "Play vs a person of similar skill",
    color: "yellow",
    id: "play_online",
    disabled: true
  },
  {
    icon: Bot,
    title: "Play Bots",
    description: "Challenge a bot from Easy to Master",
    color: "blue",
    id: "play_bots",
    disabled: true
  },
  {
    icon: Users,
    title: "Play a Friend",
    description: "Invite a friend to a game of Othello",
    color: "pink",
    id: "play_friend",
    disabled: true
  },
  {
    icon: UserPlus,
    title: "Pass and Play",
    description: "Play locally with a friend on the same device",
    color: "green",
    id: "pass_and_play"
  }
]

function RouteComponent() {
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

  return <>
    <div className="flex flex-col 2xl:flex-row gap-6 p-4 w-full h-full">
      <div className="flex items-center justify-center flex-1">
        <Board disabled/>
      </div>

      {/* Play Modes */}
      <div className="flex flex-col gap-4 shrink grow">
        <div className="flex items-center gap-3 mb-2">
          <div className="bg-primary/10 p-2 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-foreground"/>
          </div>
          <h1 className="text-3xl font-bold">Play Othello</h1>
        </div>

        <form className="flex flex-col gap-3" action={`${import.meta.env.VITE_PUBLIC_API_URL}/api/game/new`} method="POST">
          <Alert variant="default" className="border-0 text-yellow-900 bg-yellow-100">
            <MessageCircleWarningIcon/>
            <AlertTitle>Heads up!</AlertTitle>
            <AlertDescription className="text-yellow-700">
              Othello is currently in alpha. Not all features work yet.
            </AlertDescription>
          </Alert>
          <Field hidden>
            <Input id="csrf" name="csrf" type="hidden" value={csrf} readOnly required/>
          </Field>
          <FieldGroup>
            <FieldSet>
              <RadioGroup defaultValue="pass_and_play" name="game_type">
                {playModes.map((playMode) => (
                  <FieldLabel htmlFor={playMode.id}>
                    <Field className="has-disabled:opacity-50 has-disabled:cursor-not-allowed" orientation="horizontal">
                      <FieldContent>
                        <FieldTitle className="font-bold text-lg">{playMode.title}</FieldTitle>
                        <FieldDescription>
                          {playMode.description}
                        </FieldDescription>
                      </FieldContent>
                      <RadioGroupItem className="peer" disabled={playMode.disabled} value={playMode.id}
                                      id={playMode.id}/>
                    </Field>
                  </FieldLabel>
                ))}
              </RadioGroup>
            </FieldSet>
          </FieldGroup>
          <Button size="lg" type="submit">Start game</Button>
        </form>

        <div className="flex gap-3 mt-4">
          <Button size="icon-lg" variant="ghost" className="flex-1 gap-2 bg-transparent">
            <BookOpen className="w-4 h-4"/>
            Game History
          </Button>
          <Button size="icon-lg" variant="ghost" className="flex-1 gap-2 bg-transparent">
            <BarChart3 className="w-4 h-4"/>
            Leaderboard
          </Button>
        </div>
      </div>
    </div>
  </>
}
