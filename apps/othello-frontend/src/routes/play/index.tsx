import {createFileRoute, Link, useNavigate} from "@tanstack/react-router"
import Board from "@/components/game/board.tsx";
import {BookOpen, Bot, MessageCircleWarningIcon, PlayCircle, UserPlus, Users, Zap} from "lucide-react";
import {Button} from "@/components/ui/button.tsx";
import {Alert, AlertDescription, AlertTitle} from "@/components/ui/alert.tsx";
import {type ComponentType, type FormEvent, useState} from "react";
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
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList, BreadcrumbPage,
  BreadcrumbSeparator
} from "@/components/ui/breadcrumb.tsx";
import {SidebarTrigger} from "@/components/ui/sidebar.tsx";
import {Separator} from "@/components/ui/separator.tsx";
import {toast} from "sonner";
import {Spinner} from "@/components/ui/spinner.tsx";
import {getCookie} from "@/lib/utils.ts";

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

interface CreationResponse {
  id: string
}

function RouteComponent() {
  const navigate = useNavigate({from: "/play"})

  const [isLoading, setIsLoading] = useState(false);
  const [gameMode, setGameMode] = useState("pass_and_play")

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setIsLoading(true);

    fetch(`${import.meta.env.VITE_PUBLIC_API_URL}/api/games/new`, {
      method: "POST",
      credentials: "include",
      body: JSON.stringify({
        game_type: gameMode,
      }),
      headers: {
        "Content-Type": "application/json",
        "X-Csrf-Token": getCookie("csrf") ?? ""
      }
    }).then((res) => {
      if (res.status == 201) {
        toast.success("Game created successfully")

        res.json().then((data: CreationResponse) => {
          navigate({to: "/play/$gameId", params: {gameId: data.id}})
        })
      } else {
        toast.error("An error occurred, please try again")
      }
      setIsLoading(false)
    }).catch((err) => {
      console.error(err)
      setIsLoading(false)
    })
  }


  return <>
    <header
      className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
      <div className="flex items-center gap-2 px-4">
        <SidebarTrigger className="-ml-1"/>
        <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4"/>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink asChild>
                <Link to="/">Othello</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block"/>
            <BreadcrumbItem>
              <BreadcrumbPage>Play</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
    </header>
    <div className="flex flex-col 2xl:flex-row gap-6 p-4 w-full h-full">
      <div className="flex items-center justify-center flex-1">
        <Board disabled board={[
          new Array(8).fill(0),
          new Array(8).fill(0),
          new Array(8).fill(0),
          [0, 0, 0, 2, 1, 0, 0, 0],
          [0, 0, 0, 1, 2, 0, 0, 0],
          new Array(8).fill(0),
          new Array(8).fill(0),
          new Array(8).fill(0),
        ]}/>
      </div>

      {/* Play Modes */}
      <div className="flex flex-col gap-4 shrink grow">
        <div className="flex items-center gap-3 mb-2">
          <div className="bg-primary/10 p-2 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-foreground"/>
          </div>
          <h1 className="text-3xl font-bold">Play Othello</h1>
        </div>

        <form className="flex flex-col gap-3" onSubmit={(e) => onSubmit(e)}>
          <Alert variant="default" className="border-0 text-yellow-900 bg-yellow-100">
            <MessageCircleWarningIcon/>
            <AlertTitle>Heads up!</AlertTitle>
            <AlertDescription className="text-yellow-700">
              Othello is currently in alpha. Not all features work yet.
            </AlertDescription>
          </Alert>
          <FieldGroup>
            <FieldSet>
              <RadioGroup value={gameMode} onChange={(e: any) => setGameMode(e.target.value)}
                          defaultValue="pass_and_play" name="game_type">
                {playModes.map((playMode) => (
                  <FieldLabel htmlFor={playMode.id} key={playMode.id}>
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
          <Button size="lg" disabled={isLoading} type="submit">
            {isLoading && <Spinner/>} Start game
          </Button>
        </form>

        <div className="flex gap-3 mt-4">
          <Button size="icon-lg" variant="ghost" className="flex-1 gap-2 bg-transparent" asChild>
            <Link to="/play/resume-game">
              <PlayCircle className="w-4 h-4"/>
              Resume game
            </Link>
          </Button>
          <Button size="icon-lg" variant="ghost" className="flex-1 gap-2 bg-transparent" asChild>
            <Link to="/play/history">
              <BookOpen className="w-4 h-4"/>
              Game history
            </Link>
          </Button>
        </div>
      </div>
    </div>
  </>
}
