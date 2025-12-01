import {createFileRoute} from "@tanstack/react-router"
import Board from "@/components/game/board.tsx";
import {BarChart3, BookOpen, Bot, MessageCircleWarningIcon, UserPlus, Users, Zap} from "lucide-react";
import {Button} from "@/components/ui/button.tsx";
import {Alert, AlertDescription, AlertTitle} from "@/components/ui/alert.tsx";

export const Route = createFileRoute("/play/")({
  component: RouteComponent,
})

function RouteComponent() {
  return <>
  <div className="flex flex-col 2xl:flex-row gap-6 p-4 w-full h-full">
    <div className="flex items-center justify-center flex-1">
      <Board disabled />
    </div>

      {/* Play Modes */}
      <div className="flex flex-col gap-4 shrink grow">
        <div className="flex items-center gap-3 mb-2">
          <div className="bg-primary/10 p-2 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-foreground"/>
          </div>
          <h1 className="text-3xl font-bold">Play Othello</h1>
        </div>

        <div className="flex flex-col gap-3">
          <Alert variant="destructive">
            <MessageCircleWarningIcon />
            <AlertTitle>Heads up!</AlertTitle>
            <AlertDescription>
              Othello is currently in alpha. Not all features work yet.
            </AlertDescription>
          </Alert>
          <PlayModeCard
            icon={<Zap className="w-6 h-6"/>}
            title="Play Online"
            description="Play vs a person of similar skill"
            iconBg="bg-yellow-500/10"
            iconColor="text-yellow-500"
          />
          <PlayModeCard
            icon={<Bot className="w-6 h-6"/>}
            title="Play Bots"
            description="Challenge a bot from Easy to Master"
            iconBg="bg-blue-500/10"
            iconColor="text-blue-500"
          />
          <PlayModeCard
            icon={<Users className="w-6 h-6"/>}
            title="Play a Friend"
            description="Invite a friend to a game of Othello"
            iconBg="bg-pink-500/10"
            iconColor="text-pink-500"
          />
          <PlayModeCard
            icon={<UserPlus className="w-6 h-6"/>}
            title="Pass and Play"
            description="Play locally with a friend on the same device"
            iconBg="bg-green-500/10"
            iconColor="text-green-500"
          />
        </div>

        <div className="flex gap-3 mt-4">
          <Button variant="ghost" className="flex-1 gap-2 bg-transparent">
            <BookOpen className="w-4 h-4"/>
            Game History
          </Button>
          <Button variant="ghost" className="flex-1 gap-2 bg-transparent">
            <BarChart3 className="w-4 h-4"/>
            Leaderboard
          </Button>
        </div>
      </div>
    </div>
  </>
}

function PlayModeCard({
                        icon,
                        title,
                        description,
                        iconBg,
                        iconColor,
                      }: {
  icon: React.ReactNode
  title: string
  description: string
  iconBg: string
  iconColor: string
}) {
  return (
    <button
      className="flex items-start gap-4 p-4 rounded-lg bg-card border hover:bg-accent transition-colors text-left group">
      <div className={`${iconBg} ${iconColor} p-3 rounded-lg group-hover:scale-110 transition-transform`}>{icon}</div>
      <div className="flex-1">
        <h3 className="font-semibold text-lg mb-1">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
    </button>
  )
}

