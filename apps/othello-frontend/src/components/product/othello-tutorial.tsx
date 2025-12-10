import { useState, useLayoutEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Circle } from "lucide-react"
import {Link} from "@tanstack/react-router";

export function OthelloTutorial() {
  const [step, setStep] = useState(0)
  const [animationStep, setAnimationStep] = useState(0)
  const [complexAnimationStep, setComplexAnimationStep] = useState(0)

  useLayoutEffect(() => {
    if (step === 2) {
      setAnimationStep(0)
      const timer = setInterval(() => {
        setAnimationStep((prev) => (prev + 1) % 4)
      }, 1500)
      return () => clearInterval(timer)
    }
  }, [step])

  useLayoutEffect(() => {
    if (step === 3) {
      setComplexAnimationStep(0)
      const timer = setInterval(() => {
        setComplexAnimationStep((prev) => (prev + 1) % 5)
      }, 1500)
      return () => clearInterval(timer)
    }
  }, [step])

  const tutorialSteps = [
    {
      title: "Welcome to Othello!",
      description: "Also known as Reversi, Othello is a strategy board game for two players.",
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground leading-relaxed">
            The game is played on an 8×8 board with discs that are black on one side and white on the other. The goal is
            to have the majority of your color discs on the board at the end of the game.
          </p>
          <div className="flex justify-center gap-4 py-4">
            <div className="flex items-center gap-2">
              <div className="size-8 rounded-full bg-foreground" />
              <span className="text-sm">Black</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="size-8 rounded-full bg-background border-2 border-foreground" />
              <span className="text-sm">White</span>
            </div>
          </div>
          <p className="text-muted-foreground leading-relaxed">
            In this tutorial, you will learn how to play Othello in five easy steps.
          </p>
        </div>
      ),
    },
    {
      title: "Starting Position",
      description: "The game begins with four discs placed in the center of the board.",
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground leading-relaxed">
            Two black discs and two white discs are arranged in a diagonal pattern in the four center squares. Black
            always moves first.
          </p>
          <Card className="bg-muted/50">
            <CardContent className="p-4">
              <div className="grid grid-cols-4 gap-1 max-w-[200px] mx-auto">
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-background border-2 border-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-background border-2 border-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
              </div>
            </CardContent>
          </Card>
        </div>
      ),
    },
    {
      title: "How to Make a Move",
      description: "You must place your disc so that you flank (trap) your opponent's disc(s).",
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground leading-relaxed">
            A valid move must sandwich at least one opponent's disc between your new disc and another of your discs
            already on the board. The trapped discs are then flipped to your color.
          </p>
          <Card className="bg-muted/50">
            <CardContent className="p-6">
              <div className="grid grid-cols-5 gap-2 max-w-[280px] mx-auto">
                {/* Row 1 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />

                {/* Row 2 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div
                    key={step}
                    className={`size-8 rounded-full transition-all duration-500 ${
                      animationStep >= 1 ? "bg-foreground" : "bg-foreground opacity-30 scale-0"
                    }`}
                  />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div
                    key={step}
                    className={`size-8 rounded-full border-2 border-foreground transition-all duration-500 ${
                      animationStep >= 2 ? "bg-foreground rotate-180" : "bg-background"
                    }`}
                  />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />

                {/* Row 3 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
              </div>
              <div className="mt-4 text-center">
                <p className="text-sm font-medium">
                  {animationStep === 0 && "Starting position"}
                  {animationStep === 1 && "Black places disc..."}
                  {animationStep === 2 && "White disc flips!"}
                  {animationStep === 3 && "Move complete"}
                </p>
              </div>
            </CardContent>
          </Card>
          <p className="text-sm text-muted-foreground leading-relaxed">
            You can flip discs horizontally, vertically, or diagonally. Multiple lines of discs can be flipped in a
            single move.
          </p>
        </div>
      ),
    },
    {
      title: "Complex Example: Multiple Directions",
      description: "One move can flip pieces vertically and diagonally at the same time.",
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground leading-relaxed">
            Watch how placing a single black disc can flip multiple white discs in different directions simultaneously.
          </p>
          <Card className="bg-muted/50">
            <CardContent className="p-6">
              <div className="grid grid-cols-5 gap-2 max-w-[280px] mx-auto">
                {/* Row 1 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />

                {/* Row 2 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div
                    key={step}
                    className={`size-8 rounded-full border-2 border-foreground transition-all duration-500 ${
                      complexAnimationStep >= 2 ? "bg-foreground rotate-180" : "bg-background"
                    }`}
                  />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-background border-2 border-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />

                {/* Row 3 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div
                    key={step}
                    className={`size-8 rounded-full border-2 border-foreground transition-all duration-500 ${
                      complexAnimationStep >= 2 ? "bg-foreground rotate-180" : "bg-background"
                    }`}
                  />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div
                    className={`size-8 rounded-full border-2 border-foreground transition-all duration-500 ${
                      complexAnimationStep >= 3 ? "bg-foreground rotate-180" : "bg-background"
                    }`}
                  />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div className="size-8 rounded-full bg-foreground" />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />

                {/* Row 4 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm flex items-center justify-center">
                  <div
                    key={step}
                    className={`size-8 rounded-full transition-all duration-500 ${
                      complexAnimationStep >= 1 ? "bg-foreground" : "bg-foreground opacity-30 scale-0"
                    }`}
                  />
                </div>
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />

                {/* Row 5 */}
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
                <div className="size-12 bg-accent/30 rounded-sm" />
              </div>
              <div className="mt-4 text-center">
                <p className="text-sm font-medium">
                  {complexAnimationStep === 0 && "Starting position"}
                  {complexAnimationStep === 1 && "Black places disc..."}
                  {complexAnimationStep === 2 && "Both vertical discs flip!"}
                  {complexAnimationStep === 3 && "Diagonal disc flips!"}
                  {complexAnimationStep === 4 && "All flips complete"}
                </p>
              </div>
            </CardContent>
          </Card>
          <p className="text-sm text-muted-foreground leading-relaxed">
            In this example, placing black at the middle-left position flips three white discs: two vertically (going
            up) and one diagonally (going up-right). This demonstrates the power of strategic placement.
          </p>
        </div>
      ),
    },
    {
      title: "Game Rules",
      description: "Important rules to remember while playing.",
      content: (
        <div className="space-y-3">
          <Card>
            <CardContent className="p-4">
              <ul className="space-y-3 text-sm">
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Must flip discs:</strong> Every move must flip at least one opponent disc
                  </span>
                </li>
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Pass if no moves:</strong> If you have no valid moves, your turn is skipped
                  </span>
                </li>
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Game ends:</strong> When the board is full or neither player can move
                  </span>
                </li>
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Winner:</strong> The player with the most discs of their color wins
                  </span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      ),
    },
    {
      title: "Strategy Tips",
      description: "Master these strategies to improve your game.",
      content: (
        <div className="space-y-3">
          <Card>
            <CardContent className="p-4">
              <ul className="space-y-3 text-sm">
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Control corners:</strong> Corner pieces cannot be flipped and give you an advantage
                  </span>
                </li>
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Avoid edges early:</strong> Pieces next to corners can give your opponent corner access
                  </span>
                </li>
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Mobility matters:</strong> Having more valid moves gives you more control
                  </span>
                </li>
                <li className="flex gap-3">
                  <Circle className="size-4 mt-0.5 shrink-0 fill-primary text-primary" />
                  <span className="leading-relaxed">
                    <strong>Think ahead:</strong> Consider what moves you're giving your opponent
                  </span>
                </li>
              </ul>
            </CardContent>
          </Card>
          <p className="text-xs text-center text-muted-foreground">Now you're ready to play! Good luck!</p>
        </div>
      ),
    },
  ]

  const currentStep = tutorialSteps[step]

  return (
    <div className="w-full max-w-2xl mx-auto">
      <Card className="border-2">
        <CardContent className="p-8 space-y-6">
          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-balance">{currentStep.title}</h2>
            <p className="text-base text-muted-foreground text-balance">{currentStep.description}</p>
          </div>

          <div className="py-4">{currentStep.content}</div>

          <div className="flex items-center justify-between pt-4 border-t">
            <div className="flex gap-1.5">
              {tutorialSteps.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setStep(index)}
                  className={`size-2 rounded-full transition-all ${
                    index === step ? "bg-primary w-6" : "bg-muted-foreground/30 hover:bg-muted-foreground/50"
                  }`}
                  aria-label={`Go to step ${index + 1}`}
                />
              ))}
            </div>

            <div className="flex gap-2">
              {step > 0 && (
                <Button variant="outline" onClick={() => setStep(step - 1)}>
                  Previous
                </Button>
              )}
              {step < tutorialSteps.length - 1 && <Button onClick={() => setStep(step + 1)}>Next</Button>}
              {step === tutorialSteps.length - 1 && <Button asChild>
                  <Link to="/play">
                      Start a Game
                  </Link>
              </Button>}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
