import { cn } from "@/lib/utils"

interface ScoreProps {
    blackScore: number
    whiteScore: number
    currentTurn: 1 | 2
}

export default function Score({ blackScore, whiteScore, currentTurn }: ScoreProps) {
    return (
        <div className="flex items-center justify-center gap-6 mb-6">
            {/* Black Player */}
            <div
                className={cn(
                    "flex items-center gap-4 rounded-xl border-2 px-6 py-3 transition-all duration-300",
                    currentTurn === 1
                        ? "border-foreground bg-accent shadow-lg scale-105"
                        : "border-border bg-card opacity-60",
                )}
            >
                <div
                    className={cn(
                        "h-10 w-10 rounded-full border-3 transition-all duration-300 flex-shrink-0",
                        currentTurn === 1 ? "border-foreground bg-foreground shadow-md" : "border-muted bg-foreground",
                    )}
                />
                <div className="flex flex-col items-start gap-0.5">
                    <div className="text-2xl font-bold tabular-nums leading-none">{blackScore}</div>
                    <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Black</div>
                </div>
            </div>

            {/* White Player */}
            <div
                className={cn(
                    "flex items-center gap-4 rounded-xl border-2 px-6 py-3 transition-all duration-300",
                    currentTurn === 2
                        ? "border-foreground bg-accent shadow-lg scale-105"
                        : "border-border bg-card opacity-60",
                )}
            >
                <div
                    className={cn(
                        "h-10 w-10 rounded-full border-3 transition-all duration-300 flex-shrink-0",
                        currentTurn === 2 ? "border-foreground bg-background shadow-md" : "border-muted bg-background",
                    )}
                />
                <div className="flex flex-col items-start gap-0.5">
                    <div className="text-2xl font-bold tabular-nums leading-none">{whiteScore}</div>
                    <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">White</div>
                </div>
            </div>
        </div>
    )
}
