type CounterProps = {
    color?: 1 | 2
}

export default function Counter({color}: CounterProps) {
    return <div className={`${color == 1 && "bg-black"} ${color == 2 && "bg-white"} rounded-full aspect-square`}></div>
}