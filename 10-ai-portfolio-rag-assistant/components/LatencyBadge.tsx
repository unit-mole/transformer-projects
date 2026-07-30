export default function LatencyBadge({ milliseconds }: { milliseconds: number }) {
  return <span className="latency-badge">{milliseconds.toFixed(0)} ms total</span>;
}
