import type { ChatResponse } from "@/lib/types";
import LatencyBadge from "@/components/LatencyBadge";

export default function EvaluationMetricsPanel({ response }: { response: ChatResponse }) {
  return (
    <section className="metrics-grid" aria-label="Runtime metrics">
      <div><span>Retrieval</span><strong>{response.runtime.retrievalMode}</strong></div>
      <div><span>Generation</span><strong>{response.runtime.generationMode}</strong></div>
      <div><span>Support</span><strong>{response.groundedness.label}</strong></div>
      <div><span>Corpus</span><strong>{response.runtime.corpusCoverage}</strong></div>
      <div><span>Embedding</span><strong>{response.metrics.embeddingMs.toFixed(1)} ms</strong></div>
      <div><span>Vector search</span><strong>{response.metrics.retrievalMs.toFixed(1)} ms</strong></div>
      <div><span>Generation</span><strong>{response.metrics.generationMs.toFixed(1)} ms</strong></div>
      <div><span>Total latency</span><LatencyBadge milliseconds={response.metrics.totalMs} /></div>
      <div className="metrics-grid__wide"><span>Embedding model</span><strong>{response.runtime.embeddingModel}</strong></div>
      <div><span>Documents</span><strong>{response.runtime.documentCount}</strong></div>
      <div><span>Chunks</span><strong>{response.runtime.chunkCount}</strong></div>
    </section>
  );
}
