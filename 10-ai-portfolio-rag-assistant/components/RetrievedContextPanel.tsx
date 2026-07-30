import type { RetrievedChunk } from "@/lib/types";

export default function RetrievedContextPanel({ chunks }: { chunks: RetrievedChunk[] }) {
  if (!chunks.length) return null;
  return (
    <details className="context-panel">
      <summary>Inspect retrieved context ({chunks.length} chunks)</summary>
      <div className="context-list">
        {chunks.map((chunk) => (
          <article key={chunk.id}>
            <div className="context-score">
              <strong>[{chunk.citationId}] {chunk.projectName}</strong>
              <span>score {chunk.score.toFixed(3)}</span>
            </div>
            <p>{chunk.text}</p>
            <small>{chunk.sourcePath} · semantic {chunk.semanticScore.toFixed(3)} · lexical {chunk.lexicalScore.toFixed(3)}</small>
          </article>
        ))}
      </div>
    </details>
  );
}
