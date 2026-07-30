import type { Citation } from "@/lib/types";

export default function SourceCitationCard({ citation }: { citation: Citation }) {
  return (
    <article className="citation-card">
      <div className="citation-card__header">
        <span className="citation-label">[{citation.citationId}]</span>
        <span>{(citation.similarityScore * 100).toFixed(1)}% relevance</span>
      </div>
      <h4>{citation.projectName}</h4>
      <p className="citation-meta">{citation.sourceFile} · {citation.section}</p>
      <p>{citation.evidence}</p>
      <code>{citation.chunkId}</code>
    </article>
  );
}
