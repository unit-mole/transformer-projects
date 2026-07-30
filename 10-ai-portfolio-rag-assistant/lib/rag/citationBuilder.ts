import type { Citation, RetrievedChunk } from "@/lib/types";
import { safeEvidence } from "@/lib/utils/text";

export function buildCitations(chunks: RetrievedChunk[]): Citation[] {
  return chunks.map((chunk) => ({
    citationId: chunk.citationId,
    projectId: chunk.projectId,
    projectName: chunk.projectName,
    sourceFile: chunk.sourceFile,
    section: chunk.section,
    chunkId: chunk.id,
    evidence: safeEvidence(chunk.text),
    similarityScore: chunk.score,
    sourcePath: chunk.sourcePath,
    repositoryUrl: chunk.repositoryUrl,
  }));
}

export function ensureCitationPresence(answer: string, chunks: RetrievedChunk[]): string {
  if (chunks.length === 0 || /\[S\d+\]/.test(answer)) return answer;
  const references = chunks.slice(0, 3).map((chunk) => `[${chunk.citationId}]`).join(" ");
  return `${answer.trim()}\n\nSources: ${references}`;
}
