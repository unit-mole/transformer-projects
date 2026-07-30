import { cosineSimilarity } from "@/lib/embeddings/cosineSimilarity";
import { embedQuery } from "@/lib/embeddings/embeddingClient";
import { loadVectorStore } from "@/lib/data/corpusLoader";
import type {
  PortfolioChunk,
  RetrievalFilters,
  RetrievedChunk,
} from "@/lib/types";
import { tokenize } from "@/lib/utils/text";
import { elapsedMs, nowMs } from "@/lib/rag/latency";

function matchesFilters(chunk: PortfolioChunk, filters: RetrievalFilters): boolean {
  const categoryMatches =
    !filters.category || filters.category === "All" || chunk.category === filters.category;
  const deploymentMatches =
    !filters.deployment ||
    filters.deployment === "All" ||
    chunk.deployment === filters.deployment;
  const projectMatches = !filters.projectId || chunk.projectId === filters.projectId;
  return categoryMatches && deploymentMatches && projectMatches;
}

function lexicalScore(question: string, chunk: PortfolioChunk): number {
  const queryTokens = new Set(tokenize(question));
  if (queryTokens.size === 0) return 0;

  const chunkTokens = new Set(
    tokenize(`${chunk.projectName} ${chunk.section} ${chunk.text} ${chunk.keywords.join(" ")}`),
  );
  let overlap = 0;
  for (const token of queryTokens) {
    if (chunkTokens.has(token)) overlap += 1;
  }
  const coverage = overlap / queryTokens.size;
  const projectBoost = question.toLowerCase().includes(chunk.projectId.toLowerCase()) ? 0.15 : 0;
  return Math.min(1, coverage + projectBoost);
}

export interface RetrievalResult {
  chunks: RetrievedChunk[];
  embeddingMs: number;
  retrievalMs: number;
  mode: string;
  warning: string | null;
}

export async function retrievePortfolioChunks(
  question: string,
  topK: number,
  filters: RetrievalFilters,
): Promise<RetrievalResult> {
  const store = loadVectorStore();

  const embeddingStart = nowMs();
  const queryEmbedding = await embedQuery(question, store.metadata);
  const embeddingMs = elapsedMs(embeddingStart);

  const retrievalStart = nowMs();
  const semanticWeight = queryEmbedding.vector
    ? store.metadata.retrieval.hybridSemanticWeight
    : 0;
  const lexicalWeight = queryEmbedding.vector
    ? store.metadata.retrieval.hybridLexicalWeight
    : 1;

  const ranked = store.chunks
    .filter((chunk) => matchesFilters(chunk, filters))
    .map((chunk) => {
      const documentVector = store.embeddingByChunkId.get(chunk.id) ?? [];
      const semantic = queryEmbedding.vector
        ? Math.max(0, cosineSimilarity(queryEmbedding.vector, documentVector))
        : 0;
      const lexical = lexicalScore(question, chunk);
      const score = semanticWeight * semantic + lexicalWeight * lexical;
      return { ...chunk, score, semanticScore: semantic, lexicalScore: lexical };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((chunk, index) => ({
      ...chunk,
      citationId: `S${index + 1}`,
      score: Number(chunk.score.toFixed(4)),
      semanticScore: Number(chunk.semanticScore.toFixed(4)),
      lexicalScore: Number(chunk.lexicalScore.toFixed(4)),
    }));

  return {
    chunks: ranked,
    embeddingMs,
    retrievalMs: elapsedMs(retrievalStart),
    mode: queryEmbedding.mode,
    warning: queryEmbedding.warning,
  };
}
