import type {
  CorpusMetadata,
  EmbeddingRecord,
  PortfolioChunk,
} from "@/lib/types";

export interface VectorStore {
  chunks: PortfolioChunk[];
  embeddingByChunkId: Map<string, number[]>;
  metadata: CorpusMetadata;
}

export function createVectorStore(
  chunks: PortfolioChunk[],
  embeddings: EmbeddingRecord[],
  metadata: CorpusMetadata,
): VectorStore {
  const embeddingByChunkId = new Map(
    embeddings.map((record) => [record.chunkId, record.vector]),
  );

  for (const chunk of chunks) {
    const vector = embeddingByChunkId.get(chunk.id);
    if (!vector) throw new Error(`Missing embedding for chunk ${chunk.id}.`);
    if (vector.length !== metadata.embedding.dimension) {
      throw new Error(`Embedding dimension mismatch for chunk ${chunk.id}.`);
    }
  }

  return { chunks, embeddingByChunkId, metadata };
}
