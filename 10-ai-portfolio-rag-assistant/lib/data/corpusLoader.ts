import chunksJson from "@/public/data/document_chunks.json";
import embeddingsJson from "@/public/data/embeddings.json";
import metadataJson from "@/public/data/metadata.json";
import type {
  CorpusMetadata,
  EmbeddingRecord,
  PortfolioChunk,
} from "@/lib/types";
import { createVectorStore } from "@/lib/embeddings/vectorStore";

let cachedStore: ReturnType<typeof createVectorStore> | null = null;

export function loadVectorStore() {
  if (!cachedStore) {
    cachedStore = createVectorStore(
      chunksJson as PortfolioChunk[],
      embeddingsJson as EmbeddingRecord[],
      metadataJson as CorpusMetadata,
    );
  }
  return cachedStore;
}
