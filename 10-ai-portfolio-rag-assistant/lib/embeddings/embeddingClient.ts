import { InferenceClient } from "@huggingface/inference";
import type { CorpusMetadata } from "@/lib/types";
import { localHashEmbedding } from "@/lib/embeddings/localHash";

export interface QueryEmbeddingResult {
  vector: number[] | null;
  mode: string;
  warning: string | null;
}

function flattenEmbedding(output: unknown): number[] | null {
  if (!Array.isArray(output)) return null;
  if (output.every((value) => typeof value === "number")) return output as number[];
  if (
    output.length > 0 &&
    Array.isArray(output[0]) &&
    (output[0] as unknown[]).every((value) => typeof value === "number")
  ) {
    return output[0] as number[];
  }
  return null;
}

export async function embedQuery(
  question: string,
  metadata: CorpusMetadata,
): Promise<QueryEmbeddingResult> {
  if (metadata.embedding.provider === "local-hash-v1") {
    return {
      vector: localHashEmbedding(question, metadata.embedding.dimension),
      mode: "local-hash-v1",
      warning: null,
    };
  }

  const token = process.env.HF_API_TOKEN;
  if (!token) {
    return {
      vector: null,
      mode: "lexical-fallback",
      warning:
        "The corpus uses MiniLM embeddings, but HF_API_TOKEN is not configured. Retrieval used the lexical fallback.",
    };
  }

  try {
    const client = new InferenceClient(token);
    const output = await client.featureExtraction({
      model: process.env.HF_EMBEDDING_MODEL || metadata.embedding.model,
      inputs: `${metadata.embedding.queryPrefix || ""}${question}`,
      provider: "hf-inference",
      normalize: true,
    });
    const vector = flattenEmbedding(output);

    if (!vector || vector.length !== metadata.embedding.dimension) {
      return {
        vector: null,
        mode: "lexical-fallback",
        warning: "The embedding provider returned an unexpected vector shape.",
      };
    }

    return { vector, mode: "huggingface-minilm", warning: null };
  } catch (error) {
    return {
      vector: null,
      mode: "lexical-fallback",
      warning: error instanceof Error ? error.message : "Embedding request failed.",
    };
  }
}
