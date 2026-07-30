import { NextResponse } from "next/server";
import { loadVectorStore } from "@/lib/data/corpusLoader";

export const runtime = "nodejs";

export async function GET() {
  try {
    const store = loadVectorStore();
    return NextResponse.json({
      status: "ok",
      app: "AI Portfolio RAG Assistant",
      documentCount: store.metadata.documentCount,
      chunkCount: store.metadata.chunkCount,
      coverageStatus: store.metadata.coverageStatus,
      embeddingProvider: store.metadata.embedding.provider,
      embeddingModel: store.metadata.embedding.model,
      transformerReady: store.metadata.embedding.provider === "huggingface-feature-extraction",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        error: error instanceof Error ? error.message : "Health check failed.",
      },
      { status: 500 },
    );
  }
}
