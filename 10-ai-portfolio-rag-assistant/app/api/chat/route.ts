import { NextResponse } from "next/server";
import { buildCitations } from "@/lib/rag/citationBuilder";
import { generateGroundedAnswer } from "@/lib/rag/generation";
import { assessRuntimeSupport } from "@/lib/rag/groundedness";
import { elapsedMs, nowMs } from "@/lib/rag/latency";
import { retrievePortfolioChunks } from "@/lib/rag/retrieval";
import { loadVectorStore } from "@/lib/data/corpusLoader";
import { validateRequestBody } from "@/lib/utils/validation";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const totalStart = nowMs();
  try {
    const body = await request.json();
    const { question, topK, filters } = validateRequestBody(body);

    const retrieval = await retrievePortfolioChunks(question, topK, filters);
    const generationStart = nowMs();
    const generation = await generateGroundedAnswer(question, retrieval.chunks);
    const generationMs = elapsedMs(generationStart);
    const store = loadVectorStore();

    return NextResponse.json(
      {
        answer: generation.answer,
        citations: buildCitations(retrieval.chunks),
        retrievedChunks: retrieval.chunks,
        metrics: {
          embeddingMs: retrieval.embeddingMs,
          retrievalMs: retrieval.retrievalMs,
          generationMs,
          totalMs: elapsedMs(totalStart),
        },
        groundedness: assessRuntimeSupport(retrieval.chunks),
        runtime: {
          retrievalMode: retrieval.mode,
          generationMode: generation.mode,
          corpusCoverage: store.metadata.coverageStatus,
          embeddingModel: store.metadata.embedding.model,
          documentCount: store.metadata.documentCount,
          chunkCount: store.metadata.chunkCount,
        },
        warnings: [retrieval.warning, generation.warning].filter(Boolean),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "The assistant request failed." },
      { status: 400 },
    );
  }
}
