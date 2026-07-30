import { NextResponse } from "next/server";
import { retrievePortfolioChunks } from "@/lib/rag/retrieval";
import { validateRequestBody } from "@/lib/utils/validation";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { question, topK, filters } = validateRequestBody(body);
    const result = await retrievePortfolioChunks(question, topK, filters);
    return NextResponse.json(result, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Retrieval failed." },
      { status: 400 },
    );
  }
}
