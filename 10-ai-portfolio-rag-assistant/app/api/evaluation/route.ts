import { NextResponse } from "next/server";
import evaluation from "@/public/data/evaluation_summary.json";

export const runtime = "nodejs";
export const dynamic = "force-static";

export async function GET() {
  return NextResponse.json(evaluation, {
    headers: { "Cache-Control": "public, max-age=3600, s-maxage=3600" },
  });
}
