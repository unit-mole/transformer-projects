import type { RetrievedChunk } from "@/lib/types";
import {
  MODERATE_SUPPORT_THRESHOLD,
  WEAK_SUPPORT_THRESHOLD,
} from "@/lib/utils/constants";

export function assessRuntimeSupport(chunks: RetrievedChunk[]) {
  const topScores = chunks.slice(0, 3).map((chunk) => chunk.score);
  const supportScore = topScores.length
    ? topScores.reduce((sum, score) => sum + score, 0) / topScores.length
    : 0;
  const rounded = Number(supportScore.toFixed(3));

  if (supportScore >= MODERATE_SUPPORT_THRESHOLD) {
    return { label: "strong" as const, supportScore: rounded, warning: null };
  }
  if (supportScore >= WEAK_SUPPORT_THRESHOLD) {
    return {
      label: "moderate" as const,
      supportScore: rounded,
      warning: "Retrieved evidence is moderately relevant; review the cited chunks.",
    };
  }
  return {
    label: "weak" as const,
    supportScore: rounded,
    warning:
      "The indexed corpus provides weak support for this question. Treat the answer as incomplete.",
  };
}
