import {
  MAX_QUESTION_LENGTH,
  MAX_TOP_K,
  MIN_QUESTION_LENGTH,
} from "@/lib/utils/constants";
import type { RetrievalFilters } from "@/lib/types";

export interface ValidatedRequest {
  question: string;
  topK: number;
  filters: RetrievalFilters;
}

export function validateRequestBody(body: unknown): ValidatedRequest {
  if (!body || typeof body !== "object") {
    throw new Error("Request body must be a JSON object.");
  }

  const candidate = body as Record<string, unknown>;
  const question = typeof candidate.question === "string" ? candidate.question.trim() : "";

  if (question.length < MIN_QUESTION_LENGTH) {
    throw new Error("Please enter a more specific question.");
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    throw new Error(`Question must be ${MAX_QUESTION_LENGTH} characters or fewer.`);
  }

  const requestedTopK = Number(candidate.topK ?? 5);
  const topK = Number.isFinite(requestedTopK)
    ? Math.min(MAX_TOP_K, Math.max(1, Math.floor(requestedTopK)))
    : 5;

  const rawFilters =
    candidate.filters && typeof candidate.filters === "object"
      ? (candidate.filters as Record<string, unknown>)
      : {};

  const filters: RetrievalFilters = {
    category: typeof rawFilters.category === "string" ? rawFilters.category : undefined,
    deployment:
      typeof rawFilters.deployment === "string" ? rawFilters.deployment : undefined,
    projectId: typeof rawFilters.projectId === "string" ? rawFilters.projectId : undefined,
  };

  return { question, topK, filters };
}
