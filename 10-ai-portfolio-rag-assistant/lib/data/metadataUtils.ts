import type { PortfolioChunk } from "@/lib/types";

export function uniqueCategories(chunks: PortfolioChunk[]): string[] {
  return [...new Set(chunks.map((chunk) => chunk.category))].sort();
}

export function uniqueDeployments(chunks: PortfolioChunk[]): string[] {
  return [...new Set(chunks.map((chunk) => chunk.deployment))].sort();
}
