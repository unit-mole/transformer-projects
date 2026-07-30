import type { RetrievedChunk } from "@/lib/types";

export function buildGroundedPrompt(question: string, chunks: RetrievedChunk[]): string {
  const context = chunks
    .map(
      (chunk) =>
        `[${chunk.citationId}] Project: ${chunk.projectName}\n` +
        `Source: ${chunk.sourceFile} — ${chunk.section}\n` +
        `Path: ${chunk.sourcePath}\n` +
        `Evidence: ${chunk.text}`,
    )
    .join("\n\n");

  return `You are the AI Portfolio Assistant for Anmol Tripathi.

Answer the user's question using only the retrieved portfolio context.

Rules:
1. Do not invent projects, models, datasets, metrics, deployments, links, work experience, or results.
2. Cite important claims using the exact citation labels [S1], [S2], and so on.
3. If the context is insufficient, explicitly say the indexed portfolio documents do not contain enough information.
4. Keep the answer professional, concise, and recruiter-friendly.
5. Never treat the answer as employment verification or an official resume.

Retrieved context:
${context}

User question:
${question}

Grounded answer with citations:`;
}
