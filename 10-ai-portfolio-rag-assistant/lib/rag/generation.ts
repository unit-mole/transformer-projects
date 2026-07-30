import { InferenceClient } from "@huggingface/inference";
import type { RetrievedChunk } from "@/lib/types";
import { firstSentence } from "@/lib/utils/text";
import { buildGroundedPrompt } from "@/lib/rag/promptTemplates";
import { ensureCitationPresence } from "@/lib/rag/citationBuilder";

export interface GenerationResult {
  answer: string;
  mode: string;
  warning: string | null;
}

function extractiveAnswer(question: string, chunks: RetrievedChunk[]): string {
  const configuredThreshold = Number(process.env.MIN_RETRIEVAL_SCORE || "0.15");
  const threshold = Number.isFinite(configuredThreshold) ? configuredThreshold : 0.15;
  const supported = chunks.filter((chunk) => chunk.score >= threshold);
  if (supported.length === 0) {
    return "I could not find enough supporting information in the indexed portfolio documents to answer this confidently.";
  }

  const lines = supported.slice(0, 4).map((chunk) => {
    return `- ${chunk.projectName} — ${firstSentence(chunk.text)} [${chunk.citationId}]`;
  });

  const coverageNote =
    "This answer is limited to the portfolio documents currently included in the public index.";
  return `Based on the indexed portfolio evidence for “${question}”:\n\n${lines.join("\n")}\n\n${coverageNote}`;
}

export async function generateGroundedAnswer(
  question: string,
  chunks: RetrievedChunk[],
): Promise<GenerationResult> {
  const useHostedGenerator = process.env.USE_HF_GENERATOR === "true";
  const token = process.env.HF_API_TOKEN;

  if (!useHostedGenerator || !token) {
    return {
      answer: extractiveAnswer(question, chunks),
      mode: "grounded-extractive",
      warning: useHostedGenerator && !token ? "HF_API_TOKEN is missing; used extractive generation." : null,
    };
  }

  try {
    const client = new InferenceClient(token);
    const response = await client.chatCompletion({
      model: process.env.HF_GENERATOR_MODEL || "google/gemma-2-2b-it:fastest",
      messages: [
        {
          role: "user",
          content: buildGroundedPrompt(question, chunks),
        },
      ],
      max_tokens: 500,
      temperature: 0.1,
    });
    const content = response.choices?.[0]?.message?.content;
    const answer = typeof content === "string" && content.trim()
      ? content.trim()
      : extractiveAnswer(question, chunks);

    return {
      answer: ensureCitationPresence(answer, chunks),
      mode: "huggingface-instruction-model",
      warning: null,
    };
  } catch (error) {
    return {
      answer: extractiveAnswer(question, chunks),
      mode: "grounded-extractive-fallback",
      warning: error instanceof Error ? error.message : "Hosted generation failed.",
    };
  }
}
