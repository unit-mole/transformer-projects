import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

async function loadJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

const base = resolve(process.cwd(), "public/data");
const [chunks, embeddings, metadata, questions, evaluation] = await Promise.all([
  loadJson(resolve(base, "document_chunks.json")),
  loadJson(resolve(base, "embeddings.json")),
  loadJson(resolve(base, "metadata.json")),
  loadJson(resolve(base, "evaluation_questions.json")),
  loadJson(resolve(base, "evaluation_summary.json")),
]);

if (!Array.isArray(chunks) || chunks.length === 0) throw new Error("No document chunks found.");
if (!Array.isArray(embeddings) || embeddings.length !== chunks.length) throw new Error("Embedding count mismatch.");
if (!Array.isArray(questions) || questions.length < 40) throw new Error("At least 40 evaluation questions are required.");
if (!metadata || typeof metadata !== "object") throw new Error("metadata.json is invalid.");
if (metadata.chunkCount !== chunks.length) throw new Error("metadata.chunkCount mismatch.");
if (!Number.isInteger(metadata.embedding?.dimension) || metadata.embedding.dimension <= 0) {
  throw new Error("metadata.embedding.dimension must be a positive integer.");
}
if (!evaluation || !["pending", "measured"].includes(evaluation.status)) {
  throw new Error("evaluation_summary.json must have status pending or measured.");
}

const chunkIds = new Set();
for (const chunk of chunks) {
  if (!chunk.id || chunkIds.has(chunk.id)) throw new Error(`Duplicate or missing chunk id: ${chunk.id}`);
  chunkIds.add(chunk.id);
  for (const field of ["projectId", "projectName", "sourceFile", "section", "sourcePath", "text"]) {
    if (!chunk[field]) throw new Error(`Chunk ${chunk.id} is missing ${field}.`);
  }
}

const embeddingIds = new Set();
for (const record of embeddings) {
  if (!chunkIds.has(record.chunkId)) throw new Error(`Unknown chunkId: ${record.chunkId}`);
  if (embeddingIds.has(record.chunkId)) throw new Error(`Duplicate embedding: ${record.chunkId}`);
  embeddingIds.add(record.chunkId);
  if (!Array.isArray(record.vector) || record.vector.length !== metadata.embedding.dimension) {
    throw new Error(`Invalid vector for ${record.chunkId}`);
  }
  if (!record.vector.every(Number.isFinite)) throw new Error(`Non-finite vector value for ${record.chunkId}`);
}

const questionIds = new Set();
for (const item of questions) {
  if (!item.id || questionIds.has(item.id)) throw new Error(`Duplicate or missing question id: ${item.id}`);
  questionIds.add(item.id);
  if (!item.question || typeof item.answerable !== "boolean") {
    throw new Error(`Question ${item.id} is missing question or answerable.`);
  }
  if (item.answerable && (!Array.isArray(item.expected_source_project_ids) || item.expected_source_project_ids.length === 0)) {
    throw new Error(`Answerable question ${item.id} needs expected_source_project_ids.`);
  }
}

console.log(
  `Validated ${chunks.length} chunks, ${embeddings.length} embeddings, ` +
  `${questions.length} evaluation questions, and ${metadata.coverageStatus} corpus metadata.`,
);
