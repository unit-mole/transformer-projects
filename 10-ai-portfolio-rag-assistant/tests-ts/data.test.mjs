import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

async function json(path) {
  return JSON.parse(await readFile(new URL(path, import.meta.url), "utf8"));
}

test("static vector store has matching chunks and embeddings", async () => {
  const chunks = await json("../public/data/document_chunks.json");
  const embeddings = await json("../public/data/embeddings.json");
  const metadata = await json("../public/data/metadata.json");
  assert.equal(chunks.length, embeddings.length);
  assert.equal(metadata.chunkCount, chunks.length);
  assert.ok(embeddings.every((item) => item.vector.length === metadata.embedding.dimension));
});

test("every chunk contains traceable citation metadata", async () => {
  const chunks = await json("../public/data/document_chunks.json");
  for (const chunk of chunks) {
    assert.ok(chunk.id);
    assert.ok(chunk.projectId);
    assert.ok(chunk.projectName);
    assert.ok(chunk.sourceFile);
    assert.ok(chunk.section);
    assert.ok(chunk.sourcePath);
    assert.ok(chunk.repositoryUrl);
  }
});

test("evaluation set is portfolio-grade and includes refusal cases", async () => {
  const questions = await json("../public/data/evaluation_questions.json");
  assert.ok(questions.length >= 40);
  assert.ok(questions.some((item) => item.answerable === false));
  assert.ok(questions.every((item) => item.id && item.question));
});

test("evaluation summary never presents an invalid status", async () => {
  const summary = await json("../public/data/evaluation_summary.json");
  assert.ok(["pending", "measured"].includes(summary.status));
});
