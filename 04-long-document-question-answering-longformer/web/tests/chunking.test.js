import test from 'node:test';
import assert from 'node:assert/strict';

import {
  chunkDocument,
  lexicalScore,
  normalizeDocumentText,
  rankChunks,
  splitParagraphsWithOffsets,
} from '../src/chunking.js';

test('normalizeDocumentText preserves paragraph boundaries', () => {
  assert.equal(normalizeDocumentText('A\r\n\r\n\r\nB'), 'A\n\nB');
});

test('chunkDocument creates overlapping chunks with offsets', () => {
  const text = Array.from({ length: 240 }, (_, index) => `word${index}`).join(' ');
  const chunks = chunkDocument(text, 100, 20);
  assert.ok(chunks.length >= 3);
  assert.equal(chunks[0].start, 0);
  assert.ok(chunks[1].start < chunks[0].end);
  assert.equal(text.slice(chunks[1].start, chunks[1].end), chunks[1].text);
});

test('rankChunks prioritizes query evidence', () => {
  const chunks = [
    { id: 0, text: 'General background about maintenance.', start: 0, end: 37 },
    { id: 1, text: 'The effectiveness review is scheduled for 30 September 2026.', start: 38, end: 100 },
  ];
  const ranked = rankChunks('When is the effectiveness review scheduled?', chunks, 1);
  assert.equal(ranked[0].id, 1);
  assert.ok(lexicalScore('effectiveness review', chunks[1].text) > 0);
});

test('splitParagraphsWithOffsets returns mappable ranges', () => {
  const paragraphs = splitParagraphsWithOffsets('First paragraph.\n\nSecond paragraph.');
  assert.equal(paragraphs.length, 2);
  assert.equal(paragraphs[1].text, 'Second paragraph.');
  assert.equal('First paragraph.\n\nSecond paragraph.'.slice(paragraphs[1].start, paragraphs[1].end), paragraphs[1].text);
});
