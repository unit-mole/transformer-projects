import test from 'node:test';
import assert from 'node:assert/strict';

import { highlightEvidence, locateSupportingParagraph } from '../src/evidence.js';

test('locateSupportingParagraph maps an answer span', () => {
  const text = 'Paragraph one.\n\nThe action owner is Jordan Lee.';
  const start = text.indexOf('Jordan Lee');
  const result = locateSupportingParagraph(text, start, start + 'Jordan Lee'.length);
  assert.equal(result.paragraph?.id, 1);
});

test('highlightEvidence escapes content and marks answer', () => {
  const paragraph = { text: 'Owner: <Jordan Lee>', start: 10, end: 30 };
  const html = highlightEvidence(paragraph, 18, 29);
  assert.match(html, /<mark>/);
  assert.match(html, /&lt;/);
  assert.doesNotMatch(html, /<Jordan/);
});
