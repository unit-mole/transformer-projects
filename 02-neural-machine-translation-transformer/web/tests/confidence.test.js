import test from 'node:test';
import assert from 'node:assert/strict';
import { confidenceProxy } from '../src/confidence.js';

test('returns a bounded proxy', () => {
  const result = confidenceProxy('Quality matters.', 'गुणवत्ता महत्वपूर्ण है।', 'en-hi');
  assert.ok(result.score > 0 && result.score <= 0.95);
});

test('penalizes unchanged output', () => {
  const good = confidenceProxy('Quality matters.', 'गुणवत्ता महत्वपूर्ण है।', 'en-hi');
  const bad = confidenceProxy('Quality matters.', 'Quality matters.', 'en-hi');
  assert.ok(bad.score < good.score);
});
