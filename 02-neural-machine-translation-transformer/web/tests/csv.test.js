import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCsv, toCsv } from '../src/csv.js';

test('parses quoted CSV fields', () => {
  const parsed = parseCsv('text,label\n"hello, world",en\n');
  assert.equal(parsed.records[0].text, 'hello, world');
});

test('round-trips simple records', () => {
  const csv = toCsv([{ text: 'hello', label: 'en' }]);
  assert.equal(parseCsv(csv).records[0].label, 'en');
});
