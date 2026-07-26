import test from 'node:test';
import assert from 'node:assert/strict';

import { csvToDocumentText, parseCsvRows } from '../src/csv-parser.js';

test('parseCsvRows handles quoted commas', () => {
  const rows = parseCsvRows('id,text\n1,"Corrective action, due Friday"');
  assert.equal(rows[1][1], 'Corrective action, due Friday');
});

test('csvToDocumentText selects a recognized text column', () => {
  const result = csvToDocumentText('id,context,label\n1,"Evidence paragraph",ok');
  assert.equal(result, 'Evidence paragraph');
});
