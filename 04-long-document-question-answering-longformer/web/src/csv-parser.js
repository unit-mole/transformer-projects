import { normalizeDocumentText } from './chunking.js';

const TEXT_COLUMN_CANDIDATES = [
  'text', 'content', 'document', 'context', 'paragraph', 'body', 'description', 'notes',
];

export function parseCsvRows(rawText) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  const text = String(rawText ?? '');

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    const next = text[index + 1];
    if (character === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === ',' && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && next === '\n') index += 1;
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += character;
    }
  }
  row.push(cell);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

export function csvToDocumentText(rawText) {
  const rows = parseCsvRows(rawText);
  if (!rows.length) return '';
  const headers = rows[0].map((value) => value.trim().toLowerCase());
  const preferredIndex = headers.findIndex((header) => TEXT_COLUMN_CANDIDATES.includes(header));
  const dataRows = preferredIndex >= 0 ? rows.slice(1) : rows;
  const selected = dataRows
    .map((row) => {
      if (preferredIndex >= 0) return row[preferredIndex] ?? '';
      return row.map((cell) => cell.trim()).filter(Boolean).join(' | ');
    })
    .filter((value) => value.trim());
  return normalizeDocumentText(selected.join('\n\n'));
}
