import * as pdfjsLib from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

import { normalizeDocumentText } from './chunking.js';
import { csvToDocumentText } from './csv-parser.js';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

function extensionOf(filename) {
  const value = String(filename ?? '').toLowerCase();
  const index = value.lastIndexOf('.');
  return index >= 0 ? value.slice(index) : '';
}

async function extractPdfText(file) {
  const data = new Uint8Array(await file.arrayBuffer());
  const pdf = await pdfjsLib.getDocument({ data }).promise;
  const pages = [];
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item) => ('str' in item ? item.str : ''))
      .join(' ')
      .trim();
    if (pageText) pages.push(pageText);
  }
  return normalizeDocumentText(pages.join('\n\n'));
}

export async function readDocumentFile(file) {
  if (!file) throw new Error('Choose a document before reading it.');
  const extension = extensionOf(file.name);
  const allowed = new Set(['.txt', '.md', '.csv', '.pdf']);
  if (!allowed.has(extension)) {
    throw new Error(`Unsupported file type: ${extension || 'unknown'}. Use TXT, Markdown, CSV, or PDF.`);
  }

  let text;
  if (extension === '.pdf') {
    text = await extractPdfText(file);
  } else {
    const raw = await file.text();
    text = extension === '.csv' ? csvToDocumentText(raw) : normalizeDocumentText(raw);
  }
  if (!text) {
    throw new Error('No readable text was found. Scanned PDFs require OCR and are not supported in this static demo.');
  }
  return { text, sourceName: file.name, extension };
}
