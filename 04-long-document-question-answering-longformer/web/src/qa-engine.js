import { pipeline } from '@huggingface/transformers';

import { APP_CONFIG } from './config.js';
import {
  chunkDocument,
  countWords,
  normalizeDocumentText,
  rankChunks,
} from './chunking.js';
import { highlightEvidence, locateSupportingParagraph } from './evidence.js';

let cachedPipeline = null;
let cachedRuntime = null;

function normalizeProgress(progress) {
  if (!progress || typeof progress !== 'object') return { label: 'Loading model', percent: null };
  const label = progress.file
    ? `${progress.status ?? 'loading'}: ${progress.file}`
    : String(progress.status ?? 'Loading model');
  let percent = null;
  if (Number.isFinite(progress.progress)) percent = Math.round(progress.progress);
  else if (Number.isFinite(progress.loaded) && Number.isFinite(progress.total) && progress.total > 0) {
    percent = Math.round((progress.loaded / progress.total) * 100);
  }
  return { label, percent };
}

export async function loadBrowserModel({ runtime = 'wasm', onProgress = () => {} } = {}) {
  if (cachedPipeline && cachedRuntime === runtime) return cachedPipeline;

  const options = {
    dtype: runtime === 'webgpu' ? 'fp32' : 'q8',
    progress_callback: (progress) => onProgress(normalizeProgress(progress)),
  };
  if (runtime === 'webgpu') options.device = 'webgpu';

  try {
    cachedPipeline = await pipeline(
      'question-answering',
      APP_CONFIG.browserModelId,
      options,
    );
    cachedRuntime = runtime;
    return cachedPipeline;
  } catch (error) {
    if (runtime === 'webgpu') {
      onProgress({ label: 'WebGPU load failed; falling back to WASM.', percent: null });
      cachedPipeline = await pipeline(
        'question-answering',
        APP_CONFIG.browserModelId,
        {
          dtype: 'q8',
          progress_callback: (progress) => onProgress(normalizeProgress(progress)),
        },
      );
      cachedRuntime = 'wasm';
      return cachedPipeline;
    }
    throw error;
  }
}

function validPrediction(prediction) {
  return prediction
    && typeof prediction.answer === 'string'
    && prediction.answer.trim()
    && Number.isFinite(Number(prediction.score))
    && Number.isInteger(Number(prediction.start))
    && Number.isInteger(Number(prediction.end));
}

export async function answerLongDocument({
  question,
  documentText,
  chunkWords = APP_CONFIG.defaultChunkWords,
  overlapWords = APP_CONFIG.defaultOverlapWords,
  candidateChunks = APP_CONFIG.defaultCandidateChunks,
  runtime = 'wasm',
  onModelProgress = () => {},
  onInferenceProgress = () => {},
}) {
  const cleanQuestion = normalizeDocumentText(question);
  const cleanDocument = normalizeDocumentText(documentText);
  if (!cleanQuestion) throw new Error('Enter a focused question.');
  if (!cleanDocument) throw new Error('Upload, select, or paste a readable document.');
  if (cleanDocument.length > APP_CONFIG.maximumDocumentCharacters) {
    throw new Error(`The document exceeds ${APP_CONFIG.maximumDocumentCharacters.toLocaleString()} characters.`);
  }

  const chunks = chunkDocument(cleanDocument, chunkWords, overlapWords);
  const ranked = rankChunks(cleanQuestion, chunks, candidateChunks);
  if (!ranked.length) throw new Error('No candidate document chunks were created.');

  const answerer = await loadBrowserModel({ runtime, onProgress: onModelProgress });
  const started = performance.now();
  const predictions = [];

  for (let index = 0; index < ranked.length; index += 1) {
    const chunk = ranked[index];
    onInferenceProgress({
      completed: index,
      total: ranked.length,
      label: `Running extractive QA on candidate chunk ${index + 1} of ${ranked.length}`,
    });
    const output = await answerer(cleanQuestion, chunk.text);
    const prediction = Array.isArray(output) ? output[0] : output;
    if (!validPrediction(prediction)) continue;
    const localStart = Number(prediction.start);
    const localEnd = Number(prediction.end);
    const globalStart = chunk.start + localStart;
    const globalEnd = chunk.start + localEnd;
    predictions.push({
      answer: prediction.answer.trim(),
      confidenceProxy: Number(prediction.score),
      localStart,
      localEnd,
      globalStart,
      globalEnd,
      chunkId: chunk.id,
      chunkText: chunk.text,
      retrievalScore: chunk.retrievalScore,
    });
  }

  onInferenceProgress({ completed: ranked.length, total: ranked.length, label: 'Answer extraction complete' });
  const latencySeconds = (performance.now() - started) / 1000;
  predictions.sort((a, b) => (
    b.confidenceProxy - a.confidenceProxy
    || b.retrievalScore - a.retrievalScore
    || a.chunkId - b.chunkId
  ));
  const best = predictions[0];
  if (!best) throw new Error('The model did not return a valid answer span. Try a more explicit question.');

  const { paragraph, paragraphs } = locateSupportingParagraph(
    cleanDocument,
    best.globalStart,
    best.globalEnd,
  );
  return {
    answer: best.answer,
    confidenceProxy: best.confidenceProxy,
    supportingParagraph: paragraph?.text ?? best.chunkText,
    highlightedEvidenceHtml: paragraph
      ? highlightEvidence(paragraph, best.globalStart, best.globalEnd)
      : highlightEvidence({ text: best.chunkText, start: best.globalStart - best.localStart }, best.globalStart, best.globalEnd),
    paragraphIndex: paragraph?.id ?? null,
    answerStart: best.globalStart,
    answerEnd: best.globalEnd,
    documentCharacters: cleanDocument.length,
    documentWords: countWords(cleanDocument),
    totalChunks: chunks.length,
    evaluatedChunks: ranked.length,
    latencySeconds,
    runtime: cachedRuntime ?? runtime,
    browserModelId: APP_CONFIG.browserModelId,
    corePythonModelId: APP_CONFIG.pythonModelId,
    candidateResults: predictions.slice(0, 5),
    paragraphCount: paragraphs.length,
    warnings: [
      'This static demo uses a DistilBERT browser deployment baseline over retrieved chunks; it does not execute Longformer.',
      'The confidence value is an uncalibrated model score and does not guarantee correctness.',
    ],
  };
}
