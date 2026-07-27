import { canvasWithMaskedPatch } from './image_preprocessing.js';
import { findClassScore } from './inference.js';

export async function computePatchSensitivity({ image, gridSize = 4, targetLabel, baselineScore, classify, onProgress = () => {} }) {
  if (!image || !targetLabel || !Number.isFinite(baselineScore)) throw new Error('A completed baseline prediction is required.');
  const values = [];
  const total = gridSize * gridSize;
  let completed = 0;
  for (let row = 0; row < gridSize; row += 1) {
    const current = [];
    for (let col = 0; col < gridSize; col += 1) {
      const masked = canvasWithMaskedPatch(image, row, col, gridSize);
      const result = await classify(masked);
      const maskedScore = findClassScore(result.predictions, targetLabel);
      current.push(Math.max(0, baselineScore - maskedScore));
      completed += 1;
      onProgress(completed, total);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    values.push(current);
  }
  const maxValue = Math.max(...values.flat(), 1e-9);
  return values.map((row) => row.map((value) => value / maxValue));
}

export function renderSensitivityOverlay(canvas, imageElement, sensitivity) {
  const rect = imageElement.getBoundingClientRect();
  const parentRect = imageElement.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(parentRect.width * dpr);
  canvas.height = Math.round(parentRect.height * dpr);
  canvas.style.width = `${parentRect.width}px`;
  canvas.style.height = `${parentRect.height}px`;
  const context = canvas.getContext('2d');
  context.scale(dpr, dpr);
  context.clearRect(0, 0, parentRect.width, parentRect.height);
  const offsetX = rect.left - parentRect.left;
  const offsetY = rect.top - parentRect.top;
  const rows = sensitivity.length;
  const cols = sensitivity[0].length;
  const patchWidth = rect.width / cols;
  const patchHeight = rect.height / rows;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = sensitivity[row][col];
      context.fillStyle = `rgba(255, 72, 88, ${0.08 + value * 0.62})`;
      context.fillRect(offsetX + col * patchWidth, offsetY + row * patchHeight, patchWidth, patchHeight);
      context.strokeStyle = 'rgba(255,255,255,.2)';
      context.strokeRect(offsetX + col * patchWidth, offsetY + row * patchHeight, patchWidth, patchHeight);
    }
  }
}

export function clearSensitivityOverlay(canvas) {
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
}
