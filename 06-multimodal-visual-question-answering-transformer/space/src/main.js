const elements = {
  imageInput: document.querySelector("#imageInput"),
  previewImage: document.querySelector("#previewImage"),
  previewPlaceholder: document.querySelector("#previewPlaceholder"),
  imageMeta: document.querySelector("#imageMeta"),
  questionInput: document.querySelector("#questionInput"),
  askButton: document.querySelector("#askButton"),
  cancelButton: document.querySelector("#cancelButton"),
  retryButton: document.querySelector("#retryButton"),
  copyErrorButton: document.querySelector("#copyErrorButton"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  progressBar: document.querySelector("#progressBar"),
  progressDetail: document.querySelector("#progressDetail"),
  browserStatus: document.querySelector("#browserStatus"),
  modelStatus: document.querySelector("#modelStatus"),
  errorPanel: document.querySelector("#errorPanel"),
  errorMessage: document.querySelector("#errorMessage"),
  errorTechnical: document.querySelector("#errorTechnical"),
  answerOutput: document.querySelector("#answerOutput"),
  confidenceOutput: document.querySelector("#confidenceOutput"),
  questionTypeOutput: document.querySelector("#questionTypeOutput"),
  answerTypeOutput: document.querySelector("#answerTypeOutput"),
  latencyOutput: document.querySelector("#latencyOutput"),
};

let selectedBlob = null;
let worker = null;
let busy = false;
let runStartedAt = null;
let lastQuestion = "";

const MAX_MEGAPIXELS = 25;
const SUPPORTED_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

function setStatus(text, detail = "", state = "idle", progress = 0) {
  elements.statusText.textContent = text;
  elements.progressDetail.textContent = detail;
  elements.progressBar.value = Math.max(0, Math.min(100, progress || 0));
  elements.statusDot.dataset.state = state;
}

function setInitialResults() {
  elements.answerOutput.textContent = "No prediction yet";
  elements.confidenceOutput.textContent = "Not generated yet";
  elements.questionTypeOutput.textContent = "Detected after submission";
  elements.answerTypeOutput.textContent = "Detected after generation";
  elements.latencyOutput.textContent = "Measured after generation";
}

function hideError() {
  elements.errorPanel.hidden = true;
  elements.errorMessage.textContent = "";
  elements.errorTechnical.textContent = "";
}

function showError(message, technical = "") {
  elements.errorMessage.textContent = message;
  elements.errorTechnical.textContent = technical || "No additional technical details were returned.";
  elements.errorPanel.hidden = false;
}

function normalizeQuestion(value) {
  const question = String(value ?? "").replace(/\s+/g, " ").trim();
  if (!question) throw new Error("Enter a question about the selected image.");
  if (question.length > 300) throw new Error("Use a question with at most 300 characters.");
  return /[?.!]$/.test(question) ? question : `${question}?`;
}

function classifyQuestion(question) {
  const value = question.toLowerCase();
  if (/^(is|are|was|were|do|does|did|can|could|has|have|will|would)\b/.test(value)) return "Yes / no";
  if (/^(how many|what number|number of)\b/.test(value)) return "Number / counting";
  if (/\bcolou?r\b/.test(value)) return "Color";
  if (/\b(where|left|right|above|below|behind|front|next to|between)\b/.test(value)) return "Spatial";
  if (/\b(doing|happening|holding|playing|riding|eating)\b/.test(value)) return "Action";
  if (/\b(size|shape|kind|type|material|pattern)\b/.test(value)) return "Attribute";
  return "Object / other";
}

function classifyAnswer(answer) {
  const value = answer.trim().toLowerCase();
  if (value === "yes" || value === "no") return "Yes / no";
  if (/^[-+]?\d+(?:\.\d+)?$/.test(value)) return "Number";
  return "Open-ended text";
}

function updateButton() {
  elements.askButton.disabled = busy || !selectedBlob || !elements.questionInput.value.trim();
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error || new Error("The image could not be read."));
    reader.readAsDataURL(blob);
  });
}

async function validateImageBlob(blob) {
  if (!SUPPORTED_TYPES.has(blob.type)) {
    throw new Error("Use a PNG, JPEG, or WebP image.");
  }

  const bitmap = await createImageBitmap(blob);
  const megapixels = (bitmap.width * bitmap.height) / 1_000_000;
  const result = { width: bitmap.width, height: bitmap.height, megapixels };
  bitmap.close();

  if (megapixels > MAX_MEGAPIXELS) {
    throw new Error(`Image is ${megapixels.toFixed(1)} MP; maximum supported size is ${MAX_MEGAPIXELS} MP.`);
  }

  return result;
}

async function selectBlob(blob, previewUrl, filename = "sample image") {
  const meta = await validateImageBlob(blob);
  selectedBlob = blob;
  elements.previewImage.src = previewUrl;
  elements.previewImage.hidden = false;
  elements.previewPlaceholder.hidden = true;
  elements.imageMeta.textContent = `${filename} · ${meta.width} × ${meta.height}`;
  hideError();
  setStatus("Image ready", "Enter a question and start inference.", "ready", 0);
  updateButton();
}

async function checkBrowser() {
  if (!("gpu" in navigator)) {
    elements.browserStatus.textContent = "WebGPU unavailable";
    elements.browserStatus.dataset.state = "error";
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      elements.browserStatus.textContent = "No WebGPU adapter";
      elements.browserStatus.dataset.state = "error";
      return;
    }

    elements.browserStatus.textContent = "WebGPU ready · stable fp32 mode";
    elements.browserStatus.dataset.state = "ready";
  } catch (error) {
    elements.browserStatus.textContent = "WebGPU check failed";
    elements.browserStatus.dataset.state = "error";
  }
}

elements.imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    await selectBlob(file, URL.createObjectURL(file), file.name);
  } catch (error) {
    selectedBlob = null;
    showError(error.message);
    setStatus("Image rejected", error.message, "error", 0);
    updateButton();
  }
});

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      const response = await fetch(button.dataset.image);
      if (!response.ok) throw new Error(`Sample image returned HTTP ${response.status}.`);
      const blob = await response.blob();
      elements.questionInput.value = button.dataset.question;
      await selectBlob(blob, button.dataset.image, button.querySelector("span").textContent);
    } catch (error) {
      showError("The selected sample image could not be loaded.", error.message);
      setStatus("Could not load sample", error.message, "error", 0);
    }
  });
});

document.querySelectorAll("[data-question]:not(.sample)").forEach((button) => {
  button.addEventListener("click", () => {
    elements.questionInput.value = button.dataset.question;
    updateButton();
  });
});

elements.questionInput.addEventListener("input", updateButton);

function destroyWorker() {
  if (worker) worker.terminate();
  worker = null;
}

function ensureWorker() {
  if (worker) return worker;

  const workerUrl = new URL("./model-worker.js?v=3.0.0", import.meta.url);
  worker = new Worker(workerUrl, { type: "module" });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", (event) => {
    finishWithError(
      "The browser model worker stopped unexpectedly.",
      event.message || "Unknown worker error."
    );
  });
  return worker;
}

function handleWorkerMessage(event) {
  const message = event.data;

  if (message.type === "progress") {
    const numeric = Number.isFinite(message.progress) ? message.progress : elements.progressBar.value;
    setStatus(message.status || "Loading model", message.detail || "", "loading", numeric);
    elements.modelStatus.textContent = message.status || "Loading";
    elements.modelStatus.dataset.state = "loading";
    return;
  }

  if (message.type === "result") {
    busy = false;
    runStartedAt = null;
    hideError();
    elements.answerOutput.textContent = message.answer;
    elements.confidenceOutput.textContent = message.confidenceLabel || "Not calibrated";
    elements.questionTypeOutput.textContent = classifyQuestion(message.question);
    elements.answerTypeOutput.textContent = classifyAnswer(message.answer);
    elements.latencyOutput.textContent = `${message.inferenceSeconds.toFixed(2)} s inference`;
    elements.modelStatus.textContent = message.backend || "WebGPU ready";
    elements.modelStatus.dataset.state = "ready";
    setStatus(
      "Answer generated",
      `Total request time: ${message.totalSeconds.toFixed(2)} seconds. Review the answer critically.`,
      "ready",
      100
    );
    elements.cancelButton.hidden = true;
    updateButton();
    return;
  }

  if (message.type === "error") {
    finishWithError(message.error || "Inference failed.", message.technical || "", message.elapsedSeconds);
  }
}

function finishWithError(message, technical = "", elapsedSeconds = null) {
  busy = false;
  const measured = Number.isFinite(elapsedSeconds)
    ? elapsedSeconds
    : runStartedAt
      ? (performance.now() - runStartedAt) / 1000
      : null;

  elements.cancelButton.hidden = true;
  elements.answerOutput.textContent = "No answer generated";
  elements.confidenceOutput.textContent = "Unavailable because inference failed";
  elements.questionTypeOutput.textContent = lastQuestion ? classifyQuestion(lastQuestion) : "Not available";
  elements.answerTypeOutput.textContent = "Not generated";
  elements.latencyOutput.textContent = measured === null ? "Run failed" : `Failed after ${measured.toFixed(2)} s`;
  elements.modelStatus.textContent = "Model needs retry";
  elements.modelStatus.dataset.state = "error";
  showError(message, technical);
  setStatus("Unable to answer", message, "error", 0);
  destroyWorker();
  runStartedAt = null;
  updateButton();
}

elements.cancelButton.addEventListener("click", () => {
  destroyWorker();
  busy = false;
  runStartedAt = null;
  elements.cancelButton.hidden = true;
  elements.answerOutput.textContent = "Generation cancelled";
  elements.confidenceOutput.textContent = "Not generated";
  elements.answerTypeOutput.textContent = "Not generated";
  elements.latencyOutput.textContent = "Cancelled";
  elements.modelStatus.textContent = "Reset required";
  setStatus("Cancelled", "Start again when ready.", "idle", 0);
  updateButton();
});

elements.retryButton.addEventListener("click", () => {
  destroyWorker();
  hideError();
  setStatus("Ready to retry", "The browser will create a fresh model worker.", "ready", 0);
  elements.modelStatus.textContent = "Ready for a fresh load";
  elements.modelStatus.dataset.state = "ready";
  updateButton();
});

elements.copyErrorButton.addEventListener("click", async () => {
  const text = `${elements.errorMessage.textContent}\n\n${elements.errorTechnical.textContent}`;
  try {
    await navigator.clipboard.writeText(text);
    elements.copyErrorButton.textContent = "Copied";
    setTimeout(() => { elements.copyErrorButton.textContent = "Copy error details"; }, 1500);
  } catch {
    elements.copyErrorButton.textContent = "Copy failed";
  }
});

elements.askButton.addEventListener("click", async () => {
  try {
    const question = normalizeQuestion(elements.questionInput.value);
    if (!selectedBlob) throw new Error("Select an image first.");

    if (!("gpu" in navigator)) {
      throw new Error("WebGPU is unavailable. Open the app in a current desktop version of Chrome or Edge.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("The browser could not create a WebGPU adapter. Update the browser and graphics driver, then restart Chrome or Edge.");
    }

    busy = true;
    runStartedAt = performance.now();
    lastQuestion = question;
    hideError();
    updateButton();
    elements.cancelButton.hidden = false;
    elements.answerOutput.textContent = "Generating answer…";
    elements.confidenceOutput.textContent = "Pending model completion";
    elements.questionTypeOutput.textContent = classifyQuestion(question);
    elements.answerTypeOutput.textContent = "Pending";
    elements.latencyOutput.textContent = "Running";
    elements.modelStatus.textContent = "Starting model";
    elements.modelStatus.dataset.state = "loading";
    setStatus("Starting model", "The first download can take several minutes.", "loading", 1);

    const imageDataUrl = await blobToDataUrl(selectedBlob);
    ensureWorker().postMessage({
      type: "predict",
      question,
      imageDataUrl,
    });
  } catch (error) {
    finishWithError(error.message, error.stack || "");
  }
});

setInitialResults();
checkBrowser();
updateButton();
