const elements = {
  imageInput: document.querySelector("#imageInput"),
  previewImage: document.querySelector("#previewImage"),
  previewPlaceholder: document.querySelector("#previewPlaceholder"),
  imageMeta: document.querySelector("#imageMeta"),
  questionInput: document.querySelector("#questionInput"),
  askButton: document.querySelector("#askButton"),
  cancelButton: document.querySelector("#cancelButton"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  progressBar: document.querySelector("#progressBar"),
  progressDetail: document.querySelector("#progressDetail"),
  answerOutput: document.querySelector("#answerOutput"),
  confidenceOutput: document.querySelector("#confidenceOutput"),
  questionTypeOutput: document.querySelector("#questionTypeOutput"),
  answerTypeOutput: document.querySelector("#answerTypeOutput"),
  latencyOutput: document.querySelector("#latencyOutput"),
};

let selectedBlob = null;
let worker = null;
let busy = false;

const MAX_MEGAPIXELS = 25;
const SUPPORTED_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

function setStatus(text, detail = "", state = "idle", progress = 0) {
  elements.statusText.textContent = text;
  elements.progressDetail.textContent = detail;
  elements.progressBar.value = Math.max(0, Math.min(100, progress || 0));
  elements.statusDot.dataset.state = state;
}

function normalizeQuestion(value) {
  const question = String(value ?? "").replace(/\s+/g, " ").trim();
  if (!question) throw new Error("Enter a question about the selected image.");
  if (question.length > 300) throw new Error("Use a question with at most 300 characters.");
  return /[?.!]$/.test(question) ? question : `${question}?`;
}

function classifyQuestion(question) {
  const value = question.toLowerCase();
  if (/^(is|are|was|were|do|does|did|can|could|has|have|will|would)\b/.test(value)) return "yes / no";
  if (/^(how many|what number|number of)\b/.test(value)) return "number";
  if (/\bcolou?r\b/.test(value)) return "color";
  if (/\b(where|left|right|above|below|behind|front|next to|between)\b/.test(value)) return "spatial";
  if (/\b(doing|happening|holding|playing|riding|eating)\b/.test(value)) return "action";
  if (/\b(size|shape|kind|type|material|pattern)\b/.test(value)) return "attribute";
  return "object / other";
}

function classifyAnswer(answer) {
  const value = answer.trim().toLowerCase();
  if (value === "yes" || value === "no") return "yes / no";
  if (/^[-+]?\d+(?:\.\d+)?$/.test(value)) return "number";
  return "other";
}

function updateButton() {
  elements.askButton.disabled = busy || !selectedBlob || !elements.questionInput.value.trim();
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
  setStatus("Image ready", "Enter a question and start inference.", "ready", 0);
  updateButton();
}

elements.imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await selectBlob(file, URL.createObjectURL(file), file.name);
  } catch (error) {
    selectedBlob = null;
    setStatus("Image rejected", error.message, "error", 0);
    updateButton();
  }
});

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      const response = await fetch(button.dataset.image);
      const blob = await response.blob();
      elements.questionInput.value = button.dataset.question;
      await selectBlob(blob, button.dataset.image, button.querySelector("span").textContent);
    } catch (error) {
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

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("./model-worker.js", import.meta.url), { type: "module" });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", (event) => {
    finishWithError(event.message || "The model worker failed.");
  });
  return worker;
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (message.type === "progress") {
    const numeric = Number.isFinite(message.progress) ? message.progress : elements.progressBar.value;
    setStatus(message.status || "Loading model", message.detail || "", "loading", numeric);
    return;
  }
  if (message.type === "result") {
    busy = false;
    elements.answerOutput.textContent = message.answer || "No answer generated";
    elements.confidenceOutput.textContent = "N/A";
    elements.questionTypeOutput.textContent = classifyQuestion(message.question);
    elements.answerTypeOutput.textContent = classifyAnswer(message.answer || "");
    elements.latencyOutput.textContent = `${message.latencySeconds.toFixed(2)} s`;
    setStatus("Answer generated", "Review the answer critically; the model can be wrong.", "ready", 100);
    elements.cancelButton.hidden = true;
    updateButton();
    return;
  }
  if (message.type === "error") {
    finishWithError(message.error || "Inference failed.");
  }
}

function finishWithError(message) {
  busy = false;
  elements.cancelButton.hidden = true;
  setStatus("Unable to answer", message, "error", 0);
  updateButton();
}

elements.cancelButton.addEventListener("click", () => {
  if (worker) worker.terminate();
  worker = null;
  busy = false;
  elements.cancelButton.hidden = true;
  setStatus("Cancelled", "Start again when ready.", "idle", 0);
  updateButton();
});

elements.askButton.addEventListener("click", async () => {
  try {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU is unavailable. Open the Space in a current desktop version of Chrome or Edge with WebGPU enabled.");
    }
    const question = normalizeQuestion(elements.questionInput.value);
    if (!selectedBlob) throw new Error("Select an image first.");

    busy = true;
    updateButton();
    elements.cancelButton.hidden = false;
    elements.answerOutput.textContent = "…";
    elements.questionTypeOutput.textContent = classifyQuestion(question);
    elements.answerTypeOutput.textContent = "—";
    elements.latencyOutput.textContent = "—";
    setStatus("Starting model", "The first download can take several minutes.", "loading", 1);

    ensureWorker().postMessage({
      type: "predict",
      question,
      imageBlob: selectedBlob,
    });
  } catch (error) {
    finishWithError(error.message);
  }
});

updateButton();
