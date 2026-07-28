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
  evaluationRunButton: document.querySelector("#evaluationRunButton"),
  evaluationCancelButton: document.querySelector("#evaluationCancelButton"),
  evaluationDownloadButton: document.querySelector("#evaluationDownloadButton"),
  evaluationStatus: document.querySelector("#evaluationStatus"),
  evaluationProgress: document.querySelector("#evaluationProgress"),
  evaluationDetail: document.querySelector("#evaluationDetail"),
  evaluationAccuracy: document.querySelector("#evaluationAccuracy"),
  evaluationFailureRate: document.querySelector("#evaluationFailureRate"),
  evaluationAverageLatency: document.querySelector("#evaluationAverageLatency"),
  evaluationLatencyRange: document.querySelector("#evaluationLatencyRange"),
  evaluationTableBody: document.querySelector("#evaluationTableBody"),
  evaluationFailures: document.querySelector("#evaluationFailures"),
};

let selectedBlob = null;
let worker = null;
let busy = false;
let runStartedAt = null;
let lastQuestion = "";
let requestCounter = 0;
let evaluationBusy = false;
let evaluationCancelled = false;
let latestEvaluationReport = null;
const pendingRequests = new Map();

const MAX_MEGAPIXELS = 25;
const SUPPORTED_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const NUMBER_WORDS = new Map([
  ["zero", "0"], ["one", "1"], ["two", "2"], ["three", "3"], ["four", "4"],
  ["five", "5"], ["six", "6"], ["seven", "7"], ["eight", "8"], ["nine", "9"],
  ["ten", "10"],
]);

function setStatus(text, detail = "", state = "idle", progress = 0) {
  elements.statusText.textContent = text;
  elements.progressDetail.textContent = detail;
  elements.progressBar.value = Math.max(0, Math.min(100, progress || 0));
  elements.statusDot.dataset.state = state;
}

function setInitialResults() {
  elements.answerOutput.textContent = "No prediction yet";
  elements.confidenceOutput.textContent = "Not available for this generative model";
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
  if (/\b(doing|happening|holding|playing|riding|eating|flying|driving|sailing|kicking|waving)\b/.test(value)) return "Action / scene";
  if (/\b(size|shape|kind|type|material|pattern)\b/.test(value)) return "Attribute";
  return "Object / other";
}

function classifyAnswer(answer) {
  const value = answer.trim().toLowerCase().replace(/[.!?]+$/, "");
  if (value === "yes" || value === "no") return "Yes / no";
  if (/^[-+]?\d+(?:\.\d+)?$/.test(value) || NUMBER_WORDS.has(value)) return "Number";
  return "Open-ended text";
}

function updateButton() {
  elements.askButton.disabled = busy || evaluationBusy || !selectedBlob || !elements.questionInput.value.trim();
  elements.evaluationRunButton.disabled = busy || evaluationBusy;
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
  if (!SUPPORTED_TYPES.has(blob.type)) throw new Error("Use a PNG, JPEG, or WebP image.");

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
  } catch {
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

function rejectPendingRequests(reason) {
  for (const { reject } of pendingRequests.values()) reject(new Error(reason));
  pendingRequests.clear();
}

function destroyWorker(reason = "The model worker was reset.") {
  if (worker) worker.terminate();
  worker = null;
  rejectPendingRequests(reason);
}

function ensureWorker() {
  if (worker) return worker;

  const workerUrl = new URL("./model-worker.js?v=4.0.0", import.meta.url);
  worker = new Worker(workerUrl, { type: "module" });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", (event) => {
    destroyWorker(event.message || "The browser model worker stopped unexpectedly.");
    if (!evaluationBusy) {
      finishWithError("The browser model worker stopped unexpectedly.", event.message || "Unknown worker error.");
    }
  });
  return worker;
}

function handleWorkerMessage(event) {
  const message = event.data;

  if (message.type === "progress") {
    const numeric = Number.isFinite(message.progress) ? message.progress : 0;
    if (message.mode === "evaluation") {
      elements.evaluationStatus.textContent = message.status || "Running evaluation";
      elements.evaluationDetail.textContent = message.detail || "The model is processing the evaluation suite.";
    } else {
      setStatus(message.status || "Loading model", message.detail || "", "loading", numeric);
      elements.modelStatus.textContent = message.status || "Loading";
      elements.modelStatus.dataset.state = "loading";
    }
    return;
  }

  const pending = pendingRequests.get(message.requestId);
  if (!pending) return;
  pendingRequests.delete(message.requestId);

  if (message.type === "result") {
    pending.resolve(message);
  } else if (message.type === "error") {
    const error = new Error(message.error || "Inference failed.");
    error.technical = message.technical || "";
    error.elapsedSeconds = message.elapsedSeconds;
    pending.reject(error);
  }
}

function runPrediction(question, imageDataUrl, mode = "interactive") {
  const requestId = `${Date.now()}-${requestCounter += 1}`;
  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject, mode });
    ensureWorker().postMessage({ type: "predict", requestId, mode, question, imageDataUrl });
  });
}

function formatConfidence(confidence) {
  if (!confidence?.available || !Number.isFinite(confidence.percentage)) {
    return "Not available for this generative model";
  }
  return `${confidence.percentage.toFixed(1)}% token-likelihood proxy`;
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
  elements.confidenceOutput.textContent = "Not available because inference failed";
  elements.questionTypeOutput.textContent = lastQuestion ? classifyQuestion(lastQuestion) : "Not available";
  elements.answerTypeOutput.textContent = "Not generated";
  elements.latencyOutput.textContent = measured === null ? "Run failed" : `Failed after ${measured.toFixed(2)} s`;
  elements.modelStatus.textContent = "Model needs retry";
  elements.modelStatus.dataset.state = "error";
  showError(message, technical);
  setStatus("Unable to answer", message, "error", 0);
  destroyWorker("Inference failed and the worker was reset.");
  runStartedAt = null;
  updateButton();
}

elements.cancelButton.addEventListener("click", () => {
  destroyWorker("Generation was cancelled by the user.");
  busy = false;
  runStartedAt = null;
  elements.cancelButton.hidden = true;
  elements.answerOutput.textContent = "Generation cancelled";
  elements.confidenceOutput.textContent = "Not available because generation was cancelled";
  elements.answerTypeOutput.textContent = "Not generated";
  elements.latencyOutput.textContent = "Cancelled";
  elements.modelStatus.textContent = "Reset required";
  setStatus("Cancelled", "Start again when ready.", "idle", 0);
  updateButton();
});

elements.retryButton.addEventListener("click", () => {
  destroyWorker("The model was reset for a fresh retry.");
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
    if (!("gpu" in navigator)) throw new Error("WebGPU is unavailable. Open the app in a current desktop version of Chrome or Edge.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("The browser could not create a WebGPU adapter. Update the browser and graphics driver, then restart Chrome or Edge.");

    busy = true;
    runStartedAt = performance.now();
    lastQuestion = question;
    hideError();
    updateButton();
    elements.cancelButton.hidden = false;
    elements.answerOutput.textContent = "Generating answer…";
    elements.confidenceOutput.textContent = "Calculating generation confidence proxy…";
    elements.questionTypeOutput.textContent = classifyQuestion(question);
    elements.answerTypeOutput.textContent = "Pending";
    elements.latencyOutput.textContent = "Running";
    elements.modelStatus.textContent = "Starting model";
    elements.modelStatus.dataset.state = "loading";
    setStatus("Starting model", "The first download can take several minutes.", "loading", 1);

    const imageDataUrl = await blobToDataUrl(selectedBlob);
    const message = await runPrediction(question, imageDataUrl, "interactive");

    busy = false;
    runStartedAt = null;
    hideError();
    elements.answerOutput.textContent = message.answer;
    elements.confidenceOutput.textContent = formatConfidence(message.confidence);
    elements.questionTypeOutput.textContent = classifyQuestion(message.question);
    elements.answerTypeOutput.textContent = classifyAnswer(message.answer);
    elements.latencyOutput.textContent = `${message.inferenceSeconds.toFixed(2)} s inference`;
    elements.modelStatus.textContent = message.backend || "WebGPU ready";
    elements.modelStatus.dataset.state = "ready";
    setStatus(
      "Answer generated",
      `Total request time: ${message.totalSeconds.toFixed(2)} seconds. Review the answer critically.`,
      "ready",
      100,
    );
    elements.cancelButton.hidden = true;
    updateButton();
  } catch (error) {
    finishWithError(error.message, error.technical || error.stack || "", error.elapsedSeconds);
  }
});

function normalizeEvaluationAnswer(value) {
  const cleaned = String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\b(a|an|the)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const tokens = cleaned.split(" ").map((token) => NUMBER_WORDS.get(token) ?? token);
  return tokens.join(" ");
}

function scoreEvaluationPrediction(prediction, acceptedAnswers) {
  const normalizedPrediction = normalizeEvaluationAnswer(prediction);
  const predictionTokens = normalizedPrediction.split(" ").filter(Boolean);

  for (const accepted of acceptedAnswers) {
    const normalizedAccepted = normalizeEvaluationAnswer(accepted);
    if (normalizedPrediction === normalizedAccepted) return true;
    if (predictionTokens.length <= 8 && normalizedPrediction.includes(normalizedAccepted)) return true;
  }
  return false;
}

function summarizeEvaluation(rows, expectedTotal) {
  const categories = ["color", "object", "counting", "yes_no", "action_scene", "spatial"];
  const successful = rows.filter((row) => !row.error);
  const correct = rows.filter((row) => row.correct).length;
  const failures = rows.filter((row) => row.error).length;
  const latencies = successful.map((row) => row.inferenceSeconds).filter(Number.isFinite);

  const byCategory = categories.map((category) => {
    const group = rows.filter((row) => row.category === category);
    const categoryCorrect = group.filter((row) => row.correct).length;
    const categoryFailures = group.filter((row) => row.error).length;
    return {
      category,
      total: group.length,
      correct: categoryCorrect,
      failures: categoryFailures,
      accuracy: group.length ? categoryCorrect / group.length : null,
    };
  });

  return {
    expectedTotal,
    completed: rows.length,
    correct,
    failures,
    accuracy: expectedTotal ? correct / expectedTotal : null,
    failureRate: expectedTotal ? failures / expectedTotal : null,
    averageLatency: latencies.length ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length : null,
    minimumLatency: latencies.length ? Math.min(...latencies) : null,
    maximumLatency: latencies.length ? Math.max(...latencies) : null,
    byCategory,
  };
}

function renderEvaluationSummary(summary, rows) {
  elements.evaluationAccuracy.textContent = summary.accuracy === null ? "Not available" : `${(summary.accuracy * 100).toFixed(1)}%`;
  elements.evaluationFailureRate.textContent = summary.failureRate === null ? "Not available" : `${(summary.failureRate * 100).toFixed(1)}%`;
  elements.evaluationAverageLatency.textContent = summary.averageLatency === null ? "Not available" : `${summary.averageLatency.toFixed(2)} s`;
  elements.evaluationLatencyRange.textContent = summary.minimumLatency === null
    ? "Not available"
    : `${summary.minimumLatency.toFixed(2)}–${summary.maximumLatency.toFixed(2)} s`;

  elements.evaluationTableBody.innerHTML = summary.byCategory.map((row) => `
    <tr>
      <td>${row.category.replace("_", " / ")}</td>
      <td>${row.total}</td>
      <td>${row.correct}</td>
      <td>${row.failures}</td>
      <td>${row.accuracy === null ? "—" : `${(row.accuracy * 100).toFixed(1)}%`}</td>
    </tr>
  `).join("");

  const failures = rows.filter((row) => row.error || !row.correct).slice(0, 10);
  elements.evaluationFailures.innerHTML = failures.length
    ? failures.map((row) => `
      <li>
        <strong>${row.id}</strong> — ${row.question}<br />
        <span>Expected: ${row.acceptedAnswers.join(" / ")} · Predicted: ${row.prediction || "No answer"}${row.error ? ` · Error: ${row.error}` : ""}</span>
      </li>
    `).join("")
    : "<li>No failed examples were recorded.</li>";
}

async function loadEvaluationDataset() {
  const response = await fetch("./evaluation/vqa_evaluation_60.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Evaluation dataset returned HTTP ${response.status}.`);
  const records = await response.json();
  if (!Array.isArray(records) || records.length !== 60) throw new Error("The evaluation suite must contain exactly 60 records.");
  return records;
}

async function runEvaluation() {
  if (evaluationBusy || busy) return;
  evaluationBusy = true;
  evaluationCancelled = false;
  latestEvaluationReport = null;
  elements.evaluationDownloadButton.disabled = true;
  elements.evaluationCancelButton.hidden = false;
  elements.evaluationProgress.value = 0;
  elements.evaluationStatus.textContent = "Loading 60-pair evaluation suite";
  elements.evaluationDetail.textContent = "The model is evaluated in the browser using the same WebGPU pipeline as the live demo.";
  elements.modelStatus.textContent = "Evaluation running";
  elements.modelStatus.dataset.state = "loading";
  updateButton();

  const rows = [];
  try {
    const records = await loadEvaluationDataset();
    for (let index = 0; index < records.length; index += 1) {
      if (evaluationCancelled) break;
      const record = records[index];
      elements.evaluationStatus.textContent = `Evaluating ${index + 1} of ${records.length}`;
      elements.evaluationDetail.textContent = `${record.category.replace("_", " / ")} · ${record.question}`;
      elements.evaluationProgress.value = (index / records.length) * 100;

      try {
        const response = await fetch(record.image);
        if (!response.ok) throw new Error(`Image returned HTTP ${response.status}.`);
        const imageDataUrl = await blobToDataUrl(await response.blob());
        const result = await runPrediction(normalizeQuestion(record.question), imageDataUrl, "evaluation");
        rows.push({
          id: record.id,
          category: record.category,
          image: record.image,
          question: record.question,
          acceptedAnswers: record.accepted_answers,
          prediction: result.answer,
          correct: scoreEvaluationPrediction(result.answer, record.accepted_answers),
          inferenceSeconds: result.inferenceSeconds,
          totalSeconds: result.totalSeconds,
          confidence: result.confidence,
          error: null,
        });
      } catch (error) {
        if (evaluationCancelled) break;
        rows.push({
          id: record.id,
          category: record.category,
          image: record.image,
          question: record.question,
          acceptedAnswers: record.accepted_answers,
          prediction: "",
          correct: false,
          inferenceSeconds: null,
          totalSeconds: null,
          confidence: null,
          error: error.message,
        });
      }
      elements.evaluationProgress.value = ((index + 1) / records.length) * 100;
    }

    if (evaluationCancelled) {
      elements.evaluationStatus.textContent = "Evaluation cancelled";
      elements.evaluationDetail.textContent = `${rows.length} of 60 records completed. Run again for a full comparable result.`;
    } else {
      elements.evaluationStatus.textContent = "Evaluation complete";
      elements.evaluationDetail.textContent = "Results are based on 60 synthetic portfolio questions, not an official VQA v2 benchmark.";
    }

    const summary = summarizeEvaluation(rows, 60);
    latestEvaluationReport = {
      generatedAt: new Date().toISOString(),
      model: "HuggingFaceTB/SmolVLM-256M-Instruct",
      backend: "Transformers.js WebGPU fp32",
      dataset: "60-pair synthetic portfolio VQA evaluation suite",
      scoring: "Normalized accepted-answer matching with short-answer containment",
      completed: !evaluationCancelled,
      summary,
      rows,
      disclaimer: "Generation confidence proxies are token-likelihood diagnostics, not calibrated probabilities of factual correctness.",
    };
    renderEvaluationSummary(summary, rows);
    elements.evaluationDownloadButton.disabled = rows.length === 0;
    elements.modelStatus.textContent = evaluationCancelled ? "Evaluation stopped" : "Evaluation complete";
    elements.modelStatus.dataset.state = evaluationCancelled ? "error" : "ready";
  } catch (error) {
    elements.evaluationStatus.textContent = "Evaluation could not start";
    elements.evaluationDetail.textContent = error.message;
    showError("The browser evaluation could not be completed.", error.stack || error.message);
    elements.modelStatus.textContent = "Evaluation failed";
    elements.modelStatus.dataset.state = "error";
  } finally {
    evaluationBusy = false;
    elements.evaluationCancelButton.hidden = true;
    updateButton();
  }
}

elements.evaluationRunButton.addEventListener("click", runEvaluation);

elements.evaluationCancelButton.addEventListener("click", () => {
  evaluationCancelled = true;
  destroyWorker("Evaluation was cancelled by the user.");
  elements.evaluationStatus.textContent = "Stopping evaluation";
  elements.evaluationDetail.textContent = "The current model worker has been terminated.";
});

elements.evaluationDownloadButton.addEventListener("click", () => {
  if (!latestEvaluationReport) return;
  const blob = new Blob([JSON.stringify(latestEvaluationReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `project-06-smolvlm-evaluation-${new Date().toISOString().slice(0, 10)}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
});

setInitialResults();
checkBrowser();
updateButton();
