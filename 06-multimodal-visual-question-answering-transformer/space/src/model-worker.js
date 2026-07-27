import {
  AutoProcessor,
  AutoModelForVision2Seq,
  TextStreamer,
  load_image,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct";
const MODEL_LABEL = "SmolVLM-256M-Instruct";
const MAX_NEW_TOKENS = 64;

let processorPromise = null;
let modelPromise = null;

function postProgress(status, detail = "", progress = 0) {
  self.postMessage({ type: "progress", status, detail, progress });
}

function normalizeProgress(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return numeric <= 1 ? numeric * 100 : numeric;
}

function reportDownload(info) {
  const raw = normalizeProgress(info?.progress);
  const progress = raw === null ? 10 : Math.min(90, 8 + raw * 0.82);
  const file = info?.file ? String(info.file).split("/").pop() : "";

  if (info?.status === "done") {
    postProgress("Model component ready", file || "A model component finished loading.", progress);
    return;
  }

  postProgress(
    "Downloading SmolVLM",
    file || "Downloading and caching the browser model.",
    progress,
  );
}

async function ensureWebGPU() {
  if (!("gpu" in navigator)) {
    throw new Error(
      "WebGPU is unavailable. Use a current desktop version of Chrome or Edge.",
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error(
      "No WebGPU adapter was found. Update the browser and graphics driver, then restart the browser.",
    );
  }
}

async function loadRuntime() {
  await ensureWebGPU();

  processorPromise ??= AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: reportDownload,
  }).catch((error) => {
    processorPromise = null;
    throw error;
  });

  // This matches Hugging Face's official SmolVLM WebGPU example. The stable
  // fp32 profile is intentionally used instead of experimental mixed dtypes.
  modelPromise ??= AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
    dtype: "fp32",
    device: "webgpu",
    progress_callback: reportDownload,
  }).catch((error) => {
    modelPromise = null;
    throw error;
  });

  postProgress(
    "Loading vision-language model",
    "The first download can take several minutes. Later runs use the browser cache.",
    10,
  );

  const [processor, model] = await Promise.all([processorPromise, modelPromise]);
  postProgress("Model ready", `${MODEL_LABEL} is ready on WebGPU.`, 93);
  return { processor, model };
}

function cleanAnswer(value) {
  const answer = String(value ?? "")
    .replace(/<\|[^>]+\|>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!answer) {
    throw new Error("The model completed generation but returned an empty answer.");
  }

  return answer;
}

function friendlyFailure(error) {
  const technical = error instanceof Error
    ? `${error.name}: ${error.message}${error.stack ? `\n${error.stack}` : ""}`
    : String(error);
  const lower = technical.toLowerCase();

  if (lower.includes("failed to fetch") || lower.includes("network")) {
    return {
      userMessage: "The model files could not be downloaded. Check the connection, disable restrictive browser extensions, and retry.",
      technical,
    };
  }

  if (
    lower.includes("memory") ||
    lower.includes("allocation") ||
    lower.includes("out of bounds") ||
    lower.includes("device lost")
  ) {
    return {
      userMessage: "The browser or GPU did not have enough available memory. Close other tabs and applications, restart Chrome or Edge, and retry.",
      technical,
    };
  }

  if (
    lower.includes("webgpu") ||
    lower.includes("adapter") ||
    lower.includes("validation") ||
    lower.includes("shader")
  ) {
    return {
      userMessage: "WebGPU could not complete this model run. Update Chrome or Edge and the graphics driver, restart the browser, and retry.",
      technical,
    };
  }

  return {
    userMessage: "The vision-language model could not complete inference. Open the technical details for the exact browser error.",
    technical,
  };
}

self.addEventListener("message", async (event) => {
  if (event.data?.type === "reset") {
    processorPromise = null;
    modelPromise = null;
    self.postMessage({ type: "reset-complete" });
    return;
  }

  if (event.data?.type !== "predict") return;

  const overallStarted = performance.now();

  try {
    const { question, imageDataUrl } = event.data;
    if (!question || !imageDataUrl) {
      throw new Error("Both an image and a question are required.");
    }

    const { processor, model } = await loadRuntime();

    postProgress("Preprocessing image", "Preparing the image and chat prompt.", 95);
    const image = await load_image(imageDataUrl);
    const messages = [
      {
        role: "user",
        content: [
          { type: "image", image: imageDataUrl },
          { type: "text", text: question },
        ],
      },
    ];

    const text = processor.apply_chat_template(messages, {
      add_generation_prompt: true,
    });
    const inputs = await processor(text, [image], {
      do_image_splitting: false,
    });

    let streamedAnswer = "";
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        streamedAnswer += chunk;
      },
    });

    postProgress("Generating answer", "Running SmolVLM with WebGPU.", 98);
    const inferenceStarted = performance.now();

    const generation = await model.generate({
      ...inputs,
      do_sample: false,
      repetition_penalty: 1.1,
      max_new_tokens: MAX_NEW_TOKENS,
      streamer,
      return_dict_in_generate: true,
    });

    let answer = streamedAnswer.trim();
    if (!answer && generation?.sequences) {
      const decoded = processor.batch_decode(generation.sequences, {
        skip_special_tokens: true,
      });
      answer = decoded?.[0] ?? "";
      const questionIndex = answer.toLowerCase().lastIndexOf(question.toLowerCase());
      if (questionIndex >= 0) {
        answer = answer.slice(questionIndex + question.length);
      }
    }

    answer = cleanAnswer(answer);
    const inferenceSeconds = (performance.now() - inferenceStarted) / 1000;
    const totalSeconds = (performance.now() - overallStarted) / 1000;

    self.postMessage({
      type: "result",
      answer,
      question,
      inferenceSeconds,
      totalSeconds,
      model: MODEL_ID,
      backend: "SmolVLM 256M · WebGPU fp32",
      confidenceLabel: "Not calibrated",
    });
  } catch (error) {
    const failure = friendlyFailure(error);
    self.postMessage({
      type: "error",
      error: failure.userMessage,
      technical: failure.technical,
      elapsedSeconds: (performance.now() - overallStarted) / 1000,
    });
  }
});
