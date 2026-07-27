import {
  AutoProcessor,
  AutoTokenizer,
  Moondream1ForConditionalGeneration,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "Xenova/moondream2";
const MODEL_LABEL = "Moondream2 ONNX";

let assetsPromise = null;
let modelPromise = null;
let activeProfile = null;

function postProgress(status, detail = "", progress = 0) {
  self.postMessage({ type: "progress", status, detail, progress });
}

function normalizeProgress(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return numeric <= 1 ? numeric * 100 : numeric;
}

function reportDownload(prefix, info) {
  const raw = normalizeProgress(info?.progress);
  const progress = raw === null ? 10 : Math.min(90, 10 + raw * 0.8);
  const file = info?.file ? String(info.file).split("/").pop() : "";
  const status = info?.status === "done"
    ? `${prefix} component ready`
    : `Loading ${prefix.toLowerCase()}`;
  postProgress(status, file || "Downloading and caching model files.", progress);
}

async function getWebGPUProfiles() {
  if (!("gpu" in navigator)) {
    throw new Error(
      "WebGPU is not available in this browser. Use a current desktop version of Chrome or Edge."
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error(
      "A WebGPU adapter could not be created. Update the browser and graphics driver, then restart the browser."
    );
  }

  const supportsF16 = adapter.features.has("shader-f16");
  const profiles = [];

  if (supportsF16) {
    profiles.push({
      name: "WebGPU fp16/q4",
      dtype: {
        embed_tokens: "fp16",
        vision_encoder: "fp16",
        decoder_model_merged: "q4",
      },
    });
  }

  profiles.push({
    name: "WebGPU compatibility",
    dtype: {
      embed_tokens: "fp32",
      vision_encoder: "q8",
      decoder_model_merged: "q4",
    },
  });

  return profiles;
}

async function loadAssets() {
  if (!assetsPromise) {
    assetsPromise = (async () => {
      postProgress("Loading processor", "Downloading model configuration files.", 4);
      const processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: (info) => reportDownload("Processor", info),
      });

      postProgress("Loading tokenizer", "Preparing the text input pipeline.", 8);
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: (info) => reportDownload("Tokenizer", info),
      });

      return { processor, tokenizer };
    })().catch((error) => {
      assetsPromise = null;
      throw error;
    });
  }

  return assetsPromise;
}

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const { processor, tokenizer } = await loadAssets();
      const profiles = await getWebGPUProfiles();
      const failures = [];

      for (let index = 0; index < profiles.length; index += 1) {
        const profile = profiles[index];
        const attemptText = profiles.length > 1
          ? `Profile ${index + 1} of ${profiles.length}: ${profile.name}`
          : profile.name;

        try {
          postProgress(
            "Loading vision-language model",
            `${attemptText}. The first download is large and may take several minutes.`,
            10
          );

          const model = await Moondream1ForConditionalGeneration.from_pretrained(MODEL_ID, {
            device: "webgpu",
            dtype: profile.dtype,
            progress_callback: (info) => reportDownload("Model", info),
          });

          activeProfile = profile.name;
          postProgress("Model ready", `${MODEL_LABEL} loaded with ${activeProfile}.`, 92);
          return { processor, tokenizer, model, profile: activeProfile };
        } catch (error) {
          failures.push(`${profile.name}: ${error instanceof Error ? error.message : String(error)}`);
          if (index < profiles.length - 1) {
            postProgress(
              "Retrying with compatibility settings",
              "The first WebGPU precision profile was not supported by this device.",
              12
            );
          }
        }
      }

      throw new Error(
        `The browser could not load the model with the supported WebGPU profiles. ${failures.join(" | ")}`
      );
    })().catch((error) => {
      modelPromise = null;
      activeProfile = null;
      throw error;
    });
  }

  return modelPromise;
}

function extractAnswer(decoded) {
  const text = String(decoded ?? "");
  const matches = [...text.matchAll(/Answer:\s*([\s\S]*?)(?:<\|endoftext\|>|$)/gi)];
  let answer = matches.length ? matches[matches.length - 1][1] : text;

  answer = answer
    .replaceAll("<|endoftext|>", "")
    .replace(/^\s*Answer:\s*/i, "")
    .trim();

  if (!answer) {
    throw new Error("The model completed generation but returned an empty answer.");
  }

  return answer;
}

function friendlyFailure(error) {
  const technical = error instanceof Error ? error.message : String(error);
  const lower = technical.toLowerCase();

  if (lower.includes("failed to fetch") || lower.includes("network")) {
    return {
      userMessage: "The model files could not be downloaded. Check the internet connection, disable restrictive extensions, and retry.",
      technical,
    };
  }

  if (lower.includes("memory") || lower.includes("allocation") || lower.includes("out of bounds")) {
    return {
      userMessage: "The browser did not have enough GPU or system memory for this model. Close other tabs and applications, restart Chrome or Edge, and retry.",
      technical,
    };
  }

  if (lower.includes("webgpu") || lower.includes("adapter") || lower.includes("shader")) {
    return {
      userMessage: "WebGPU could not initialize the model. Update Chrome or Edge and the graphics driver, then restart the browser.",
      technical,
    };
  }

  return {
    userMessage: "The model could not complete inference. Use the retry button, then check the technical details if the issue continues.",
    technical,
  };
}

self.addEventListener("message", async (event) => {
  if (event.data?.type === "reset") {
    modelPromise = null;
    assetsPromise = null;
    activeProfile = null;
    self.postMessage({ type: "reset-complete" });
    return;
  }

  if (event.data?.type !== "predict") return;

  const overallStarted = performance.now();

  try {
    const { question, imageBlob } = event.data;
    const { processor, tokenizer, model, profile } = await loadModel();

    postProgress("Preprocessing image", "Converting the upload into model-ready RGB pixels.", 94);
    const image = (await RawImage.fromBlob(imageBlob)).rgb();
    const visionInputs = await processor(image);

    const prompt = `<image>\n\nQuestion: ${question}\n\nAnswer:`;
    const textInputs = tokenizer(prompt);

    postProgress("Generating answer", "Running the vision-language Transformer with WebGPU.", 97);
    const inferenceStarted = performance.now();
    const output = await model.generate({
      ...textInputs,
      ...visionInputs,
      do_sample: false,
      max_new_tokens: 64,
    });

    const decoded = tokenizer.batch_decode(output, { skip_special_tokens: false })[0];
    const inferenceSeconds = (performance.now() - inferenceStarted) / 1000;
    const totalSeconds = (performance.now() - overallStarted) / 1000;
    const answer = extractAnswer(decoded);

    self.postMessage({
      type: "result",
      answer,
      question,
      inferenceSeconds,
      totalSeconds,
      model: MODEL_ID,
      backend: profile,
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
