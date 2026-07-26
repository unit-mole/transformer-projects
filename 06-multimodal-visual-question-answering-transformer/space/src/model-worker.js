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
let modelPromise = null;

function postProgress(status, detail = "", progress = 0) {
  self.postMessage({ type: "progress", status, detail, progress });
}

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      postProgress("Loading processor", "Downloading small configuration files.", 4);
      const processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: (info) => reportDownload("Processor", info),
      });

      postProgress("Loading tokenizer", "Preparing the text input pipeline.", 8);
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: (info) => reportDownload("Tokenizer", info),
      });

      postProgress("Loading vision-language model", "Large quantized ONNX files are downloading.", 10);
      const model = await Moondream1ForConditionalGeneration.from_pretrained(MODEL_ID, {
        device: "webgpu",
        dtype: {
          embed_tokens: "int8",
          vision_encoder: "q4",
          decoder_model_merged: "q4f16",
        },
        progress_callback: (info) => reportDownload("Model", info),
      });
      postProgress("Model ready", "Running image-question inference.", 92);
      return { processor, tokenizer, model };
    })();
  }
  return modelPromise;
}

function reportDownload(prefix, info) {
  const raw = Number(info?.progress);
  const progress = Number.isFinite(raw) ? 10 + raw * 0.8 : 10;
  const file = info?.file ? String(info.file).split("/").pop() : "";
  const status = info?.status === "done" ? `${prefix} component ready` : `Loading ${prefix.toLowerCase()}`;
  postProgress(status, file || "Downloading and caching model files.", progress);
}

function extractAnswer(decoded, question) {
  const marker = "Answer:";
  const index = decoded.lastIndexOf(marker);
  let answer = index >= 0 ? decoded.slice(index + marker.length) : decoded;
  answer = answer.replaceAll("<|endoftext|>", "").trim();
  if (answer.startsWith(question)) answer = answer.slice(question.length).trim();
  return answer || "No answer generated";
}

self.addEventListener("message", async (event) => {
  if (event.data?.type !== "predict") return;
  try {
    const { question, imageBlob } = event.data;
    const { processor, tokenizer, model } = await loadModel();

    postProgress("Preprocessing image", "Converting the upload into model-ready pixels.", 94);
    const image = await RawImage.fromBlob(imageBlob);
    const visionInputs = await processor(image);

    const prompt = `<image>\n\nQuestion: ${question}\n\nAnswer:`;
    const textInputs = tokenizer(prompt);

    postProgress("Generating answer", "Running the vision-language Transformer with WebGPU.", 97);
    const started = performance.now();
    const output = await model.generate({
      ...textInputs,
      ...visionInputs,
      do_sample: false,
      max_new_tokens: 48,
    });
    const decoded = tokenizer.batch_decode(output, { skip_special_tokens: false })[0];
    const latencySeconds = (performance.now() - started) / 1000;
    const answer = extractAnswer(decoded, question);

    self.postMessage({ type: "result", answer, question, latencySeconds });
  } catch (error) {
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
