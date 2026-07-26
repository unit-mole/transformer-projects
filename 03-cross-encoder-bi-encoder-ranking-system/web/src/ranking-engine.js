import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  pipeline,
} from "@huggingface/transformers";

import { MODEL_IDS } from "./constants.js";

env.allowLocalModels = false;
env.useBrowserCache = true;

function dotProduct(left, right) {
  let total = 0;
  const length = Math.min(left.length, right.length);

  for (let index = 0; index < length; index += 1) {
    total += left[index] * right[index];
  }

  return total;
}

export class BrowserRankingEngine {
  constructor(documents, onProgress = () => {}) {
    this.documents = documents;
    this.onProgress = onProgress;
    this.embedder = null;
    this.documentEmbeddings = null;
    this.rerankerTokenizer = null;
    this.rerankerModel = null;
  }

  setProgressHandler(handler) {
    this.onProgress = handler ?? (() => {});
  }

  async ensureBiEncoder() {
    let setupMs = 0;

    if (!this.embedder) {
      const started = performance.now();
      this.embedder = await pipeline(
        "feature-extraction",
        MODEL_IDS.biEncoder,
        {
          dtype: "q8",
          progress_callback: (event) =>
            this.onProgress(event, "Loading MiniLM bi-encoder"),
        },
      );
      setupMs += performance.now() - started;
    }

    if (!this.documentEmbeddings) {
      const started = performance.now();
      this.onProgress(
        {
          status: "encoding_documents",
          progress: 0.68,
          file: `${this.documents.length} documents`,
        },
        "Building document embeddings",
      );

      const output = await this.embedder(
        this.documents.map((document) => document.search_text),
        {
          pooling: "mean",
          normalize: true,
        },
      );

      this.documentEmbeddings = output.tolist();
      setupMs += performance.now() - started;
    }

    return setupMs;
  }

  async ensureCrossEncoder() {
    if (this.rerankerTokenizer && this.rerankerModel) {
      return 0;
    }

    const started = performance.now();

    this.rerankerTokenizer =
      await AutoTokenizer.from_pretrained(
        MODEL_IDS.crossEncoder,
        {
          progress_callback: (event) =>
            this.onProgress(
              event,
              "Loading reranker tokenizer",
            ),
        },
      );

    this.rerankerModel =
      await AutoModelForSequenceClassification.from_pretrained(
        MODEL_IDS.crossEncoder,
        {
          dtype: "q8",
          progress_callback: (event) =>
            this.onProgress(
              event,
              "Loading MS MARCO cross-encoder",
            ),
        },
      );

    return performance.now() - started;
  }

  async retrieve(query, candidateK) {
    const queryEmbeddingStarted = performance.now();

    const queryOutput = await this.embedder(query, {
      pooling: "mean",
      normalize: true,
    });

    const queryEmbedding = queryOutput.tolist()[0];
    const queryEmbeddingMs =
      performance.now() - queryEmbeddingStarted;

    const retrievalStarted = performance.now();

    const candidates = this.documents
      .map((document, index) => ({
        ...document,
        bi_encoder_score: dotProduct(
          queryEmbedding,
          this.documentEmbeddings[index],
        ),
      }))
      .sort(
        (left, right) =>
          right.bi_encoder_score - left.bi_encoder_score,
      )
      .slice(0, candidateK)
      .map((document, index) => ({
        ...document,
        retrieval_rank: index + 1,
      }));

    return {
      candidates,
      queryEmbeddingMs,
      retrievalMs: performance.now() - retrievalStarted,
    };
  }

  async rerank(query, candidates, rerankK) {
    const selectedCandidates = candidates.slice(0, rerankK);

    if (selectedCandidates.length === 0) {
      return {
        results: [],
        rerankingMs: 0,
      };
    }

    const started = performance.now();

    const features = this.rerankerTokenizer(
      Array(selectedCandidates.length).fill(query),
      {
        text_pair: selectedCandidates.map(
          (document) => document.search_text,
        ),
        padding: true,
        truncation: true,
        max_length: 512,
      },
    );

    const output = await this.rerankerModel(features);
    const scores = Array.from(output.logits.data);

    const results = selectedCandidates
      .map((document, index) => ({
        ...document,
        cross_encoder_score: scores[index],
      }))
      .sort(
        (left, right) =>
          right.cross_encoder_score -
          left.cross_encoder_score,
      )
      .map((document, index) => ({
        ...document,
        reranked_rank: index + 1,
        rank_movement:
          document.retrieval_rank - (index + 1),
      }));

    return {
      results,
      rerankingMs: performance.now() - started,
    };
  }

  async search({
    query,
    candidateK,
    rerankK,
    mode = "two-stage",
  }) {
    const totalStarted = performance.now();

    const biEncoderSetupMs = await this.ensureBiEncoder();
    const retrieval = await this.retrieve(query, candidateK);

    let crossEncoderSetupMs = 0;
    let reranking = {
      results: [],
      rerankingMs: 0,
    };

    if (mode === "two-stage") {
      crossEncoderSetupMs = await this.ensureCrossEncoder();
      reranking = await this.rerank(
        query,
        retrieval.candidates,
        rerankK,
      );
    }

    return {
      candidates: retrieval.candidates,
      rerankedResults: reranking.results,
      latency: {
        setupMs: biEncoderSetupMs + crossEncoderSetupMs,
        queryEmbeddingMs: retrieval.queryEmbeddingMs,
        retrievalMs: retrieval.retrievalMs,
        rerankingMs: reranking.rerankingMs,
        totalMs: performance.now() - totalStarted,
      },
    };
  }
}
