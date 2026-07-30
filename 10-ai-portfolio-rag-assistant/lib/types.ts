export type PortfolioCategory =
  | "ANN"
  | "Simple RNN"
  | "LSTM"
  | "BiLSTM"
  | "CNN"
  | "Transformer"
  | "Portfolio";

export type DeploymentPlatform =
  | "Hugging Face"
  | "GitHub Pages"
  | "Vercel"
  | "Streamlit"
  | "Gradio"
  | "TensorFlow.js"
  | "Not specified";

export interface PortfolioChunk {
  id: string;
  projectId: string;
  projectName: string;
  category: PortfolioCategory;
  deployment: DeploymentPlatform;
  sourceFile: string;
  section: string;
  sourcePath: string;
  repository?: string;
  repositoryUrl: string;
  text: string;
  keywords: string[];
  documentId?: string;
  checksumSha256?: string;
  startWord?: number;
  endWord?: number;
}

export interface EmbeddingRecord {
  chunkId: string;
  vector: number[];
}

export interface CorpusMetadata {
  schemaVersion: string;
  corpusName: string;
  generatedAt: string;
  documentCount: number;
  chunkCount: number;
  coverageStatus: "starter" | "partial" | "complete";
  categories: string[];
  deployments: string[];
  sourceRepositories?: string[];
  chunking: {
    strategy: string;
    sizeWords: number;
    overlapWords: number;
  };
  embedding: {
    provider: "pending" | "local-hash-v1" | "huggingface-feature-extraction";
    model: string;
    dimension: number;
    normalized: boolean;
    queryPrefix?: string;
    passagePrefix?: string;
    generatedAt?: string;
    documentEmbeddingsPrecomputed?: boolean;
  };
  retrieval: {
    similarity: "cosine";
    defaultTopK: number;
    hybridSemanticWeight: number;
    hybridLexicalWeight: number;
  };
  notes: string[];
}

export interface RetrievalFilters {
  category?: string;
  deployment?: string;
  projectId?: string;
}

export interface RetrievedChunk extends PortfolioChunk {
  score: number;
  semanticScore: number;
  lexicalScore: number;
  citationId: string;
}

export interface Citation {
  citationId: string;
  projectId: string;
  projectName: string;
  sourceFile: string;
  section: string;
  chunkId: string;
  evidence: string;
  similarityScore: number;
  sourcePath: string;
  repositoryUrl: string;
}

export interface LatencyMetrics {
  embeddingMs: number;
  retrievalMs: number;
  generationMs: number;
  totalMs: number;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  retrievedChunks: RetrievedChunk[];
  metrics: LatencyMetrics;
  groundedness: {
    label: "strong" | "moderate" | "weak";
    supportScore: number;
    warning: string | null;
  };
  runtime: {
    retrievalMode: string;
    generationMode: string;
    corpusCoverage: "starter" | "partial" | "complete";
    embeddingModel: string;
    documentCount: number;
    chunkCount: number;
  };
  warnings?: string[];
}

export interface EvaluationSummary {
  status: "pending" | "partial" | "measured";
  retrieval?: {
    best_method?: string | null;
    hit_rate_at_5?: number | null;
    precision_at_5?: number | null;
    recall_at_5?: number | null;
    mrr_at_5?: number | null;
    map_at_5?: number | null;
    ndcg_at_5?: number | null;
    question_count?: number | null;
  };
  groundedness?: {
    mean_groundedness?: number;
    refusal_accuracy?: number | null;
  };
  citations?: {
    mean_citation_precision?: number;
    mean_citation_completeness?: number;
  };
  latency?: {
    median_ms?: number;
    p95_ms?: number;
  };
  quality_gates?: {
    passed?: number;
    total?: number;
    all_passed?: boolean;
  };
}
