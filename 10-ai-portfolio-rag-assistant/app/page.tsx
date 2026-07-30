import ChatInterface from "@/components/ChatInterface";
import DisclaimerBanner from "@/components/DisclaimerBanner";
import PortfolioEvaluationSummary from "@/components/PortfolioEvaluationSummary";
import metadata from "@/public/data/metadata.json";

/**
 * Converts a Vercel hostname into a complete external URL.
 *
 * Vercel system variables commonly contain:
 * 10-ai-portfolio-rag-assistant.vercel.app
 *
 * Browsers require:
 * https://10-ai-portfolio-rag-assistant.vercel.app
 */
function normalizeExternalUrl(
  value: string | undefined,
  fallback: string,
): string {
  const candidate = value?.trim() || fallback;

  if (
    candidate.startsWith("https://") ||
    candidate.startsWith("http://")
  ) {
    return candidate;
  }

  return `https://${candidate}`;
}

export default function HomePage() {
  const githubUrl = normalizeExternalUrl(
    process.env.NEXT_PUBLIC_GITHUB_URL,
    "github.com/unit-mole/transformer-projects",
  );

  /**
   * Priority:
   *
   * 1. Manually configured public production URL
   * 2. Vercel production-domain system variable
   * 3. Current Vercel deployment-domain variable
   * 4. Confirmed permanent production URL
   *
   * Server-side Vercel variables are included because this page is
   * rendered as a Server Component.
   */
  const vercelUrl = normalizeExternalUrl(
    process.env.NEXT_PUBLIC_VERCEL_PROJECT_PRODUCTION_URL ||
      process.env.VERCEL_PROJECT_PRODUCTION_URL ||
      process.env.NEXT_PUBLIC_VERCEL_URL ||
      process.env.VERCEL_URL,
    "10-ai-portfolio-rag-assistant.vercel.app",
  );

  const transformerReady =
    metadata.embedding.provider ===
    "huggingface-feature-extraction";

  return (
    <main>
      <header className="site-header">
        <a
          className="brand"
          href="#top"
          aria-label="AI Portfolio RAG Assistant home"
        >
          <span className="brand-mark">R</span>
          <span>Portfolio RAG</span>
        </a>

        <nav>
          <a href="#architecture">Architecture</a>
          <a href="#evaluation">Evaluation</a>

          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </nav>
      </header>

      <section
        className="hero"
        id="top"
      >
        <div className="hero-copy">
          <span className="eyebrow">
            Transformer Project 10 · Vercel Capstone
          </span>

          <h1>AI Portfolio RAG Assistant</h1>

          <p className="hero-lead">
            Ask evidence-backed questions about machine learning,
            NLP, computer vision, deployment, and portfolio
            experience. Every answer exposes retrieved evidence,
            source citations, relevance scores, and runtime latency.
          </p>

          <div className="hero-actions">
            <a
              className="primary-link"
              href="#assistant"
            >
              Try the assistant
            </a>

            <a
              className="secondary-link"
              href={vercelUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open the live Vercel application in a new tab"
            >
              Live Vercel URL
            </a>
          </div>
        </div>

        <div
          className="hero-card"
          aria-label="Portfolio corpus summary"
        >
          <span>
            {metadata.coverageStatus} public corpus
          </span>

          <strong>
            {metadata.documentCount} documents
          </strong>

          <strong>
            {metadata.chunkCount} chunks
          </strong>

          <p>{metadata.embedding.model}</p>

          <small
            className={
              transformerReady
                ? "status-ready"
                : "status-pending"
            }
          >
            {transformerReady
              ? "Real Transformer embeddings are active."
              : "Starter fallback is active; run the evaluation notebook before final launch."}
          </small>
        </div>
      </section>

      <DisclaimerBanner />

      <section
        id="assistant"
        className="assistant-section"
      >
        <div className="section-heading">
          <span className="eyebrow">
            Retriever + grounded generator
          </span>

          <h2>Search the portfolio</h2>

          <p>
            Use filters or ask a natural-language question.
            Answers are limited to indexed public evidence.
          </p>
        </div>

        <ChatInterface />
      </section>

      <section
        className="info-grid"
        id="architecture"
      >
        <article>
          <span className="step-number">01</span>

          <h3>Public corpus</h3>

          <p>
            Safe Markdown documents are collected, cleaned,
            section-chunked, and exported with traceable metadata.
          </p>
        </article>

        <article>
          <span className="step-number">02</span>

          <h3>Transformer retrieval</h3>

          <p>
            Precomputed MiniLM or E5 embeddings are combined with
            lexical matching for efficient Vercel retrieval.
          </p>
        </article>

        <article>
          <span className="step-number">03</span>

          <h3>Grounded generation</h3>

          <p>
            A hosted instruction model can generate cited answers,
            while the extractive composer remains a safe fallback.
          </p>
        </article>

        <article>
          <span className="step-number">04</span>

          <h3>Evidence and evaluation</h3>

          <p>
            The interface exposes source chunks and latency, while
            offline scripts measure retrieval and answer quality.
          </p>
        </article>
      </section>

      <section
        className="evaluation-section"
        id="evaluation"
      >
        <div className="section-heading">
          <span className="eyebrow">
            Measured, not claimed
          </span>

          <h2>Portfolio evaluation</h2>

          <p>
            These values are generated from committed evaluation
            artifacts after the corpus and models are run locally.
          </p>
        </div>

        <PortfolioEvaluationSummary />

        <div className="evaluation-list">
          <div>
            <strong>Retrieval quality</strong>

            <span>
              Hit Rate@K, Precision@K, Recall@K, MRR, MAP, and
              nDCG@K.
            </span>
          </div>

          <div>
            <strong>Answer groundedness</strong>

            <span>
              Claim-level NLI support against retrieved evidence.
            </span>
          </div>

          <div>
            <strong>Citation correctness</strong>

            <span>
              Citation precision, completeness, and
              unsupported-claim rate.
            </span>
          </div>

          <div>
            <strong>Response latency</strong>

            <span>
              Embedding, retrieval, generation, total, median,
              P90, and P95.
            </span>
          </div>
        </div>

        <p className="evaluation-note">
          Pending values remain visibly labeled until the full
          evaluation notebook has been run. No placeholder result
          is presented as a final score.
        </p>
      </section>

      <footer>
        <p>
          Built as a Vercel-ready portfolio demonstration by
          Anmol Tripathi.
        </p>

        <p>
          Next.js · TypeScript · Transformer retrieval · RAG ·
          source citations
        </p>
      </footer>
    </main>
  );
}