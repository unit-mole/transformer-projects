import evaluationJson from "@/public/data/evaluation_summary.json";
import type { EvaluationSummary } from "@/lib/types";

function metric(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" ? value.toFixed(digits) : "Pending";
}

export default function PortfolioEvaluationSummary() {
  const evaluation = evaluationJson as EvaluationSummary;
  const gateLabel = typeof evaluation.quality_gates?.passed === "number"
    ? `${evaluation.quality_gates.passed}/${evaluation.quality_gates.total}`
    : "Pending";

  return (
    <div className="portfolio-evaluation-card">
      <div>
        <span>Evaluation status</span>
        <strong>{evaluation.status}</strong>
      </div>
      <div>
        <span>Best retriever</span>
        <strong>{evaluation.retrieval?.best_method || "Pending evaluation"}</strong>
      </div>
      <div>
        <span>Recall@5</span>
        <strong>{metric(evaluation.retrieval?.recall_at_5)}</strong>
      </div>
      <div>
        <span>nDCG@5</span>
        <strong>{metric(evaluation.retrieval?.ndcg_at_5)}</strong>
      </div>
      <div>
        <span>Groundedness</span>
        <strong>{metric(evaluation.groundedness?.mean_groundedness)}</strong>
      </div>
      <div>
        <span>Citation precision</span>
        <strong>{metric(evaluation.citations?.mean_citation_precision)}</strong>
      </div>
      <div>
        <span>Refusal accuracy</span>
        <strong>{metric(evaluation.groundedness?.refusal_accuracy)}</strong>
      </div>
      <div>
        <span>Quality gates</span>
        <strong>{gateLabel}</strong>
      </div>
      <div>
        <span>Local P95 latency</span>
        <strong>{typeof evaluation.latency?.p95_ms === "number" ? `${evaluation.latency.p95_ms.toFixed(1)} ms` : "Pending"}</strong>
      </div>
    </div>
  );
}
