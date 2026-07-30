"use client";

import { useMemo, useState } from "react";
import type { ChatResponse } from "@/lib/types";
import { EXAMPLE_QUESTIONS } from "@/lib/utils/constants";
import EvaluationMetricsPanel from "@/components/EvaluationMetricsPanel";
import ProjectFilter from "@/components/ProjectFilter";
import RetrievedContextPanel from "@/components/RetrievedContextPanel";
import SourceCitationCard from "@/components/SourceCitationCard";

export default function ChatInterface() {
  const [question, setQuestion] = useState(EXAMPLE_QUESTIONS[0]);
  const [category, setCategory] = useState("All");
  const [deployment, setDeployment] = useState("All");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const canSubmit = useMemo(() => question.trim().length >= 3 && !loading, [question, loading]);

  async function askQuestion() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setCopied(false);

    try {
      const result = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          topK: 5,
          filters: { category, deployment },
        }),
      });
      const payload = await result.json();
      if (!result.ok) throw new Error(payload.error || "The assistant request failed.");
      setResponse(payload as ChatResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unexpected error.");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  }

  async function copyAnswer() {
    if (!response) return;
    await navigator.clipboard.writeText(response.answer);
    setCopied(true);
  }

  return (
    <section className="chat-shell">
      <div className="ask-panel">
        <div className="example-row" aria-label="Example questions">
          {EXAMPLE_QUESTIONS.map((example) => (
            <button key={example} type="button" className="example-chip" onClick={() => setQuestion(example)}>
              {example}
            </button>
          ))}
        </div>

        <ProjectFilter
          category={category}
          deployment={deployment}
          onCategoryChange={setCategory}
          onDeploymentChange={setDeployment}
        />

        <label className="question-label" htmlFor="portfolio-question">Ask about the portfolio</label>
        <textarea
          id="portfolio-question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") askQuestion();
          }}
          placeholder="Example: Which projects demonstrate semantic search and retrieval skills?"
          rows={5}
          maxLength={700}
        />
        <div className="ask-actions">
          <span>{question.length}/700 · Ctrl/Cmd + Enter</span>
          <button type="button" className="primary-button" disabled={!canSubmit} onClick={askQuestion}>
            {loading ? "Retrieving evidence…" : "Ask portfolio assistant"}
          </button>
        </div>
      </div>

      {error && <div className="error-panel" role="alert">{error}</div>}

      {response && (
        <div className="answer-stack">
          <section className="answer-panel">
            <div className="answer-header">
              <div>
                <span className="eyebrow">Grounded response</span>
                <h2>Answer</h2>
              </div>
              <button type="button" className="secondary-button" onClick={copyAnswer}>
                {copied ? "Copied" : "Copy answer"}
              </button>
            </div>
            <div className="answer-text">
              {response.answer.split("\n").map((line, index) =>
                line ? <p key={`${line}-${index}`}>{line}</p> : <br key={`break-${index}`} />
              )}
            </div>
            {response.groundedness.warning && (
              <div className="warning-panel">{response.groundedness.warning}</div>
            )}
          </section>

          <EvaluationMetricsPanel response={response} />

          <section>
            <div className="section-heading">
              <span className="eyebrow">Traceable evidence</span>
              <h2>Source citations</h2>
            </div>
            <div className="citation-grid">
              {response.citations.map((citation) => (
                <SourceCitationCard key={citation.chunkId} citation={citation} />
              ))}
            </div>
          </section>

          <RetrievedContextPanel chunks={response.retrievedChunks} />
        </div>
      )}
    </section>
  );
}
