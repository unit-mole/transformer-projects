function dcg(grades) {
  return grades.reduce(
    (total, grade, index) =>
      total + (2 ** grade - 1) / Math.log2(index + 2),
    0,
  );
}

export function recallAtK(rankedIds, relevance, k) {
  const relevantIds = [...relevance.entries()]
    .filter(([, grade]) => grade > 0)
    .map(([documentId]) => documentId);

  if (relevantIds.length === 0) {
    return 0;
  }

  const retrieved = new Set(rankedIds.slice(0, k));
  const hits = relevantIds.filter((documentId) =>
    retrieved.has(documentId),
  );
  return hits.length / relevantIds.length;
}

export function reciprocalRankAtK(
  rankedIds,
  relevance,
  k = 10,
) {
  const limit = Math.min(k, rankedIds.length);
  for (let index = 0; index < limit; index += 1) {
    if ((relevance.get(rankedIds[index]) ?? 0) > 0) {
      return 1 / (index + 1);
    }
  }
  return 0;
}

export function ndcgAtK(rankedIds, relevance, k = 10) {
  const observed = rankedIds
    .slice(0, k)
    .map((documentId) => relevance.get(documentId) ?? 0);

  const ideal = [...relevance.values()]
    .sort((left, right) => right - left)
    .slice(0, k);

  const idealDcg = dcg(ideal);
  return idealDcg === 0 ? 0 : dcg(observed) / idealDcg;
}

export function calculateQueryMetrics({
  queryId,
  qrelsByQuery,
  candidates,
  rerankedResults,
  candidateK,
  mode,
}) {
  if (!queryId || !qrelsByQuery.has(queryId)) {
    return null;
  }

  const relevance = qrelsByQuery.get(queryId);
  const retrievalIds = candidates.map(
    (document) => document.document_id,
  );

  const rerankedIds = rerankedResults.map(
    (document) => document.document_id,
  );
  const rerankedSet = new Set(rerankedIds);

  const finalIds =
    mode === "two-stage"
      ? [
          ...rerankedIds,
          ...candidates
            .filter(
              (candidate) =>
                !rerankedSet.has(candidate.document_id),
            )
            .map((document) => document.document_id),
        ]
      : retrievalIds;

  const beforeMrr = reciprocalRankAtK(
    retrievalIds,
    relevance,
    10,
  );
  const afterMrr = reciprocalRankAtK(finalIds, relevance, 10);
  const beforeNdcg = ndcgAtK(retrievalIds, relevance, 10);
  const afterNdcg = ndcgAtK(finalIds, relevance, 10);

  return {
    relevantDocumentCount: [...relevance.values()].filter(
      (grade) => grade > 0,
    ).length,
    recallAtK: recallAtK(retrievalIds, relevance, candidateK),
    beforeMrr,
    afterMrr,
    mrrImprovement: afterMrr - beforeMrr,
    beforeNdcg,
    afterNdcg,
    ndcgImprovement: afterNdcg - beforeNdcg,
  };
}
