const dataUrl = (filename) =>
  `${import.meta.env.BASE_URL}data/${filename}`;

async function fetchJson(filename) {
  const response = await fetch(dataUrl(filename));
  if (!response.ok) {
    throw new Error(
      `Could not load ${filename}. HTTP status: ${response.status}.`,
    );
  }
  return response.json();
}

export function buildQrelsLookup(qrels) {
  const lookup = new Map();

  for (const row of qrels) {
    if (!lookup.has(row.query_id)) {
      lookup.set(row.query_id, new Map());
    }
    lookup
      .get(row.query_id)
      .set(row.document_id, Number(row.relevance));
  }

  return lookup;
}

export async function loadDemoData() {
  const [documents, queries, qrels] = await Promise.all([
    fetchJson("sample_documents.json"),
    fetchJson("sample_queries.json"),
    fetchJson("sample_qrels.json"),
  ]);

  if (!Array.isArray(documents) || documents.length === 0) {
    throw new Error("The document dataset is empty.");
  }
  if (!Array.isArray(queries) || queries.length === 0) {
    throw new Error("The query dataset is empty.");
  }
  if (!Array.isArray(qrels) || qrels.length === 0) {
    throw new Error("The relevance dataset is empty.");
  }

  const preparedDocuments = documents.map((document) => ({
    ...document,
    search_text: `${document.title}. ${document.document}`,
  }));

  return {
    documents: preparedDocuments,
    queries,
    qrels,
    queryById: new Map(queries.map((row) => [row.query_id, row])),
    qrelsByQuery: buildQrelsLookup(qrels),
  };
}
