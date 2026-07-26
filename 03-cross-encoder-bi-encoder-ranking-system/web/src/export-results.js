export function downloadJson(payload, filenamePrefix = "docrank360-results") {
  if (!payload) {
    throw new Error("Run a search before exporting results.");
  }

  const blob = new Blob(
    [JSON.stringify(payload, null, 2)],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${filenamePrefix}-${Date.now()}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
