const data = window.PROJECT05_DATA;
const records = data.records;
let filteredRecords = [...records];
let selectedId = records[0]?.id;

const metricNames = {
  instruction_adherence: "Instruction adherence",
  quality_rubric: "Quality rubric",
  rouge_l_f1: "ROUGE-L F1",
  semantic_similarity: "Semantic similarity",
  bertscore_f1: "BERTScore F1",
  hallucination_flag_rate: "Hallucination-risk flags",
};

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const decimal = (value) => Number(value).toFixed(3);
const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function fillSummary() {
  const s = data.summary;
  document.querySelector("#kpi-prompts").textContent = s.benchmark_prompts;
  document.querySelector("#kpi-exp1").textContent = s.experiment1_wins;
  document.querySelector("#kpi-exp2").textContent = s.experiment2_wins;
  document.querySelector("#kpi-ties").textContent = s.ties;

  const container = document.querySelector("#metric-bars");
  container.innerHTML = Object.entries(s.automated_metrics).map(([key, values]) => {
    const isRisk = key === "hallucination_flag_rate";
    const max = Math.max(...Object.values(values), 0.001);
    const format = isRisk ? pct : decimal;
    return `
      <div class="metric-row">
        <div class="metric-label">
          <strong>${metricNames[key]}</strong>
          <small>${isRisk ? "Lower is better" : "Higher is better"}</small>
        </div>
        <div class="bar-group">
          ${["base", "experiment1", "experiment2"].map(model => `
            <div class="bar-line">
              <span>${model === "base" ? "Base" : model === "experiment1" ? "Experiment 1" : "Experiment 2"}</span>
              <div class="bar-track"><div class="bar-fill ${model}" style="width:${(values[model] / max) * 100}%"></div></div>
              <strong>${format(values[model])}</strong>
            </div>`).join("")}
        </div>
      </div>`;
  }).join("");
}

function populateFilters() {
  const category = document.querySelector("#category-filter");
  const topic = document.querySelector("#topic-filter");

  data.categories.forEach(value => category.insertAdjacentHTML("beforeend", `<option value="${esc(value)}">${esc(value)}</option>`));
  data.topics.forEach(value => topic.insertAdjacentHTML("beforeend", `<option value="${esc(value)}">${esc(value)}</option>`));
}

function preferenceLabel(value) {
  return value === "experiment1" ? "Experiment 1" : value === "experiment2" ? "Experiment 2" : "Tie";
}

function renderRecordList() {
  const list = document.querySelector("#record-list");
  document.querySelector("#record-count").textContent = `${filteredRecords.length} shown`;

  if (!filteredRecords.length) {
    list.innerHTML = `<div class="empty-state">No prompts match these filters.</div>`;
    document.querySelector("#record-detail").innerHTML = `<div class="empty-state">Change the filters to see benchmark records.</div>`;
    return;
  }

  if (!filteredRecords.some(r => r.id === selectedId)) selectedId = filteredRecords[0].id;

  list.innerHTML = filteredRecords.map(record => `
    <button class="record-button ${record.id === selectedId ? "active" : ""}" data-id="${esc(record.id)}">
      <strong>${esc(record.topic)}</strong>
      <small>
        <span>${esc(record.category)}</span>
        <span class="tag ${esc(record.human_preferred_model)}">${preferenceLabel(record.human_preferred_model)}</span>
      </small>
    </button>
  `).join("");

  list.querySelectorAll(".record-button").forEach(button => {
    button.addEventListener("click", () => {
      selectedId = button.dataset.id;
      renderRecordList();
      renderRecordDetail();
    });
  });
}

function renderRecordDetail() {
  const record = records.find(r => r.id === selectedId);
  if (!record) return;

  const hallucination = value => String(value).toLowerCase() === "true";
  const metricRows = [
    ["Instruction adherence", record.metrics.instruction_adherence],
    ["Quality rubric", record.metrics.quality_rubric],
    ["ROUGE-L F1", record.metrics.rouge_l_f1],
    ["Semantic similarity", record.metrics.semantic_similarity],
    ["BERTScore F1", record.metrics.bertscore_f1],
  ];

  document.querySelector("#record-detail").innerHTML = `
    <div class="detail-top">
      <div>
        <div class="detail-meta">${esc(record.id)} · ${esc(record.category)} · ${esc(record.difficulty)}</div>
        <h3>${esc(record.prompt)}</h3>
      </div>
      <div class="preference-badge ${esc(record.human_preferred_model)}">
        Human choice: ${preferenceLabel(record.human_preferred_model)}
      </div>
    </div>

    <div class="answer-grid">
      <section class="answer-card reference">
        <h4>Reference answer</h4>
        <p>${esc(record.reference_answer)}</p>
      </section>

      <section class="answer-card">
        <h4>Untouched FLAN-T5-base</h4>
        <p>${esc(record.base_answer)}</p>
      </section>

      <section class="answer-card">
        <h4>Experiment 1 · Selected</h4>
        <p>${esc(record.experiment1_answer)}</p>
        ${hallucination(record.experiment1_hallucination_flag) ? '<div class="risk">Automated hallucination-risk flag</div>' : ""}
      </section>

      <section class="answer-card">
        <h4>Experiment 2 · Not promoted</h4>
        <p>${esc(record.experiment2_answer)}</p>
        ${hallucination(record.experiment2_hallucination_flag) ? '<div class="risk">Automated hallucination-risk flag</div>' : ""}
      </section>
    </div>

    <div class="review-note">
      <h4>Human reviewer note</h4>
      <p>${esc(record.human_notes)}</p>
    </div>

    <div class="example-metrics">
      <table>
        <thead><tr><th>Per-example metric</th><th>Experiment 1</th><th>Experiment 2</th><th>Delta (Exp2 − Exp1)</th></tr></thead>
        <tbody>
          ${metricRows.map(([name, values]) => `
            <tr>
              <td>${name}</td>
              <td>${decimal(values.experiment1)}</td>
              <td>${decimal(values.experiment2)}</td>
              <td>${(values.experiment2 - values.experiment1).toFixed(3)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function applyFilters() {
  const search = document.querySelector("#search-input").value.trim().toLowerCase();
  const category = document.querySelector("#category-filter").value;
  const topic = document.querySelector("#topic-filter").value;
  const preference = document.querySelector("#preference-filter").value;

  filteredRecords = records.filter(record => {
    const searchable = [
      record.prompt, record.topic, record.category, record.reference_answer,
      record.base_answer, record.experiment1_answer, record.experiment2_answer,
      record.human_notes,
    ].join(" ").toLowerCase();

    return (!search || searchable.includes(search))
      && (!category || record.category === category)
      && (!topic || record.topic === topic)
      && (!preference || record.human_preferred_model === preference);
  });

  renderRecordList();
  renderRecordDetail();
}



function fillTrainingComparison() {
  const results = data.summary.training_results;
  const exp1 = results.experiment1;
  const exp2 = results.experiment2;

  document.querySelector("#training-comparison").innerHTML = `
    <div class="table-wrap">
      <table class="training-table">
        <thead>
          <tr>
            <th>Training result</th>
            <th>Experiment 1</th>
            <th>Experiment 2</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Decision</td><td class="selected-cell">${exp1.decision}</td><td class="not-promoted-cell">${exp2.decision}</td></tr>
          <tr><td>Dataset records</td><td>${exp1.records}</td><td>${exp2.records}</td></tr>
          <tr><td>Train / validation / test</td><td>${exp1.train_records} / ${exp1.validation_records} / ${exp1.test_records}</td><td>${exp2.train_records} / ${exp2.validation_records} / ${exp2.test_records}</td></tr>
          <tr><td>Training epochs</td><td>${exp1.epochs}</td><td>${exp2.epochs}</td></tr>
          <tr><td>LoRA rank / alpha</td><td>${exp1.lora_rank} / ${exp1.lora_alpha}</td><td>${exp2.lora_rank} / ${exp2.lora_alpha}</td></tr>
          <tr><td>Trainable parameters</td><td>${exp1.trainable_parameters.toLocaleString()} (${exp1.trainable_percentage.toFixed(2)}%)</td><td>${exp2.trainable_parameters.toLocaleString()} (${exp2.trainable_percentage.toFixed(2)}%)</td></tr>
          <tr><td>Validation loss</td><td class="selected-cell">${exp1.validation_loss.toFixed(4)}</td><td>${exp2.validation_loss.toFixed(4)}</td></tr>
          <tr><td>Test loss</td><td class="selected-cell">${exp1.test_loss.toFixed(4)}</td><td>${exp2.test_loss.toFixed(4)}</td></tr>
          <tr><td>Validation perplexity</td><td class="selected-cell">${exp1.validation_perplexity.toFixed(3)}</td><td>${exp2.validation_perplexity.toFixed(3)}</td></tr>
        </tbody>
      </table>
    </div>
  `;

  const reviews = data.summary.human_review;
  document.querySelector("#human-review-summary").innerHTML = `
    <div class="review-stat"><span>Experiment 1 vs base</span><strong>${reviews.base_vs_experiment1.experiment1} wins</strong></div>
    <div class="review-stat"><span>Experiment 1 vs Experiment 2</span><strong>${reviews.experiment1_vs_experiment2.experiment1} wins</strong></div>
    <div class="review-stat"><span>Experiment 2 wins</span><strong>${reviews.experiment1_vs_experiment2.experiment2}</strong></div>
    <div class="review-stat"><span>Ties</span><strong>${reviews.experiment1_vs_experiment2.ties}</strong></div>
  `;
}

function fillHeadlineMetricsTable() {
  const metrics = data.summary.automated_metrics;
  const tbody = document.querySelector("#headline-metrics-table tbody");

  tbody.innerHTML = Object.entries(metrics).map(([key, values]) => {
    const lowerIsBetter = key === "hallucination_flag_rate";
    const entries = [
      ["Base", values.base],
      ["Experiment 1", values.experiment1],
      ["Experiment 2", values.experiment2],
    ];
    const best = entries.reduce((winner, current) =>
      lowerIsBetter
        ? (current[1] < winner[1] ? current : winner)
        : (current[1] > winner[1] ? current : winner)
    );
    const formatter = lowerIsBetter
      ? value => `${(value * 100).toFixed(2)}%`
      : value => Number(value).toFixed(4);

    return `
      <tr>
        <td><strong>${metricNames[key]}</strong></td>
        <td>${formatter(values.base)}</td>
        <td>${formatter(values.experiment1)}</td>
        <td>${formatter(values.experiment2)}</td>
        <td><span class="best-badge">${best[0]}</span></td>
      </tr>
    `;
  }).join("");
}

function fillWorkflow() {
  document.querySelector("#workflow-timeline").innerHTML =
    data.summary.workflow.map(item => `
      <article class="workflow-card">
        <span class="workflow-number">${item.step}</span>
        <h3>${esc(item.title)}</h3>
        <p>${esc(item.description)}</p>
      </article>
    `).join("");
}

fillSummary();
fillTrainingComparison();
fillHeadlineMetricsTable();
fillWorkflow();
populateFilters();
renderRecordList();
renderRecordDetail();

["search-input", "category-filter", "topic-filter", "preference-filter"].forEach(id => {
  document.querySelector(`#${id}`).addEventListener(id === "search-input" ? "input" : "change", applyFilters);
});
