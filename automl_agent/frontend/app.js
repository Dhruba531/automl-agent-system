const state = {
  schema: null,
  metadata: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value ?? "-";
}

function showNotice(message, tone = "warn") {
  const notice = $("#notice");
  notice.textContent = message;
  notice.dataset.tone = tone;
  notice.classList.toggle("is-hidden", !message);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail));
  }
  return response.json();
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function renderKeyValues(selector, values) {
  const node = $(selector);
  node.innerHTML = "";
  Object.entries(values || {}).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    node.append(dt, dd);
  });
}

function sampleRows(multiplier = 1) {
  const columns = state.schema?.feature_columns || [];
  const row = {};
  columns.forEach((column, index) => {
    row[column] = Number(((index + 1) * multiplier).toFixed(3));
  });
  return { rows: [row] };
}

function renderOverview() {
  const metadata = state.metadata || {};
  const schema = state.schema || {};
  setText("#modelTitle", metadata.model_name || schema.model_name || "Model console");
  setText("#modelName", metadata.model_name || schema.model_name);
  setText("#modelVersion", metadata.model_version || schema.model_version || "unversioned");
  setText("#targetName", metadata.target || schema.target);
  setText("#featureCount", String((schema.feature_columns || []).length || "-"));
  setText("#taskBadge", metadata.task_type || schema.task_type || "Task unknown");
  renderKeyValues("#metricsList", metadata.metrics || schema.metrics || {});
  renderKeyValues("#profileList", {
    rows: metadata.profile?.rows,
    columns: metadata.profile?.columns,
    target: metadata.profile?.target,
    task: metadata.profile?.task_type,
    numeric_features: metadata.profile?.numeric_features?.length,
    categorical_features: metadata.profile?.categorical_features?.length,
  });
}

function renderExplainability() {
  const explainability = state.metadata?.explainability;
  const list = $("#importanceList");
  list.innerHTML = "";
  setText("#explainMethod", explainability?.method || "Unavailable");
  const importances = explainability?.importances || [];
  const max = Math.max(...importances.map((item) => Math.abs(item.importance_mean)), 0.000001);
  if (!importances.length) {
    list.textContent = "No explainability artifact is available for this model.";
    return;
  }
  importances.forEach((item) => {
    const row = document.createElement("div");
    row.className = "importance-row";
    const label = document.createElement("strong");
    label.textContent = item.feature;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(2, Math.abs(item.importance_mean / max) * 100)}%`;
    const score = document.createElement("span");
    score.textContent = item.importance_mean.toFixed(4);
    track.append(fill);
    row.append(label, track, score);
    list.append(row);
  });
}

async function loadAuth() {
  try {
    const auth = await fetchJson("/auth/status");
    const label = auth.enabled
      ? auth.authenticated
        ? auth.user?.email || "Signed in"
        : "Google sign-in required"
      : "Auth disabled";
    setText("#authState", label);
    $("#loginLink").classList.toggle("is-hidden", !auth.enabled || auth.authenticated);
    $("#logoutButton").classList.toggle("is-hidden", !auth.enabled || !auth.authenticated);
  } catch {
    setText("#authState", "Unknown");
  }
}

async function loadModel() {
  const health = await fetchJson("/health");
  setText("#healthBadge", `Healthy · ${health.bundle}`);
  state.schema = await fetchJson("/schema");
  state.metadata = await fetchJson("/metadata");
  renderOverview();
  renderExplainability();
  const rows = sampleRows(1);
  $("#predictInput").value = pretty(rows);
  $("#driftInput").value = pretty(rows);
  showNotice("");
}

function bindNavigation() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      $$(".nav-item").forEach((item) => item.classList.toggle("is-active", item === button));
      $$(".view").forEach((panel) => panel.classList.toggle("is-active", panel.dataset.panel === view));
    });
  });
}

function bindActions() {
  $("#sampleButton").addEventListener("click", () => {
    $("#predictInput").value = pretty(sampleRows(1));
  });
  $("#driftSampleButton").addEventListener("click", () => {
    $("#driftInput").value = pretty(sampleRows(50));
  });
  $("#predictButton").addEventListener("click", async () => {
    try {
      const payload = JSON.parse($("#predictInput").value);
      $("#predictionOutput").textContent = pretty(await fetchJson("/predict", { method: "POST", body: JSON.stringify(payload) }));
    } catch (error) {
      $("#predictionOutput").textContent = error.message;
    }
  });
  $("#driftButton").addEventListener("click", async () => {
    try {
      const payload = JSON.parse($("#driftInput").value);
      $("#driftOutput").textContent = pretty(await fetchJson("/drift", { method: "POST", body: JSON.stringify(payload) }));
    } catch (error) {
      $("#driftOutput").textContent = error.message;
    }
  });
  $("#logoutButton").addEventListener("click", async () => {
    await fetchJson("/auth/logout", { method: "POST" });
    window.location.reload();
  });
}

async function boot() {
  bindNavigation();
  bindActions();
  await loadAuth();
  try {
    await loadModel();
  } catch (error) {
    showNotice(error.message === "Google login required." ? "Sign in with Google to view model details." : error.message);
  }
}

boot();
