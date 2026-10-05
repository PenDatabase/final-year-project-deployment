const form = document.querySelector("#analysis-form");
const processingPanel = document.querySelector("#processing-panel");
const reportRegion = document.querySelector("#report-region");
const runButton = document.querySelector("#run-button");

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function showFile(input) {
  const zone = input.closest("[data-drop-zone]");
  const output = zone.querySelector("[data-file-name]");
  const file = input.files[0];
  if (!file) return;
  output.textContent = `${file.name} · ${formatBytes(file.size)}`;
  zone.classList.add("has-file");
}

document.querySelectorAll("[data-drop-zone]").forEach((zone) => {
  const input = zone.querySelector("input[type=file]");
  input.addEventListener("change", () => showFile(input));
  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  });
  zone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (!files.length) return;
    input.files = files;
    showFile(input);
  });
});

function updateProgress(event) {
  const stages = ["queued", "validating", "training", "forecasting", "optimizing"];
  const currentIndex = stages.indexOf(event.stage);
  document.querySelectorAll("[data-stage]").forEach((item) => {
    const index = stages.indexOf(item.dataset.stage);
    item.classList.toggle("is-active", index === currentIndex);
    item.classList.toggle("is-complete", index >= 0 && index < currentIndex);
    if (index === currentIndex) item.querySelector("small").textContent = event.message;
  });
}

async function runAnalysis(event) {
  event.preventDefault();
  if (!form.reportValidity()) return;
  processingPanel.hidden = false;
  reportRegion.replaceChildren();
  runButton.disabled = true;
  runButton.innerHTML = "Working <span class=\"button-loader\">◌</span>";
  processingPanel.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const response = await fetch(form.action, { method: "POST", body: new FormData(form) });
    if (!response.ok || !response.body) throw new Error("The analysis could not be started.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const eventData = JSON.parse(line);
        if (eventData.type === "progress") updateProgress(eventData);
        if (eventData.type === "complete") {
          document.querySelectorAll("[data-stage]").forEach((item) => item.classList.add("is-complete"));
          reportRegion.innerHTML = eventData.html;
          reportRegion.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        if (eventData.type === "error") throw new Error(eventData.message);
      }
      if (done) break;
    }
  } catch (error) {
    reportRegion.innerHTML = `<section class="inline-error page-shell"><p class="eyebrow">Needs attention</p><h2>${error.message}</h2><a class="back-link" href="/optimize">Try again ↗</a></section>`;
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = "Run analysis <span>↗</span>";
  }
}

form.addEventListener("submit", runAnalysis);
