import { predictSingle, predictBatch } from "./api.js";

const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const batchBtn = document.getElementById("batchPredictBtn");

const resultState = document.getElementById("resultState");
const resultContent = document.getElementById("resultContent");
const classBadge = document.getElementById("classBadge");
const topConfidence = document.getElementById("topConfidence");

const barGood = document.getElementById("barGood");
const barAverage = document.getElementById("barAverage");
const barBad = document.getElementById("barBad");

const valGood = document.getElementById("valGood");
const valAverage = document.getElementById("valAverage");
const valBad = document.getElementById("valBad");

const batchResults = document.getElementById("batchResults");

let selectedFiles = [];

imageInput?.addEventListener("change", (e) => {
  selectedFiles = Array.from(e.target.files || []);
  if (!selectedFiles.length) return;

  fileName.textContent =
    selectedFiles.length === 1
      ? selectedFiles[0].name
      : `${selectedFiles.length} files selected`;

  const first = selectedFiles[0];
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImage.src = ev.target.result;
    previewWrap.style.display = "block";
  };
  reader.readAsDataURL(first);
});

predictBtn?.addEventListener("click", async () => {
  if (!selectedFiles.length) return alert("Upload at least one image.");
  await runSinglePrediction(selectedFiles[0]);
});

batchBtn?.addEventListener("click", async () => {
  if (!selectedFiles.length) return alert("Upload images first.");
  await runBatchPrediction(selectedFiles);
});

async function runSinglePrediction(file) {
  setLoading(predictBtn, true, "Predicting...");
  try {
    const data = await predictSingle(file);
    const probs = data.probabilities || {};
    renderSingleResult(data.predicted_class, toPercent(data.confidence), {
      good: toPercent(probs.Good ?? 0),
      average: toPercent(probs.Average ?? 0),
      bad: toPercent(probs.Bad ?? 0)
    });
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(predictBtn, false, "Predict Quality");
  }
}

async function runBatchPrediction(files) {
  if (!batchResults) return;
  setLoading(batchBtn, true, "Batch Predicting...");
  try {
    const data = await predictBatch(files);
    renderBatchResults(data.items || []);
  } catch (err) {
    batchResults.innerHTML = `<p class="error">${err.message}</p>`;
  } finally {
    setLoading(batchBtn, false, "Batch Predict");
  }
}

function renderSingleResult(predictedClass, confidence, p) {
  resultState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  classBadge.textContent = predictedClass;
  topConfidence.textContent = `${confidence}%`;

  barGood.style.width = `${p.good}%`;
  barAverage.style.width = `${p.average}%`;
  barBad.style.width = `${p.bad}%`;

  valGood.textContent = `${p.good}%`;
  valAverage.textContent = `${p.average}%`;
  valBad.textContent = `${p.bad}%`;
}

function renderBatchResults(items) {
  batchResults.innerHTML = "";
  if (!items.length) {
    batchResults.innerHTML = "<p>No batch results found.</p>";
    return;
  }

  const table = document.createElement("table");
  table.className = "result-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Filename</th>
        <th>Class</th>
        <th>Confidence</th>
        <th>Time</th>
      </tr>
    </thead>
    <tbody>
      ${items
        .map(
          (it) => `
          <tr>
            <td>${it.filename}</td>
            <td>${it.predicted_class}</td>
            <td>${Math.round((it.confidence || 0) * 100)}%</td>
            <td>${new Date(it.created_at).toLocaleString()}</td>
          </tr>
        `
        )
        .join("")}
    </tbody>
  `;
  batchResults.appendChild(table);
}

function showError(message) {
  resultContent.classList.add("hidden");
  resultState.classList.remove("hidden");
  resultState.textContent = message;
}

function setLoading(button, state, loadingText) {
  if (!button) return;
  button.disabled = state;
  button.textContent = state ? loadingText : button.dataset.defaultText || button.textContent;
  if (!button.dataset.defaultText) button.dataset.defaultText = button.textContent;
}

function toPercent(v) {
  return Math.round(Number(v) * 100);
}