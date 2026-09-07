import { getHistory, clearHistory } from "./api.js";

const historyBody = document.getElementById("historyBody");
const refreshBtn = document.getElementById("refreshHistoryBtn");
const clearBtn = document.getElementById("clearHistoryBtn");

refreshBtn?.addEventListener("click", loadHistory);
clearBtn?.addEventListener("click", async () => {
  if (!confirm("Clear all history?")) return;
  await clearHistory();
  await loadHistory();
});

async function loadHistory() {
  historyBody.innerHTML = `<tr><td colspan="5">Loading...</td></tr>`;
  try {
    const data = await getHistory(50);
    if (!data.items.length) {
      historyBody.innerHTML = `<tr><td colspan="5">No history found.</td></tr>`;
      return;
    }

    historyBody.innerHTML = data.items
      .map(
        (item) => `
      <tr>
        <td>${item.filename}</td>
        <td>${item.predicted_class}</td>
        <td>${Math.round(item.confidence * 100)}%</td>
        <td>${JSON.stringify(item.probabilities)}</td>
        <td>${new Date(item.created_at).toLocaleString()}</td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    historyBody.innerHTML = `<tr><td colspan="5">${err.message}</td></tr>`;
  }
}

loadHistory();