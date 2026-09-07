const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

async function handleResponse(response) {
  if (!response.ok) {
    let message = "Request failed";
    try {
      const data = await response.json();
      message = data.detail || JSON.stringify(data);
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

export async function predictSingle(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData
  });

  return handleResponse(response);
}

export async function predictBatch(files) {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  const response = await fetch(`${API_BASE_URL}/predict/batch`, {
    method: "POST",
    body: formData
  });

  return handleResponse(response);
}

export async function getHistory(limit = 20) {
  const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`);
  return handleResponse(response);
}

export async function clearHistory() {
  const response = await fetch(`${API_BASE_URL}/history`, {
    method: "DELETE"
  });
  return handleResponse(response);
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(response);
}