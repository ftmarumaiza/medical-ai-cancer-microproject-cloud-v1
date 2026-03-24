const form = document.getElementById("predict-form");
const fileInput = document.getElementById("image-input");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview");
const predictBtn = document.getElementById("predict-btn");

const resultEmpty = document.getElementById("result-empty");
const resultLoading = document.getElementById("result-loading");
const resultError = document.getElementById("result-error");
const resultCard = document.getElementById("result-card");

const diagnosisBadge = document.getElementById("diagnosis-badge");
const diagnosisText = document.getElementById("diagnosis-text");
const confidenceText = document.getElementById("confidence-text");
const topLabelText = document.getElementById("top-label-text");
const objectsText = document.getElementById("objects-text");
const timeText = document.getElementById("time-text");
const filenameText = document.getElementById("filename-text");
const storageText = document.getElementById("storage-text");
const resultRaw = document.getElementById("result-raw");

function resetResultState() {
  resultEmpty.classList.add("hidden");
  resultLoading.classList.add("hidden");
  resultError.classList.add("hidden");
  resultCard.classList.add("hidden");
}

function showLoading() {
  resetResultState();
  resultLoading.classList.remove("hidden");
}

function showError(message) {
  resetResultState();
  resultError.textContent = message;
  resultError.classList.remove("hidden");
}

function formatConfidence(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatProcessingTime(ms) {
  if (typeof ms !== "number") {
    return "-";
  }

  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} s`;
  }
  return `${ms.toFixed(2)} ms`;
}

function renderPrediction(payload) {
  resetResultState();

  const diagnosis = payload.result || (payload.cancer_present ? "Cancer Detected" : "No Cancer Detected");
  const objectCount = payload.metrics?.total_objects ?? payload.detections?.length ?? 0;
  const processingMs = payload.metrics?.processing_time_ms;
  const topDetection = payload.top_detection || payload.detections?.[0] || null;
  const topLabel = topDetection?.label || "-";

  diagnosisText.textContent = diagnosis;
  confidenceText.textContent = formatConfidence(payload.confidence);
  topLabelText.textContent = topLabel;
  objectsText.textContent = String(objectCount);
  timeText.textContent = formatProcessingTime(processingMs);
  filenameText.textContent = payload.filename || "-";
  storageText.textContent = payload.storage_uri || "-";

  diagnosisBadge.textContent = payload.cancer_present ? "Cancer Detected" : "No Cancer Detected";
  diagnosisBadge.classList.remove("badge-positive", "badge-negative");
  diagnosisBadge.classList.add(payload.cancer_present ? "badge-positive" : "badge-negative");

  resultRaw.textContent = JSON.stringify(payload, null, 2);
  resultCard.classList.remove("hidden");
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    previewWrap.classList.add("hidden");
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewWrap.classList.remove("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    showError("Please select an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";
  showLoading();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    renderPrediction(payload);
  } catch (error) {
    showError(`Error: ${error.message}`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});
