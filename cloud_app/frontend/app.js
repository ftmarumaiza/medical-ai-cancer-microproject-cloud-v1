const form = document.getElementById("predict-form");
const fileInput = document.getElementById("image-input");
const resultBox = document.getElementById("result");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview");
const predictBtn = document.getElementById("predict-btn");

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
    resultBox.textContent = "Please select an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";
  resultBox.textContent = "Sending image to server and running inference...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    resultBox.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    resultBox.textContent = `Error: ${error.message}`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});
