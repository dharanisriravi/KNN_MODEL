// static/script.js
const colorPicker = document.getElementById("colorPicker");
const colorPreview = document.getElementById("colorPreview");
const sizeValue = document.getElementById("sizeValue");
const sizeRange = document.getElementById("sizeRange");

function updateColorPreview() {
  const c = colorPicker.value;
  colorPreview.style.background = c;
}

function updateSizeValue(v) {
  sizeValue.textContent = parseFloat(v).toFixed(1) + " cm";
}

colorPicker.addEventListener("input", updateColorPreview);
window.updateSizeValue = updateSizeValue;

// initialize
updateColorPreview();
updateSizeValue(sizeRange.value);
