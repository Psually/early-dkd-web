const groupsEl = document.getElementById('input-groups');
const form = document.getElementById('predict-form');
const statusEl = document.getElementById('form-status');
const button = document.getElementById('predict-button');
let schema;

function fieldHtml(field) {
  const required = field.required ? '<b> *</b>' : '';
  const unit = field.unit ? `<span class="unit">${field.unit}</span>` : '';
  let control;
  if (field.kind === 'number') {
    const range = field.min === undefined ? '' : `min="${field.min}" max="${field.max}"`;
    control = `<input id="${field.name}" name="${field.name}" type="number" step="any" ${range} placeholder="Leave blank if unavailable">`;
  } else if (field.kind === 'sex') {
    control = `<select id="${field.name}" name="${field.name}"><option value="">Select</option><option value="1">Male</option><option value="0">Female</option></select>`;
  } else {
    control = `<select id="${field.name}" name="${field.name}"><option value="">Unavailable</option><option value="0">Negative / No</option><option value="1">Positive / Yes</option></select>`;
  }
  return `<div class="field"><label for="${field.name}">${field.label}${required}</label><div class="control">${control}${unit}</div><span class="field-error" id="${field.name}-error" hidden></span></div>`;
}

function renderSchema(data) {
  schema = data;
  groupsEl.innerHTML = data.groups.map(group => `<section class="input-section"><h3>${group.title}</h3><div class="field-grid">${group.fields.map(fieldHtml).join('')}</div></section>`).join('');
}

function collectValues() {
  const values = {};
  for (const group of schema.groups) for (const field of group.fields) {
    const value = document.getElementById(field.name).value;
    values[field.name] = value === '' ? null : Number(value);
  }
  return values;
}

function validateInputs() {
  let firstInvalid = null;
  for (const group of schema.groups) for (const field of group.fields) {
    const input = document.getElementById(field.name);
    const error = document.getElementById(`${field.name}-error`);
    const raw = input.value.trim();
    let message = '';
    if (field.required && raw === '') message = `${field.label} is required.`;
    else if (raw !== '' && field.kind === 'number') {
      const value = Number(raw);
      if (input.validity.badInput || !Number.isFinite(value)) message = `Enter a valid number for ${field.label}.`;
      else if (field.min !== undefined && (value < field.min || value > field.max))
        message = `${field.label} must be between ${field.min} and ${field.max}.`;
    }
    error.textContent = message;
    error.hidden = !message;
    input.setAttribute('aria-invalid', message ? 'true' : 'false');
    if (message && !firstInvalid) firstInvalid = input;
  }
  if (firstInvalid) {
    showError('Please complete the highlighted fields.');
    firstInvalid.focus();
    return false;
  }
  document.getElementById('error-box').hidden = true;
  return true;
}

function clearFieldError(event) {
  const input = event.target;
  if (!input.id || !schema) return;
  const error = document.getElementById(`${input.id}-error`);
  if (error) { error.hidden = true; error.textContent = ''; input.setAttribute('aria-invalid', 'false'); }
  document.getElementById('error-box').hidden = true;
}
form.addEventListener('input', clearFieldError);
form.addEventListener('change', clearFieldError);

function percent(n, decimals = 1) { return `${(100 * n).toFixed(decimals)}%`; }
function makeTextCell(text, className = '') { const td = document.createElement('td'); td.textContent = text; if (className) td.className = className; return td; }

function renderResult(data) {
  document.getElementById('empty-result').hidden = true;
  document.getElementById('error-box').hidden = true;
  document.getElementById('result-content').hidden = false;
  document.getElementById('risk-number').textContent = percent(data.probability);
  const high = data.probability >= data.threshold;
  const level = document.getElementById('risk-level');
  level.textContent = high ? 'Above threshold' : 'Below threshold';
  level.classList.toggle('high', high);
  const fill = document.getElementById('risk-fill');
  fill.style.width = `${data.percent}%`;
  fill.classList.toggle('high', high);
  document.getElementById('threshold-note').textContent = `Fixed threshold ${percent(data.threshold, 1)} · Training cross-validation Youden J`;
  document.getElementById('baseline-number').textContent = percent(data.baseline);
  document.getElementById('missing-number').textContent = `${data.missing_count} / 30`;
  document.getElementById('waterfall-image').src = data.waterfall_png;
  const tbody = document.getElementById('all-contributions');
  tbody.replaceChildren();
  for (const item of data.all_contributions) {
    const tr = document.createElement('tr');
    tr.append(makeTextCell(item.name), makeTextCell(item.value.toFixed(3)),
              makeTextCell(`${item.shap_probability >= 0 ? '+' : ''}${percent(item.shap_probability, 2)}`,
                           item.shap_probability >= 0 ? 'positive' : 'negative'));
    tbody.append(tr);
  }
}

function showError(message) {
  const box = document.getElementById('error-box');
  box.textContent = message; box.hidden = false;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!schema || !validateInputs()) return;
  button.disabled = true; statusEl.textContent = 'Calculating prediction and SHAP explanation…';
  document.getElementById('error-box').hidden = true;
  try {
    const response = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(collectValues())});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');
    renderResult(data); statusEl.textContent = 'Calculation complete';
    document.getElementById('results').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { showError(error.message); statusEl.textContent = ''; }
  finally { button.disabled = false; }
});

fetch('/api/schema').then(r => r.json()).then(renderSchema).catch(() => showError('The model could not be loaded. Check that the local server is running.'));
