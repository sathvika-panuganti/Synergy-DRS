const $ = (selector) => document.querySelector(selector);
// Live Server uses a separate port from Flask; Flask serves the API on 5000.
const API = location.port && location.port !== '5000' ? 'http://127.0.0.1:5000/api' : '/api';
const FINAL_FAMILY = 'DRS';
const FAMILY_COLORS = {Structure: '#8797a5', DS: '#20a68a', [FINAL_FAMILY]: '#b4530a'};
const FAMILY_ORDER = ['Structure', 'DS', FINAL_FAMILY];

let suggestions = [];
let recommendations = {synergistic: [], not_synergistic: []};

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const display = value => value == null || !Number.isFinite(Number(value)) ? 'N/A' : Number(value).toFixed(4);
const shorten = (value, limit = 28) => String(value).length > limit ? `${String(value).slice(0, limit - 1)}…` : String(value);
const drugLabel = (item, side) => item[`drug${side}_label`] || shorten(item[`drug${side}`]);

$('#theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  localStorage.setItem('synergy-theme', dark ? 'dark' : 'light');
});
document.documentElement.dataset.theme = localStorage.getItem('synergy-theme') || 'light';

/* ------------------------------------------------------------------ */
/* Options, suggestions, and pooled combination examples               */
/* ------------------------------------------------------------------ */

async function loadOptions() {
  const select = $('#cell-line');
  const apply = data => {
    suggestions = data.suggestions || [];
    recommendations = data.recommendations || recommendations;
    if (!data.cell_lines?.length) throw new Error('No cell lines available');
    fillCellLines(data.cell_lines);
  };
  try {
    const response = await fetch(`${API}/options`);
    if (!response.ok) throw new Error('Options API is unavailable');
    apply(await response.json());
  } catch (apiError) {
    // Keep the cell-line selector usable when opened through a static server.
    try {
      const response = await fetch('data/options.json');
      if (!response.ok) throw new Error('Local options file could not be loaded');
      apply(await response.json());
    } catch (fallbackError) {
      select.innerHTML = '<option value="">Could not load cell lines — start Flask or use the project frontend path</option>';
      throw fallbackError;
    }
  }
  bindSuggestions('drug1', 'suggestions1');
  bindSuggestions('drug2', 'suggestions2');
  bindRecommendations();
  renderRecommendations();
}

function fillCellLines(cellLines) {
  $('#cell-line').innerHTML = '<option value="">Choose a cell line</option>' +
    cellLines.map(name => `<option>${escapeHtml(name)}</option>`).join('');
}

function bindSuggestions(inputId, boxId) {
  const input = document.getElementById(inputId), box = document.getElementById(boxId);
  input.addEventListener('input', () => {
    const query = input.value.trim().toLowerCase();
    const matches = suggestions.filter(item =>
      item.name.toLowerCase().includes(query) || (item.label || '').toLowerCase().includes(query)).slice(0, 6);
    box.innerHTML = query && matches.length ? matches.map(item =>
      `<button type="button" class="suggestion" data-name="${escapeHtml(item.name)}">${escapeHtml(item.label || shorten(item.name, 36))}<small>${escapeHtml(shorten(item.name, 44))}</small></button>`).join('') : '';
  });
  box.addEventListener('click', event => {
    const choice = event.target.closest('.suggestion');
    if (!choice) return;
    input.value = choice.dataset.name;
    box.innerHTML = '';
  });
}

function matchesQuery(item, query) {
  if (!query) return true;
  return [item.drug1, item.drug2, item.drug1_label, item.drug2_label, item.cell_line]
    .filter(Boolean).some(value => String(value).toLowerCase().includes(query));
}

function renderRecommendations() {
  const groups = $('#recommendation-groups');
  const query = ($('#recommendation-search')?.value || '').trim().toLowerCase();
  let shown = 0;
  const renderGroup = (key, title, style) => {
    const items = (recommendations[key] || []).filter(item => matchesQuery(item, query));
    shown += items.length;
    const buttons = items.map(item => {
      const index = (recommendations[key] || []).indexOf(item);
      return `<button type="button" class="recommendation" data-category="${key}" data-index="${index}" title="${escapeHtml(item.drug1)} + ${escapeHtml(item.drug2)} · ${escapeHtml(item.cell_line)}"><b>${escapeHtml(drugLabel(item, 1))} + ${escapeHtml(drugLabel(item, 2))}</b><small>Cell line ${escapeHtml(item.cell_line)} — click to use</small></button>`;
    }).join('');
    return `<section class="recommendation-group ${style}"><h3>${title}</h3><div class="recommendation-list">${buttons || '<p class="notice">No examples match this filter.</p>'}</div></section>`;
  };
  groups.innerHTML = renderGroup('synergistic', 'Synergistic examples', 'positive') +
    renderGroup('not_synergistic', 'Non-synergistic examples', 'negative');
  const total = (recommendations.synergistic || []).length + (recommendations.not_synergistic || []).length;
  const counter = $('#recommendation-count');
  if (counter) counter.textContent = `${shown} of ${total} pooled examples shown`;
}

function bindRecommendations() {
  $('#recommendation-search')?.addEventListener('input', renderRecommendations);
  $('#recommendation-groups').addEventListener('click', event => {
    const button = event.target.closest('.recommendation');
    if (!button) return;
    const item = recommendations[button.dataset.category]?.[Number(button.dataset.index)];
    if (!item) return;
    $('#drug1').value = item.drug1;
    $('#drug2').value = item.drug2;
    $('#cell-line').value = item.cell_line;
    $('#predict-form').scrollIntoView({behavior: 'smooth', block: 'center'});
  });
}

/* ------------------------------------------------------------------ */
/* Prediction                                                          */
/* ------------------------------------------------------------------ */

$('#predict-form').addEventListener('submit', async event => {
  event.preventDefault();
  const result = $('#prediction-result');
  const button = $('#predict-form button[type="submit"]');
  result.className = 'result';
  result.innerHTML = '<span class="spinner" aria-hidden="true"></span> Scoring the combination…';
  button.disabled = true;
  try {
    const drug1 = $('#drug1').value, drug2 = $('#drug2').value;
    const response = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({drug1, drug2, cell_line: $('#cell-line').value}),
    });
    let data;
    try {
      data = await response.json();
    } catch (parseError) {
      throw new Error(`The server returned an unreadable response (HTTP ${response.status}).`);
    }
    if (!response.ok) throw new Error(data.error || 'Prediction request failed.');
    renderPrediction(result, data);
  } catch (error) {
    result.className = 'result error';
    result.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

function renderPrediction(result, data) {
  const synergistic = Number(data.score) > 10;
  const usedNameLookup = data.drug1_resolved || data.drug2_resolved;
  const families = data.family_scores || [];
  const cards = families.map(item => `
    <div class="model-score ${item.final ? 'final' : ''}" style="--family-color:${FAMILY_COLORS[item.family] || '#267f73'}">
      <span class="model-score-name">${escapeHtml(item.family)}${item.final ? '<em>FINAL MODEL</em>' : ''}</span>
      <b>${display(item.score)}</b>
    </div>`).join('');
  result.innerHTML = `
    <div class="result-main">
      <div>
        <span class="eyebrow">FINAL PREDICTED SYNERGY SCORE</span>
        <div class="result-score">${display(data.score)}</div>
        <span class="result-unit">${escapeHtml(data.selected_model || 'XGBoost · DRS')}</span>
      </div>
      <span class="classification ${synergistic ? 'positive' : 'negative'}">${escapeHtml(data.classification)}</span>
    </div>
    <div class="result-details">
      <span><b>Drug 1</b>${escapeHtml(shorten(data.drug1_input || data.drug1_smiles, 40))}</span>
      <span><b>Drug 2</b>${escapeHtml(shorten(data.drug2_input || data.drug2_smiles, 40))}</span>
      <span><b>Cell line</b>${escapeHtml(data.cell_line)}</span>
    </div>
    <div class="model-comparison">
      <span class="eyebrow">FEATURE-FAMILY MODEL SCORES FOR THIS COMBINATION</span>
      ${families.length ? `<div class="model-score-grid">${cards}</div>`
        : `<p class="notice">This Flask process is running an older backend: it returned no
           per-family scores. Stop it and run <code>python backend/app.py</code> again to see the
           Structure, DS and DRS comparison.</p>`}
    </div>
    ${usedNameLookup ? '<small class="result-model">Drug name resolved to a molecular structure through PubChem.</small><br>' : ''}
    <small class="result-model">The final score comes from the Drug Resistance Signature (DRS) model, the paper\u2019s method.</small>`;
}

/* ------------------------------------------------------------------ */
/* SVG chart helpers                                                   */
/* ------------------------------------------------------------------ */

const svg = (width, height, label, body) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${escapeHtml(label)}">${body}</svg>`;

function axisTicks(minimum, maximum, toY, left, right, count = 5) {
  const step = (maximum - minimum) / (count - 1);
  return Array.from({length: count}, (unused, index) => {
    const value = minimum + step * index, y = toY(value);
    return `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" class="chart-gridline"/>` +
      `<text x="${left - 8}" y="${y + 4}" text-anchor="end" class="chart-tick">${Number(value.toPrecision(3))}</text>`;
  }).join('');
}

function legend(entries, x, y) {
  return entries.map(([name, color], index) =>
    `<g transform="translate(${x + index * 150} ${y})"><rect width="11" height="11" rx="2" fill="${color}"/><text x="17" y="10" class="chart-label-small">${escapeHtml(name)}</text></g>`).join('');
}

/* ------------------------------------------------------------------ */
/* Evaluation dashboard                                                */
/* ------------------------------------------------------------------ */

function normaliseReportRow(row) {
  const rmse = Number(row.RMSE);
  return {
    dataset: row.dataset,
    feature_family: row.feature_family,
    model: row.model,
    final: row.feature_family === FINAL_FAMILY,
    mse_mean: Number.isFinite(Number(row.MSE)) ? Number(row.MSE) : (Number.isFinite(rmse) ? rmse * rmse : null),
    rmse_mean: rmse,
    mae_mean: Number(row.MAE),
    r2_mean: Number(row.R2),
    pearson_mean: Number(row.Pearson),
    spearman_mean: Number(row.Spearman),
    n_features: row.n_features,
  };
}

function mergeRows(data) {
  const reportRows = (data.report?.metrics || []).map(normaliseReportRow);
  // XGBoost numbers all come from the signature-model run so every XGBoost row
  // shares one pipeline; the other algorithms stay from the earlier comparison.
  const legacy = (data.metrics || [])
    .filter(row => !(reportRows.length && row.model === 'XGBoost'))
    .map(row => ({...row, final: false}));
  return [...reportRows, ...legacy];
}

async function loadMetrics() {
  let data;
  try {
    const response = await fetch(`${API}/dashboard`);
    if (!response.ok) throw new Error('Dashboard API unavailable');
    data = await response.json();
  } catch (apiError) {
    // The evaluation report is bundled so Live Server can display it too.
    const [metrics, report] = await Promise.all([
      fetch('data/metrics.json').then(response => response.ok ? response.json() : null),
      fetch('data/model_report.json').then(response => response.ok ? response.json() : null).catch(() => null),
    ]);
    if (!metrics) throw new Error('Could not load the evaluation report');
    data = {...metrics, report};
  }

  const rows = mergeRows(data);
  const report = data.report;
  renderSummaryCard(rows, data, report);
  renderComparisonTable(rows, data);
  if (report) renderResearchFigures(report);
  else document.querySelector('.figure-grid')?.remove();
}

function renderSummaryCard(rows, data, report) {
  const finalRows = rows.filter(row => row.final);
  if (!finalRows.length) {
    // Say what is actually missing instead of printing "0 datasets" and "N/A".
    $('#best-card').innerHTML = `<div class="best-icon">≋</div><div>
        <span class="eyebrow">DRS MODEL REPORT NOT LOADED</span>
        <h3>Showing the earlier feature-family comparison only</h3>
        <small>The DRS model report was not returned by the server. Train the model with
        <code>python backend/services/combined_feature_training.py</code> and restart Flask, so
        <code>frontend/data/model_report.json</code> is served by <code>/api/dashboard</code>.
        ${escapeHtml(data.evaluation?.limitations || '')}</small>
      </div>`;
    return;
  }
  const byDataset = new Map();
  rows.filter(row => row.model === 'XGBoost').forEach(row => {
    if (!byDataset.has(row.dataset)) byDataset.set(row.dataset, {});
    byDataset.get(row.dataset)[row.feature_family] = row;
  });
  let drsWins = 0, compared = 0;
  for (const group of byDataset.values()) {
    const values = FAMILY_ORDER.map(family => group[family]?.rmse_mean).filter(Number.isFinite);
    if (values.length !== FAMILY_ORDER.length) continue;
    compared += 1;
    if (group[FINAL_FAMILY].rmse_mean <= Math.min(...values)) drsWins += 1;
  }
  const meanR2 = finalRows.length
    ? finalRows.reduce((total, row) => total + (Number(row.r2_mean) || 0), 0) / finalRows.length : NaN;
  const share = report ? Object.values(report.diagnostics || {})
    .map(entry => {
      // The trainer reports the DRS gain split across the drug-1 and drug-2
      // halves, so add them back together for the headline share.
      const byGroup = entry.importance?.group_share || {};
      const signature = Object.entries(byGroup)
        .filter(([name]) => name.endsWith('signature'))
        .reduce((total, [, value]) => total + value, 0);
      return signature || undefined;
    }).filter(Number.isFinite) : [];
  const meanDrsShare = share.length ? share.reduce((a, b) => a + b, 0) / share.length : NaN;
  $('#best-card').innerHTML = `<div class="best-icon">≋</div><div>
      <span class="eyebrow">FINAL MODEL · XGBOOST ON DRS</span>
      <h3>${finalRows.length} dataset models · mean held-out R² ${Number.isFinite(meanR2) ? meanR2.toFixed(3) : 'N/A'}</h3>
      <small>Lowest RMSE of the three feature families in ${drsWins} of ${compared} datasets.
      ${Number.isFinite(meanDrsShare) ? `The two drug signature blocks carry ${(meanDrsShare * 100).toFixed(0)}% of the model's gain-based importance on average. ` : ''}
      ${escapeHtml(report?.protocol || data.evaluation?.protocol || '')} ${escapeHtml(data.evaluation?.limitations || '')}</small>
    </div>`;
}

function renderComparisonTable(rows, data) {
  const datasets = [...new Set(rows.map(row => row.dataset))].sort();
  const models = [...new Set(rows.map(row => row.model))].sort();
  $('#dataset-filter').innerHTML = '<option value="all">All datasets</option>' +
    datasets.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
  $('#model-filter').innerHTML = '<option value="all">All models</option>' +
    models.map(name => `<option value="${escapeHtml(name)}" ${name === 'XGBoost' ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('');

  const familyRank = family => {
    const index = FAMILY_ORDER.indexOf(family);
    return index === -1 ? FAMILY_ORDER.length : index;
  };
  const draw = () => {
    // Fall back to "all" so the table still fills if a selector has no value yet.
    const dataset = $('#dataset-filter').value || 'all', model = $('#model-filter').value || 'all';
    const metric = $('#metric-filter').value || 'rmse_mean';
    const selected = rows
      .filter(row => (dataset === 'all' || row.dataset === dataset) && (model === 'all' || row.model === model))
      .sort((a, b) => a.dataset.localeCompare(b.dataset) || a.model.localeCompare(b.model) ||
        familyRank(a.feature_family) - familyRank(b.feature_family));

    $('#metrics-body').innerHTML = selected.map(row => `<tr class="${row.final ? 'final-row' : ''}">
        <td>${escapeHtml(row.dataset)}</td>
        <td>${escapeHtml(row.feature_family)}${row.final ? ' <span class="tag final">final model</span>' : ''}</td>
        <td>${escapeHtml(row.model)}</td>
        <td>${display(row.mse_mean)}</td><td>${display(row.rmse_mean)}</td><td>${display(row.mae_mean)}</td>
        <td>${display(row.r2_mean)}</td><td>${display(row.pearson_mean)}</td><td>${display(row.spearman_mean)}</td>
      </tr>`).join('') || '<tr><td colspan="9">No recorded evaluation metrics found for this selection.</td></tr>';
    $('#row-count').textContent = `(${selected.length})`;

    $('#comparison-chart').hidden = dataset === 'all';
    $('#chart-note').hidden = dataset !== 'all';
    if (dataset === 'all') {
      $('#chart-note').textContent = 'Choose one dataset to plot comparable scores. The table below lists all five datasets.';
      $('#comparison-chart').innerHTML = '';
      return;
    }
    const values = selected.map(row => Number(row[metric])).filter(Number.isFinite);
    if (!values.length) {
      $('#comparison-chart').innerHTML = '<p class="notice">No values available for this metric.</p>';
      return;
    }
    const width = 1150, height = 320, left = 70, right = 20, top = 16, bottom = 112;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    let minimum = Math.min(0, ...values), maximum = Math.max(0, ...values);
    if (minimum === maximum) maximum = minimum + 1;
    const toY = value => top + (maximum - value) / (maximum - minimum) * plotHeight;
    const zeroY = toY(0);
    const slot = plotWidth / selected.length, barWidth = Math.min(46, slot * 0.64);
    const bars = selected.map((row, index) => {
      const value = Number(row[metric]);
      if (!Number.isFinite(value)) return '';
      const x = left + slot * index + (slot - barWidth) / 2, valueY = toY(value);
      const label = `${row.model} · ${row.feature_family}`;
      return `<g><title>${escapeHtml(`${label}: ${value.toFixed(4)}`)}</title>
        <rect x="${x}" y="${Math.min(zeroY, valueY)}" width="${barWidth}" height="${Math.max(1, Math.abs(zeroY - valueY))}" rx="4" fill="${FAMILY_COLORS[row.feature_family] || '#267f73'}" ${row.final ? 'stroke="#1f2937" stroke-width="1.5"' : ''}/>
        <text transform="translate(${x + barWidth / 2},${height - bottom + 18}) rotate(-38)" text-anchor="end" class="chart-label">${escapeHtml(label)}</text></g>`;
    }).join('');
    const metricLabel = $('#metric-filter').selectedOptions[0]?.textContent || metric;
    $('#comparison-chart').innerHTML = svg(width, height, `${metricLabel} scores for ${dataset}`,
      `<text transform="translate(16 ${top + plotHeight / 2}) rotate(-90)" text-anchor="middle" class="chart-axis-title">${escapeHtml(metricLabel)}</text>
       ${axisTicks(minimum, maximum, toY, left, width - right)}${bars}`);
  };
  $('#dataset-filter').addEventListener('change', draw);
  $('#model-filter').addEventListener('change', draw);
  $('#metric-filter').addEventListener('change', draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* Research-style figures from the DRS model run                       */
/* ------------------------------------------------------------------ */

function renderResearchFigures(report) {
  const datasets = [...new Set(report.metrics.map(row => row.dataset))];
  const select = $('#family-dataset');
  select.innerHTML = datasets.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
  const draw = () => {
    const dataset = select.value || datasets[0];
    drawFamilyChart(report, dataset);
    drawScatter(report, dataset);
    drawResiduals(report, dataset);
  };
  select.addEventListener('change', draw);
  draw();
  drawDatasetChart(report);
}

function drawFamilyChart(report, dataset) {
  const rows = FAMILY_ORDER
    .map(family => report.metrics.find(row => row.dataset === dataset && row.feature_family === family))
    .filter(Boolean);
  if (!rows.length) return;
  const metrics = [['MAE', row => row.MAE], ['RMSE', row => row.RMSE], ['R²', row => row.R2]];
  const width = 760, height = 330, left = 62, right = 16, top = 20, bottom = 74;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const panelWidth = plotWidth / metrics.length;
  let body = '';
  metrics.forEach(([name, read], panelIndex) => {
    const values = rows.map(read).map(Number).filter(Number.isFinite);
    const maximum = Math.max(0, ...values) * 1.15 || 1;
    const minimum = Math.min(0, ...values);
    const panelLeft = left + panelWidth * panelIndex;
    const toY = value => top + (maximum - value) / (maximum - minimum) * plotHeight;
    const slot = (panelWidth - 24) / rows.length, barWidth = Math.min(34, slot * 0.7);
    body += `<text x="${panelLeft + panelWidth / 2 - 12}" y="${height - bottom + 34}" text-anchor="middle" class="chart-axis-title">${escapeHtml(name)}</text>`;
    body += `<line x1="${panelLeft}" y1="${toY(Math.max(0, minimum))}" x2="${panelLeft + panelWidth - 24}" y2="${toY(Math.max(0, minimum))}" class="chart-axis"/>`;
    rows.forEach((row, index) => {
      const value = Number(read(row));
      if (!Number.isFinite(value)) return;
      const x = panelLeft + slot * index + (slot - barWidth) / 2;
      const valueY = toY(value), zeroY = toY(0);
      body += `<g><title>${escapeHtml(`${row.feature_family} ${name}: ${value.toFixed(4)}`)}</title>
        <rect x="${x}" y="${Math.min(zeroY, valueY)}" width="${barWidth}" height="${Math.max(1, Math.abs(zeroY - valueY))}" rx="3" fill="${FAMILY_COLORS[row.feature_family]}"/>
        <text x="${x + barWidth / 2}" y="${Math.min(zeroY, valueY) - 5}" text-anchor="middle" class="chart-value">${value.toFixed(2)}</text></g>`;
    });
  });
  $('#family-chart').innerHTML = svg(width, height, `Feature family comparison for ${dataset}`,
    `${body}${legend(FAMILY_ORDER.map(family => [family, FAMILY_COLORS[family]]), left - 40, height - 22)}`);
}

function drawScatter(report, dataset) {
  const points = report.diagnostics?.[dataset]?.scatter || [];
  if (!points.length) {
    $('#scatter-chart').innerHTML = '<p class="notice">No held-out predictions recorded.</p>';
    return;
  }
  const width = 760, height = 360, left = 64, right = 20, top = 20, bottom = 56;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const all = points.flat();
  let minimum = Math.min(...all), maximum = Math.max(...all);
  const pad = (maximum - minimum) * 0.05 || 1;
  minimum -= pad; maximum += pad;
  const toX = value => left + (value - minimum) / (maximum - minimum) * plotWidth;
  const toY = value => top + (maximum - value) / (maximum - minimum) * plotHeight;
  const dots = points.map(([actual, predicted]) =>
    `<circle cx="${toX(actual).toFixed(1)}" cy="${toY(predicted).toFixed(1)}" r="2.4" fill="${FAMILY_COLORS[FINAL_FAMILY]}" fill-opacity="0.38"/>`).join('');
  const xTicks = Array.from({length: 5}, (unused, index) => {
    const value = minimum + (maximum - minimum) * index / 4;
    return `<text x="${toX(value)}" y="${height - bottom + 20}" text-anchor="middle" class="chart-tick">${Number(value.toPrecision(3))}</text>`;
  }).join('');
  $('#scatter-chart').innerHTML = svg(width, height, `Actual versus predicted synergy for ${dataset}`,
    `${axisTicks(minimum, maximum, toY, left, width - right)}
     <line x1="${toX(minimum)}" y1="${toY(minimum)}" x2="${toX(maximum)}" y2="${toY(maximum)}" class="chart-reference"/>
     ${dots}${xTicks}
     <text x="${left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" class="chart-axis-title">Actual synergy score</text>
     <text transform="translate(14 ${top + plotHeight / 2}) rotate(-90)" text-anchor="middle" class="chart-axis-title">Predicted synergy score</text>
     <text x="${width - right - 6}" y="${top + 14}" text-anchor="end" class="chart-label-small">${points.length} held-out pairs · dashed line = ideal</text>`);
}

function drawResiduals(report, dataset) {
  const residuals = report.diagnostics?.[dataset]?.residuals;
  if (!residuals?.counts?.length) {
    $('#residual-chart').innerHTML = '<p class="notice">No residual distribution recorded.</p>';
    return;
  }
  const {counts, edges, mean, std} = residuals;
  const width = 760, height = 330, left = 64, right = 20, top = 20, bottom = 60;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const maximum = Math.max(...counts) || 1;
  const toY = value => top + (maximum - value) / maximum * plotHeight;
  const toX = value => left + (value - edges[0]) / (edges[edges.length - 1] - edges[0]) * plotWidth;
  const barWidth = plotWidth / counts.length;
  const bars = counts.map((count, index) =>
    `<g><title>${escapeHtml(`${edges[index].toFixed(2)} to ${edges[index + 1].toFixed(2)}: ${count} pairs`)}</title>
      <rect x="${(left + barWidth * index).toFixed(1)}" y="${toY(count).toFixed(1)}" width="${(barWidth - 1).toFixed(1)}" height="${(plotHeight + top - toY(count)).toFixed(1)}" fill="${FAMILY_COLORS[FINAL_FAMILY]}" fill-opacity="0.75"/></g>`).join('');
  const zeroX = toX(0);
  const xTicks = Array.from({length: 5}, (unused, index) => {
    const value = edges[0] + (edges[edges.length - 1] - edges[0]) * index / 4;
    return `<text x="${toX(value)}" y="${height - bottom + 20}" text-anchor="middle" class="chart-tick">${Number(value.toPrecision(3))}</text>`;
  }).join('');
  $('#residual-chart').innerHTML = svg(width, height, `Prediction error distribution for ${dataset}`,
    `${axisTicks(0, maximum, toY, left, width - right)}${bars}
     <line x1="${zeroX}" y1="${top}" x2="${zeroX}" y2="${top + plotHeight}" class="chart-reference"/>
     ${xTicks}
     <text x="${left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" class="chart-axis-title">Prediction error (predicted − actual)</text>
     <text transform="translate(14 ${top + plotHeight / 2}) rotate(-90)" text-anchor="middle" class="chart-axis-title">Held-out pairs</text>
     <text x="${width - right - 6}" y="${top + 14}" text-anchor="end" class="chart-label-small">mean ${mean.toFixed(2)} · SD ${std.toFixed(2)}</text>`);
}

function drawDatasetChart(report) {
  const datasets = [...new Set(report.metrics.map(row => row.dataset))];
  const width = 1100, height = 340, left = 64, right = 20, top = 26, bottom = 74;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const values = report.metrics.map(row => Number(row.R2)).filter(Number.isFinite);
  const maximum = Math.max(0.05, ...values) * 1.18;
  const minimum = Math.min(0, ...values);
  const toY = value => top + (maximum - value) / (maximum - minimum) * plotHeight;
  const slot = plotWidth / datasets.length, barWidth = Math.min(28, (slot - 16) / FAMILY_ORDER.length);
  let body = '';
  datasets.forEach((dataset, datasetIndex) => {
    const base = left + slot * datasetIndex + (slot - barWidth * FAMILY_ORDER.length) / 2;
    FAMILY_ORDER.forEach((family, familyIndex) => {
      const row = report.metrics.find(item => item.dataset === dataset && item.feature_family === family);
      const value = Number(row?.R2);
      if (!Number.isFinite(value)) return;
      const x = base + barWidth * familyIndex, valueY = toY(value), zeroY = toY(0);
      body += `<g><title>${escapeHtml(`${dataset} · ${family}: R² ${value.toFixed(4)}`)}</title>
        <rect x="${x}" y="${Math.min(zeroY, valueY)}" width="${barWidth - 2}" height="${Math.max(1, Math.abs(zeroY - valueY))}" rx="3" fill="${FAMILY_COLORS[family]}"/></g>`;
    });
    body += `<text x="${left + slot * datasetIndex + slot / 2}" y="${height - bottom + 22}" text-anchor="middle" class="chart-label-small">${escapeHtml(dataset)}</text>`;
  });
  $('#dataset-chart').innerHTML = svg(width, height, 'Held-out R² per feature family across datasets',
    `<text transform="translate(16 ${top + plotHeight / 2}) rotate(-90)" text-anchor="middle" class="chart-axis-title">Held-out R²</text>
     ${axisTicks(minimum, maximum, toY, left, width - right)}${body}
     ${legend(FAMILY_ORDER.map(family => [family, FAMILY_COLORS[family]]), left, height - 16)}`);
}

Promise.all([loadOptions(), loadMetrics()]).catch(error => {
  console.error(error);
  $('#best-card').textContent = 'Could not load evaluation data. Check that the project data files are available.';
  $('#metrics-body').innerHTML = '<tr><td colspan="12">Evaluation data could not be loaded.</td></tr>';
});
