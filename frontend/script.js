const $ = (selector) => document.querySelector(selector);
// Live Server uses a separate port from Flask; Flask serves the API on 5000.
const API = location.port && location.port !== '5000' ? 'http://127.0.0.1:5000/api' : '/api';
let suggestions = [];
let recommendations = {synergistic:[], not_synergistic:[]};
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const display = value => value == null || !Number.isFinite(Number(value)) ? 'N/A' : Number(value).toFixed(4);

$('#theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  localStorage.setItem('synergy-theme', dark ? 'dark' : 'light');
});
document.documentElement.dataset.theme = localStorage.getItem('synergy-theme') || 'light';

async function loadOptions() {
  const select = $('#cell-line');
  try {
    let response = await fetch(`${API}/options`);
    if (!response.ok) throw new Error('Options API is unavailable');
    let data = await response.json();
    if (!data.cell_lines?.length) throw new Error('No cell lines returned by API');
    suggestions = data.suggestions || [];
    recommendations = data.recommendations || recommendations;
    fillCellLines(data.cell_lines);
  } catch (apiError) {
    // Keep the cell-line selector usable when opened through a static server.
    try {
      const response = await fetch('data/options.json');
      if (!response.ok) throw new Error('Local options file could not be loaded');
      const data = await response.json();
      suggestions = data.suggestions || [];
      recommendations = data.recommendations || recommendations;
      if (!data.cell_lines?.length) throw new Error('No cell lines in local options file');
      fillCellLines(data.cell_lines);
    } catch (fallbackError) {
      select.innerHTML = '<option value="">Could not load cell lines — start Flask or use the project frontend path</option>';
      throw fallbackError;
    }
  }
  bindSuggestions('drug1', 'suggestions1'); bindSuggestions('drug2', 'suggestions2');
  renderRecommendations();
}
function fillCellLines(cellLines) {
  $('#cell-line').innerHTML = '<option value="">Choose a cell line</option>' + cellLines.map(x => `<option>${escapeHtml(x)}</option>`).join('');
}
function bindSuggestions(inputId, boxId) {
  const input = document.getElementById(inputId), box = document.getElementById(boxId);
  input.addEventListener('input', () => {
    const query = input.value.trim().toLowerCase();
    const matches = suggestions.filter(x => x.name.toLowerCase().includes(query)).slice(0, 5);
    box.innerHTML = query && matches.length ? matches.map(x => `<button type="button" class="suggestion" data-name="${escapeHtml(x.name)}">${escapeHtml(x.name)} <small>${escapeHtml(x.dataset)} · score ${display(x.score)}</small></button>`).join('') : '';
  });
  box.addEventListener('click', event => {
    const choice = event.target.closest('.suggestion');
    if (!choice) return;
    input.value = choice.dataset.name;
    box.innerHTML = '';
  });
}
function renderRecommendations() {
  const groups = $('#recommendation-groups');
  const renderGroup = (key, title, style) => {
    const items = recommendations[key] || [];
    return `<section class="recommendation-group ${style}"><h3>${title}</h3><div class="recommendation-list">${items.map((item, index) => `<button type="button" class="recommendation" data-category="${key}" data-index="${index}" title="${escapeHtml(item.drug1)} + ${escapeHtml(item.drug2)}"><b>${escapeHtml(item.drug1)} + ${escapeHtml(item.drug2)}</b><small>Model score ${display(item.score)} · ${escapeHtml(item.cell_line)} · ${escapeHtml(item.dataset)} — click to use</small></button>`).join('') || '<p class="notice">No examples available.</p>'}</div></section>`;
  };
  groups.innerHTML = renderGroup('synergistic', 'Synergistic examples', 'positive') + renderGroup('not_synergistic', 'Non-synergistic examples', 'negative');
  groups.addEventListener('click', event => {
    const button = event.target.closest('.recommendation');
    if (!button) return;
    const item = recommendations[button.dataset.category]?.[Number(button.dataset.index)];
    if (!item) return;
    $('#drug1').value = item.drug1;
    $('#drug2').value = item.drug2;
    $('#cell-line').value = item.cell_line;
    $('#predict-form').scrollIntoView({behavior:'smooth', block:'center'});
  });
}
$('#predict-form').addEventListener('submit', async event => {
  event.preventDefault(); const result = $('#prediction-result');
  result.className = 'result'; result.textContent = 'Submitting…';
  try {
    const drug1 = $('#drug1').value, drug2 = $('#drug2').value;
    const response = await fetch(`${API}/predict`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({drug1, drug2, drug1_smiles:drug1, drug2_smiles:drug2, cell_line:$('#cell-line').value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction request failed.');
    const synergistic = Number(data.score) > 10;
    const usedNameLookup = data.drug1_resolved || data.drug2_resolved;
    const individualScores = Object.entries(data.scores || {}).map(([family, value]) => `<div class="family-score"><span>${escapeHtml(family)}</span><b>${display(value)}</b></div>`).join('');
    const sourceScores = (data.score_sources || []).map(item => `${escapeHtml(item.dataset)}: ${display(item.score)}`).join(' · ');
    const observedNote = data.observed_score == null
      ? 'No matching recorded pair/cell-line score; the dataset was selected by precedence and model coverage.'
      : `Recorded score selected by dataset precedence: ${escapeHtml(data.observed_dataset)} ${display(data.observed_score)}${sourceScores ? ` · Other sources: ${sourceScores}` : ''}`;
    result.innerHTML = `<div class="result-main"><div><span class="eyebrow">FINAL PREDICTED SYNERGY SCORE</span><div class="result-score">${display(data.score)}</div><span class="result-unit">${escapeHtml(data.selected_model)} · ${escapeHtml(data.dataset)} dataset</span></div><span class="classification ${synergistic ? 'positive' : 'negative'}">${escapeHtml(data.classification)}</span></div><div class="score-breakdown">${individualScores}</div><p class="result-reference">${observedNote}</p><div class="result-details"><span><b>Drug 1</b>${escapeHtml(data.drug1_input || data.drug1_smiles)}</span><span><b>Drug 2</b>${escapeHtml(data.drug2_input || data.drug2_smiles)}</span><span><b>Cell line</b>${escapeHtml(data.cell_line)}</span></div>${usedNameLookup ? '<small class="result-model">Drug name resolved to a molecular structure through PubChem.</small><br>' : ''}<small class="result-model">Final score uses the feature family with the lowest single-holdout RMSE for the selected dataset.</small>`;
  } catch (error) { result.className = 'result error'; result.textContent = error.message; }
});
async function loadMetrics() {
  let data;
  try {
    const response = await fetch(`${API}/dashboard`);
    if (!response.ok) throw new Error('Dashboard API unavailable');
    data = await response.json();
  } catch (apiError) {
    // The evaluation report is bundled so Live Server can display it too.
    const response = await fetch('data/metrics.json');
    if (!response.ok) throw new Error('Could not load the evaluation report');
    data = await response.json();
  }
  const rows = data.metrics || [];
  const datasets = [...new Set(rows.map(r => r.dataset))].sort();
  const models = [...new Set(rows.map(r => r.model))].sort();
  const rankTotals = Object.fromEntries(models.map(name => [name, 0]));
  const rankGroups = new Map();
  rows.forEach(row => {
    const key = `${row.dataset}|${row.feature_family}`;
    if (!rankGroups.has(key)) rankGroups.set(key, []);
    rankGroups.get(key).push(row);
  });
  let bestModelWins = 0;
  for (const group of rankGroups.values()) {
    const ordered = [...group].sort((a,b) => Number(a.rmse_mean) - Number(b.rmse_mean));
    ordered.forEach((row,index) => { rankTotals[row.model] += index + 1; });
    const groupBest = ordered[0];
    if (groupBest) rankTotals[`__wins_${groupBest.model}`] = (rankTotals[`__wins_${groupBest.model}`] || 0) + 1;
  }
  const bestModel = models.reduce((best,name) => !best || rankTotals[name] < rankTotals[best] ? name : best, '');
  const bestRank = rankGroups.size ? rankTotals[bestModel] / rankGroups.size : 0;
  bestModelWins = rankTotals[`__wins_${bestModel}`] || 0;
  const xgbByDataset = new Map();
  rows.filter(row => row.model === 'XGBoost').forEach(row => {
    if (!xgbByDataset.has(row.dataset)) xgbByDataset.set(row.dataset, {});
    xgbByDataset.get(row.dataset)[row.feature_family] = row.rmse_mean;
  });
  const xgbCompared = [...xgbByDataset.values()].filter(group => Number.isFinite(Number(group.DRS)) && Number.isFinite(Number(group.DS)) && Number.isFinite(Number(group.Structure)));
  const drsVsDsWins = xgbCompared.filter(group => Number(group.DRS) < Number(group.DS)).length;
  const drsVsStructureWins = xgbCompared.filter(group => Number(group.DRS) < Number(group.Structure)).length;
  $('#best-card').innerHTML = `<div class="best-icon">≋</div><div><span class="eyebrow">BEST COMPARISON MODEL · EVALUATED ${rows.length} COMBINATIONS</span><h3>${escapeHtml(bestModel || 'No model results')}</h3><small>Mean within-dataset RMSE rank ${bestRank.toFixed(2)}; lowest RMSE in ${bestModelWins} of ${rankGroups.size} dataset × feature-family comparisons. For XGBoost, DRS beats DS in ${drsVsDsWins}/${xgbCompared.length} datasets and Structure in ${drsVsStructureWins}/${xgbCompared.length}. ${escapeHtml(data.evaluation?.protocol || '')} ${escapeHtml(data.evaluation?.limitations || '')}</small></div>`;
  $('#dataset-filter').innerHTML = '<option value="all">All datasets</option>' + datasets.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
  $('#model-filter').innerHTML = '<option value="all">All models</option>' + models.map(name => `<option value="${escapeHtml(name)}" ${name === bestModel ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('');
  const drawChart = () => {
    const dataset = $('#dataset-filter').value;
    const model = $('#model-filter').value;
    const selected = rows.filter(r => (dataset === 'all' || r.dataset === dataset) && (model === 'all' || r.model === model));
    const metric = $('#metric-filter').value;
    const value = metricValue => display(metricValue);
    $('#metrics-body').innerHTML = selected.map(r => `<tr><td>${escapeHtml(r.dataset)}</td><td>${escapeHtml(r.feature_family)}</td><td>${escapeHtml(r.model)}</td><td>${value(r.mse_mean)}</td><td>${value(r.rmse_mean)}</td><td>${value(r.mae_mean)}</td><td>${value(r.r2_mean)}</td><td>${value(r.explainedvariance_mean)}</td><td>${value(r.pearson_mean)}</td><td>${value(r.spearman_mean)}</td><td>${value(r.auc_mean)}</td><td>${r.test_rows}</td></tr>`).join('') || '<tr><td colspan="12">No recorded evaluation metrics found for this selection.</td></tr>';
    $('#row-count').textContent = `(${selected.length})`;
    $('#comparison-chart').hidden = dataset === 'all';
    $('#chart-note').hidden = dataset !== 'all';
    if (dataset === 'all') {
      $('#chart-note').textContent = 'Choose one dataset to plot comparable scores. The complete table below lists results for all five datasets.';
      $('#comparison-chart').innerHTML = '';
      return;
    }
    const colors = {Structure:'#8797a5', DS:'#20a68a', DRS:'#267f73'};
    const values = selected.map(r => Number(r[metric])).filter(Number.isFinite);
    if (!values.length) { $('#comparison-chart').innerHTML = '<p class="notice">No values available for this metric.</p>'; return; }
    let minimum = Math.min(0, ...values), maximum = Math.max(0, ...values);
    if (minimum === maximum) maximum = minimum + 1;
    const width = 1150, height = 300, left = 64, right = 20, top = 14, bottom = 100;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const y = value => top + (maximum - value) / (maximum - minimum) * plotHeight;
    const zeroY = y(0), step = (maximum - minimum) / 4;
    const grid = Array.from({length:5}, (_, i) => {
      const tick = minimum + step * i, yPos = y(tick);
      return `<line x1="${left}" y1="${yPos}" x2="${width-right}" y2="${yPos}" stroke="#dfe7e3"/><text x="${left-9}" y="${yPos+4}" text-anchor="end" class="chart-tick">${Number(tick.toPrecision(3))}</text>`;
    }).join('');
    const slot = plotWidth / selected.length, barWidth = Math.min(44, slot * 0.64);
    const bars = selected.map((row, i) => {
      const value = Number(row[metric]);
      if (!Number.isFinite(value)) return '';
      const x = left + slot * i + (slot - barWidth) / 2, valueY = y(value);
      const barY = Math.min(zeroY, valueY), barHeight = Math.max(1, Math.abs(zeroY - valueY));
      const label = `${row.model} · ${row.feature_family}`;
      const title = `${label}: ${value.toFixed(4)}`;
      return `<g><title>${escapeHtml(title)}</title><rect x="${x}" y="${barY}" width="${barWidth}" height="${barHeight}" rx="4" fill="${colors[row.feature_family] || '#267f73'}"/><text transform="translate(${x+barWidth/2},${height-bottom+18}) rotate(-38)" text-anchor="end" class="chart-label">${escapeHtml(label)}</text></g>`;
    }).join('');
    const metricLabel = $('#metric-filter').selectedOptions[0]?.textContent || metric;
    $('#comparison-chart').innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metricLabel)} scores for ${escapeHtml(dataset)}"><text transform="translate(16 ${top+plotHeight/2}) rotate(-90)" text-anchor="middle" class="chart-axis-title">${escapeHtml(metricLabel)}</text>${grid}${bars}</svg>`;
  };
  $('#dataset-filter').addEventListener('change', drawChart);
  $('#model-filter').addEventListener('change', drawChart);
  $('#metric-filter').addEventListener('change', drawChart);
  drawChart();
}
Promise.all([loadOptions(),loadMetrics()]).catch(error => { console.error(error); $('#best-card').textContent = 'Could not load evaluation data. Check that the project data files are available.'; $('#metrics-body').innerHTML = '<tr><td colspan="12">Evaluation data could not be loaded.</td></tr>'; });
