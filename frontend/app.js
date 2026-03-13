const el = (id) => document.getElementById(id);
let _lastStatus = { running: null, last_finished_at: null, last_exit_code: null };
let _isSyncingLatest = false;

const log = (msg) => {
  const now = new Date().toLocaleTimeString();
  el('logBox').textContent = `[${now}] ${msg}\n` + el('logBox').textContent;
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function renderStatus(data) {
  el('statusText').textContent = data.running ? 'Running' : 'Idle';
  el('exitCode').textContent = data.last_exit_code ?? '-';
  el('lastError').textContent = data.last_error || '-';
  setRunningUI(Boolean(data.running));
}

function setRunningUI(isRunning) {
  const loadingCard = el('loadingCard');
  const runBtn = el('runBtn');
  const latestBtn = el('latestBtn');
  const hint = el('loadingHint');

  if (loadingCard) {
    loadingCard.style.display = isRunning ? 'block' : 'none';
  }

  if (runBtn) {
    runBtn.disabled = isRunning;
    runBtn.style.opacity = isRunning ? '0.65' : '1';
    runBtn.style.cursor = isRunning ? 'not-allowed' : 'pointer';
    runBtn.textContent = isRunning ? 'Running...' : 'Run Forecast';
  }

  if (latestBtn) {
    latestBtn.disabled = isRunning;
    latestBtn.style.opacity = isRunning ? '0.65' : '1';
    latestBtn.style.cursor = isRunning ? 'not-allowed' : 'pointer';
  }

  if (hint) {
    hint.textContent = isRunning
      ? 'Computing Prophet + XGBoost + GARCH + Monte Carlo. Auto-updates when finished.'
      : 'Latest results loaded automatically after run completion.';
  }
}

function formatMaybeNumber(value, decimals = 4) {
  if (value === null || value === undefined || value === '') return '-';
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(decimals) : String(value);
}

function renderSummary(data) {
  const grid = el('summaryGrid');
  grid.innerHTML = '';

  const stats = data.stats || {};
  const entries = [
    ['Rows', data.rows],
    ['Start Date', data.start_date],
    ['End Date', data.end_date],
    ['Downloads', `${(data.csvList || []).map((c) => `<a href="${c}" target="_blank" style="color:var(--accent)">${c.split('/').pop()}</a>`).join('<br>')}`],
  ];

  entries.forEach(([k, v]) => {
    const div = document.createElement('div');
    div.className = 'metric';
    div.innerHTML = `<strong>${k}</strong><br>${v}`;
    grid.appendChild(div);
  });

  el('lastRow').textContent = JSON.stringify(data.last_row, null, 2);
  renderModelStats(stats);
  renderPlotlyChart();
}

function renderModelStats(stats) {
  const grid = el('modelStatsGrid');
  if (!grid) return;

  grid.innerHTML = '';
  if (!stats || !Object.keys(stats).length) {
    const div = document.createElement('div');
    div.className = 'metric';
    div.innerHTML = '<strong>Status</strong><br>No statistics available yet. Run forecast first.';
    grid.appendChild(div);
    return;
  }

  const entries = [
    ['Directional Accuracy', formatMaybeNumber(stats.directional_accuracy, 4)],
    ['p-value', formatMaybeNumber(stats.p_value, 6)],
    ['Significant', stats.significant ? 'Yes' : 'No'],
    ['Monte Carlo Runs', stats.monte_carlo_runs ?? '-'],
    ['95% CI (Target)', `${formatMaybeNumber(stats.ci95_low_target, 2)} → ${formatMaybeNumber(stats.ci95_high_target, 2)}`],
    ['MAP Cutoff', stats.map_cutoff ?? '-'],
    ['MAP Posterior', formatMaybeNumber(stats.map_posterior, 6)],
    ['Ensemble Members', stats.ensemble_members ?? '-'],
    ['Regime', `${stats.regime ?? '-'} ${stats.regime_emoji ?? ''}`.trim()],
    ['Residual Test MSE', formatMaybeNumber(stats.residual_test_mse, 6)],
    ['Forecast 30D', formatMaybeNumber(stats.forecast_30d, 2)],
    ['Forecast Target', formatMaybeNumber(stats.forecast_target, 2)],
  ];

  entries.forEach(([k, v]) => {
    const div = document.createElement('div');
    div.className = 'metric';
    div.innerHTML = `<strong>${k}</strong><br>${v}`;
    grid.appendChild(div);
  });

  const cutoffPosteriors = Array.isArray(stats.cutoff_posteriors) ? [...stats.cutoff_posteriors] : [];
  if (cutoffPosteriors.length) {
    cutoffPosteriors.sort((a, b) => Number(b.posterior || 0) - Number(a.posterior || 0));
    const topRows = cutoffPosteriors.slice(0, 8)
      .map((row, index) => `${index + 1}. ${row.cutoff} | acc=${formatMaybeNumber(row.accuracy, 4)} | posterior=${formatMaybeNumber(row.posterior, 6)}`)
      .join('<br>');

    const basis = document.createElement('div');
    basis.className = 'metric wide';
    basis.innerHTML =
      `<strong>Cutoff Selection Basis</strong><br>` +
      `Selected cutoff (MAP): <b>${stats.map_cutoff ?? '-'}</b><br>` +
      `Scoring objective: directional accuracy on validation slice (Bayesian weighted)<br>` +
      `Grid frequency: ${stats.cutoff_freq ?? '-'} | Min train days: ${stats.min_cutoff_train_days ?? '-'} | Evaluated: ${stats.cutoff_candidates_evaluated ?? cutoffPosteriors.length}<br><br>` +
      `<strong>Top cutoff rankings</strong><br>${topRows}`;
    grid.appendChild(basis);
  }
}

function renderArtifacts(data) {
  const list = el('artifactsList');
  list.innerHTML = '';
  if (!data.files?.length) {
    list.innerHTML = '<li>No artifacts yet.</li>';
    return;
  }
  data.files.forEach((f) => {
    const li = document.createElement('li');
    const link = f.url ? `<a href="${f.url}" target="_blank" style="color:var(--accent)">${f.name}</a>` : f.name;
    li.innerHTML = `${link} (${f.size_bytes} bytes)`;
    list.appendChild(li);
  });
}

async function refreshArtifacts() {
  try {
    const artifacts = await api('/artifacts');
    renderArtifacts(artifacts);
  } catch (err) {
    log(`Artifacts error: ${err.message}`);
  }
}

async function syncLatestAndArtifacts(reason = '') {
  if (_isSyncingLatest) return;
  _isSyncingLatest = true;
  try {
    const loaded = await loadLatest();
    await refreshArtifacts();
    if (loaded && reason) {
      log(`Auto-updated latest (${reason}).`);
    }
  } finally {
    _isSyncingLatest = false;
  }
}

async function fetchCSV(url) {
  try {
    const res = await fetch(`${url}?t=${Date.now()}`);
    if (!res.ok) return null;
    const text = await res.text();
    const lines = text.trim().split('\n');
    if (lines.length < 2) return null;

    const headers = lines[0].split(',').map((h) => h.trim());
    return lines.slice(1).map((line) => {
      const values = line.split(',');
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] ? values[index].trim() : null;
      });
      return row;
    });
  } catch (_) {
    return null;
  }
}

async function renderPlotlyChart() {
  const history = await fetchCSV('/out/historical_prices.csv');
  const forecast = await fetchCSV('/out/nextgen_hybrid_forecast_results_montecarlo.csv');

  if (!history || !forecast || !history.length || !forecast.length) {
    el('plotlyChart').style.display = 'none';
    return;
  }

  const histDate = history.map((d) => d.date);
  const histPrice = history.map((d) => Number(d.close));

  const fcDate = forecast.map((d) => d.date);
  const fcPrice = forecast.map((d) => Number(d.combined_price));
  const fcLow = forecast.map((d) => Number(d.combined_low_mc));
  const fcHigh = forecast.map((d) => Number(d.combined_high_mc));
  const fcProphet = forecast.map((d) => Number(d.prophet_baseline));

  const lastHistDate = histDate[histDate.length - 1];
  const lastHistPrice = histPrice[histPrice.length - 1];
  const connectedFcDate = [lastHistDate, ...fcDate];
  const connectedHybrid = [lastHistPrice, ...fcPrice];
  const connectedProphet = [lastHistPrice, ...fcProphet];
  const connectedHigh = [lastHistPrice, ...fcHigh];
  const connectedLow = [lastHistPrice, ...fcLow];

  const traceHist = {
    x: histDate,
    y: histPrice,
    type: 'scatter',
    mode: 'lines',
    name: 'Historical (365d)',
    line: { color: '#6d8dff', width: 2 },
  };

  const traceProphet = {
    x: connectedFcDate,
    y: connectedProphet,
    type: 'scatter',
    mode: 'lines',
    name: 'Prophet Baseline',
    line: { color: '#ffa94d', width: 2, dash: 'dot' },
  };

  const traceHybrid = {
    x: connectedFcDate,
    y: connectedHybrid,
    type: 'scatter',
    mode: 'lines',
    name: 'Hybrid (Prophet + XGB)',
    line: { color: '#ff6b6b', width: 2 },
  };

  const traceCI = {
    x: connectedFcDate.concat(connectedFcDate.slice().reverse()),
    y: connectedHigh.concat(connectedLow.slice().reverse()),
    fill: 'toself',
    fillColor: 'rgba(255, 107, 107, 0.15)',
    line: { color: 'transparent' },
    name: '95% MC + GARCH CI',
    showlegend: true,
    type: 'scatter',
  };

  const layout = {
    title: 'BTC-USD Forecast (Bayesian + Hybrid + GARCH)',
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#e8ecff' },
    xaxis: { gridcolor: '#2a3350' },
    yaxis: { gridcolor: '#2a3350', tickprefix: '$' },
    hovermode: 'x unified',
    margin: { l: 60, r: 20, t: 50, b: 40 },
    legend: { orientation: 'h', y: -0.2 },
  };

  el('plotlyChart').style.display = 'block';
  Plotly.newPlot('plotlyChart', [traceCI, traceHist, traceProphet, traceHybrid], layout, { responsive: true });
}

async function refreshStatus() {
  try {
    const data = await api('/status');
    renderStatus(data);

    const justFinished = _lastStatus.running === true && data.running === false;
    const finishedAtChanged =
      data.last_finished_at &&
      _lastStatus.last_finished_at &&
      data.last_finished_at !== _lastStatus.last_finished_at;
    const newExitCode =
      data.last_exit_code !== null &&
      data.last_exit_code !== undefined &&
      data.last_exit_code !== _lastStatus.last_exit_code;

    if (justFinished || finishedAtChanged || (data.running === false && newExitCode)) {
      await syncLatestAndArtifacts('run finished');
    }

    _lastStatus = {
      running: data.running,
      last_finished_at: data.last_finished_at,
      last_exit_code: data.last_exit_code,
    };
  } catch (err) {
    log(`Status error: ${err.message}`);
  }
}

async function loadLatest() {
  try {
    const data = await api('/latest');
    renderSummary(data);
    log('Loaded latest forecast summary.');
    return true;
  } catch (err) {
    log(`Latest error: ${err.message}`);
    const statusCode = String(err.message || '').trim().split(' ')[0];
    if (statusCode === '404') {
      return false;
    }
  }

  return false;
}

async function runForecast() {
  const payload = {
    ticker: el('ticker').value,
    horizon_days: Number(el('horizon').value),
    test_last_days: Number(el('testdays').value),
    monte_carlo_runs: Number(el('mc').value),
    bayesian_temperature: Number(el('temp').value),
    max_lag: Number(el('maxlag').value),
    random_state: Number(el('seed').value),
  };

  try {
    setRunningUI(true);
    await api('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    log('Forecast job started.');
    await refreshStatus();
    return true;
  } catch (err) {
    setRunningUI(false);
    log(`Run error: ${err.message}`);
    return false;
  }
}

async function waitForRunToFinish(maxWaitMs = 25 * 60 * 1000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    const state = await api('/status');
    renderStatus(state);
    if (!state.running) {
      return state;
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
  throw new Error('Auto-run timeout waiting for forecast completion.');
}

async function autoBootstrap() {
  const hasLatest = await loadLatest();
  if (hasLatest) {
    await refreshArtifacts();
    return;
  }

  log('No latest forecast found. Auto-starting forecast...');
  const started = await runForecast();
  if (!started) {
    throw new Error('Unable to auto-start forecast.');
  }
  await waitForRunToFinish();
  await loadLatest();
  await refreshArtifacts();
  log('Auto-run finished and latest results loaded.');
}

el('runBtn').addEventListener('click', runForecast);
el('refreshBtn').addEventListener('click', refreshStatus);
el('latestBtn').addEventListener('click', loadLatest);

refreshStatus();
autoBootstrap().catch((err) => log(`Auto bootstrap error: ${err.message}`));
setInterval(refreshStatus, 5000);
