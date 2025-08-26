// ===== Global =====
let uploadedFiles = [];
let analysisData = null;
let currentBundleHash = null;

// ===== Palette & layout =====
const VIBRANT_CATEGORICAL_PALETTE = [
  '#FF1744','#FF9100','#FFC400','#00E676','#00B0FF',
  '#D500F9','#FF4081','#00FFFF','#FFD600','#FF6D00'
];

const plotlyLayoutConfig = {
  font: { family: 'Poppins, system-ui, -apple-system, Segoe UI, Roboto, sans-serif', color: '#E2E8F0' },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 56, r: 40, b: 56, t: 56 },
  xaxis: { gridcolor: '#2A3444', zerolinecolor: '#2A3444', type: 'category', tickangle: -15 },
  yaxis: { gridcolor: '#2A3444', zerolinecolor: '#2A3444', automargin: true },
  legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
  hoverlabel: { bgcolor: '#0f172a', bordercolor: '#334155', font: { color: '#E2E8F0' } },
  transition: { duration: 450, easing: 'cubic-in-out' }
};

// ===== Utils =====
const hexToRgba = (hex, alpha = 0.25) => {
  const h = hex.replace('#', '');
  const bigint = parseInt(h, 16);
  const r = (bigint >> 16) & 255, g = (bigint >> 8) & 255, b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const _chartColorIndex = {};
let _nextColorIndex = 0;
function getBaseColorIndexForChart(elementId) {
  if (!elementId) return 0;
  if (_chartColorIndex[elementId] === undefined) {
    _chartColorIndex[elementId] = _nextColorIndex % VIBRANT_CATEGORICAL_PALETTE.length;
    _nextColorIndex++;
  }
  return _chartColorIndex[elementId];
}

// Compute bins client-side if needed (legacy)
function computeHistogram(values, binCount = 20) {
  const data = (values || []).filter(v => typeof v === 'number' && isFinite(v));
  if (data.length === 0) return { xs: [], ys: [], labels: [] };
  const min = Math.min(...data), max = Math.max(...data);
  if (min === max) return { xs: [min], ys: [data.length], labels: [`${min.toFixed(2)}`] };
  const step = (max - min) / binCount;
  const edges = Array.from({ length: binCount + 1 }, (_, i) => min + i * step);
  const counts = new Array(binCount).fill(0);
  for (const v of data) {
    let idx = Math.floor((v - min) / step);
    if (idx < 0) idx = 0;
    if (idx >= binCount) idx = binCount - 1;
    counts[idx]++;
  }
  const centers = edges.slice(0, -1).map((e) => e + step / 2);
  const labels = edges.slice(0, -1).map((e, i) => `${e.toFixed(2)}–${(e + step).toFixed(2)}`);
  return { xs: centers, ys: counts, labels };
}

function unwrapResult(payload) {
  return payload && payload.result ? payload.result : payload;
}

// ===== Boot =====
document.addEventListener('DOMContentLoaded', () => {
  initializeUploadArea();
  document.getElementById('analyze-btn')?.addEventListener('click', analyzeData);
  document.getElementById('download-pdf-btn')?.addEventListener('click', downloadPDFReport);
  document.getElementById('savedBtn')?.addEventListener('click', toggleSavedPanel);
  document.getElementById('savedCloseBtn')?.addEventListener('click', toggleSavedPanel);

  window.addEventListener('renderAnalysis', (e) => {
    const result = unwrapResult(e.detail);
    analysisData = result;
    document.getElementById('upload-section').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');
    populateDashboard(result);
  });
});

// ===== Upload =====
function initializeUploadArea() {
  const uploadArea = document.getElementById('upload-area');
  const fileInput = document.getElementById('file-input');
  if (!uploadArea || !fileInput) return;

  uploadArea.addEventListener('click', () => fileInput.click());
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
}

function handleFiles(files) {
  const fileList = document.getElementById('file-list');
  const analyzeBtn = document.getElementById('analyze-btn');

  Array.from(files).forEach(file => {
    const isCSV = file.type === 'text/csv' || file.name.toLowerCase().endsWith('.csv');
    const isXLSX = file.name.toLowerCase().endsWith('.xlsx');
    if ((isCSV || isXLSX) && !uploadedFiles.find(f => f.name === file.name && f.size === file.size)) {
      uploadedFiles.push(file);
    }
  });

  fileList.innerHTML = '';
  uploadedFiles.forEach((file, index) => {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    const icon = file.name.toLowerCase().endsWith('.xlsx') ? 'fa-file-excel' : 'fa-file-csv';
    fileItem.innerHTML = `<span><i class="fas ${icon}"></i> ${file.name}</span>
      <button onclick="removeFile(${index})" class="icon-btn danger" title="Remove">
        <i class="fas fa-times"></i>
      </button>`;
    fileList.appendChild(fileItem);
  });

  analyzeBtn.disabled = uploadedFiles.length < 1;
}

function removeFile(index) {
  uploadedFiles.splice(index, 1);
  handleFiles([]); // re-render
}

// ===== Analyze =====
async function analyzeData() {
  const spinner = document.getElementById('loading-spinner');
  spinner.classList.remove('hidden');
  document.getElementById('dashboard').classList.add('hidden');

  const formData = new FormData();
  uploadedFiles.forEach(file => formData.append('files', file));

  try {
    const response = await fetch('/analyze/', { method: 'POST', body: formData });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
      throw new Error(error.detail);
    }
    const payload = await response.json();
    currentBundleHash = payload?.bundle_hash || null;
    const result = unwrapResult(payload);
    analysisData = result;

    populateDashboard(result);
    document.getElementById('upload-section').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');
  } catch (error) {
    console.error('Analysis error:', error);
    alert(`Analysis failed: ${error.message}`);
  } finally {
    spinner.classList.add('hidden');
  }
}

// ===== Saved panel =====
async function toggleSavedPanel() {
  const panel = document.getElementById('savedPanel');
  const nowHidden = !panel.classList.contains('hidden');
  panel.classList.toggle('hidden');
  panel.setAttribute('aria-hidden', nowHidden ? 'true' : 'false');

  if (!panel.classList.contains('hidden')) {
    await refreshSavedLists();
  }
}

async function refreshSavedLists() {
  const bundlesList = document.getElementById('bundlesList');
  const filesList = document.getElementById('filesList');
  bundlesList.innerHTML = '<li class="muted">Loading…</li>';
  filesList.innerHTML = '<li class="muted">Loading…</li>';

  try {
    const [bRes, fRes] = await Promise.all([fetch('/saved/bundles'), fetch('/saved/files')]);
    const [bundles, files] = await Promise.all([bRes.json(), fRes.json()]);

    // Bundles
    bundlesList.innerHTML = '';
    if (!bundles || bundles.length === 0) {
      bundlesList.innerHTML = '<li class="muted">No saved analyses yet.</li>';
    } else {
      bundles.forEach(b => {
        const li = document.createElement('li');
        li.className = 'saved-item';
        const names = (b.file_names || []).join(', ');
        const time = b.created_at ? new Date(b.created_at).toLocaleString() : '';
        li.innerHTML = `
          <div class="saved-meta">
            <div class="saved-title"><i class="fa-solid fa-chart-line"></i> ${names || '(files)'}</div>
            <div class="saved-sub">${time}</div>
          </div>
          <div class="saved-actions">
            <button class="icon-btn" data-open="${b.bundle_hash}" title="Open analysis">
              <i class="fa-solid fa-arrow-up-right-from-square"></i>
            </button>
          </div>
        `;
        li.querySelector('[data-open]').addEventListener('click', async (e) => {
          const hash = e.currentTarget.getAttribute('data-open');
          const r = await fetch('/saved/bundles/' + hash).then(x => x.json());
          window.dispatchEvent(new CustomEvent('renderAnalysis', { detail: r }));
          document.getElementById('savedPanel').classList.add('hidden');
          document.getElementById('savedPanel').setAttribute('aria-hidden', 'true');
        });
        bundlesList.appendChild(li);
      });
    }

    // Files
    filesList.innerHTML = '';
    if (!files || files.length === 0) {
      filesList.innerHTML = '<li class="muted">No saved files yet.</li>';
    } else {
      files.forEach(f => {
        const li = document.createElement('li');
        li.className = 'saved-item';
        const sizeKB = (f.size / 1024).toFixed(1);
        const time = f.uploaded_at ? new Date(f.uploaded_at).toLocaleString() : '';
        li.innerHTML = `
          <div class="saved-meta">
            <div class="saved-title"><i class="fa-regular fa-file-lines"></i> ${f.original_name}</div>
            <div class="saved-sub">${sizeKB} KB • ${time}</div>
          </div>
          <div class="saved-actions">
            <a class="icon-btn" href="/saved/files/${f.hash}/download" title="Download original">
              <i class="fa-solid fa-download"></i>
            </a>
          </div>
        `;
        filesList.appendChild(li);
      });
    }
  } catch (err) {
    console.error('Saved lists error:', err);
    bundlesList.innerHTML = '<li class="muted">Failed to load saved analyses.</li>';
    filesList.innerHTML = '<li class="muted">Failed to load saved files.</li>';
  }
}

// ===== Dashboard =====
function populateDashboard(data) {
  if (!data) return;
  renderDailySummary(data.daily_summary);

  const dashboardGrid = document.getElementById('dashboard-grid');
  dashboardGrid.innerHTML = '';
  document.getElementById('kpi-grid').innerHTML = '';

  const { charts } = data || {};
  if (!charts) return;

  if (charts.cumulative_profit) {
    createPlotCard('cumulative-profit', 'Cumulative Profit Over Time', 'grid-col-6 min-h-400');
    renderLineChart(charts.cumulative_profit, 'cumulative-profit', { yaxis: { title: 'Profit (Units)' }, hovermode: 'x unified' });
  }
  if (charts.rolling_roi) {
    createPlotCard('rolling-roi', '30-Day Rolling ROI Over Time', 'grid-col-6 min-h-400');
    renderLineChart(charts.rolling_roi, 'rolling-roi', { yaxis: { title: 'ROI (%)' }, hovermode: 'x unified' });
  }
  if (charts.roi_by_tipster) {
    createPlotCard('roi-by-tipster', 'ROI by Tipster', 'grid-col-6 min-h-400');
    renderBarChart(charts.roi_by_tipster, 'roi-by-tipster', { yaxis: { title: 'ROI (%)' }, xaxis: { type: 'category' } });
  }
  if (charts.roi_by_odds) {
    createPlotCard('roi-by-odds', 'ROI by Odds Band', 'grid-col-6 min-h-400');
    renderBarChart(charts.roi_by_odds, 'roi-by-odds', { yaxis: { title: 'ROI (%)' }, xaxis: { type: 'category' } });
  }
  if (charts.price_movement_histogram) {
    createPlotCard('price-movement-histogram', 'Price Movement Distribution', 'grid-col-6 min-h-400');
    renderRainbowHistogram(charts.price_movement_histogram, 'price-movement-histogram', {
      xaxis: { title: 'Price Movement (fraction)', type: 'category' },
      yaxis: { title: 'Frequency' }
    });
  }
  if (charts.clv_trend) {
    createPlotCard('clv-trend', 'CLV Trend Over Time', 'grid-col-6 min-h-400');
    renderLineChart(charts.clv_trend, 'clv-trend', { yaxis: { title: 'CLV (%)' }, hovermode: 'x unified' });
  }
  if (charts.win_rate_vs_field_size) {
    createPlotCard('win-rate-vs-field-size', 'Win Rate vs Field Size', 'grid-col-12 min-h-400');
    renderBarChart(charts.win_rate_vs_field_size, 'win-rate-vs-field-size', { xaxis: { title: 'Field Size', type: 'category' }, yaxis: { title: 'Win Rate (%)' } });
  }
}

// ===== Daily Summary =====
function renderDailySummary(summaryData) {
  const tableContainer = document.getElementById('daily-summary-table');
  if (!summaryData || summaryData.length === 0) { tableContainer.innerHTML = ''; return; }
  let tableHTML = `<h3>Daily Summary</h3><div style="overflow-x:auto;"><table><thead><tr>
      <th>Date</th><th>Bets Placed</th><th>Units Staked</th><th>Units Returned</th><th>ROI %</th>
      <th>Win Rate %</th><th>Avg Odds</th><th>CLV %</th><th>Drifters %</th><th>Steamers %</th>
      </tr></thead><tbody>`;
  summaryData.forEach(day => {
    const ur = Number.isFinite(day['Units Returned']) ? day['Units Returned'] : 0;
    const roi = Number.isFinite(day['ROI %']) ? day['ROI %'] : 0;
    const wr  = Number.isFinite(day['Win Rate %']) ? day['Win Rate %'] : 0;
    const ao  = Number.isFinite(day['Avg Odds']) ? day['Avg Odds'] : 0;
    const clv = Number.isFinite(day['CLV']) ? day['CLV'] : 0;
    const drf = Number.isFinite(day['Drifters %']) ? day['Drifters %'] : 0;
    const stm = Number.isFinite(day['Steamers %']) ? day['Steamers %'] : 0;

    tableHTML += `<tr>
        <td>${day.Date || 'N/A'}</td>
        <td>${day['Bets Placed'] ?? 0}</td>
        <td>${day['Units Staked'] ?? 0}</td>
        <td>${ur.toFixed(2)}</td>
        <td>${roi.toFixed(2)}</td>
        <td>${wr.toFixed(2)}</td>
        <td>${ao.toFixed(2)}</td>
        <td>${clv.toFixed(2)}</td>
        <td>${drf.toFixed(2)}</td>
        <td>${stm.toFixed(2)}</td>
      </tr>`;
  });
  tableHTML += '</tbody></table></div>';
  tableContainer.innerHTML = tableHTML;
}

// ===== Charts =====
function createPlotCard(id, title, gridClass) {
  const card = document.createElement('div');
  card.className = `plot-card ${gridClass}`;
  card.innerHTML = `<h3 class="plot-title">${title}</h3><div id="${id}" style="height: calc(100% - 40px); width: 100%;"></div>`;
  document.getElementById('dashboard-grid').appendChild(card);
}

function renderBarChart(data, elementId, options = {}) {
  if (!data || !data.labels) return;
  const labels = data.labels || [];
  const values = (Array.isArray(data.data) && data.data.length === labels.length)
    ? data.data
    : labels.map(() => 0);
  if (labels.length === 0) return;

  const layout = { ...plotlyLayoutConfig, ...options, title: '', bargap: 0.18, barmode: 'group' };
  const baseIdx = getBaseColorIndexForChart(elementId);
  const barColors = labels.map((_, i) =>
    VIBRANT_CATEGORICAL_PALETTE[(baseIdx + i) % VIBRANT_CATEGORICAL_PALETTE.length]
  );

  const plotData = [{
    x: options.orientation === 'h' ? values : labels,
    y: options.orientation === 'h' ? labels : values,
    type: 'bar',
    orientation: options.orientation || 'v',
    marker: { color: barColors, line: { color: '#0f172a', width: 1.2 }, opacity: 0.95 },
    hovertemplate: options?.yaxis?.title?.includes('%')
      ? '%{y:.2f}%<extra>%{x}</extra>'
      : '%{y:.2f}<extra>%{x}</extra>'
  }];

  Plotly.newPlot(elementId, plotData, layout, { responsive: true, displayModeBar: 'hover' });
}

function renderLineChart(chartData, elementId, options = {}) {
  if (!chartData || !chartData.labels) return;

  let datasets = Array.isArray(chartData.datasets) ? chartData.datasets : [];
  if (datasets.length === 0 && chartData.labels.length > 0) {
    const d = (Array.isArray(chartData.data) && chartData.data.length === chartData.labels.length)
      ? chartData.data
      : chartData.labels.map(() => 0);
    datasets = [{ name: 'Value', data: d }];
  }
  if (chartData.labels.length === 0) return;

  const baseIdx = getBaseColorIndexForChart(elementId);

  const plotData = datasets.map((dataset, i) => {
    const col = VIBRANT_CATEGORICAL_PALETTE[(baseIdx + i) % VIBRANT_CATEGORICAL_PALETTE.length];
    return {
      x: chartData.labels,
      y: dataset.data || [],
      name: dataset.name || `Series ${i+1}`,
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: col, width: 3.25, shape: 'spline', smoothing: 0.7 },
      marker: { color: col, size: 6, line: { color: '#0f172a', width: 1 } },
      fill: 'tozeroy',
      fillcolor: hexToRgba(col, 0.12),
      hovertemplate: '%{y:.2f}<extra>%{fullData.name} — %{x}</extra>'
    };
  });

  const layout = { ...plotlyLayoutConfig, ...options, title: '', hovermode: options.hovermode || 'x unified' };
  layout.xaxis = { ...(layout.xaxis || {}), type: 'category' };
  Plotly.newPlot(elementId, plotData, layout, { responsive: true, displayModeBar: 'hover' });
}

function renderRainbowHistogram(chartData, elementId, options = {}) {
  if (!chartData) return;

  // Prefer server-binned histogram (labels = bins, data = counts)
  let labels = Array.isArray(chartData.labels) ? chartData.labels.slice() : null;
  let counts = Array.isArray(chartData.data) ? chartData.data.slice() : null;

  // Fallback to client-side binning if needed
  if (!labels || !counts || labels.length === 0 || counts.length === 0) {
    const h = computeHistogram(chartData.data || [], 20);
    labels = h.labels;
    counts = h.ys;
  }
  if (!labels || labels.length === 0) return;

  const baseIdx = getBaseColorIndexForChart(elementId);
  const colors = labels.map((_, i) =>
    VIBRANT_CATEGORICAL_PALETTE[(baseIdx + i) % VIBRANT_CATEGORICAL_PALETTE.length]
  );

  const layout = { ...plotlyLayoutConfig, ...options, title: '', bargap: 0.06 };
  const plotData = [{
    x: labels,
    y: counts,
    type: 'bar',
    marker: { color: colors, line: { color: '#0f172a', width: 1.2 }, opacity: 0.95 },
    hovertemplate: 'Count: %{y}<extra>%{x}</extra>'
  }];

  Plotly.newPlot(elementId, plotData, layout, { responsive: true, displayModeBar: 'hover' });
}

// ===== PDF export =====
async function exportChartJPEG(elementId, scale = 1) {
  const el = document.getElementById(elementId);
  if (!el) return null;
  const gd = el.querySelector('.plotly, .js-plotly-plot') || el;
  if (!gd || typeof Plotly?.toImage !== 'function') return null;
  try {
    return await Plotly.toImage(gd, {
      format: 'jpeg',
      scale,
      width: gd.clientWidth || 800,
      height: gd.clientHeight || 450
    });
  } catch {
    return null;
  }
}

async function downloadPDFReport() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'p', unit: 'pt', format: 'a4', compress: true });

  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const margin = 24;
  const maxW = pageW - margin * 2;

  let yPos = margin + 16;

  // COVER
  doc.setFontSize(22);
  doc.text("Betting Insights Performance Report", margin, yPos);
  yPos += 26;
  doc.setFontSize(12);
  doc.text(`Generated: ${new Date().toLocaleString()}`, margin, yPos);

  // New page
  doc.addPage();
  yPos = margin;

  // DAILY SUMMARY TABLE
  const tableElement = document.getElementById("daily-summary-table");
  if (tableElement) {
    const tableCanvas = await html2canvas(tableElement, { scale: 1, backgroundColor: null, useCORS: true, logging: false });
    const tableDataURL = tableCanvas.toDataURL("image/jpeg", 0.75);
    const aspect = tableCanvas.width / tableCanvas.height || (16/9);
    const targetW = maxW;
    const targetH = targetW / aspect;

    if (yPos + targetH > pageH - margin) {
      doc.addPage(); yPos = margin;
    }
    doc.addImage(tableDataURL, 'JPEG', margin, yPos, targetW, targetH);
    yPos += targetH + 18;
  }

  // CHARTS
  const chartIds = [
    "cumulative-profit","rolling-roi","roi-by-tipster","roi-by-odds",
    "price-movement-histogram","clv-trend","win-rate-vs-field-size"
  ];

  for (const id of chartIds) {
    const el = document.getElementById(id);
    if (!el) continue;

    let imgData = await exportChartJPEG(id, 1);
    if (!imgData) {
      const canvas = await html2canvas(el, { scale: 1, backgroundColor: null, useCORS: true, logging: false });
      imgData = canvas.toDataURL('image/jpeg', 0.8);
    }

    const rect = el.getBoundingClientRect();
    const aspect = (rect.width > 0 && rect.height > 0) ? (rect.width / rect.height) : (16/9);
    const targetW = maxW;
    const targetH = targetW / aspect;

    if (yPos + targetH > pageH - margin) {
      doc.addPage(); yPos = margin;
    }
    doc.addImage(imgData, 'JPEG', margin, yPos, targetW, targetH);
    yPos += targetH + 18;
  }

  doc.save(`BettingInsightsReport_${new Date().toISOString().split('T')[0]}.pdf`);
}
