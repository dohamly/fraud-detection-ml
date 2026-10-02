  // "" = same-origin requests. Works out of the box when this page is served
  // BY the FastAPI backend itself (uvicorn app.web.backend.main:app), which is
  // the normal way to run this app. If you ever open index.html directly as a
  // file instead (no backend serving it), set this to "http://localhost:8000".
  const API_BASE = "";

  function showTab(name, btn) {
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (name === 'insights') loadInsights();
  }

  let flagged = false;
  function setFlag(v) {
    flagged = v;
    document.getElementById('btnClear').className = 'toggle-btn' + (!v ? ' on-clear' : '');
    document.getElementById('btnFlag').className = 'toggle-btn' + (v ? ' on-flag' : '');
  }

  async function analyze() {
    const btn = document.querySelector('.analyze-btn');
    const status = document.getElementById('apiStatus');
    btn.disabled = true;
    status.textContent = 'Analyzing…';
    status.className = 'api-status';

    const payload = {
      step: parseInt(document.getElementById('step').value) || 1,
      tx_type: document.getElementById('txType').value,
      amount: parseFloat(document.getElementById('amount').value) || 0,
      oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value) || 0,
      newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value) || 0,
      oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value) || 0,
      newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value) || 0,
      isFlaggedFraud: flagged,
    };

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API error ${res.status}`);
      }
      const data = await res.json();
      const pct = data.probability * 100;
      const isFraud = data.prediction === 'FRAUD';

      document.getElementById('pct').textContent = pct.toFixed(1) + '%';
      document.getElementById('pct').style.color = isFraud ? 'var(--red)' : 'var(--green)';
      document.getElementById('gauge').style.borderColor = isFraud ? 'rgba(248,113,113,0.5)' : 'rgba(74,222,128,0.5)';
      document.getElementById('placeholderText').textContent = isFraud ? '🚨 FRAUD' : '✅ LEGITIMATE';
      document.getElementById('infoModel').textContent = data.model;

      const factorsList = document.getElementById('factorsList');
      factorsList.innerHTML = data.top_factors.map((f, i) => {
        const up = f.value > 0;
        return `<div class="info-row"><span class="info-label">${i + 1}. ${f.feature}</span>
          <span style="color:${up ? 'var(--red)' : 'var(--green)'}; font-weight:700;">
          ${up ? '⬆' : '⬇'} ${f.value >= 0 ? '+' : ''}${f.value.toFixed(3)}</span></div>`;
      }).join('');
      document.getElementById('methodNote').textContent = data.method;
      document.getElementById('explainPanel').style.display = 'block';

      status.textContent = '';
    } catch (e) {
      status.textContent = `⚠️ Could not reach the API (${e.message}). Is the backend running on ${API_BASE}?`;
      status.className = 'api-status err';
    } finally {
      btn.disabled = false;
    }
  }

  async function loadModelInfo() {
    try {
      const res = await fetch(`${API_BASE}/model-info`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      document.getElementById('infoModel').textContent = data.model;
      if (data.roc_auc) document.getElementById('infoRocAuc').textContent = data.roc_auc.toFixed(3);
    } catch (e) {
      // Backend not reachable yet — fields keep their placeholder "—".
    }
  }

  let insightsLoaded = false;
  async function loadInsights() {
    if (insightsLoaded) return;
    const list = document.getElementById('insightsList');
    try {
      const res = await fetch(`${API_BASE}/model-info`);
      const data = await res.json();
      if (!data.figures || !data.figures.length) {
        list.innerHTML = `<div class="panel"><div class="insight-placeholder">No figures found in reports/figures/. Run your training script first.</div></div>`;
        return;
      }
      list.innerHTML = data.figures.map(fname => `
        <div class="panel insight-card">
          <div class="insight-title">${fname}</div>
          <div class="insight-placeholder"><img src="${API_BASE}/figures/${fname}" alt="${fname}"
               onerror="this.parentElement.textContent='Could not load ${fname}'"></div>
        </div>`).join('');
      insightsLoaded = true;
    } catch (e) {
      list.innerHTML = `<div class="panel"><div class="insight-placeholder">⚠️ Could not reach the API at ${API_BASE}. Start the backend, then reopen this tab.</div></div>`;
    }
  }

  loadModelInfo();
