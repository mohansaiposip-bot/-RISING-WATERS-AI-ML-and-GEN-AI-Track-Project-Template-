// ---------- Slider readouts ----------

const sliderConfig = {
  annual_rainfall_mm: { suffix: " mm", decimals: 0 },
  seasonal_rainfall_mm: { suffix: " mm", decimals: 0 },
  cloud_visibility_km: { suffix: " km", decimals: 1 },
  humidity_percent: { suffix: "%", decimals: 0 },
  temperature_c: { suffix: "°C", decimals: 1 },
  days_continuous_rainfall: { suffix: "", decimals: 0 },
  river_discharge_cumecs: { suffix: "", decimals: 0 },
};

function refreshReadout(id) {
  const input = document.getElementById(id);
  const out = document.getElementById("out_" + id);
  const cfg = sliderConfig[id];
  const val = parseFloat(input.value).toFixed(cfg.decimals);
  out.textContent = val + cfg.suffix;
}

Object.keys(sliderConfig).forEach((id) => {
  const input = document.getElementById(id);
  input.addEventListener("input", () => refreshReadout(id));
  refreshReadout(id);
});

// ---------- Gauge tick marks (stage gauge, 0-5m) ----------

const gaugeTicks = document.getElementById("gaugeTicks");
for (let m = 5; m >= 0; m--) {
  const tick = document.createElement("div");
  tick.className = "tick";
  tick.innerHTML = `<span class="line"></span><span class="num">${m}m</span>`;
  gaugeTicks.appendChild(tick);
}

// ---------- Presets ----------

const presets = {
  calm: {
    annual_rainfall_mm: 950, seasonal_rainfall_mm: 480, cloud_visibility_km: 8.6,
    humidity_percent: 52, temperature_c: 33, days_continuous_rainfall: 0,
    river_discharge_cumecs: 900,
  },
  monsoon: {
    annual_rainfall_mm: 2200, seasonal_rainfall_mm: 1650, cloud_visibility_km: 3.8,
    humidity_percent: 82, temperature_c: 27, days_continuous_rainfall: 6,
    river_discharge_cumecs: 4800,
  },
  severe: {
    annual_rainfall_mm: 3600, seasonal_rainfall_mm: 3100, cloud_visibility_km: 1.1,
    humidity_percent: 96, temperature_c: 24, days_continuous_rainfall: 13,
    river_discharge_cumecs: 9800,
  },
};

document.querySelectorAll(".preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = presets[btn.dataset.preset];
    Object.entries(preset).forEach(([id, val]) => {
      document.getElementById(id).value = val;
      refreshReadout(id);
    });
  });
});

// ---------- Prediction ----------

const analyzeBtn = document.getElementById("analyzeBtn");
const gaugeFill = document.getElementById("gaugeFill");
const gaugeMarker = document.getElementById("gaugeMarker");
const riskPercent = document.getElementById("riskPercent");
const riskBand = document.getElementById("riskBand");
const advisory = document.getElementById("advisory");

async function runAnalysis() {
  analyzeBtn.classList.add("loading");
  analyzeBtn.querySelector("span").textContent = "Analysing…";

  const payload = { region: document.getElementById("region").value || "Unspecified region" };
  Object.keys(sliderConfig).forEach((id) => {
    payload[id] = parseFloat(document.getElementById(id).value);
  });

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderResult(data);
  } catch (err) {
    riskBand.textContent = "Error";
    advisory.textContent = err.message;
  } finally {
    analyzeBtn.classList.remove("loading");
    analyzeBtn.querySelector("span").textContent = "Run flood risk analysis";
  }
}

function renderResult(data) {
  const pct = data.probability_percent;
  gaugeFill.style.height = Math.max(pct, 3) + "%";
  gaugeMarker.style.bottom = Math.max(pct, 3) + "%";
  gaugeFill.className = "gauge-fill risk-" + data.risk_band;

  riskPercent.textContent = pct.toFixed(1) + "%";
  riskBand.textContent = data.risk_band + " risk — " + (data.flood_predicted ? "flood likely" : "no flood expected");
  riskBand.className = "risk-band " + data.risk_band;
  advisory.textContent = `${data.region}: ${data.advisory}. (Model: ${data.model_used})`;
}

analyzeBtn.addEventListener("click", runAnalysis);
