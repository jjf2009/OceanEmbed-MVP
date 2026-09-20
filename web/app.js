const state = { meta: null, dates: [], date: null, depth: 100, marker: null, fieldLayer: null };

const $ = (id) => document.getElementById(id);

function dateRange(start, end) {
  const out = [];
  for (let d = new Date(start); d <= new Date(end); d.setDate(d.getDate() + 1)) {
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
}

const map = L.map("map", { minZoom: 3 });
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "&copy; OpenStreetMap, &copy; CARTO", maxZoom: 12,
}).addTo(map);

// Blue -> red ramp for the predicted temperature field.
function color(v, lo, hi) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const r = Math.round(30 + 210 * t), g = Math.round(90 + 60 * Math.sin(Math.PI * t)), b = Math.round(220 - 180 * t);
  return `rgb(${r},${g},${b})`;
}

async function loadMeta() {
  state.meta = await (await fetch("/meta")).json();
  const r = state.meta.region;
  map.fitBounds([[r.lat_min, r.lon_min], [r.lat_max, r.lon_max]]);
  L.rectangle([[r.lat_min, r.lon_min], [r.lat_max, r.lon_max]],
    { color: "#4fb3d9", weight: 1, fill: false, dashArray: "4 4" }).addTo(map);

  state.dates = dateRange(state.meta.dates.start, state.meta.dates.end);
  const slider = $("dateSlider");
  slider.max = state.dates.length - 1;
  slider.value = state.dates.length - 1;

  $("depthSelect").innerHTML = state.meta.depths
    .map((d) => `<option value="${d}"${d === 100 ? " selected" : ""}>${d} m</option>`).join("");

  if (state.meta.synthetic) $("banner").textContent = "SYNTHETIC DATA — not real GLORYS";
  renderMetrics();
  setDate(state.dates[state.dates.length - 1]);
}

function renderMetrics() {
  const m = state.meta.metrics;
  if (!m) { $("metrics").textContent = "No metrics yet — run evaluate.py."; return; }
  const rows = m.model.map((x, i) =>
    `<tr><td>${x.depth}</td><td>${x.rmse.toFixed(3)}</td>` +
    `<td>${m.climatology[i].rmse.toFixed(3)}</td><td>${x.r2.toFixed(3)}</td></tr>`).join("");
  $("metrics").innerHTML =
    `<p>Test-block skill — model beats climatology at ${m.beats_climatology_at_levels}/${m.model.length} levels.</p>` +
    `<table><tr><th>depth (m)</th><th>model RMSE</th><th>clim RMSE</th><th>R²</th></tr>${rows}</table>`;
}

async function setDate(date) {
  state.date = date;
  $("dateLabel").textContent = date;
  await drawField();
  if (state.marker) await drawProfile(state.marker.getLatLng());
}

async function drawField() {
  const res = await fetch(`/predict/field?date=${state.date}&depth=${state.depth}`);
  const f = await res.json();
  const flat = f.values.flat().filter((v) => v !== null);
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const dLat = (f.lat[1] - f.lat[0]) / 2, dLon = (f.lon[1] - f.lon[0]) / 2;

  if (state.fieldLayer) map.removeLayer(state.fieldLayer);
  const cells = [];
  f.values.forEach((row, i) => row.forEach((v, j) => {
    if (v === null) return;
    cells.push(L.rectangle(
      [[f.lat[i] - dLat, f.lon[j] - dLon], [f.lat[i] + dLat, f.lon[j] + dLon]],
      { stroke: false, fillColor: color(v, lo, hi), fillOpacity: 0.75 }
    ));
  }));
  state.fieldLayer = L.layerGroup(cells).addTo(map);
  $("subtitle").textContent =
    `Predicted temperature at ${f.depth} m on ${f.date} — ${lo.toFixed(2)} to ${hi.toFixed(2)} °C. Click a cell for its profile.`;
}

async function drawProfile(latlng) {
  const res = await fetch(`/predict/profile?lat=${latlng.lat}&lon=${latlng.lng}&date=${state.date}`);
  if (!res.ok) { Plotly.purge("profile"); return; }
  const p = await res.json();
  const upper = p.mu.map((v, i) => v + p.sigma[i]);
  const lower = p.mu.map((v, i) => v - p.sigma[i]);

  Plotly.newPlot("profile", [
    { x: upper.concat([...lower].reverse()), y: p.depths.concat([...p.depths].reverse()),
      fill: "toself", fillcolor: "rgba(79,179,217,0.20)", line: { width: 0 },
      name: "±1σ", hoverinfo: "skip" },
    { x: p.mu, y: p.depths, mode: "lines+markers", name: "predicted",
      line: { color: "#4fb3d9", width: 2 } },
    { x: p.truth, y: p.depths, mode: "lines", name: "GLORYS truth",
      line: { color: "#e0b050", width: 2, dash: "dot" } },
  ], {
    title: `${p.lat.toFixed(2)}°N, ${p.lon.toFixed(2)}°E — ${p.date}`,
    paper_bgcolor: "#16212e", plot_bgcolor: "#16212e",
    font: { color: "#e7eef6" }, margin: { t: 50, r: 20, b: 45, l: 55 },
    xaxis: { title: "temperature (°C)", gridcolor: "#2a3a4d" },
    yaxis: { title: "depth (m)", autorange: "reversed", gridcolor: "#2a3a4d" },
    legend: { orientation: "h", y: -0.15 },
  }, { responsive: true });
}

map.on("click", (e) => {
  if (state.marker) map.removeLayer(state.marker);
  state.marker = L.circleMarker(e.latlng, { radius: 6, color: "#fff", weight: 2, fillOpacity: 0 }).addTo(map);
  drawProfile(e.latlng);
});

$("dateSlider").addEventListener("change", (e) => setDate(state.dates[Number(e.target.value)]));
$("depthSelect").addEventListener("change", (e) => { state.depth = Number(e.target.value); drawField(); });

loadMeta();
