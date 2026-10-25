"use strict";

const MODEL_LABELS = {
  PriceFM_Plus: "PriceFM+", TabPFN_TS_3: "TabPFN-TS 3", TiRex_2: "TiRex-2",
  TimesFM_3: "TimesFM-3", Chronos_2: "Chronos-2", MLP: "MLP", LSTM: "LSTM",
  Transformer: "Transformer", Persistence_1: "Persistence 1",
  Persistence_2: "Persistence 2", Persistence_3: "Persistence 3",
};
const COLORS = ["#386649", "#778A4F", "#8DCB20", "#BAF343", "#F9FD17",
  "#FCBC19", "#FA7B15", "#FF5357", "#626A6E", "#92999D", "#BDC2C5"];
const DEFAULT = new Set(["PriceFM_Plus", "TabPFN_TS_3", "TiRex_2", "TimesFM_3", "Chronos_2"]);
const $ = id => document.getElementById(id);
let manifest;
const cache = new Map();

function formatNumber(value) { return value.toFixed(2); }

async function loadResults() {
  const response = await fetch("data/results.csv");
  if (!response.ok) throw new Error("Results data unavailable");
  const lines = (await response.text()).trim().split(/\r?\n/).slice(1);
  const rows = lines.map(line => {
    const [model, feature_case, Pinball, Interval, MAE, RMSE] = line.split(",");
    return { model, feature_case, values: [Pinball, Interval, MAE, RMSE].map(Number) };
  });
  const lookup = new Map(rows.map(row => [`${row.model}/${row.feature_case}`, row.values]));
  const cases = ["Realistic", "TSO", "Oracle"];
  const best = new Map();
  for (const featureCase of cases) for (let metric = 0; metric < 4; metric++) {
    const sorted = rows.filter(r => r.feature_case === featureCase).map(r => r.values[metric]).sort((a, b) => a - b);
    best.set(`${featureCase}/${metric}`, [sorted[0], sorted[1]]);
  }
  const order = ["Persistence_1", "Persistence_2", "Persistence_3", "MLP", "LSTM", "Transformer",
    "TabPFN_TS_3", "TiRex_2", "TimesFM_3", "Chronos_2", "PriceFM_Plus"];
  for (const featureCase of cases) {
    const table = $(`results-${featureCase.toLowerCase()}`).querySelector("tbody");
    for (const model of order) {
      const tr = document.createElement("tr");
      if (["MLP", "TabPFN_TS_3", "PriceFM_Plus"].includes(model)) tr.className = "sep";
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = MODEL_LABELS[model];
      tr.appendChild(name);
      const values = lookup.get(`${model}/${featureCase}`);
      if (!values) throw new Error(`Missing results row: ${model}/${featureCase}`);
      values.forEach((v, metric) => {
        const td = document.createElement("td");
        td.textContent = formatNumber(v);
        const rank = best.get(`${featureCase}/${metric}`);
        if (v === rank[0]) td.classList.add("best");
        else if (v === rank[1]) td.classList.add("second");
        tr.appendChild(td);
      });
      if (model === order[order.length - 1]) tr.classList.add("rule");
      table.appendChild(tr);
    }
  }
}

function selectedModels() {
  return [...document.querySelectorAll("#models input:checked")].map(input => Number(input.value));
}

function makeModelControls() {
  const box = $("models");
  manifest.models.forEach((model, index) => {
    const label = document.createElement("label");
    label.className = "chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = index;
    input.checked = DEFAULT.has(model);
    input.addEventListener("change", () => render().catch(showError));
    const span = document.createElement("span");
    span.textContent = MODEL_LABELS[model];
    label.append(input, span);
    box.append(label);
  });
}

function setLegend(selected) {
  const box = $("key");
  box.replaceChildren();
  const truth = document.createElement("span");
  truth.innerHTML = '<i class="swatch" style="background:#111"></i>Observed price';
  box.append(truth);
  selected.forEach(index => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = COLORS[index];
    item.append(swatch, MODEL_LABELS[manifest.models[index]]);
    box.append(item);
  });
}

async function loadShard(caseName, day) {
  const month = day.slice(0, 7);
  const key = `${caseName}/${month}`;
  const meta = manifest.shards[key];
  if (!meta || !meta.dates.includes(day)) throw new Error(`No saved test forecasts for ${day} (${caseName}).`);
  if (cache.has(key)) return cache.get(key);
  $("status").textContent = `Loading ${caseName} forecasts for ${month}…`;
  const response = await fetch(meta.file);
  if (!response.ok) throw new Error(`Forecast file unavailable: ${meta.file}`);
  const payload = await response.arrayBuffer();
  const header = new Uint8Array(payload, 0, Math.min(2, payload.byteLength));
  const isGzip = header[0] === 0x1f && header[1] === 0x8b;
  if (isGzip && !window.DecompressionStream) throw new Error("This browser needs gzip DecompressionStream support.");
  const buffer = isGzip
    ? await new Response(new Blob([payload]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()
    : payload;
  const data = meta.dtype === "int32" ? new Int32Array(buffer) : new Int16Array(buffer);
  const days = meta.dates.length;
  const truthLength = days * 40 * 96;
  const expected = truthLength * (1 + manifest.models.length * 3);
  if (data.length !== expected) throw new Error(`Forecast file has ${data.length} values; expected ${expected}.`);
  const stride = manifest.models.length * 3;
  for (let cell = 0; cell < days * 40; cell++) {
    const base = cell * 96;
    for (let slot = 1; slot < 96; slot++) {
      const t = base + slot;
      if (data[t] !== meta.missing && data[t - 1] !== meta.missing) data[t] += data[t - 1];
      const cur = truthLength + t * stride;
      const prev = cur - stride;
      for (let j = 0; j < stride; j++)
        if (data[cur + j] !== meta.missing && data[prev + j] !== meta.missing)
          data[cur + j] += data[prev + j];
    }
  }
  const shard = { data, dates:meta.dates, truthLength, missing:meta.missing };
  cache.set(key, shard);
  while (cache.size > 2) cache.delete(cache.keys().next().value);
  return shard;
}

function value(raw, shard, scale) { return raw === shard.missing ? NaN : raw / scale; }

function holdHourly(values) {
  const present = values.flatMap((v, i) => Number.isFinite(v) ? [i] : []);
  if (present.length !== 24 || present.some((slot, hour) => slot !== hour * 4)) return;
  for (let hour = 0; hour < 24; hour++)
    for (let quarter = 1; quarter < 4; quarter++)
      values[hour * 4 + quarter] = values[hour * 4];
}

function series(shard, dayIndex, regionIndex, modelIndex) {
  const dayBase = (dayIndex * 40 + regionIndex) * 96;
  const truth = new Array(96);
  const quantiles = Array.from({ length: 3 }, () => new Array(96));
  for (let slot = 0; slot < 96; slot++) {
    truth[slot] = value(shard.data[dayBase + slot], shard, manifest.truth_scale);
    if (modelIndex !== null) for (let q = 0; q < 3; q++) {
      const p = shard.truthLength + (dayBase + slot) * manifest.models.length * 3 + modelIndex * 3 + q;
      quantiles[q][slot] = value(shard.data[p], shard, manifest.forecast_scale);
    }
  }
  holdHourly(truth);
  if (modelIndex !== null) quantiles.forEach(holdHourly);
  return { truth, quantiles };
}

function paths(values, x, y) {
  const result = [];
  let run = [];
  const flush = () => { if (run.length > 1) result.push("M" + run.map(([i, v]) => `${x(i)},${y(v)}`).join("L")); run = []; };
  values.forEach((v, i) => {
    if (!Number.isFinite(v)) return;
    if (run.length && i - run[run.length - 1][0] > 4) flush();
    run.push([i, v]);
  });
  flush();
  return result;
}

function bands(lo, hi, x, y) {
  const result = [];
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      const top = run.map(i => `${x(i)},${y(hi[i])}`);
      const bottom = run.slice().reverse().map(i => `${x(i)},${y(lo[i])}`);
      result.push("M" + top.join("L") + "L" + bottom.join("L") + "Z");
    }
    run = [];
  };
  lo.forEach((v, i) => {
    if (!Number.isFinite(v) || !Number.isFinite(hi[i])) return;
    if (run.length && i - run[run.length - 1] > 4) flush();
    run.push(i);
  });
  flush();
  return result;
}

function chart(shard, dayIndex, regionIndex, selected) {
  const region = manifest.regions[regionIndex];
  const base = series(shard, dayIndex, regionIndex, null);
  const models = selected.map(i => ({ index:i, ...series(shard, dayIndex, regionIndex, i) }));
  const available = base.truth.filter(Number.isFinite);
  for (const item of models) for (const q of item.quantiles) available.push(...q.filter(Number.isFinite));
  const min = available.length ? Math.min(...available) : 0;
  const max = available.length ? Math.max(...available) : 100;
  const pad = Math.max(4, (max - min) * .08);
  const low = min - pad, high = max + pad;
  const x = slot => 36 + slot / 95 * 314;
  const y = v => 152 - (v - low) / (high - low) * 139;
  const pieces = [`<line class="axis" x1="36" y1="152" x2="350" y2="152"/>`,
    `<line class="axis" x1="36" y1="13" x2="36" y2="152"/>`,
    `<text x="3" y="17">${Math.round(high)}</text><text x="3" y="153">${Math.round(low)}</text>`];
  for (let hour = 0; hour <= 24; hour += 6) {
    const xx = hour === 24 ? 350 : x(hour * 4);
    pieces.push(`<line class="axis" x1="${xx}" y1="152" x2="${xx}" y2="156"/><text x="${xx - 7}" y="172">${hour}</text>`);
  }
  for (const item of models) {
    const color = COLORS[item.index];
    for (const d of bands(item.quantiles[0], item.quantiles[2], x, y))
      pieces.push(`<path d="${d}" fill="${color}" fill-opacity=".10"/>`);
  }
  for (const item of models) {
    const color = COLORS[item.index];
    for (const d of paths(item.quantiles[1], x, y))
      pieces.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="2" opacity=".95"/>`);
  }
  for (const d of paths(base.truth, x, y)) pieces.push(`<path class="truth" d="${d}"/>`);
  if (models.length && models.every(item => item.quantiles.every(q => q.every(v => !Number.isFinite(v)))))
    pieces.push('<text x="115" y="32">No saved forecast for this day</text>');
  const article = document.createElement("article");
  article.className = "chart";
  const title = document.createElement("h3");
  title.textContent = region;
  article.append(title);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 360 180");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${region}: observed price and selected forecasts over the day`);
  svg.innerHTML = pieces.join("");
  svg.addEventListener("pointermove", event => {
    const rect = svg.getBoundingClientRect();
    const slot = Math.max(0, Math.min(95, Math.round(((event.clientX - rect.left) / rect.width * 360 - 36) / 314 * 95)));
    const hour = String(Math.floor(slot / 4)).padStart(2, "0");
    const minute = String(slot % 4 * 15).padStart(2, "0");
    const fmt = v => Number.isFinite(v) ? v.toFixed(1) : "—";
    const tip = $("tip");
    tip.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = `${region} · ${hour}:${minute} UTC`;
    tip.append(heading, document.createElement("br"), `Observed: ${fmt(base.truth[slot])} €/MWh`);
    for (const item of models) {
      const line = document.createElement("div");
      line.style.color = COLORS[item.index];
      const q = item.quantiles;
      line.textContent = `${MODEL_LABELS[manifest.models[item.index]]}: ${fmt(q[0][slot])} / ${fmt(q[1][slot])} / ${fmt(q[2][slot])}`;
      tip.append(line);
    }
    tip.style.display = "block";
    tip.style.left = `${Math.min(event.clientX + 13, window.innerWidth - 280)}px`;
    tip.style.top = `${Math.min(event.clientY + 13, window.innerHeight - tip.offsetHeight - 8)}px`;
  });
  svg.addEventListener("pointerleave", () => { $("tip").style.display = "none"; });
  article.append(svg);
  return article;
}

async function render() {
  if (!manifest) return;
  const day = $("day").value;
  const caseName = $("case").value;
  const selected = selectedModels();
  setLegend(selected);
  const shard = await loadShard(caseName, day);
  const dayIndex = shard.dates.indexOf(day);
  const grid = $("grid");
  const fragment = document.createDocumentFragment();
  for (let r = 0; r < 40; r++) fragment.append(chart(shard, dayIndex, r, selected));
  grid.replaceChildren(fragment);
  $("status").textContent = `${day} · ${caseName} · 40 regions · ${selected.length} models`;
}

function showError(error) { $("status").textContent = error.message || String(error); console.error(error); }

async function main() {
  loadResults().catch(showError);
  const response = await fetch("data/manifest.json");
  if (!response.ok) throw new Error("Forecast index unavailable");
  manifest = await response.json();
  const days = Object.values(manifest.shards).flatMap(shard => shard.dates).sort();
  $("day").min = days[0];
  $("day").max = days[days.length - 1];
  makeModelControls();
  $("day").addEventListener("change", () => render().catch(showError));
  $("case").addEventListener("change", () => render().catch(showError));
  await render();
}

main().catch(showError);
