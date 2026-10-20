"use strict";
/* Predicción and Seguimiento screens. Loaded before app.js, which owns the helpers (api, fmt, icon, combo, ...) used here at call time. */

const forecastState = { planta: "", sku: "", mes: "" };
const trackState = { mes: "", origen: "" };
const MODEL_NAMES = { ml_mean: "Modelo principal (promedio)", lgbm: "LightGBM", xgb: "XGBoost", rf: "Random Forest", naive: "Modelo simple (repite el mes anterior)" };
const ORANGE = "#f97316";
const FIXED = new Intl.NumberFormat("es-CL", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const fx = (v) => (v === null || v === undefined ? "–" : FIXED.format(v));
let pollTimer = null;

async function apiPost(path, body, headers = {}) {
  const response = await fetch(`/api/${path}`, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });
  return { ok: response.ok, status: response.status, body: await response.json().catch(() => ({})) };
}

const arrow = (v) => (v === null || v === undefined ? "" : v >= 0 ? "▲ " : "▼ ");
const biasText = (pct) => (pct === null || pct === undefined ? "" : Math.abs(pct) < 0.05 ? "sin sesgo" : pct > 0 ? `el pronóstico quedó ${fmt(Math.abs(pct))}% por debajo` : `el pronóstico quedó ${fmt(Math.abs(pct))}% por arriba`);

function bindPicks(container, error, state, rerender) {
  container.querySelectorAll("[data-pick]").forEach((button) => (button.onclick = () => {
    const field = (error.body.campo || "").includes("planta") ? "planta" : "sku";
    state[field] = button.dataset.pick;
    if (field === "planta") state.sku = "";
    rerender();
  }));
}

/* ---- Predicción ------------------------------------------------------------------------------------------------------------- */
async function renderForecast(view) {
  clearTimeout(pollTimer);
  const [status, list] = await Promise.all([api("estado"), api("pronosticos")]);
  if (!status.datos) { view.innerHTML = '<div class="card empty"><h3>Todavía no hay datos</h3><p>Carga los exportes de SAP en la sección Datos.</p></div>'; return; }
  const next = status.proximo_pronostico;
  const have = list.pronosticos.find((f) => f.mes === next);
  view.innerHTML = `<div class="stack"><div class="card gen" id="gen-card"></div><div id="fc-body"></div></div>`;
  renderGenCard(status, have);
  if (list.pronosticos.length) await renderForecastBody();
  else $("#fc-body").innerHTML = `<div class="card empty"><h3>Todavía no hay pronósticos</h3><p>Genera el de ${monthLong(next)} con el botón de arriba.</p></div>`;
  const running = (await api("trabajos/activo")).trabajo;
  if (running) watchJob(running.id);
}

function renderGenCard(status, have, extra = "") {
  const data = status.datos, next = status.proximo_pronostico;
  const partial = data.mes_parcial;
  const text = partial
    ? `El último mes de los datos (${monthLong(partial)}) está incompleto. Carga el mes completo en la sección Datos para poder pronosticar.`
    : have
      ? `Generado el ${dateLong(have.creado_en)}. Usa los datos hasta ${monthLong(data.ultimo_mes_completo)}.`
      : `Todavía no se genera. Usa los datos hasta ${monthLong(data.ultimo_mes_completo)} y tarda menos de un minuto.`;
  $("#gen-card").innerHTML = `<div><h2>Pronóstico de ${monthLong(next)}</h2><p>${text}</p></div>
    <button class="btn primary" id="gen-btn" ${partial ? "disabled" : ""}>${have ? "Generar de nuevo" : "Generar pronóstico"}</button>
    <div class="grow" id="gen-extra">${extra}</div>`;
  $("#gen-btn").onclick = () => (have ? askOverwrite(status, have) : startGenerate(false, status, have));
}

function askOverwrite(status, have) {
  $("#gen-extra").innerHTML = `<div class="confirm">Ya existe el pronóstico de ${monthLong(have.mes)}. Si lo generas de nuevo, se reemplaza.
    <button class="btn primary" id="yes">Sí, generar de nuevo</button><button class="btn" id="no">Cancelar</button></div>`;
  $("#yes").onclick = () => startGenerate(true, status, have);
  $("#no").onclick = () => ($("#gen-extra").innerHTML = "");
}

async function startGenerate(overwrite, status, have) {
  $("#gen-btn").disabled = true;
  const reply = await apiPost("pronostico/generar", { sobrescribir: overwrite });
  if (!reply.ok) {
    $("#gen-btn").disabled = false;
    $("#gen-extra").innerHTML = `<div class="banner warn">${icon("alert")}<div>${esc(reply.body.detalle || "No se pudo empezar.")}</div></div>`;
    return;
  }
  watchJob(reply.body.trabajo);
}

function watchJob(jobId) {
  const tick = async () => {
    const extra = $("#gen-extra");
    if (!extra) return; // the user left the screen; the task goes on and the screen picks it up when opened again
    const job = await api(`trabajos/${jobId}`);
    if ($("#gen-btn")) $("#gen-btn").disabled = job.estado === "corriendo";
    if (job.estado === "corriendo") {
      extra.innerHTML = `<div class="progress"><div class="bar" style="width:${Math.max(4, Math.round(job.progreso * 100))}%"></div></div><p class="note">${esc(job.mensaje)} · ${fmt(job.segundos, 0)} s</p>`;
      pollTimer = setTimeout(tick, 700);
    } else if (job.estado === "listo") {
      forecastState.mes = job.resultado.mes;
      route();
    } else {
      extra.innerHTML = `<div class="banner warn">${icon("alert")}<div><b>No se pudo generar el pronóstico.</b><br>${esc(job.error)}</div></div>`;
    }
  };
  tick();
}

async function renderForecastBody() {
  const s = forecastState;
  $("#fc-body").innerHTML = `<div class="stack"><div class="card">${filtersHtml(s)}</div><div id="fc-result"></div></div>`;
  bindFilters(s, refreshForecast);
  await refreshForecast();
}

async function refreshForecast() {
  const s = forecastState;
  const result = $("#fc-result");
  if (!result) return;
  result.innerHTML = '<div class="loading">Cargando…</div>';
  let d;
  try {
    d = await api("pronostico", { mes: s.mes, planta: s.planta, sku: s.sku });
  } catch (error) {
    disposeCharts();
    result.innerHTML = notFoundHtml(error);
    bindPicks(result, error, s, () => renderForecastBody());
    return;
  }
  s.mes = d.mes;
  disposeCharts();
  result.innerHTML = forecastHtml(d);
  const element = $("#fc-chart");
  mountChart(element, forecastOption(d, element.clientWidth));
  const select = $("#fc-mes");
  if (select) select.onchange = () => { s.mes = select.value; refreshForecast(); };
  result.querySelectorAll("tr[data-planta]").forEach((row) => (row.onclick = () => { s.planta = row.dataset.planta; s.sku = ""; renderForecastBody(); }));
}

function forecastHtml(d) {
  const f = d.pronostico, last = d.ultimo_real;
  const card = (label, value, sub) => `<div class="card stat" style="min-height:0"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const who = [d.filtro.planta || "Todas las plantas", d.filtro.sku || "todos los SKU"].join(" · ");
  const months = d.disponibles.length > 1 ? `<label class="hint" for="fc-mes">Pronóstico de </label><select class="pick" id="fc-mes">${d.disponibles.map((m) => `<option value="${m}" ${m === d.mes ? "selected" : ""}>${monthLong(m)}</option>`).join("")}</select>` : "";
  const t = d.tabla;
  const head = t.tipo === "planta" ? "<th>Planta</th>" : "<th>Planta</th><th>SKU</th>";
  const rows = t.filas.map((r) => `<tr ${t.tipo === "planta" ? `data-planta="${esc(r.planta)}"` : ""}>${t.tipo === "planta" ? `<td>${esc(r.planta)}</td>` : `<td>${esc(r.planta)}</td><td>${esc(r.sku)}</td>`}<td>${fx(r.pronostico_TO)}</td><td>${fx(r.real_ultimo_TO)}</td><td>${r.cambio_pct === null ? "–" : arrow(r.cambio_pct) + fmt(Math.abs(r.cambio_pct)) + "%"}</td></tr>`).join("");
  const more = t.total_filas > t.filas.length ? `<p class="note">Mostrando los ${t.filas.length} mayores de ${fmt(t.total_filas, 0)}. Filtra por planta o SKU para ver otros.</p>` : "";
  return `<div class="stack"><div class="grid cols-3">
      ${card(`Pronóstico de ${monthLong(d.mes)}`, `${fmt(f.ml_mean_TO, 0)}<small>TO</small>`, "Promedio de los tres modelos (LightGBM, XGBoost y Random Forest).")}
      ${card("Contra el último mes real", last.cambio_pct === null ? "–" : `${arrow(last.cambio_pct)}${fmt(Math.abs(last.cambio_pct))}%`, `${fmt(last.TO, 0)} TO en ${monthLong(last.mes)}.`)}
      ${card("Rango entre modelos", `${fmt(f.minimo_TO, 0)} – ${fmt(f.maximo_TO, 0)}<small>TO</small>`, "Si los modelos se separan mucho, el pronóstico es menos seguro.")}
    </div>
    <div class="card"><div class="toolbar"><div><h2>Consumo real y pronóstico · ${esc(who)}</h2><p class="hint">Toneladas (TO), últimos 24 meses. La escala no empieza en cero. La franja naranja es el rango entre los tres modelos.</p></div><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">${months}<button class="btn" data-png="pronostico">Imagen (PNG)</button></div></div><div class="chart" id="fc-chart"></div></div>
    <div class="card"><h2>${t.tipo === "planta" ? "Pronóstico por planta" : "Pronóstico por SKU"}</h2><p class="hint">${t.tipo === "planta" ? "Toca una planta para ver sus SKU." : "Ordenado de mayor a menor pronóstico."}</p>
      <div class="tablewrap" style="margin-top:12px"><table class="data ${t.tipo === "planta" ? "clickable" : "two-left"}"><thead><tr>${head}<th>Pronóstico (TO)</th><th>Último mes real (TO)</th><th>Cambio</th></tr></thead><tbody>${rows}</tbody></table></div>${more}</div></div>`;
}

function forecastOption(d, width) {
  const f = d.pronostico;
  const cut = d.historia.etiquetas.indexOf(d.mes);
  const labelsAll = cut >= 0 ? d.historia.etiquetas.slice(0, cut) : d.historia.etiquetas;
  const valuesAll = d.historia.valores.slice(0, labelsAll.length);
  const keep = 24;
  const labels = [...labelsAll.slice(-keep), d.mes];
  const real = [...valuesAll.slice(-keep), null];
  let lastIdx = real.length - 2;
  while (lastIdx >= 0 && real[lastIdx] === null) lastIdx--;
  const connector = labels.map((l, i) => {
    if (i === lastIdx) return { value: real[lastIdx], symbol: "none" };
    if (i === labels.length - 1) return { value: f.ml_mean_TO, symbol: "circle", symbolSize: 11, itemStyle: { color: ORANGE, borderColor: SURFACE, borderWidth: 2 },
      label: { show: true, position: "top", color: INK, fontWeight: 600, fontSize: 12, formatter: () => fmt(f.ml_mean_TO, 0) } };
    return null;
  });
  const option = baseOption(labels, monthShort);
  option.legend = { top: 0, left: 8, icon: "circle", itemWidth: 8, itemHeight: 8, itemGap: 16, textStyle: { color: INK2, fontSize: 12 } };
  option.grid.top = 40;
  option.yAxis.scale = true; // a line chart need not start at zero: this keeps the forecast and its range visible
  option.series = [
    { name: "Consumo real", type: "line", data: real, symbol: "circle", symbolSize: 8, showSymbol: false, lineStyle: { width: 2, color: SERIES_COLORS[0], cap: "round", join: "round" },
      itemStyle: { color: SERIES_COLORS[0], borderColor: SURFACE, borderWidth: 2 } },
    { name: "Pronóstico", type: "line", data: connector, connectNulls: true, lineStyle: { width: 2, color: ORANGE, type: "dashed" }, itemStyle: { color: ORANGE },
      markLine: { silent: true, symbol: "none", label: { show: false }, lineStyle: { color: ORANGE, width: 14, opacity: 0.28, type: "solid" }, data: [[{ coord: [d.mes, f.minimo_TO] }, { coord: [d.mes, f.maximo_TO] }]] } },
  ];
  option.tooltip.formatter = (items) => {
    const label = items[0].axisValue;
    if (label === d.mes) {
      const rows = [{ name: MODEL_NAMES.ml_mean, value: f.ml_mean_TO, color: ORANGE }, ...["lgbm", "xgb", "rf"].map((m) => ({ name: MODEL_NAMES[m], value: f.modelos_TO[m], color: INK3 }))];
      return tooltipHtml(`${monthLong(label)} · pronóstico`, rows);
    }
    return tooltipHtml(monthLong(label), [{ name: "Consumo real", value: items.find((i) => i.seriesName === "Consumo real")?.value ?? null, color: SERIES_COLORS[0] }]);
  };
  return option;
}

/* ---- Seguimiento ------------------------------------------------------------------------------------------------------------ */
const ORIGIN_TEXT = { produccion: "Producción", backtest: "Prueba histórica" };

async function renderTracking(view) {
  const history = await api("seguimiento");
  const s = trackState;
  let detail = null;
  try { detail = await api("seguimiento/detalle", { mes: s.mes, origen: s.origen }); } catch (error) { if (error.status !== 404) throw error; s.mes = s.origen = ""; }
  const banners = history.por_medir.map((m) => `<div class="banner warn">${icon("alert")}<div class="grow"><b>${monthLong(m)} ya cerró.</b> Mide qué tanto acertó el pronóstico de ese mes contra el consumo real.</div>
      <button class="btn primary" data-measure="${m}">Medir acierto</button></div>`).join("");
  const hasHistory = history.backtest.length || history.produccion.length;
  view.innerHTML = `<div class="stack">${banners ? `<div class="stack" id="measure-box">${banners}</div>` : ""}
    <div class="card"><div class="toolbar"><h2>Acierto por mes</h2><button class="btn" data-png="acierto_por_mes">Imagen (PNG)</button></div><p class="hint">Del modelo principal, contra el consumo real de cada mes. Los rombos son meses reales medidos en producción; los puntos pequeños, la prueba histórica.</p>
      ${hasHistory ? '<div class="chart small" id="tr-chart"></div>' : '<div class="empty">Todavía no hay seguimiento.</div>'}
      <p class="note">Se marca una alerta si el acierto baja de ${fmt(history.umbrales.sku, 0)}% por SKU, ${fmt(history.umbrales.planta, 0)}% por planta o ${fmt(history.umbrales.total, 0)}% en el total, o si el modelo queda peor que repetir el mes anterior.</p></div>
    <div id="tr-detail">${detail ? detailHtml(detail, history) : ""}</div></div>`;
  if (hasHistory) mountChart($("#tr-chart"), trackingOption(history));
  if (detail) bindDetail(detail, history);
  view.querySelectorAll("[data-measure]").forEach((button) => (button.onclick = async () => {
    button.parentElement.querySelectorAll(".error-text").forEach((old) => old.remove());  // one message at a time, not one more per click
    button.disabled = true;
    button.textContent = "Midiendo…";
    const reply = await apiPost("seguimiento/medir", { mes: button.dataset.measure });
    if (!reply.ok) {
      button.disabled = false;
      button.textContent = "Medir acierto";
      button.parentElement.insertAdjacentHTML("beforeend", `<div class="grow error-text">${esc(reply.body.detalle || "No se pudo medir.")}</div>`);
      return;
    }
    s.mes = button.dataset.measure;
    s.origen = "produccion";
    route();
  }));
}

function trackingOption(h) {
  const byMonth = new Map();
  h.backtest.forEach((r) => byMonth.set(r.mes, { ...r, prod: false }));
  h.produccion.forEach((r) => byMonth.set(r.mes, { ...r, prod: true }));
  const rows = [...byMonth.values()].sort((a, b) => (a.mes < b.mes ? -1 : 1));
  const labels = rows.map((r) => r.mes);
  const option = baseOption(labels, monthShort);
  option.legend = { top: 0, left: 8, icon: "circle", itemWidth: 8, itemHeight: 8, itemGap: 16, textStyle: { color: INK2, fontSize: 12 } };
  option.grid.top = 40;
  option.yAxis.min = (value) => Math.max(0, Math.floor((value.min - 5) / 5) * 5);
  option.yAxis.max = 100;
  option.yAxis.axisLabel.formatter = (v) => `${v}%`;
  const levels = [["sku", "Por SKU"], ["planta", "Por planta"], ["total", "Total"]];
  option.series = levels.map(([key, name], i) => ({
    name, type: "line", data: rows.map((r) => (r[key] === null ? null : r.prod ? { value: r[key], symbol: "diamond", symbolSize: 13 } : r[key])),
    symbol: "circle", symbolSize: 6, lineStyle: { width: 2, color: SERIES_COLORS[i] }, itemStyle: { color: SERIES_COLORS[i], borderColor: SURFACE, borderWidth: 2 },
  }));
  option.tooltip.formatter = (items) => {
    const row = byMonth.get(items[0].axisValue);
    const lines = items.map((it) => ({ name: it.seriesName, value: null, color: it.color, text: `${fmt(it.value)}%` }));
    const html = lines.map((l) => `<div style="display:flex;justify-content:space-between;gap:18px;align-items:center"><span><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${l.color};margin-right:7px"></i>${esc(l.name)}</span><b>${l.text}</b></div>`).join("");
    return `<div style="font-size:12.5px;color:${INK};min-width:170px"><div style="color:${INK2};margin-bottom:5px">${monthLong(items[0].axisValue)} · ${row.prod ? "producción" : "prueba histórica"}</div>${html}</div>`;
  };
  return option;
}

function scoreBadge(value, threshold) {
  if (value === null || value === undefined) return "";
  return value < threshold ? `<span class="badge warn">${icon("alert")}Bajo ${fmt(threshold, 0)}%</span>` : `<span class="badge ok">${icon("check")}Dentro de lo esperado</span>`;
}

function detailHtml(d, h) {
  const main = d.modelos.find((m) => m.modelo === "ml_mean") || d.modelos[0];
  const t = d.total;
  const card = (label, value, sub) => `<div class="card stat" style="min-height:0"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const prodMonths = new Set(h.produccion.map((r) => r.mes)), backMonths = new Set(h.backtest.map((r) => r.mes));
  const origins = ["produccion", "backtest"].filter((o) => (o === "produccion" ? prodMonths : backMonths).has(d.mes));
  const seg = origins.length > 1 ? `<div class="seg" id="tr-origen">${origins.map((o) => `<button data-v="${o}" class="${o === d.origen ? "on" : ""}">${ORIGIN_TEXT[o]}</button>`).join("")}</div>` : `<span class="badge neutral">${ORIGIN_TEXT[d.origen]}</span>`;
  const select = `<select class="pick" id="tr-mes">${d.disponibles.map((m) => `<option value="${m}" ${m === d.mes ? "selected" : ""}>${monthLong(m)}</option>`).join("")}</select>`;
  const models = d.modelos.map((m) => `<tr ${m.modelo === "ml_mean" ? 'style="font-weight:650"' : ""}><td>${MODEL_NAMES[m.modelo] || m.modelo}</td><td>${fmt(m.sku)}%</td><td>${fmt(m.planta)}%</td><td>${fmt(m.total)}%</td><td>${m.sesgo_total_pct === null ? "–" : (m.sesgo_total_pct > 0 ? "+" : "") + fmt(m.sesgo_total_pct) + "%"}</td></tr>`).join("");
  const plantas = d.plantas.map((p) => `<tr><td>${esc(p.planta)}</td><td><div class="barcell"><div class="track"><div class="fill ${p.acierto < h.umbrales.planta_revisar ? "low" : ""}" style="width:${Math.max(0, Math.min(100, p.acierto))}%"></div></div><span style="min-width:48px">${fmt(p.acierto)}%</span></div></td><td>${fmt(p.real_TO, 0)}</td><td>${fmt(p.pronostico_TO, 0)}</td></tr>`).join("");
  const alerts = d.alertas.length
    ? d.alertas.map((a) => `<span class="badge warn">${icon("alert")}${esc(ALERT_TEXT[a.codigo] || a.codigo)}${a.planta ? ": " + esc(a.planta) : ""}</span>`).join("")
    : `<span class="badge ok">${icon("check")}Sin alertas</span>`;
  const missing = d.sin_pronostico ? `<p class="note">${fmt(d.sin_pronostico.series, 0)} series consumieron en el mes y no tenían pronóstico (${fmt(d.sin_pronostico.real_TO, 0)} TO).</p>` : "";
  const errors = d.errores.length ? `<div class="card"><h2>Donde más falló</h2><p class="hint">Las series con mayor error en toneladas. Error = pronóstico − real.</p><div class="tablewrap" style="margin-top:12px"><table class="data"><thead><tr><th>Planta y SKU</th><th>Real (TO)</th><th>Pronóstico (TO)</th><th>Error (TO)</th></tr></thead><tbody>${d.errores.map((e) => `<tr><td>${esc(e.planta)} · ${esc(e.sku)}</td><td>${fx(e.real_TO)}</td><td>${fx(e.pronostico_TO)}</td><td>${e.error_TO > 0 ? "+" : ""}${fx(e.error_TO)}</td></tr>`).join("")}</tbody></table></div></div>` : "";
  return `<div class="stack">
    <div class="card"><div class="toolbar"><div><h2>Detalle de ${monthLong(d.mes)}</h2><p class="hint">Acierto = 100% − error total ÷ consumo real. Sesgo positivo: el pronóstico quedó por debajo del real.</p></div><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">${select}${seg}</div></div>
      <div class="grid cols-4" style="margin-top:12px">
        ${card("Acierto por SKU", `${fmt(main.sku)}%`, scoreBadge(main.sku, h.umbrales.sku))}
        ${card("Acierto por planta", `${fmt(main.planta)}%`, scoreBadge(main.planta, h.umbrales.planta))}
        ${card("Acierto del total", `${fmt(main.total)}%`, scoreBadge(main.total, h.umbrales.total))}
        ${t ? card("Total del mes", `${fmt(t.real_TO, 0)}<small>TO</small>`, `Pronóstico ${fmt(t.pronostico_TO, 0)} TO; ${biasText(t.sesgo_pct)}.`) : ""}
      </div>
      <div class="alertlist" style="flex-direction:row;flex-wrap:wrap">${alerts}</div>${missing}</div>
    <div class="split">
      <div class="card"><h2>Por modelo</h2><div class="tablewrap" style="margin-top:12px"><table class="data"><thead><tr><th>Modelo</th><th>SKU</th><th>Planta</th><th>Total</th><th>Sesgo</th></tr></thead><tbody>${models}</tbody></table></div></div>
      <div class="card"><h2>Por planta</h2><p class="hint">De la que peor acertó a la mejor.</p><div class="tablewrap" style="margin-top:12px;max-height:340px"><table class="data"><thead><tr><th>Planta</th><th>Acierto</th><th>Real (TO)</th><th>Pronóstico (TO)</th></tr></thead><tbody>${plantas}</tbody></table></div></div>
    </div>${errors}</div>`;
}

function bindDetail(d, h) {
  const s = trackState;
  $("#tr-mes").onchange = (event) => {
    s.mes = event.target.value;
    const prod = h.produccion.some((r) => r.mes === s.mes);
    s.origen = prod ? "produccion" : "backtest";
    route();
  };
  const seg = $("#tr-origen");
  if (seg) seg.onclick = (event) => { const b = event.target.closest("button"); if (b) { s.mes = d.mes; s.origen = b.dataset.v; route(); } };
}
