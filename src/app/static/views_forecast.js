"use strict";
/* Predicción and Seguimiento screens. Loaded before app.js, which owns the helpers (api, fmt, icon, combo, ...) used here at call time. */

const forecastState = { planta: "", sku: "", mes: "", estrategia: "" };
const STRATEGY_NAMES = {
  automatica: "Automática", h1: "Un mes de anticipación", directa: "Dos meses, directa", parcial: "Dos meses, directa y días del mes en curso",
  nowcast_media: "Dos meses, mes en curso completado con la media", iterada: "Dos meses, iterada",
};
const trackState = { mes: "", origen: "", estrategia: "", grafico: "" };
const STRATEGY_ORDER = ["h1", "directa", "parcial", "nowcast_media", "iterada"];
const STRATEGY_SHORT = { h1: "Un mes", directa: "Directa", parcial: "Directa + mes en curso", nowcast_media: "Mes completado", iterada: "Iterada" };
const strategyName = (id) => STRATEGY_NAMES[id] || id;
const MODEL_NAMES = { ml_mean: "Modelo principal (promedio)", lgbm: "LightGBM", xgb: "XGBoost", rf: "Random Forest", naive: "Modelo simple (repite el mes anterior)" };
const ORANGE = "#f97316";
const FIXED = new Intl.NumberFormat("es-CL", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const fx = (v) => (v === null || v === undefined ? "–" : FIXED.format(v));
let pollTimer = null;

async function apiPost(path, body, headers = {}) {
  const response = await fetch(`/api/${path}`, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });
  return { ok: response.ok, status: response.status, body: await response.json().catch(() => ({})) };
}

const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
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
  const plan = status.plan;
  const have = plan && plan.estrategia ? list.pronosticos.find((f) => f.mes === plan.mes_pronosticado && f.estrategia === plan.estrategia) : null;
  const next = status.proximo_pronostico;
  view.innerHTML = `<div class="stack"><div class="card gen" id="gen-card"></div><div id="fc-body"></div></div>`;
  renderGenCard(status, have);
  if (list.pronosticos.length) await renderForecastBody();
  else $("#fc-body").innerHTML = `<div class="card empty"><h3>Todavía no hay pronósticos</h3><p>Genera el de ${monthLong(next)} con el botón de arriba.</p></div>`;
  const running = (await api("trabajos/activo")).trabajo;
  if (running) watchJob(running.id);
}

function renderGenCard(status, have, extra = "") {
  const data = status.datos, plan = status.plan;
  const blocked = !plan || !plan.estrategia;
  const open = data.mes_parcial ? ` ${cap(monthLong(data.mes_parcial))} sigue en curso.` : "";
  const text = blocked
    ? (plan && plan.error ? esc(plan.error) : "Todavía no hay datos.")
    : have
      ? `Generado el ${dateLong(have.creado_en)}. Usa los datos hasta el ${dateLong(data.ultimo_dia)}.${open}`
      : `Todavía no se genera. Usa los datos hasta el ${dateLong(data.ultimo_dia)} y tarda menos de un minuto.${open}`;
  const how = blocked ? "" : `<p class="how"><b>${esc(STRATEGY_NAMES[plan.estrategia])}</b>${plan.corte ? ` · corte el día ${plan.corte}` : ""} · parte de ${monthLong(plan.mes_origen)} · <a href="#/ajustes">Cambiar en Ajustes</a></p>
    ${plan.avisos.map((n) => `<p class="how warn-text">${esc(n)}</p>`).join("")}`;
  $("#gen-card").innerHTML = `<div><h2>${blocked ? "Pronóstico" : `Pronóstico de ${monthLong(plan.mes_pronosticado)}`}</h2><p>${text}</p>${how}</div>
    <button class="btn primary" id="gen-btn" ${blocked ? "disabled" : ""}>${have ? "Generar de nuevo" : "Generar pronóstico"}</button>
    <div class="grow" id="gen-extra">${extra}</div>`;
  $("#gen-btn").onclick = () => (have ? askOverwrite(status, have) : startGenerate(false, status, have));
}

function askOverwrite(status, have) {
  $("#gen-extra").innerHTML = `<div class="confirm">Ya existe el pronóstico de ${monthLong(have.mes)} (${esc(STRATEGY_NAMES[have.estrategia] || have.estrategia)}). Si lo generas de nuevo, se reemplaza.
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
      forecastState.estrategia = job.resultado.estrategia;
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
    d = await api("pronostico", { mes: s.mes, estrategia: s.estrategia, planta: s.planta, sku: s.sku });
  } catch (error) {
    disposeCharts();
    result.innerHTML = notFoundHtml(error);
    bindPicks(result, error, s, () => renderForecastBody());
    return;
  }
  s.mes = d.mes;
  s.estrategia = d.estrategia;
  disposeCharts();
  result.innerHTML = forecastHtml(d);
  const element = $("#fc-chart");
  mountChart(element, forecastOption(d, element.clientWidth));
  const select = $("#fc-mes");
  if (select) select.onchange = () => { [s.mes, s.estrategia] = select.value.split("|"); refreshForecast(); };
  result.querySelectorAll("tr[data-planta]").forEach((row) => (row.onclick = () => { s.planta = row.dataset.planta; s.sku = ""; renderForecastBody(); }));
}

function forecastHtml(d) {
  const f = d.pronostico, last = d.ultimo_real;
  const card = (label, value, sub) => `<div class="card stat" style="min-height:0"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const who = [d.filtro.planta || "Todas las plantas", d.filtro.sku || "todos los SKU"].join(" · ");
  const months = d.disponibles_detalle.length > 1 ? `<label class="hint" for="fc-mes">Pronóstico de </label><select class="pick" id="fc-mes">${d.disponibles_detalle.map((m) => `<option value="${m.mes}|${m.estrategia}" ${m.mes === d.mes && m.estrategia === d.estrategia ? "selected" : ""}>${monthLong(m.mes)} · ${esc(STRATEGY_NAMES[m.estrategia] || m.estrategia)}</option>`).join("")}</select>` : "";
  const open = last.mes_anterior_abierto;
  const current = d.mes_en_curso ? currentMonthHtml(d.mes_en_curso, card) : "";
  const t = d.tabla;
  const head = t.tipo === "planta" ? "<th>Planta</th>" : "<th>Planta</th><th>SKU</th>";
  const rows = t.filas.map((r) => `<tr ${t.tipo === "planta" ? `data-planta="${esc(r.planta)}"` : ""}>${t.tipo === "planta" ? `<td>${esc(r.planta)}</td>` : `<td>${esc(r.planta)}</td><td>${esc(r.sku)}</td>`}<td>${fx(r.pronostico_TO)}</td><td>${fx(r.real_ultimo_TO)}</td><td>${r.cambio_pct === null ? "–" : arrow(r.cambio_pct) + fmt(Math.abs(r.cambio_pct)) + "%"}</td></tr>`).join("");
  const more = t.total_filas > t.filas.length ? `<p class="note">Mostrando los ${t.filas.length} mayores de ${fmt(t.total_filas, 0)}. Filtra por planta o SKU para ver otros.</p>` : "";
  return `<div class="stack"><div class="grid cols-3">
      ${card(`Pronóstico de ${monthLong(d.mes)}`, `${fmt(f.ml_mean_TO, 0)}<small>TO</small>`, `${esc(STRATEGY_NAMES[d.estrategia] || d.estrategia)}${d.corte ? ` · corte el día ${d.corte}` : ""}. Promedio de LightGBM, XGBoost y Random Forest.`)}
      ${card(open ? "Contra el último mes completo" : "Contra el último mes real", last.cambio_pct === null ? "–" : `${arrow(last.cambio_pct)}${fmt(Math.abs(last.cambio_pct))}%`, `${fmt(last.TO, 0)} TO en ${monthLong(last.mes)}.${open ? ` ${cap(monthLong(open))} sigue en curso.` : ""}`)}
      ${card("Rango entre modelos", `${fmt(f.minimo_TO, 0)} – ${fmt(f.maximo_TO, 0)}<small>TO</small>`, "Si los modelos se separan mucho, el pronóstico es menos seguro.")}
    </div>${current}
    <div class="card"><div class="toolbar"><div><h2>Consumo real y pronóstico · ${esc(who)}</h2><p class="hint">Toneladas (TO), últimos 24 meses. La escala no empieza en cero. La franja naranja es el rango entre los tres modelos.</p></div><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">${months}<button class="btn" data-png="pronostico">Imagen (PNG)</button></div></div><div class="chart" id="fc-chart"></div></div>
    <div class="card"><h2>${t.tipo === "planta" ? "Pronóstico por planta" : "Pronóstico por SKU"}</h2><p class="hint">${t.tipo === "planta" ? "Toca una planta para ver sus SKU." : "Ordenado de mayor a menor pronóstico."}</p>
      <div class="tablewrap" style="margin-top:12px"><table class="data ${t.tipo === "planta" ? "clickable" : "two-left"}"><thead><tr>${head}<th>Pronóstico (TO)</th><th>Último mes real (TO)</th><th>Cambio</th></tr></thead><tbody>${rows}</tbody></table></div>${more}</div></div>`;
}

function currentMonthHtml(m, card) {
  const closed = m.real_completo_TO !== null && m.real_completo_TO !== undefined;
  return `<div class="card"><h2>El mes en curso: ${monthLong(m.mes)}</h2>
    <p class="hint" style="margin-top:6px">Esto es lo que el pronóstico sabía de ${monthLong(m.mes)} al día ${m.corte}: ${fmt(m.dias_vistos, 0)} días hábiles vistos y ${fmt(m.dias_restantes, 0)} por venir. Los días que faltan se estiman ${m.metodo === "montecarlo" ? "con una simulación Monte Carlo de la historia y los días vistos" : "con la media diaria de los vistos"}; es una estimación, no un dato.
      ${m.metodo_fijo ? "Este pronóstico se hizo con la media, así que se muestra con la media." : `Con ${m.otro_metodo.metodo === "montecarlo" ? "la simulación Monte Carlo" : "la media diaria"} el mes completo sería ${fmt(m.otro_metodo.total_estimado_TO, 0)} TO. Se cambia en Ajustes.`}</p>
    <div class="grid cols-3" style="margin-top:14px">
      ${card(`Real hasta el día ${m.corte}`, `${fmt(m.real_hasta_corte_TO, 0)}<small>TO</small>`, "Consumo ya ocurrido.")}
      ${card("Resto del mes, estimado", `${fmt(m.resto_estimado_TO, 0)}<small>TO</small>`, m.metodo === "montecarlo" ? "Promedio de 200 escenarios simulados de los días que faltan." : "Media diaria de los días vistos por los días que faltan.")}
      ${card("Mes completo, estimado", `${fmt(m.total_estimado_TO, 0)}<small>TO</small>`, closed ? `El mes ya cerró: el real fue ${fmt(m.real_completo_TO, 0)} TO.` : "Real hasta el corte más el resto estimado.")}
    </div></div>`;
}

function forecastOption(d, width) {
  const f = d.pronostico;
  const cut = d.historia.etiquetas.indexOf(d.mes);
  const labelsAll = cut >= 0 ? d.historia.etiquetas.slice(0, cut) : d.historia.etiquetas;
  const open = d.historia.mes_parcial; // a month with only some of its days is not a real total: leave it out of the line
  const valuesAll = d.historia.valores.slice(0, labelsAll.length).map((v, i) => (labelsAll[i] === open ? null : v));
  const keep = 24;
  const labels = [...labelsAll.slice(-keep), d.mes];
  const real = [...valuesAll.slice(-keep), null];
  let lastIdx = real.length - 2;
  while (lastIdx >= 0 && real[lastIdx] === null) lastIdx--;
  const now = d.mes_en_curso;
  const connector = labels.map((l, i) => {
    if (i === lastIdx) return { value: real[lastIdx], symbol: "none" };
    if (now && l === now.mes) return { value: now.total_estimado_TO, symbol: "circle", symbolSize: 9, itemStyle: { color: SURFACE, borderColor: ORANGE, borderWidth: 2 } };
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
    if (d.mes_en_curso && label === d.mes_en_curso.mes) {
      const m = d.mes_en_curso;
      return tooltipHtml(`${monthLong(label)} · mes en curso`, [{ name: `Real hasta el día ${m.corte}`, value: m.real_hasta_corte_TO, color: SERIES_COLORS[0] }, { name: "Mes completo, estimado", value: m.total_estimado_TO, color: ORANGE }]);
    }
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
  try { detail = await api("seguimiento/detalle", { mes: s.mes, origen: s.origen, estrategia: s.estrategia }); } catch (error) { if (error.status !== 404) throw error; s.mes = s.origen = s.estrategia = ""; }
  const banners = history.por_medir.map((m) => `<div class="banner warn">${icon("alert")}<div class="grow"><b>${cap(monthLong(m.mes))} ya cerró.</b> Mide qué tanto acertó contra el consumo real el pronóstico de ese mes hecho con «${esc(strategyName(m.estrategia))}».</div>
      <button class="btn primary" data-measure="${m.mes}" data-strategy="${m.estrategia}">Medir acierto</button></div>`).join("");
  const hasHistory = history.backtest.length || history.produccion.length;
  const present = new Set([...history.produccion, ...history.backtest].map((r) => r.estrategia));
  const strategies = STRATEGY_ORDER.filter((id) => present.has(id));
  const chosen = strategies.includes(s.grafico) ? s.grafico : (detail && strategies.includes(detail.estrategia) ? detail.estrategia : (history.produccion.length ? history.produccion[history.produccion.length - 1].estrategia : (strategies.includes("h1") ? "h1" : strategies[0])));
  s.grafico = chosen;
  const thresholds = chosen === "h1" ? history.umbrales : history.umbrales_dos_meses;
  const picker = strategies.length > 1 ? `<div class="seg" id="tr-estr">${strategies.map((id) => `<button data-v="${id}" class="${id === chosen ? "on" : ""}" title="${esc(strategyName(id))}">${esc(STRATEGY_SHORT[id] || id)}</button>`).join("")}</div>` : "";
  view.innerHTML = `<div class="stack">${banners ? `<div class="stack" id="measure-box">${banners}</div>` : ""}
    <div class="card"><div class="toolbar"><div><h2>Acierto por mes${strategies.length ? ` · ${esc(strategyName(chosen))}` : ""}</h2><p class="hint">Del modelo principal, contra el consumo real de cada mes. Los rombos son meses reales medidos en producción; los puntos pequeños, la prueba histórica de esa estrategia.</p></div><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">${picker}<button class="btn" data-png="acierto_por_mes">Imagen (PNG)</button></div></div>
      ${hasHistory ? '<div class="chart small" id="tr-chart"></div>' : '<div class="empty">Todavía no hay seguimiento.</div>'}
      <p class="note">Se marca una alerta si el acierto baja de ${fmt(thresholds.sku, 0)}% por SKU, ${fmt(thresholds.planta, 0)}% por planta o ${fmt(thresholds.total, 0)}% en el total, o si el modelo queda peor que repetir el mes anterior. ${chosen === "h1" ? "" : "Los pronósticos a dos meses tienen umbrales más bajos porque aciertan menos."}</p></div>
    ${comparisonHtml(history)}
    <div id="tr-detail">${detail ? detailHtml(detail, history) : ""}</div></div>`;
  if (hasHistory) mountChart($("#tr-chart"), trackingOption(history, chosen));
  if ($("#tr-estr")) $("#tr-estr").onclick = (event) => { const b = event.target.closest("button"); if (b) { s.grafico = b.dataset.v; route(); } };
  if (detail) bindDetail(detail, history);
  view.querySelectorAll("[data-measure]").forEach((button) => (button.onclick = async () => {
    button.parentElement.querySelectorAll(".error-text").forEach((old) => old.remove());  // one message at a time, not one more per click
    button.disabled = true;
    button.textContent = "Midiendo…";
    const reply = await apiPost("seguimiento/medir", { mes: button.dataset.measure, estrategia: button.dataset.strategy || "h1" });
    if (!reply.ok) {
      button.disabled = false;
      button.textContent = "Medir acierto";
      button.parentElement.insertAdjacentHTML("beforeend", `<div class="grow error-text">${esc(reply.body.detalle || "No se pudo medir.")}</div>`);
      return;
    }
    s.mes = button.dataset.measure;
    s.origen = "produccion";
    s.estrategia = button.dataset.strategy || "h1";
    s.grafico = s.estrategia;
    route();
  }));
}

function comparisonHtml(h) {
  if (!h.comparacion.length) return "";
  const cell = (x, key) => (x ? fmt(x[key]) + "%" : "–");
  const rows = h.comparacion.map((c) => {
    const p = c.produccion, b = c.prueba_historica;
    const few = p && p.meses < 3 ? ` <span class="badge neutral" title="Con menos de 3 meses el promedio no dice mucho">pocos meses</span>` : "";
    return `<tr><td>${esc(strategyName(c.estrategia))}</td><td>${p ? fmt(p.meses, 0) : "–"}${few}</td><td>${cell(p, "sku")}</td><td>${cell(p, "planta")}</td><td>${cell(p, "total")}</td><td>${b ? fmt(b.meses, 0) : "–"}</td><td>${cell(b, "sku")}</td><td>${cell(b, "planta")}</td><td>${cell(b, "total")}</td></tr>`;
  }).join("");
  return `<div class="card"><h2>Comparación de estrategias</h2><p class="hint">Promedio del acierto de los meses medidos, por SKU, planta y total. La prueba histórica son los últimos 12 meses con lo que se sabía en cada momento. Cada estrategia se mide sobre sus propias series, y un pronóstico de mitad de mes ve menos información que uno hecho con el mes anterior completo: no se comparan de uno en uno.</p>
    <div class="tablewrap" style="margin-top:12px;max-height:none"><table class="data left3"><thead><tr><th rowspan="2">Estrategia</th><th colspan="4" style="text-align:center">En producción</th><th colspan="4" style="text-align:center">Prueba histórica</th></tr>
      <tr><th>Meses</th><th>SKU</th><th>Planta</th><th>Total</th><th>Meses</th><th>SKU</th><th>Planta</th><th>Total</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function trackingOption(h, strategy) {
  const byMonth = new Map();
  h.backtest.filter((r) => r.estrategia === strategy).forEach((r) => byMonth.set(r.mes, { ...r, prod: false }));
  h.produccion.filter((r) => r.estrategia === strategy).forEach((r) => byMonth.set(r.mes, { ...r, prod: true }));
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
  const th = d.estrategia === "h1" ? h.umbrales : h.umbrales_dos_meses;
  const how = `${esc(strategyName(d.estrategia))}${d.corte ? ` · corte el día ${d.corte}` : ""}`;
  const strategySelect = d.estrategias.length > 1 ? `<select class="pick" id="tr-estr-detalle">${d.estrategias.map((id) => `<option value="${id}" ${id === d.estrategia ? "selected" : ""}>${esc(STRATEGY_SHORT[id] || id)}</option>`).join("")}</select>` : "";
  const others = d.otras_estrategias.length ? `<div class="card"><h2>El mismo mes con otras estrategias</h2><p class="hint">Acierto del modelo principal en ${monthLong(d.mes)}. No son del todo comparables: no vieron la misma información.</p>
    <div class="tablewrap" style="margin-top:12px;max-height:none"><table class="data left3"><thead><tr><th>Estrategia</th><th>SKU</th><th>Planta</th><th>Total</th></tr></thead><tbody>${d.otras_estrategias.map((o) => `<tr><td>${esc(strategyName(o.estrategia))}</td><td>${fmt(o.sku)}%</td><td>${fmt(o.planta)}%</td><td>${fmt(o.total)}%</td></tr>`).join("")}</tbody></table></div></div>` : "";
  const origins = ["produccion", "backtest"].filter((o) => (o === "produccion" ? prodMonths : backMonths).has(d.mes));
  const seg = origins.length > 1 ? `<div class="seg" id="tr-origen">${origins.map((o) => `<button data-v="${o}" class="${o === d.origen ? "on" : ""}">${ORIGIN_TEXT[o]}</button>`).join("")}</div>` : `<span class="badge neutral">${ORIGIN_TEXT[d.origen]}</span>`;
  const select = `<select class="pick" id="tr-mes">${d.disponibles.map((m) => `<option value="${m}" ${m === d.mes ? "selected" : ""}>${monthLong(m)}</option>`).join("")}</select>`;
  const models = d.modelos.map((m) => `<tr ${m.modelo === "ml_mean" ? 'style="font-weight:650"' : ""}><td>${MODEL_NAMES[m.modelo] || m.modelo}</td><td>${fmt(m.sku)}%</td><td>${fmt(m.planta)}%</td><td>${fmt(m.total)}%</td><td>${m.sesgo_total_pct === null ? "–" : (m.sesgo_total_pct > 0 ? "+" : "") + fmt(m.sesgo_total_pct) + "%"}</td></tr>`).join("");
  const plantas = d.plantas.map((p) => `<tr><td>${esc(p.planta)}</td><td><div class="barcell"><div class="track"><div class="fill ${p.acierto < th.planta_revisar ? "low" : ""}" style="width:${Math.max(0, Math.min(100, p.acierto))}%"></div></div><span style="min-width:48px">${fmt(p.acierto)}%</span></div></td><td>${fmt(p.real_TO, 0)}</td><td>${fmt(p.pronostico_TO, 0)}</td></tr>`).join("");
  const alerts = d.alertas.length
    ? d.alertas.map((a) => `<span class="badge warn">${icon("alert")}${esc(ALERT_TEXT[a.codigo] || a.codigo)}${a.planta ? ": " + esc(a.planta) : ""}</span>`).join("")
    : `<span class="badge ok">${icon("check")}Sin alertas</span>`;
  const missing = d.sin_pronostico ? `<p class="note">${fmt(d.sin_pronostico.series, 0)} series consumieron en el mes y no tenían pronóstico (${fmt(d.sin_pronostico.real_TO, 0)} TO).</p>` : "";
  const errors = d.errores.length ? `<div class="card"><h2>Donde más falló</h2><p class="hint">Las series con mayor error en toneladas. Error = pronóstico − real.</p><div class="tablewrap" style="margin-top:12px"><table class="data"><thead><tr><th>Planta y SKU</th><th>Real (TO)</th><th>Pronóstico (TO)</th><th>Error (TO)</th></tr></thead><tbody>${d.errores.map((e) => `<tr><td>${esc(e.planta)} · ${esc(e.sku)}</td><td>${fx(e.real_TO)}</td><td>${fx(e.pronostico_TO)}</td><td>${e.error_TO > 0 ? "+" : ""}${fx(e.error_TO)}</td></tr>`).join("")}</tbody></table></div></div>` : "";
  return `<div class="stack">
    <div class="card"><div class="toolbar"><div><h2>Detalle de ${monthLong(d.mes)} · ${how}</h2><p class="hint">Acierto = 100% − error total ÷ consumo real. Sesgo positivo: el pronóstico quedó por debajo del real.</p></div><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">${select}${strategySelect}${seg}</div></div>
      <div class="grid cols-4" style="margin-top:12px">
        ${card("Acierto por SKU", `${fmt(main.sku)}%`, scoreBadge(main.sku, th.sku))}
        ${card("Acierto por planta", `${fmt(main.planta)}%`, scoreBadge(main.planta, th.planta))}
        ${card("Acierto del total", `${fmt(main.total)}%`, scoreBadge(main.total, th.total))}
        ${t ? card("Total del mes", `${fmt(t.real_TO, 0)}<small>TO</small>`, `Pronóstico ${fmt(t.pronostico_TO, 0)} TO; ${biasText(t.sesgo_pct)}.`) : ""}
      </div>
      <div class="alertlist" style="flex-direction:row;flex-wrap:wrap">${alerts}</div>${missing}</div>
    <div class="split">
      <div class="card"><h2>Por modelo</h2><div class="tablewrap" style="margin-top:12px"><table class="data"><thead><tr><th>Modelo</th><th>SKU</th><th>Planta</th><th>Total</th><th>Sesgo</th></tr></thead><tbody>${models}</tbody></table></div></div>
      <div class="card"><h2>Por planta</h2><p class="hint">De la que peor acertó a la mejor.</p><div class="tablewrap" style="margin-top:12px;max-height:340px"><table class="data"><thead><tr><th>Planta</th><th>Acierto</th><th>Real (TO)</th><th>Pronóstico (TO)</th></tr></thead><tbody>${plantas}</tbody></table></div></div>
    </div>${others}${errors}</div>`;
}

function bindDetail(d, h) {
  const s = trackState;
  $("#tr-mes").onchange = (event) => {
    s.mes = event.target.value;
    const prod = h.produccion.some((r) => r.mes === s.mes);
    s.origen = prod ? "produccion" : "backtest";
    s.estrategia = "";
    route();
  };
  const strategy = $("#tr-estr-detalle");
  if (strategy) strategy.onchange = (event) => { s.mes = d.mes; s.origen = d.origen; s.estrategia = event.target.value; route(); };
  const seg = $("#tr-origen");
  if (seg) seg.onclick = (event) => { const b = event.target.closest("button"); if (b) { s.mes = d.mes; s.origen = b.dataset.v; s.estrategia = ""; route(); } };
}
