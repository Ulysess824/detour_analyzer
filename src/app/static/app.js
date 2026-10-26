"use strict";

/* ---- helpers ---------------------------------------------------------------------------------------------------------------- */
const $ = (selector, root = document) => root.querySelector(selector);
const NUM = new Intl.NumberFormat("es-CL", { maximumFractionDigits: 1 });
const NUM0 = new Intl.NumberFormat("es-CL", { maximumFractionDigits: 0 });
const MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];
const fmt = (v, digits = 1) => (v === null || v === undefined ? "–" : digits === 0 ? NUM0.format(v) : NUM.format(v));
const monthLong = (ym) => (ym ? `${MONTHS[+ym.slice(5, 7) - 1]} ${ym.slice(0, 4)}` : "–");
const monthShort = (ym) => `${MONTHS[+ym.slice(5, 7) - 1].slice(0, 3)} ${ym.slice(2, 4)}`;
const dateLong = (iso) => {
  if (!iso) return "–";
  const d = new Date(iso.slice(0, 10) + "T12:00:00");
  return `${d.getDate()} ${MONTHS[d.getMonth()].slice(0, 3)} ${d.getFullYear()}`;
};
const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const ICONS = {
  home: '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  trend: '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
  sliders: '<path d="M4 6h8M18 6h2M4 12h2M12 12h8M4 18h10M20 18h0"/><circle cx="15" cy="6" r="2.2"/><circle cx="9" cy="12" r="2.2"/><circle cx="17" cy="18" r="2.2"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5M4 21h16"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 1-1 1.7M12 17h0"/>',
  alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18h0"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h0"/>',
  cross: '<path d="M6 6l12 12M18 6L6 18"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

async function api(path, params = {}) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v !== null && v !== undefined));
  const response = await fetch(`/api/${path}${query.toString() ? "?" + query : ""}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.detalle || "Error"), { status: response.status, body });
  return body;
}

/* ---- navigation ------------------------------------------------------------------------------------------------------------- */
const SECTIONS = [
  { id: "inicio", label: "Inicio", icon: "home", title: "Inicio", subtitle: "El estado del modelo de un vistazo." },
  { id: "consumo", label: "Consumo real", icon: "chart", title: "Consumo real", subtitle: "Lo que se consumió, por planta y por SKU." },
  { id: "prediccion", label: "Predicción", icon: "trend", title: "Predicción", subtitle: "El pronóstico con los datos que hay." },
  { id: "seguimiento", label: "Seguimiento", icon: "check", title: "Seguimiento", subtitle: "" },
  { id: "ajustes", label: "Ajustes", icon: "gear", title: "Ajustes", subtitle: "" },
  { id: "ajuste", label: "Hiperparámetros", icon: "sliders", title: "Hiperparámetros", subtitle: "Reafinar el modelo cada seis meses." },
  { id: "datos", label: "Datos", icon: "database", title: "Datos", subtitle: "Cargar los exportes de SAP." },
  { id: "descargar", label: "Descargar", icon: "download", title: "Descargar", subtitle: "Gráficos en imagen y tablas en Excel." },
  { id: "historial", label: "Historial", icon: "clock", title: "Historial de pronósticos", subtitle: "Los pronósticos generados antes." },
  { id: "ayuda", label: "Ayuda", icon: "help", title: "Ayuda", subtitle: "Cómo usar la app." },
];
const SEPARATE_BEFORE = new Set(["descargar"]);

let current = null;
let charts = [];

function buildNav() {
  $("#nav").innerHTML = SECTIONS.map(
    (s) => `${SEPARATE_BEFORE.has(s.id) ? '<div class="nav-sep"></div>' : ""}<button class="nav-item${s.soon ? " soon" : ""}" data-id="${s.id}" ${s.soon ? 'title="Disponible en una próxima versión"' : ""}>${icon(s.icon)}<span>${s.label}</span></button>`
  ).join("");
  $("#nav").addEventListener("click", (event) => {
    const item = event.target.closest(".nav-item");
    if (item) location.hash = `#/${item.dataset.id}`;
  });
}

function disposeCharts() {
  charts.forEach((chart) => chart.dispose());
  charts = [];
}

async function route() {
  const id = (location.hash.match(/^#\/(\w+)/) || [])[1] || "inicio";
  const section = SECTIONS.find((s) => s.id === id) || SECTIONS[0];
  current = section.id;
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.id === section.id));
  $("#title").textContent = section.title;
  $("#subtitle").textContent = section.subtitle;
  disposeCharts();
  const view = $("#view");
  view.innerHTML = '<div class="loading">Cargando…</div>';
  try {
    if (section.id === "inicio") await renderHome(view);
    else if (section.id === "consumo") await renderConsumption(view);
    else if (section.id === "prediccion") await renderForecast(view);
    else if (section.id === "seguimiento") await renderTracking(view);
    else if (section.id === "ajustes") await renderSettings(view);
    else if (section.id === "ajuste") await renderTuning(view);
    else if (section.id === "datos") await renderData(view);
    else if (section.id === "descargar") await renderDownloads(view);
    else if (section.id === "historial") await renderHistory(view);
    else if (section.id === "ayuda") await renderHelp(view);
    else renderSoon(view, section);
  } catch (error) {
    view.innerHTML = `<div class="card empty"><h3>No se pudo mostrar esta sección</h3><p>${esc(error.message)}</p></div>`;
  }
}

function renderSoon(view, section) {
  view.innerHTML = `<div class="card empty"><h3>${esc(section.title)}</h3><p>Esta sección estará disponible en una próxima versión de la app.</p></div>`;
}

/* ---- charts ----------------------------------------------------------------------------------------------------------------- */
// Categorical colours in a fixed order (validated palette); the last one is the neutral grey of "Otros".
const SERIES_COLORS = ["#4f46e5", "#f97316", "#14b8a6", "#e11d8a", "#0ea5e9", "#a3a30c", "#7c3aed"];
const OTHER_COLOR = "#94a3b8";
const INK = "#1b1f3b", INK2 = "#5b6078", INK3 = "#8a90a8", GRID = "#eceef6", SURFACE = "#ffffff";

const compact = (v) => (Math.abs(v) >= 1000 ? `${NUM.format(v / 1000)} mil` : NUM.format(v));

function colorOf(name, index) {
  return name === "Otros" ? OTHER_COLOR : SERIES_COLORS[index % SERIES_COLORS.length];
}

function tooltipHtml(title, rows, total) {
  const lines = rows
    .filter((r) => r.value !== null && r.value !== undefined)
    .map((r) => `<div style="display:flex;justify-content:space-between;gap:18px;align-items:center"><span><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${r.color};margin-right:7px"></i>${esc(r.name)}</span><b>${fmt(r.value)} TO</b></div>`)
    .join("");
  const foot = total === undefined ? "" : `<div style="display:flex;justify-content:space-between;gap:18px;border-top:1px solid ${GRID};margin-top:5px;padding-top:5px"><span>Total</span><b>${fmt(total)} TO</b></div>`;
  return `<div style="font-size:12.5px;color:${INK};min-width:170px"><div style="color:${INK2};margin-bottom:5px">${esc(title)}</div>${lines}${foot}</div>`;
}

function baseOption(labels, formatLabel) {
  return {
    animationDuration: 500,
    textStyle: { fontFamily: "Segoe UI, system-ui, sans-serif" },
    grid: { left: 8, right: 22, top: 30, bottom: 8, containLabel: true },
    xAxis: {
      type: "category", data: labels, boundaryGap: true,
      axisLine: { lineStyle: { color: GRID } }, axisTick: { show: false },
      axisLabel: { color: INK3, fontSize: 11.5, formatter: formatLabel, hideOverlap: true },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: INK3, fontSize: 11.5, formatter: compact },
      splitLine: { lineStyle: { color: GRID, width: 1 } },
    },
    tooltip: { trigger: "axis", backgroundColor: SURFACE, borderColor: GRID, borderWidth: 1, padding: 12, textStyle: { color: INK }, extraCssText: "box-shadow:0 8px 24px rgba(16,24,40,.14);border-radius:12px", axisPointer: { type: "line", lineStyle: { color: INK3, width: 1 } } },
  };
}

function legendRows(names, width) {
  let rows = 1, used = 0;
  for (const name of names) {
    const need = name.length * 6.6 + 34;
    if (used + need > width - 30) { rows++; used = 0; }
    used += need;
  }
  return rows;
}

function monthlyOption(data, mode, width = 900) {
  const labels = data.etiquetas;
  const partial = data.resumen.mes_parcial;
  const breakdown = data.desglose && mode === "desglose" ? data.desglose.series : null;
  const option = baseOption(labels, monthShort);
  const markArea = partial
    ? { silent: true, itemStyle: { color: "rgba(249,115,22,.08)" }, label: { show: true, position: "insideTop", color: INK2, fontSize: 11, formatter: "mes parcial" }, data: [[{ xAxis: partial }, { xAxis: partial }]] }
    : undefined;
  if (breakdown) {
    option.legend = { top: 0, left: 8, right: 8, icon: "circle", itemWidth: 8, itemHeight: 8, itemGap: 16, textStyle: { color: INK2, fontSize: 12 } };
    option.grid.top = 14 + 24 * legendRows(breakdown.map((s) => s.nombre), width);
    option.series = breakdown.map((s, i) => ({
      name: s.nombre, type: "bar", stack: "total", data: s.valores, barMaxWidth: 24,
      itemStyle: { color: colorOf(s.nombre, i), borderColor: SURFACE, borderWidth: 1, borderRadius: i === breakdown.length - 1 ? [4, 4, 0, 0] : 0 },
      markArea: i === 0 ? markArea : undefined,
    }));
    option.tooltip.formatter = (items) => {
      const rows = items.map((it) => ({ name: it.seriesName, value: it.value, color: it.color })).reverse();
      return tooltipHtml(monthLong(items[0].axisValue), rows, items.reduce((a, it) => a + (it.value || 0), 0));
    };
    return option;
  }
  option.series = [{
    name: "Consumo", type: "line", data: data.valores, smooth: false, symbol: "circle", symbolSize: 8, showSymbol: false,
    lineStyle: { width: 2, color: SERIES_COLORS[0], cap: "round", join: "round" },
    itemStyle: { color: SERIES_COLORS[0], borderColor: SURFACE, borderWidth: 2 },
    areaStyle: { color: SERIES_COLORS[0], opacity: 0.1 },
    emphasis: { focus: "series" },
    markArea,
    markPoint: lastPoint(data.valores, labels),
  }];
  option.tooltip.formatter = (items) => tooltipHtml(monthLong(items[0].axisValue), [{ name: "Consumo", value: items[0].value, color: SERIES_COLORS[0] }]);
  return option;
}

function lastPoint(values, labels) {
  let i = values.length - 1;
  while (i >= 0 && values[i] === null) i--;
  if (i < 0) return undefined;
  return {
    silent: true, symbol: "circle", symbolSize: 10, itemStyle: { color: SERIES_COLORS[0], borderColor: SURFACE, borderWidth: 2 },
    label: { show: true, position: "top", color: INK, fontWeight: 600, fontSize: 12, formatter: () => fmt(values[i], 0) },
    data: [{ coord: [labels[i], values[i]] }],
  };
}

function dailyOption(data) {
  const option = baseOption(data.etiquetas, (iso) => `${+iso.slice(8, 10)} ${MONTHS[+iso.slice(5, 7) - 1].slice(0, 3)}`);
  option.series = [{
    name: "Consumo", type: "bar", data: data.valores, barMaxWidth: 14,
    itemStyle: { color: SERIES_COLORS[0], borderRadius: [4, 4, 0, 0] },
  }];
  option.tooltip.formatter = (items) => tooltipHtml(dateLong(items[0].axisValue), [{ name: "Consumo", value: items[0].value, color: SERIES_COLORS[0] }]);
  return option;
}

function mountChart(element, option) {
  const chart = echarts.init(element, null, { renderer: "canvas" });
  chart.setOption(option);
  charts.push(chart);
  new ResizeObserver(() => chart.resize()).observe(element);
  return chart;
}

function tableHtml(data, mode) {
  const daily = data.nivel === "diario";
  const label = (l) => (daily ? dateLong(l) : monthLong(l));
  const breakdown = !daily && mode === "desglose" && data.desglose ? data.desglose.series : null;
  const head = breakdown ? breakdown.map((s) => `<th>${esc(s.nombre)}</th>`).join("") + "<th>Total</th>" : "<th>Consumo (TO)</th>";
  const rows = data.etiquetas.map((l, i) => {
    const cells = breakdown
      ? breakdown.map((s) => `<td>${fmt(s.valores[i])}</td>`).join("") + `<td><b>${fmt(data.valores[i])}</b></td>`
      : `<td>${fmt(data.valores[i])}</td>`;
    return `<tr><td>${label(l)}</td>${cells}</tr>`;
  });
  return `<div class="tablewrap"><table class="data"><thead><tr><th>${daily ? "Día" : "Mes"}</th>${head}</tr></thead><tbody>${rows.reverse().join("")}</tbody></table></div>`;
}

/* ---- Inicio ----------------------------------------------------------------------------------------------------------------- */
const ALERT_TEXT = {
  sku_bajo: "Acierto por SKU bajo",
  planta_bajo: "Acierto por planta bajo",
  total_bajo: "Acierto del total bajo",
  peor_que_naive: "Peor que el modelo simple",
  planta_revisar: "Hay una planta por revisar",
};
const RETUNE = {
  al_dia: { cls: "ok", icon: "check", text: "Al día" },
  por_vencer: { cls: "warn", icon: "alert", text: "Por vencer" },
  vencido: { cls: "bad", icon: "cross", text: "Vencido" },
  sin_registro: { cls: "neutral", icon: "info", text: "Sin registro" },
};

function statCard(iconName, tone, label, value, sub, foot = "") {
  return `<div class="card stat"><div class="label"><span class="icon ${tone}">${icon(iconName)}</span>${label}</div><div class="value">${value}</div><div class="sub">${sub}</div>${foot ? `<div class="foot">${foot}</div>` : ""}</div>`;
}

async function renderHome(view) {
  const status = await api("estado");
  const data = status.datos;
  const forecast = status.pronostico;
  const tracking = status.seguimiento;
  const retune = status.ajuste;
  const notes = [];
  if (!data) notes.push(`<div class="banner warn">${icon("alert")}<div><b>Todavía no hay datos de consumo.</b> Carga los exportes de SAP en la sección Datos para empezar.</div></div>`);
  if (status.seguimiento_pendiente) notes.push(`<div class="banner warn">${icon("alert")}<div><b>${cap(monthLong(status.seguimiento_pendiente.mes))} ya cerró.</b> Falta medir qué tanto acertó el pronóstico de ese mes hecho con «${esc(STRATEGY_NAMES[status.seguimiento_pendiente.estrategia] || status.seguimiento_pendiente.estrategia)}» (sección Seguimiento).</div></div>`);
  if (retune.candidato) notes.push(`<div class="banner info">${icon("info")}<div><b>Hay parámetros nuevos esperando una decisión.</b> Revísalos en la sección Hiperparámetros.</div></div>`);
  if (retune.estado === "por_vencer" || retune.estado === "vencido") notes.push(`<div class="banner warn">${icon("alert")}<div><b>${retune.estado === "vencido" ? "Los hiperparámetros están vencidos." : "Los hiperparámetros vencen pronto."}</b> Próximo ajuste: ${dateLong(retune.proximo)}.</div></div>`);

  const dataCard = data
    ? statCard("database", "", "Datos", monthLong(data.ultimo_mes_completo), `Último mes completo.<br>${fmt(data.filas, 0)} filas · ${fmt(data.series, 0)} series · ${data.plantas} plantas`, `<span class="badge neutral">Último dato: ${dateLong(data.ultimo_dia)}</span>${data.mes_parcial ? `<span class="badge warn">Mes parcial: ${monthLong(data.mes_parcial)}</span>` : ""}`)
    : statCard("database", "", "Datos", "Sin datos", "Carga los exportes de SAP.");

  const plan = status.plan;
  const ready = forecast && plan && plan.estrategia && forecast.mes === plan.mes_pronosticado && forecast.estrategia === plan.estrategia;
  const how = plan && plan.estrategia ? `<br>${esc(STRATEGY_NAMES[plan.estrategia] || plan.estrategia)}${plan.corte ? ` · corte el día ${plan.corte}` : ""}` : "";
  const forecastCard = plan && plan.error
    ? statCard("trend", "orange", "Próximo pronóstico", "Sin definir", "Los ajustes guardados no sirven con los datos actuales.", '<span class="badge warn">' + icon("alert") + "Revisa Ajustes</span>")
    : statCard("trend", "orange", "Próximo pronóstico", monthLong(status.proximo_pronostico),
      (ready ? `Generado el ${dateLong(forecast.creado_en)}.<br>Total estimado: <b>${fmt(forecast.total_ml_mean_TO, 0)} TO</b>` : "Todavía no se genera este pronóstico.") + how,
      ready ? '<span class="badge ok">' + icon("check") + "Listo</span>" : '<span class="badge warn">' + icon("alert") + "Pendiente</span>");

  let trackingCard;
  if (tracking) {
    const scores = `<div class="scores"><div><b>${fmt(tracking.acierto_sku)}%</b><span>SKU</span></div><div><b>${fmt(tracking.acierto_planta)}%</b><span>planta</span></div><div><b>${fmt(tracking.acierto_total)}%</b><span>total</span></div></div>`;
    const alerts = tracking.alertas.length ? tracking.alertas.map((a) => `<span class="badge warn">${esc(ALERT_TEXT[a] || a)}</span>`).join("") : '<span class="badge ok">' + icon("check") + "Sin alertas</span>";
    trackingCard = statCard("check", "teal", "Último seguimiento", monthLong(tracking.mes), (tracking.origen === "backtest" ? "Prueba histórica (aún no hay un mes real medido)." : `Pronóstico guardado contra el consumo real.<br>${esc(STRATEGY_NAMES[tracking.estrategia] || tracking.estrategia)}`) + scores, alerts);
  } else {
    trackingCard = statCard("check", "teal", "Último seguimiento", "Sin medir", "Cuando cierre el mes pronosticado se compara con el real.");
  }

  const r = RETUNE[retune.estado];
  const retuneCard = statCard("sliders", "", "Hiperparámetros", retune.proximo ? dateLong(retune.proximo) : "–",
    retune.ultimo ? `Próximo ajuste. El último fue el ${dateLong(retune.ultimo)}; se repite cada ${retune.intervalo_meses} meses.` : "No hay un ajuste registrado.",
    `<span class="badge ${r.cls}">${icon(r.icon)}${r.text}${retune.dias_restantes !== null && retune.estado !== "vencido" ? ` · faltan ${fmt(retune.dias_restantes, 0)} días` : ""}</span>`);

  view.innerHTML = `<div class="stack">${notes.join("")}
    <div class="grid cols-4">${dataCard}${forecastCard}${trackingCard}${retuneCard}</div>
    <div class="card"><div class="toolbar"><div><h2>Consumo total por mes</h2><p class="hint">Todas las plantas, toneladas (TO).</p></div><div style="display:flex;gap:10px"><button class="btn" data-png="consumo_total">Imagen (PNG)</button><button class="btn" id="to-consumo">Ver con filtros</button></div></div><div class="chart small" id="home-chart"></div></div></div>`;
  $("#to-consumo").onclick = () => (location.hash = "#/consumo");
  if (data) {
    const series = await api("consumo");
    mountChart($("#home-chart"), monthlyOption({ ...series, desglose: null }, "total"));
  } else {
    $("#home-chart").innerHTML = '<div class="empty">Sin datos para mostrar.</div>';
  }
}

/* ---- Consumo real ----------------------------------------------------------------------------------------------------------- */
const consumptionState = { planta: "", sku: "", nivel: "mensual", modo: "desglose", vista: "grafico", desde: "", hasta: "", ultimoDia: "" };

function combo(root, { field, getParams, onPick }) {
  const input = $("input", root), list = $(".options", root), clear = $(".clear", root);
  let items = [], highlighted = -1, timer = null, seq = 0;
  const open = () => list.classList.add("open");
  const close = () => { seq++; clearTimeout(timer); list.classList.remove("open"); highlighted = -1; };
  const sync = () => root.classList.toggle("has-value", input.value !== "");
  async function load() {
    const mine = ++seq;
    let found = [];
    try {
      found = (await api("sugerencias", { campo: field, q: input.value, limite: 8, ...getParams() })).sugerencias;
    } catch (_) { /* no list */ }
    if (mine !== seq) return;
    items = found;
    list.innerHTML = items.length ? items.map((name, i) => `<button type="button" data-i="${i}">${esc(name)}</button>`).join("") : '<div class="none">Sin coincidencias</div>';
    open();
  }
  const pick = (name) => { input.value = name; sync(); close(); onPick(name); };
  input.addEventListener("focus", load);
  input.addEventListener("input", () => { sync(); clearTimeout(timer); timer = setTimeout(load, 120); });
  input.addEventListener("keydown", (event) => {
    const buttons = [...list.querySelectorAll("button")];
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      highlighted = (highlighted + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % Math.max(buttons.length, 1);
      buttons.forEach((b, i) => b.classList.toggle("on", i === highlighted));
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (highlighted >= 0 && items[highlighted]) pick(items[highlighted]); else { close(); onPick(input.value.trim()); }
    } else if (event.key === "Escape") close();
  });
  list.addEventListener("mousedown", (event) => {
    const button = event.target.closest("button");
    if (button) { event.preventDefault(); pick(items[+button.dataset.i]); }
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
  clear.addEventListener("click", () => { input.value = ""; sync(); close(); onPick(""); });
  return { set(value) { input.value = value; sync(); } };
}

function filtersHtml(s) {
  return `<div class="filters">
        <div class="field"><label for="f-planta">Planta</label><div class="input-wrap" id="w-planta"><input id="f-planta" autocomplete="off" placeholder="Todas" value="${esc(s.planta)}"><button class="clear" type="button" aria-label="Quitar">×</button><div class="options"></div></div></div>
        <div class="field"><label for="f-sku">Nombre del SKU</label><div class="input-wrap" id="w-sku"><input id="f-sku" autocomplete="off" placeholder="Todos · escribe parte del nombre, por ejemplo HP2 125 2440" value="${esc(s.sku)}"><button class="clear" type="button" aria-label="Quitar">×</button><div class="options"></div></div></div>
        <button class="btn" id="reset" type="button">Limpiar filtros</button>
      </div>`;
}

function bindFilters(s, onChange) {
  const wp = $("#w-planta"), ws = $("#w-sku");
  wp.classList.toggle("has-value", !!s.planta);
  ws.classList.toggle("has-value", !!s.sku);
  const comboPlanta = combo(wp, { field: "planta", getParams: () => ({}), onPick: (v) => { s.planta = v; if (s.sku) { s.sku = ""; comboSku.set(""); } onChange(); } });
  const comboSku = combo(ws, { field: "sku", getParams: () => ({ planta: s.planta }), onPick: (v) => { s.sku = v; onChange(); } });
  $("#reset").onclick = () => { s.planta = s.sku = ""; comboPlanta.set(""); comboSku.set(""); onChange(); };
}

async function renderConsumption(view) {
  const s = consumptionState;
  if (!s.ultimoDia) {
    const status = await api("estado");
    if (!status.datos) { view.innerHTML = '<div class="card empty"><h3>Todavía no hay datos</h3><p>Carga los exportes de SAP en la sección Datos.</p></div>'; return; }
    s.ultimoDia = status.datos.ultimo_dia;
  }
  view.innerHTML = `<div class="stack"><div class="card">${filtersHtml(s)}<div id="range"></div></div><div id="result"></div></div>`;
  bindFilters(s, refresh);
  await refresh();
}

function rangeHtml() {
  const s = consumptionState;
  const presets = [[30, "30 días"], [90, "90 días"], [180, "6 meses"], [365, "1 año"]];
  return `<div class="daterow"><div class="seg" id="presets">${presets.map(([d, t]) => `<button type="button" data-days="${d}" class="${s.hasta === s.ultimoDia && s.desde === shiftDays(s.ultimoDia, -(d - 1)) ? "on" : ""}">${t}</button>`).join("")}</div>
    <div class="field"><label for="d-desde">Desde</label><input type="date" id="d-desde" value="${s.desde}" max="${s.ultimoDia}"></div>
    <div class="field"><label for="d-hasta">Hasta</label><input type="date" id="d-hasta" value="${s.hasta}" max="${s.ultimoDia}"></div></div>`;
}

function shiftDays(iso, days) {
  const d = new Date(iso + "T12:00:00");
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

async function refresh() {
  const s = consumptionState;
  const result = $("#result");
  if (!result) return;
  result.innerHTML = '<div class="loading">Cargando…</div>';
  if (s.nivel === "diario" && !s.hasta) { s.hasta = s.ultimoDia; s.desde = shiftDays(s.ultimoDia, -89); }
  $("#range").innerHTML = s.nivel === "diario" ? rangeHtml() : "";
  if (s.nivel === "diario") {
    $("#presets").onclick = (event) => { const b = event.target.closest("button"); if (b) { s.hasta = s.ultimoDia; s.desde = shiftDays(s.ultimoDia, -(+b.dataset.days - 1)); refresh(); } };
    $("#d-desde").onchange = (event) => { s.desde = event.target.value; refresh(); };
    $("#d-hasta").onchange = (event) => { s.hasta = event.target.value; refresh(); };
  }
  let data;
  try {
    data = await api("consumo", { planta: s.planta, sku: s.sku, nivel: s.nivel, desde: s.nivel === "diario" ? s.desde : "", hasta: s.nivel === "diario" ? s.hasta : "" });
  } catch (error) {
    disposeCharts();
    result.innerHTML = notFoundHtml(error);
    result.querySelectorAll("[data-pick]").forEach((b) => (b.onclick = () => {
      const field = error.body.campo.includes("planta") ? "planta" : "sku";
      s[field] = b.dataset.pick;
      if (field === "planta") s.sku = "";
      renderConsumption($("#view"));
    }));
    return;
  }
  disposeCharts();
  result.innerHTML = resultHtml(data);
  bindResult(data);
}

function notFoundHtml(error) {
  if (error.status === 404) {
    const chips = (error.body.sugerencias || []).map((n) => `<button class="chip" data-pick="${esc(n)}">${esc(n)}</button>`).join("");
    return `<div class="card empty"><h3>${esc(error.message)}</h3>${chips ? `<p>¿Quisiste decir alguno de estos?</p><div class="chips">${chips}</div>` : "<p>Revisa el nombre o usa las sugerencias al escribir.</p>"}</div>`;
  }
  return `<div class="card empty"><h3>No se pudo cargar el consumo</h3><p>${esc(error.message)}</p></div>`;
}

function kpis(data) {
  const r = data.resumen;
  const card = (label, value, sub) => `<div class="card stat" style="min-height:0"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  if (data.nivel === "diario") {
    const days = data.etiquetas.length;
    return [
      card("Total del periodo", `${fmt(r.total_TO, 0)}<small>TO</small>`, `${dateLong(r.desde)} a ${dateLong(r.hasta)}`),
      card("Días con consumo", `${r.dias_con_consumo}<small>de ${days}</small>`, "Los domingos con consumo también cuentan."),
      card("Promedio por día", `${fmt(r.total_TO / Math.max(days, 1))}<small>TO</small>`, "Sobre todos los días del periodo."),
    ].join("");
  }
  const change = r.cambio_vs_mes_anterior_pct;
  const arrow = change === null ? "" : change >= 0 ? "▲ " : "▼ ";
  return [
    card("Último mes completo", `${fmt(r.ultimo_mes_TO, 0)}<small>TO</small>`, monthLong(r.ultimo_mes_completo)),
    card("Cambio contra el mes anterior", change === null ? "–" : `${arrow}${fmt(Math.abs(change))}%`, "Del último mes completo."),
    card("Promedio mensual", `${fmt(r.promedio_mensual_TO, 0)}<small>TO</small>`, `Desde ${monthLong(r.primer_mes)}${r.mes_parcial ? ", sin el mes parcial" : ""}.`),
  ].join("");
}

function resultHtml(data) {
  const s = consumptionState;
  const who = [data.filtro.planta || "Todas las plantas", data.filtro.sku || "todos los SKU"].join(" · ");
  const hasBreakdown = data.nivel === "mensual" && data.desglose;
  const by = hasBreakdown ? (data.desglose.por === "planta" ? "por planta" : "por SKU") : "";
  const partial = data.resumen.mes_parcial ? `<p class="note">${monthLong(data.resumen.mes_parcial)} está incompleto (los datos llegan hasta el ${dateLong(data.resumen.ultimo_dato)}) y no entra en el resumen.</p>` : "";
  return `<div class="stack"><div class="grid cols-3">${kpis(data)}</div>
    <div class="card"><div class="toolbar"><div><h2>Consumo ${data.nivel} · ${esc(who)}</h2><p class="hint">Toneladas (TO). Pasa el cursor sobre el gráfico para ver cada valor.</p></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <div class="seg" id="seg-nivel"><button data-v="mensual" class="${s.nivel === "mensual" ? "on" : ""}">Por mes</button><button data-v="diario" class="${s.nivel === "diario" ? "on" : ""}">Por día</button></div>
        <div class="seg" id="seg-modo"><button data-v="total" class="${s.modo === "total" ? "on" : ""}" ${hasBreakdown ? "" : "disabled"}>Total</button><button data-v="desglose" class="${s.modo === "desglose" ? "on" : ""}" ${hasBreakdown ? "" : "disabled"}>${by ? "Desglose " + by : "Desglose"}</button></div>
        <div class="seg" id="seg-vista"><button data-v="grafico" class="${s.vista === "grafico" ? "on" : ""}">Gráfico</button><button data-v="tabla" class="${s.vista === "tabla" ? "on" : ""}">Tabla</button></div>
        <button class="btn" data-png="consumo_real" ${s.vista === "tabla" ? "disabled" : ""}>Imagen (PNG)</button>
      </div></div>
      <div id="plot"></div>${partial}</div></div>`;
}

function bindResult(data) {
  const s = consumptionState;
  const hasBreakdown = data.nivel === "mensual" && data.desglose;
  const mode = hasBreakdown ? s.modo : "total";
  const plot = $("#plot");
  if (s.vista === "tabla") plot.innerHTML = tableHtml(data, mode);
  else {
    plot.innerHTML = '<div class="chart" id="chart"></div>';
    const element = $("#chart");
    mountChart(element, data.nivel === "diario" ? dailyOption(data) : monthlyOption(data, mode, element.clientWidth));
  }
  const bind = (id, key, after) => { $(id).onclick = (event) => { const b = event.target.closest("button"); if (b && !b.disabled) { s[key] = b.dataset.v; after(); } }; };
  bind("#seg-nivel", "nivel", refresh);
  bind("#seg-modo", "modo", () => bindResultAgain(data));
  bind("#seg-vista", "vista", () => bindResultAgain(data));
}

function bindResultAgain(data) {
  disposeCharts();
  $("#result").innerHTML = resultHtml(data);
  bindResult(data);
}

/* ---- start ------------------------------------------------------------------------------------------------------------------ */
async function loadFooter() {
  try {
    const status = await api("estado");
    if (status.datos) {
      $("#top-chip").innerHTML = `<span class="dot ${status.datos.mes_parcial ? "warn" : ""}"></span>Datos al ${dateLong(status.datos.ultimo_dia)}`;
      $("#side-foot").innerHTML = `${fmt(status.datos.series, 0)} series<br>${status.datos.plantas} plantas`;
    } else {
      $("#top-chip").innerHTML = '<span class="dot warn"></span>Sin datos';
    }
  } catch (_) { /* the screens show their own message */ }
}

buildNav();
window.addEventListener("hashchange", route);
loadFooter();
route();
