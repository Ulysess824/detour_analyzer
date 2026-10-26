"use strict";
/* Ajustes: which strategy the forecast uses and the cutoff day. Loaded before app.js, which owns the helpers (api, fmt, icon, mountChart, ...) used here at call time. */

const settingsState = { data: null, draft: null, message: "", open: { rest: false, curve: false } };
const HIDDEN_STRATEGIES = ["iterada"];  // not recommended and never better than the direct one: only listed while it is the saved choice
const REFERENCE_LINES = { directa: "Directa, sin el mes en curso", h1: "Un mes de anticipación" };

async function apiPut(path, body) {
  const response = await fetch(`/api/${path}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { ok: response.ok, status: response.status, body: await response.json().catch(() => ({})) };
}

async function renderSettings(view) {
  const data = await api("ajustes");
  settingsState.data = data;
  settingsState.draft = { ...data.ajustes };
  settingsState.message = "";
  paintSettings();
}

const optionName = (id) => (STRATEGY_NAMES[id] || id);
const sameSettings = (a, b) => a.estrategia === b.estrategia && a.corte_manual === b.corte_manual && a.metodo_resto === b.metodo_resto;

function accuracyChips(a) {
  if (!a) return "";
  const chip = (label, v) => (v === null || v === undefined ? "" : `<span class="badge neutral">${label} ${fmt(v)}%</span>`);
  return `<div class="chiprow">${chip("SKU", a.sku)}${chip("planta", a.planta)}${chip("total", a.total)}</div>`;
}

function settingsStateCard(d) {
  const st = d.estado, plan = d.plan;
  if (!st) {
    return `<div class="card"><h2>Próximo pronóstico</h2><p class="hint" style="margin-top:6px">Todavía no hay datos. Carga los exportes de SAP en la sección Datos; mientras tanto puedes elegir la estrategia.</p></div>`;
  }
  let line = "";
  if (d.error) {
    line = `<div class="banner warn">${icon("alert")}<div><b>Los ajustes guardados no sirven con estos datos.</b><br>${esc(d.error)}</div></div>`;
  } else if (plan) {
    const cutoff = plan.corte ? ` · corte día ${plan.corte}` : "";
    line = `<div class="banner info">${icon("info")}<div><b>${monthLong(plan.mes_pronosticado)}</b> · «${esc(optionName(plan.estrategia))}»${cutoff}</div></div>${plan.avisos.map((n) => `<div class="banner warn">${icon("alert")}<div>${esc(n)}</div></div>`).join("")}`;
  }
  return `<div class="card"><h2>Próximo pronóstico</h2><div class="stack" style="margin-top:12px;gap:10px">${line}</div></div>`;
}

function strategyCard(d, draft) {
  const st = d.estado;
  const shown = d.opciones.filter((o) => !HIDDEN_STRATEGIES.includes(o.id) || o.id === d.ajustes.estrategia || o.id === draft.estrategia);
  const rows = shown.map((o) => {
    const note = o.mes_en_curso && st && st.estado === "complete" ? '<small class="note">Con el mes completo se usa un mes de anticipación.</small>' : "";
    return `<label title="${esc(o.texto)}"><input type="radio" name="s-est" value="${o.id}" ${draft.estrategia === o.id ? "checked" : ""}><b>${esc(o.nombre)}</b> ${o.recomendada ? '<span class="badge ok">Recomendada</span>' : ""}
      ${accuracyChips(o.acierto)}${note}</label>`;
  }).join("");
  return `<div class="card"><h2>Estrategia de pronóstico</h2><div class="choice" id="s-options">${rows}</div></div>`;
}

function cutoffCard(d, draft) {
  const st = d.estado;
  const off = !st || st.estado === "complete";
  const manual = draft.corte_manual !== null;
  const max = st && st.dia_de_corte ? st.dia_de_corte : 31;
  const value = manual && !Number.isNaN(draft.corte_manual) ? draft.corte_manual : "";
  const low = manual && draft.corte_manual < d.corte_minimo;
  const bad = manual && !(Number.isInteger(draft.corte_manual) && draft.corte_manual >= 1 && draft.corte_manual <= max);
  const text = off ? '<p class="hint" style="margin-top:6px">El último mes está completo, así que no hace falta un día de corte.</p>' : "";
  return `<div class="card"><h2>Día de corte</h2>${text}
    <div class="cutoffrow"><div class="seg" id="s-mode"><button data-v="auto" class="${manual ? "" : "on"}" ${off ? "disabled" : ""}>Automático${st && st.dia_de_corte ? ` (día ${st.dia_de_corte})` : ""}</button>
      <button data-v="manual" class="${manual ? "on" : ""}" ${off ? "disabled" : ""}>Manual</button></div>
      <label class="hint" for="s-cut">Día</label><input class="num" type="number" id="s-cut" min="1" max="${max}" step="1" value="${value}" ${!manual || off ? "disabled" : ""}></div>
    ${bad ? `<p class="note error-text">El día debe ser un número entre 1 y ${max}.</p>` : ""}
    ${low && !bad ? `<p class="note">Bajo el día ${d.corte_minimo} el mes en curso aporta poco: con la estrategia automática se usa la directa en ese caso.</p>` : ""}</div>`;
}

function foldCard(key, title, body) {
  return `<div class="card"><details class="howto" data-fold="${key}" ${settingsState.open[key] ? "open" : ""}><summary>${title}</summary>${body}</details></div>`;
}

function restMethodCard(d, draft) {
  const rows = d.metodos_resto.map((m) => `<label title="${esc(m.texto)}"><input type="radio" name="s-met" value="${m.id}" ${draft.metodo_resto === m.id ? "checked" : ""}><b>${esc(m.nombre)}</b> ${m.recomendada ? '<span class="badge ok">Recomendada</span>' : ""}
      ${accuracyChips(m.acierto)}</label>`).join("");
  return foldCard("rest", "Cómo se estima el resto del mes en curso", `<p class="hint">Solo cambia la tarjeta "El mes en curso" de Predicción; el pronóstico no cambia.</p><div class="choice" id="s-methods">${rows}</div>`);
}

function curveOption(d) {
  const days = d.curva_parcial.map((r) => String(r.dia));
  const option = baseOption(days, (v) => `día ${v}`);
  option.legend = { top: 0, left: 8, icon: "circle", itemWidth: 8, itemHeight: 8, itemGap: 16, textStyle: { color: INK2, fontSize: 12 } };
  option.grid.top = 40;
  option.yAxis.scale = true;
  option.yAxis.min = 64;
  option.yAxis.axisLabel.formatter = (v) => `${NUM0.format(v)}%`;
  const references = d.opciones.filter((o) => REFERENCE_LINES[o.id] && o.acierto);
  const sku = {
    name: "SKU", type: "line", data: d.curva_parcial.map((r) => r.sku), symbol: "circle", symbolSize: 7, lineStyle: { width: 2, color: SERIES_COLORS[0] }, itemStyle: { color: SERIES_COLORS[0] },
    markLine: { silent: true, symbol: "none", lineStyle: { type: "dashed", color: SERIES_COLORS[0], opacity: 0.55, width: 1.4 },
      label: { color: INK2, fontSize: 11, position: "insideEndBottom" },
      data: [...references.map((o) => ({ yAxis: o.acierto.sku, label: { formatter: `${REFERENCE_LINES[o.id]} (${fmt(o.acierto.sku)}%)`, position: "insideEndTop" } })),
        { xAxis: String(d.corte_minimo), lineStyle: { type: "solid", color: ORANGE, opacity: 0.5 }, label: { formatter: `desde el día ${d.corte_minimo}`, position: "insideStartTop", color: "#b34a00" } }] },
  };
  option.series = [
    sku,
    { name: "Planta", type: "line", data: d.curva_parcial.map((r) => r.planta), symbol: "circle", symbolSize: 7, lineStyle: { width: 2, color: SERIES_COLORS[2] }, itemStyle: { color: SERIES_COLORS[2] } },
    { name: "Total", type: "line", data: d.curva_parcial.map((r) => r.total), symbol: "circle", symbolSize: 7, lineStyle: { width: 2, color: SERIES_COLORS[1] }, itemStyle: { color: SERIES_COLORS[1] } },
  ];
  option.tooltip.formatter = (items) => {
    const rows = items.map((i) => `<div style="display:flex;justify-content:space-between;gap:18px"><span><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${i.color};margin-right:7px"></i>${esc(i.seriesName)}</span><b>${fmt(i.value)}%</b></div>`).join("");
    return `<div style="font-size:12.5px;color:${INK};min-width:150px"><div style="color:${INK2};margin-bottom:5px">Con ${items[0].axisValue} días del mes en curso</div>${rows}</div>`;
  };
  return option;
}

function paintSettings() {
  const { data: d, draft } = settingsState;
  const saved = d.ajustes;
  const dirty = !sameSettings(draft, saved);
  const st = d.estado;
  const max = st && st.dia_de_corte ? st.dia_de_corte : 31;
  const invalid = draft.corte_manual !== null && !(Number.isInteger(draft.corte_manual) && draft.corte_manual >= 1 && draft.corte_manual <= max);
  disposeCharts();
  $("#view").innerHTML = `<div class="stack">${settingsStateCard(d)}${strategyCard(d, draft)}${cutoffCard(d, draft)}${restMethodCard(d, draft)}
    ${foldCard("curve", "Cuánto mejora el pronóstico según el día de corte", `<div class="toolbar"><p class="hint">Pruebas de los últimos 12 meses. Las líneas punteadas son la directa (sin el mes en curso) y un mes de anticipación.</p><button class="btn" data-png="acierto_por_dia_de_corte">Imagen (PNG)</button></div><div class="chart" id="s-chart"></div>`)}
    <div class="card gen"><div><h2>Guardar los ajustes</h2><p>${dirty ? "Hay cambios sin guardar." : "Guardado."}</p></div>
      <button class="btn primary" id="s-save" ${!dirty || invalid ? "disabled" : ""}>Guardar</button><button class="btn" id="s-reset" ${saved.estrategia === "automatica" && saved.corte_manual === null && saved.metodo_resto === "media" && !dirty ? "disabled" : ""}>Volver a automático</button>
      <div class="grow" id="s-msg">${settingsState.message}</div></div></div>`;
  const mountCurve = () => { const el = $("#s-chart"); if (el && !el.firstChild && el.clientWidth > 0) mountChart(el, curveOption(d)); };
  document.querySelectorAll("details[data-fold]").forEach((el) => {
    el.addEventListener("toggle", () => { settingsState.open[el.dataset.fold] = el.open; if (el.open && el.dataset.fold === "curve") mountCurve(); });
  });
  mountCurve();

  $("#s-options").onchange = (event) => { if (event.target.name === "s-est") { draft.estrategia = event.target.value; paintSettings(); } };
  $("#s-methods").onchange = (event) => { if (event.target.name === "s-met") { draft.metodo_resto = event.target.value; paintSettings(); } };
  const mode = $("#s-mode");
  mode.onclick = (event) => {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    draft.corte_manual = button.dataset.v === "manual" ? (draft.corte_manual === null ? (st && st.dia_de_corte) || 15 : draft.corte_manual) : null;
    paintSettings();
  };
  $("#s-cut").onchange = (event) => { draft.corte_manual = event.target.value === "" ? NaN : Number(event.target.value); paintSettings(); };
  $("#s-save").onclick = () => saveSettings(draft);
  $("#s-reset").onclick = () => saveSettings({ estrategia: "automatica", corte_manual: null, metodo_resto: "media" });
}

async function saveSettings(values) {
  $("#s-save").disabled = true;
  const reply = await apiPut("ajustes", values);
  if (!reply.ok) {
    $("#s-msg").innerHTML = `<div class="banner warn">${icon("alert")}<div>${esc(reply.body.detalle || "No se pudo guardar.")}</div></div>`;
    $("#s-save").disabled = false;
    return;
  }
  settingsState.data = reply.body;
  settingsState.draft = { ...reply.body.ajustes };
  settingsState.message = "";
  paintSettings();
}
