"use strict";
/* Ajustes: which strategy the forecast uses and the cutoff day. Loaded before app.js, which owns the helpers (api, fmt, icon, mountChart, ...) used here at call time. */

const settingsState = { data: null, draft: null, message: "" };
const MONTH_STATE = { complete: ["ok", "check", "Mes completo"], in_progress: ["warn", "alert", "Mes en curso"] };
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
const sameSettings = (a, b) => a.estrategia === b.estrategia && a.corte_manual === b.corte_manual;

function accuracyChips(a) {
  if (!a) return "";
  const chip = (label, v) => (v === null || v === undefined ? "" : `<span class="badge neutral">${label} ${fmt(v)}%</span>`);
  return `<div class="chiprow">${chip("SKU", a.sku)}${chip("planta", a.planta)}${chip("total", a.total)}</div>`;
}

function settingsStateCard(d) {
  const st = d.estado, plan = d.plan;
  if (!st) {
    return `<div class="card"><h2>Estado de los datos</h2><p class="hint" style="margin-top:6px">Todavía no hay datos. Carga los exportes de SAP en la sección Datos; mientras tanto puedes elegir la estrategia.</p></div>`;
  }
  const [cls, ic, label] = MONTH_STATE[st.estado];
  const facts = `<div class="facts"><div><span>Último registro</span><b>${dateLong(st.ultimo_registro)}</b></div>
    <div><span>Último mes</span><b>${monthLong(st.mes)} <span class="badge ${cls}">${icon(ic)}${label}</span></b></div>
    <div><span>Día de corte detectado</span><b>${st.dia_de_corte === null ? "No hace falta" : `Día ${st.dia_de_corte}`}</b></div>
    <div><span>Días hábiles del mes en curso</span><b>${st.estado === "complete" ? "–" : `${st.dias_vistos} vistos · ${st.dias_restantes} por venir`}</b></div></div>`;
  let line;
  if (d.error) {
    line = `<div class="banner warn">${icon("alert")}<div><b>Los ajustes guardados no sirven con estos datos.</b><br>${esc(d.error)}</div></div>`;
  } else if (plan) {
    const uses = plan.corte ? ` y los primeros ${plan.corte} días de ${monthLong(st.mes)}` : "";
    line = `<div class="banner info">${icon("info")}<div>Con lo que está guardado, el próximo pronóstico es el de <b>${monthLong(plan.mes_pronosticado)}</b>, con «${esc(optionName(plan.estrategia))}»,
      partiendo de ${monthLong(plan.mes_origen)}${uses}.</div></div>${plan.avisos.map((n) => `<div class="banner warn">${icon("alert")}<div>${esc(n)}</div></div>`).join("")}`;
  } else {
    line = "";
  }
  return `<div class="card"><h2>Estado de los datos</h2><p class="hint" style="margin-top:6px">La app lee el último registro de los datos para saber si el mes terminó y cuántos días lleva. Así no hace falta pronosticar un día fijo del mes.</p>${facts}<div class="stack" style="margin-top:14px;gap:10px">${line}</div></div>`;
}

function strategyCard(d, draft) {
  const st = d.estado;
  const rows = d.opciones.map((o) => {
    const note = o.mes_en_curso && st && st.estado === "complete" ? '<small class="note">Con el mes completo se usa un mes de anticipación.</small>' : "";
    return `<label><input type="radio" name="s-est" value="${o.id}" ${draft.estrategia === o.id ? "checked" : ""}><b>${esc(o.nombre)}</b> ${o.recomendada ? '<span class="badge ok">Recomendada</span>' : ""}
      <small>${esc(o.texto)}</small>${accuracyChips(o.acierto)}${note}</label>`;
  }).join("");
  return `<div class="card"><h2>Estrategia de pronóstico</h2><p class="hint" style="margin-top:6px">Cómo se calcula el pronóstico cuando el planificador lo pide a mitad de mes. El acierto es el de la prueba de los últimos 12 meses, por SKU, planta y total.
    Para reentrenar los modelos usa la sección Hiperparámetros.</p><div class="choice" id="s-options">${rows}</div></div>`;
}

function cutoffCard(d, draft) {
  const st = d.estado;
  const off = !st || st.estado === "complete";
  const manual = draft.corte_manual !== null;
  const max = st && st.dia_de_corte ? st.dia_de_corte : 31;
  const value = manual && !Number.isNaN(draft.corte_manual) ? draft.corte_manual : "";
  const low = manual && draft.corte_manual < d.corte_minimo;
  const bad = manual && !(Number.isInteger(draft.corte_manual) && draft.corte_manual >= 1 && draft.corte_manual <= max);
  const text = off
    ? "El último mes está completo, así que no hace falta un día de corte."
    : `Cuántos días del mes en curso usa el pronóstico. En automático son todos los que hay (hasta el día ${st.dia_de_corte}). Uno manual sirve para repetir un pronóstico con menos información; no puede pasar del último registro.`;
  return `<div class="card"><h2>Día de corte</h2><p class="hint" style="margin-top:6px">${text}</p>
    <div class="cutoffrow"><div class="seg" id="s-mode"><button data-v="auto" class="${manual ? "" : "on"}" ${off ? "disabled" : ""}>Automático${st && st.dia_de_corte ? ` (día ${st.dia_de_corte})` : ""}</button>
      <button data-v="manual" class="${manual ? "on" : ""}" ${off ? "disabled" : ""}>Manual</button></div>
      <label class="hint" for="s-cut">Día</label><input class="num" type="number" id="s-cut" min="1" max="${max}" step="1" value="${value}" ${!manual || off ? "disabled" : ""}></div>
    ${bad ? `<p class="note error-text">El día debe ser un número entre 1 y ${max}.</p>` : ""}
    ${low && !bad ? `<p class="note">Bajo el día ${d.corte_minimo} el mes en curso aporta poco: con la estrategia automática se usa la directa en ese caso.</p>` : ""}</div>`;
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
  $("#view").innerHTML = `<div class="stack">${settingsStateCard(d)}${strategyCard(d, draft)}${cutoffCard(d, draft)}
    <div class="card"><div class="toolbar"><div><h2>Cuánto mejora el pronóstico según el día de corte</h2><p class="hint">Acierto de «Dos meses, directa y días del mes en curso» según los días del mes en curso que se conocen. Pruebas de los últimos 12 meses.
      Las líneas punteadas son la directa (sin el mes en curso) y el pronóstico con un mes de anticipación.</p></div><button class="btn" data-png="acierto_por_dia_de_corte">Imagen (PNG)</button></div><div class="chart" id="s-chart"></div></div>
    <div class="card gen"><div><h2>Guardar los ajustes</h2><p>${dirty ? "Hay cambios sin guardar." : "Los ajustes están guardados."} El próximo pronóstico que generes usa lo que está guardado.</p></div>
      <button class="btn primary" id="s-save" ${!dirty || invalid ? "disabled" : ""}>Guardar</button><button class="btn" id="s-reset" ${saved.estrategia === "automatica" && saved.corte_manual === null && !dirty ? "disabled" : ""}>Volver a automático</button>
      <div class="grow" id="s-msg">${settingsState.message}</div></div></div>`;
  mountChart($("#s-chart"), curveOption(d));

  $("#s-options").onchange = (event) => { if (event.target.name === "s-est") { draft.estrategia = event.target.value; paintSettings(); } };
  const mode = $("#s-mode");
  mode.onclick = (event) => {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    draft.corte_manual = button.dataset.v === "manual" ? (draft.corte_manual === null ? (st && st.dia_de_corte) || 15 : draft.corte_manual) : null;
    paintSettings();
  };
  $("#s-cut").onchange = (event) => { draft.corte_manual = event.target.value === "" ? NaN : Number(event.target.value); paintSettings(); };
  $("#s-save").onclick = () => saveSettings(draft);
  $("#s-reset").onclick = () => saveSettings({ estrategia: "automatica", corte_manual: null });
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
  settingsState.message = '<span class="badge ok">' + icon("check") + "Guardado</span>";
  paintSettings();
}
