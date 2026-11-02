"use strict";
/* Ajustes: which strategy the forecast uses and the cutoff day. Loaded before app.js, which owns the helpers (api, fmt, icon, mountChart, ...) used here at call time. */

const settingsState = { data: null, draft: null, message: "" };
const HIDDEN_STRATEGIES = ["iterada"];  // not recommended: only listed while it is the saved choice

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
    ${low && !bad ? `<p class="note">Bajo el día ${d.corte_minimo} el mes en curso aporta poco: la media diaria con que se completa el mes es poco fiable.</p>` : ""}</div>`;
}

function paintSettings() {
  const { data: d, draft } = settingsState;
  const saved = d.ajustes;
  const dirty = !sameSettings(draft, saved);
  const st = d.estado;
  const max = st && st.dia_de_corte ? st.dia_de_corte : 31;
  const invalid = draft.corte_manual !== null && !(Number.isInteger(draft.corte_manual) && draft.corte_manual >= 1 && draft.corte_manual <= max);
  $("#view").innerHTML = `<div class="stack">${settingsStateCard(d)}${strategyCard(d, draft)}${cutoffCard(d, draft)}
    <div class="card gen"><div><h2>Guardar los ajustes</h2><p>${dirty ? "Hay cambios sin guardar." : "Guardado."}</p></div>
      <button class="btn primary" id="s-save" ${!dirty || invalid ? "disabled" : ""}>Guardar</button><button class="btn" id="s-reset" ${saved.estrategia === "automatica" && saved.corte_manual === null && !dirty ? "disabled" : ""}>Volver a automático</button>
      <div class="grow" id="s-msg">${settingsState.message}</div></div></div>`;

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
  settingsState.message = "";
  paintSettings();
}
