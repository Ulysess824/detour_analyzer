"use strict";
/* Datos: load new SAP exports. The files are uploaded, reviewed against the current table (nothing is written), and applied only on request. */

const LEVEL_BADGE = { error: ["bad", "cross", "Error"], warning: ["warn", "alert", "Aviso"] };
let reviewState = { id: null, politica: "detener" };

async function renderData(view) {
  clearTimeout(pollTimer);
  const s = await api("datos");
  const a = s.actual;
  const current = a
    ? `<div class="facts"><div><span>Último día cargado</span><b>${dateLong(a.ultimo_dia)}</b></div><div><span>Desde</span><b>${dateLong(a.primer_dia)}</b></div><div><span>Tamaño</span><b>${fmt(a.filas, 0)} filas</b></div></div>
       <p class="note">${fmt(a.series, 0)} series en ${a.plantas} plantas.${a.mes_parcial ? ` El mes ${monthLong(a.mes_parcial)} está incompleto.` : ""}</p>`
    : `<p class="hint" style="margin-top:6px">Todavía no hay datos. Carga el primer exporte de SAP.</p>`;
  const last = s.respaldos[0];
  const undo = last ? `<div class="rowbtns"><button class="btn" id="d-undo">Deshacer la última carga</button><span class="note" style="margin:0">Vuelve a los datos que había antes (${dateLong(last.creado_en)}).</span></div><div id="d-undo-box"></div>` : "";
  view.innerHTML = `<div class="stack">
    <div class="card"><h2>Datos actuales</h2>${current}${undo}</div>
    <div class="card"><h2>Cargar datos nuevos</h2>
      <p class="hint" style="margin-top:6px;line-height:1.5">Elige el exporte de consumo de SAP (Excel, .xlsx) con los días nuevos. Antes de guardar nada se revisa: los días que ya estaban deben traer los mismos valores y se aplican las mismas validaciones de siempre. Los datos de antes se guardan como respaldo.</p>
      ${formatHelpHtml()}
      <label class="drop" id="d-drop"><input type="file" id="d-files" accept=".xlsx" multiple><b>Elige los archivos</b> o suéltalos aquí<span id="d-list" class="note"></span></label>
      <div class="rowbtns"><button class="btn primary" id="d-review" disabled>Revisar archivos</button></div>
      <div id="d-progress"></div></div>
    <div id="d-result"></div></div>`;
  bindData();
  showFormatExample();
  if (reviewState.id) {  // coming back to the screen with a review that was not applied yet
    const box = $("#d-result");
    try { box.innerHTML = '<div class="loading">Cargando…</div>'; const running = (await api("trabajos/activo")).trabajo; if (running && running.tipo === "datos") watchData(running.id); else box.innerHTML = ""; } catch (_) { box.innerHTML = ""; }
  }
  const undoButton = $("#d-undo");
  if (undoButton) undoButton.onclick = () => {
    $("#d-undo-box").innerHTML = `<div class="confirm">Los datos vuelven a como estaban antes de la última carga y se pierden los días que esa carga agregó. ¿Seguro?<button class="btn primary" id="d-yes">Sí, deshacer</button><button class="btn" id="d-no">Cancelar</button></div>`;
    $("#d-no").onclick = () => ($("#d-undo-box").innerHTML = "");
    $("#d-yes").onclick = async () => {
      const reply = await apiPost("datos/deshacer", {});
      if (!reply.ok) { $("#d-undo-box").innerHTML = `<div class="banner warn" style="margin-top:12px">${icon("alert")}<div>${esc(reply.body.detalle || "No se pudo deshacer.")}</div></div>`; return; }
      loadFooter();
      route();
    };
  };
}

/* how the raw export must look, with an example of invented data (the rows are in src/app/example_service.py) */
function formatHelpHtml() {
  return `<details class="howto" open><summary>Cómo debe ser el archivo (con un ejemplo)</summary>
    <ol>
      <li><b>Un archivo de Excel (.xlsx).</b> Puedes subir varios a la vez. Se lee la primera hoja, o la hoja llamada <code>RESULT</code>.</li>
      <li><b>Una fila por planta, material y día con consumo.</b> Los días sin consumo no hacen falta.</li>
      <li><b>La fila 1 lleva los títulos</b> <code>Customer</code>, <code>Material</code>, <code>Day</code> y <code>TO</code>, escritos así. Justo después de <code>Material</code> hay una columna sin título con la <b>descripción del SKU</b> (por ejemplo <code>K/01/125gsm/2000mm/1200-1400</code>), no el código numérico.</li>
      <li class="caution"><b><code>Day</code> debe ser texto con el formato <code>dd.mm.aaaa</code></b>, por ejemplo <code>17.02.2026</code>. Una fecha de Excel o un texto como <code>2026-02-17</code> no se lee: esas filas se descartan sin aviso.</li>
      <li><b><code>TO</code> es el consumo en toneladas</b>, un número con punto decimal. Los valores negativos cuentan como 0 y las filas sin valor se ignoran.</li>
      <li><b>Días ya cargados:</b> pueden repetirse, pero con los mismos valores. No deben quedar meses enteros sin filas entre lo que ya está y lo nuevo.</li>
    </ol>
    <p class="note">Así se ve el archivo (los datos son inventados: plantas <code>DEMO1</code> y <code>DEMO2</code>).</p>
    <div id="d-format"></div>
    <p class="note">También se acepta la tabla dinámica de SAP en formato ancho (hoja <code>RESULT</code>, un día por columna).</p>
  </details>`;
}

async function showFormatExample() {
  const box = $("#d-format");
  try {
    const f = await api("datos/formato");
    box.innerHTML = `<div class="tablewrap"><table class="data sheet"><thead><tr><th class="rownum"></th>${f.columnas.map((c) => `<th>${c.letra}</th>`).join("")}</tr></thead>
      <tbody><tr class="titles"><td class="rownum">1</td>${f.columnas.map((c) => `<td>${c.titulo ? esc(c.titulo) : '<span class="untitled">(sin título)</span>'}</td>`).join("")}</tr>
      ${f.filas.map((row, i) => `<tr><td class="rownum">${i + 2}</td>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  } catch (_) { box.innerHTML = ""; }
}

function bindData() {
  const input = $("#d-files"), list = $("#d-list"), button = $("#d-review"), drop = $("#d-drop");
  let chosen = [];
  const show = () => { list.textContent = chosen.length ? chosen.map((f) => f.name).join(", ") : ""; button.disabled = !chosen.length; };
  input.onchange = () => { chosen = [...input.files]; show(); };
  drop.ondragover = (event) => { event.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (event) => { event.preventDefault(); drop.classList.remove("over"); chosen = [...event.dataTransfer.files]; show(); };
  button.onclick = async () => {
    button.disabled = true;
    $("#d-result").innerHTML = "";
    $("#d-progress").innerHTML = '<p class="note">Subiendo los archivos…</p>';
    const form = new FormData();
    chosen.forEach((file) => form.append("files", file));
    const response = await fetch("/api/datos/subir", { method: "POST", body: form });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) { button.disabled = false; $("#d-progress").innerHTML = `<div class="banner warn" style="margin-top:14px">${icon("alert")}<div>${esc(body.detalle || "No se pudieron subir los archivos.")}</div></div>`; return; }
    reviewState = { id: body.id, politica: "detener" };
    startReview();
  };
}

async function startReview() {
  const reply = await apiPost("datos/revisar", { id: reviewState.id, politica: reviewState.politica });
  if (!reply.ok) { $("#d-progress").innerHTML = `<div class="banner warn" style="margin-top:14px">${icon("alert")}<div>${esc(reply.body.detalle)}</div></div>`; return; }
  watchData(reply.body.trabajo);
}

function watchData(jobId) {
  const tick = async () => {
    const box = $("#d-progress");
    if (!box) return;
    const job = await api(`trabajos/${jobId}`);
    if (job.estado === "corriendo") {
      box.innerHTML = `<div class="progress"><div class="bar" style="width:${Math.max(4, Math.round(job.progreso * 100))}%"></div></div><p class="note">${esc(job.mensaje)} · ${fmt(job.segundos, 0)} s</p>`;
      pollTimer = setTimeout(tick, 600);
    } else if (job.estado === "listo") {
      box.innerHTML = "";
      $("#d-result").innerHTML = reviewHtml(job.resultado);
      bindReview(job.resultado);
    } else {
      box.innerHTML = `<div class="banner warn" style="margin-top:14px">${icon("alert")}<div><b>No se pudo revisar.</b><br>${esc(job.error)}</div></div>`;
      $("#d-review").disabled = false;
    }
  };
  tick();
}

function findingHtml(f) {
  const [cls, ic, label] = LEVEL_BADGE[f.nivel];
  let examples = "";
  if (f.ejemplos.length) {
    const columns = Object.keys(f.ejemplos[0]);
    examples = `<div class="tablewrap" style="margin-top:8px"><table class="data left3"><thead><tr>${columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${f.ejemplos.map((r) => `<tr>${columns.map((c) => `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>${f.cantidad > f.ejemplos.length ? `<p class="note">Se muestran ${f.ejemplos.length} de ${fmt(f.cantidad, 0)}.</p>` : ""}`;
  }
  return `<div class="finding"><span class="badge ${cls}">${icon(ic)}${label}</span><div><b>${esc(f.titulo)}</b><p>${esc(f.mensaje)}${f.consejo ? " " + esc(f.consejo) : ""}</p>${examples}</div></div>`;
}

function reviewHtml(r) {
  const s = r.resumen;
  const kpi = (label, value, sub) => `<div class="card stat" style="min-height:0"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const files = r.archivos.map((f) => `${esc(f.nombre)}: ${fmt(f.filas, 0)} filas, del ${dateLong(f.desde)} al ${dateLong(f.hasta)}`).join("<br>");
  const conflict = s.conflictos > 0 ? `<div class="rowbtns"><b style="font-size:13.5px">¿Qué valor es el correcto?</b>
      ${r.politica === "detener" ? '<button class="btn" id="d-use-new">Usar los valores del exporte nuevo</button>' : '<button class="btn" id="d-keep-old">Volver a no cargar nada</button>'}</div>` : "";
  const verdict = r.puede_aplicar
    ? `<span class="badge ok">${icon("check")}Se puede cargar</span>`
    : `<span class="badge bad">${icon("cross")}No se puede cargar todavía</span>`;
  return `<div class="stack"><div class="card"><h2>Revisión de los archivos</h2><p class="hint" style="margin-top:6px">${files}</p><div style="margin-top:12px">${verdict}</div>
      <div class="grid cols-4" style="margin-top:14px">
        ${kpi("Filas nuevas", fmt(s.filas_nuevas, 0), s.meses_nuevos.length ? `Agrega ${s.meses_nuevos.map(monthLong).join(", ")}.` : "No agrega meses nuevos.")}
        ${kpi("Días que ya estaban", fmt(s.filas_repetidas_iguales, 0), "Con el mismo valor: se dejan una sola vez.")}
        ${kpi("Valores distintos", fmt(s.conflictos, 0), s.conflictos ? "Hay que elegir cuál es el correcto." : "Ninguno.")}
        ${kpi("Último día", dateLong(s.ultimo_dia_resultado), `Antes: ${s.ultimo_dia_actual ? dateLong(s.ultimo_dia_actual) : "sin datos"}.`)}
      </div>${conflict}</div>
    ${r.hallazgos.length ? `<div class="card"><h2>Lo que encontró la revisión</h2><div class="findings">${r.hallazgos.map(findingHtml).join("")}</div></div>` : ""}
    <div class="card"><div class="rowbtns" style="margin-top:0"><button class="btn primary" id="d-apply" ${r.puede_aplicar ? "" : "disabled"}>Cargar estos datos</button><button class="btn" id="d-cancel">Cancelar</button>
      <span class="note" style="margin:0">Los datos de antes quedan como respaldo y se pueden recuperar con "Deshacer la última carga".</span></div><div id="d-apply-box"></div></div></div>`;
}

function bindReview(r) {
  const again = (policy) => { reviewState.politica = policy; $("#d-result").innerHTML = ""; startReview(); };
  if ($("#d-use-new")) $("#d-use-new").onclick = () => again("usar_nuevo");
  if ($("#d-keep-old")) $("#d-keep-old").onclick = () => again("detener");
  $("#d-cancel").onclick = async () => { await apiPost("datos/cancelar", { id: r.id }); reviewState = { id: null, politica: "detener" }; route(); };
  $("#d-apply").onclick = async () => {
    $("#d-apply").disabled = true;
    const reply = await apiPost("datos/aplicar", { id: r.id });
    if (!reply.ok) { $("#d-apply").disabled = false; $("#d-apply-box").innerHTML = `<div class="banner warn" style="margin-top:12px">${icon("alert")}<div>${esc(reply.body.detalle || "No se pudo cargar.")}</div></div>`; return; }
    reviewState = { id: null, politica: "detener" };
    loadFooter();
    const done = reply.body;
    $("#d-result").innerHTML = `<div class="card"><h2>Listo: datos cargados</h2><p class="hint" style="margin-top:6px">${fmt(done.filas_nuevas, 0)} filas nuevas. Los datos llegan hasta el ${dateLong(done.ultimo_dia)}. ${done.respaldo ? "Los datos de antes quedaron en el respaldo " + esc(done.respaldo) + "." : ""}</p>
      <div class="rowbtns"><button class="btn primary" id="d-go">Ir a Predicción</button><button class="btn" id="d-again">Cargar otro archivo</button></div></div>`;
    $("#d-go").onclick = () => (location.hash = "#/prediccion");
    $("#d-again").onclick = () => route();
  };
}
