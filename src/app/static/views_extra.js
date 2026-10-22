"use strict";
/* Descargar, Historial and Ayuda. */

const downloadState = { planta: "", sku: "", nivel: "mensual", mesPron: "", mesSeg: "" };

/* inside the desktop window a native "save as" dialog is used (python side: WindowApi.save_file); in a browser a normal download */
function nativeSave() { return window.pywebview && window.pywebview.api && window.pywebview.api.save_file ? window.pywebview.api.save_file : null; }
function toBase64(blob) {
  return new Promise((resolve) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(",")[1]); reader.readAsDataURL(blob); });
}

async function saveBlob(blob, name) {
  const native = nativeSave();
  if (native) { await native(name, await toBase64(blob)); return; }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 5000);
}

async function downloadExcel(path, params, messageBox) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v));
  messageBox.innerHTML = '<span class="note">Preparando el archivo…</span>';
  const response = await fetch(`/api/descargas/${path}?${query}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    messageBox.innerHTML = `<span class="note error-text">${esc(body.detalle || "No se pudo preparar el archivo.")}${(body.sugerencias || []).length ? " ¿Quisiste decir " + body.sugerencias.map(esc).join(", ") + "?" : ""}</span>`;
    return;
  }
  const name = /filename="([^"]+)"/.exec(response.headers.get("Content-Disposition") || "")?.[1] || "descarga.xlsx";
  saveBlob(await response.blob(), name);
  messageBox.innerHTML = `<span class="note">Listo: ${esc(name)}</span>`;
}

/* a chart as an image: every chart has a button with data-png="file name" */
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-png]");
  if (!button) return;
  if (!charts.length) { button.title = "Cambia a la vista de gráfico para guardar la imagen."; return; }
  const url = charts[0].getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#ffffff" });
  const native = nativeSave();
  if (native) { native(`${button.dataset.png}.png`, url.split(",")[1]); return; }
  const link = document.createElement("a");
  link.href = url;
  link.download = `${button.dataset.png}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
});

/* ---- Descargar -------------------------------------------------------------------------------------------------------------- */
async function renderDownloads(view) {
  const [list, tracking] = await Promise.all([api("pronosticos"), api("seguimiento")]);
  const d = downloadState;
  const months = list.pronosticos.map((f) => `${f.mes}|${f.estrategia}`);
  const trackItems = [...tracking.produccion.map((r) => ({ ...r, origen: "produccion" })), ...tracking.backtest.map((r) => ({ ...r, origen: "backtest" }))];
  const trackMonths = trackItems.sort((a, b) => (a.mes < b.mes ? 1 : a.mes > b.mes ? -1 : a.origen < b.origen ? 1 : -1)).map((r) => `${r.mes}|${r.estrategia}|${r.origen}`);
  const options = (items, selected) => items.map((m) => {
    const [month, strategy, origin] = m.split("|");
    return `<option value="${m}" ${m === selected ? "selected" : ""}>${monthLong(month)}${strategy ? ` · ${esc(STRATEGY_NAMES[strategy] || strategy)}` : ""}${origin === "backtest" ? " · prueba histórica" : ""}</option>`;
  }).join("");
  view.innerHTML = `<div class="stack"><div class="tiles">
    <div class="card tile"><h2>Pronóstico (Excel)</h2><p>El pronóstico de un mes: resumen, por planta y cada SKU con los tres modelos, su rango y el último mes real.</p>
      ${months.length ? `<select class="pick" id="x-pron">${options(months, d.mesPron)}</select><div class="rowbtns"><button class="btn primary" id="x-pron-go">Descargar Excel</button></div>` : '<p class="note">Todavía no hay pronósticos.</p>'}<div id="x-pron-msg"></div></div>
    <div class="card tile"><h2>Seguimiento (Excel)</h2><p>El acierto de cada mes y el detalle de uno: modelos, plantas y las series donde más falló.</p>
      ${trackMonths.length ? `<select class="pick" id="x-seg">${options(trackMonths, d.mesSeg)}</select><div class="rowbtns"><button class="btn primary" id="x-seg-go">Descargar Excel</button></div>` : '<p class="note">Todavía no hay seguimiento.</p>'}<div id="x-seg-msg"></div></div>
    <div class="card tile"><h2>Gráficos (imagen)</h2><p>Cada gráfico de la app tiene un botón <b>Imagen (PNG)</b> arriba a la derecha. Guarda el gráfico tal como lo ves, con los filtros que tengas puestos.</p>
      <div class="rowbtns"><button class="btn" id="x-go-consumo">Ir a Consumo real</button></div></div></div>
    <div class="card"><h2>Consumo real (Excel)</h2><p class="hint">Con los filtros que elijas: el total por mes y una fila por serie (o los días de un rango).</p><div style="margin-top:14px">${filtersHtml(d)}</div>
      <div class="rowbtns"><div class="seg" id="x-nivel"><button data-v="mensual" class="${d.nivel === "mensual" ? "on" : ""}">Por mes</button><button data-v="diario" class="${d.nivel === "diario" ? "on" : ""}">Últimos 90 días</button></div>
      <button class="btn primary" id="x-cons-go">Descargar Excel</button></div><div id="x-cons-msg"></div></div></div>`;
  bindFilters(d, () => {});
  const select = (id, key) => { const el = $(id); if (el) { d[key] = el.value; el.onchange = () => (d[key] = el.value); } };
  select("#x-pron", "mesPron"); select("#x-seg", "mesSeg");
  if ($("#x-pron-go")) $("#x-pron-go").onclick = () => { const [mes, estrategia] = (d.mesPron || $("#x-pron").value).split("|"); downloadExcel("pronostico", { mes, estrategia }, $("#x-pron-msg")); };
  if ($("#x-seg-go")) $("#x-seg-go").onclick = () => { const [mes, estrategia, origen] = (d.mesSeg || $("#x-seg").value).split("|"); downloadExcel("seguimiento", { mes, estrategia, origen }, $("#x-seg-msg")); };
  $("#x-nivel").onclick = (event) => { const b = event.target.closest("button"); if (b) { d.nivel = b.dataset.v; $("#x-nivel").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); } };
  $("#x-cons-go").onclick = () => downloadExcel("consumo", { planta: $("#f-planta").value, sku: $("#f-sku").value, nivel: d.nivel }, $("#x-cons-msg"));
  $("#x-go-consumo").onclick = () => (location.hash = "#/consumo");
}

/* ---- Historial -------------------------------------------------------------------------------------------------------------- */
async function renderHistory(view) {
  const list = (await api("historial")).pronosticos;
  if (!list.length) { view.innerHTML = '<div class="card empty"><h3>Todavía no hay pronósticos guardados</h3><p>Genera el primero en la sección Predicción.</p></div>'; return; }
  const rows = list.map((r) => {
    const done = r.medido ? `<span class="badge ok">${icon("check")}Medido: ${fmt(r.medido.sku)} / ${fmt(r.medido.planta)} / ${fmt(r.medido.total)}%</span>`
      : r.por_medir ? `<span class="badge warn">${icon("alert")}Por medir</span>` : '<span class="badge neutral">Mes sin cerrar</span>';
    return `<tr><td>${monthLong(r.mes)}</td><td style="white-space:normal;min-width:170px;max-width:230px">${esc(STRATEGY_NAMES[r.estrategia] || r.estrategia)}</td><td>${r.creado_en ? dateLong(r.creado_en) : "–"}</td><td>${fmt(r.total_TO, 0)}</td><td>${r.datos_hasta ? dateLong(r.datos_hasta) : "–"}</td><td>${esc(r.parametros || "–")}</td><td>${done}</td>
      <td><button class="btn" data-view="${r.mes}|${r.estrategia}" style="height:32px">Ver</button> <button class="btn" data-xlsx="${r.mes}|${r.estrategia}" style="height:32px">Excel</button></td></tr>`;
  }).join("");
  view.innerHTML = `<div class="card"><h2>Pronósticos guardados</h2><p class="hint">Cada pronóstico queda con su fecha, los datos y los parámetros con que se hizo. Si generas de nuevo el mismo mes con la misma estrategia, el anterior se reemplaza. El acierto es por SKU / planta / total.</p>
    <div class="tablewrap" style="margin-top:12px;max-height:none"><table class="data left3"><thead><tr><th>Mes</th><th>Estrategia</th><th>Generado el</th><th>Total (TO)</th><th>Datos hasta</th><th>Parámetros</th><th>Resultado</th><th></th></tr></thead><tbody>${rows}</tbody></table></div><div id="h-msg"></div></div>`;
  view.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => { [forecastState.mes, forecastState.estrategia] = b.dataset.view.split("|"); location.hash = "#/prediccion"; }));
  view.querySelectorAll("[data-xlsx]").forEach((b) => (b.onclick = () => { const [mes, estrategia] = b.dataset.xlsx.split("|"); downloadExcel("pronostico", { mes, estrategia }, $("#h-msg")); }));
}

/* ---- Ayuda ------------------------------------------------------------------------------------------------------------------ */
const GLOSSARY = [
  ["TO", "Toneladas. Todas las cifras de la app están en toneladas."],
  ["Serie", "Un SKU en una planta. Hay unas 2.200."],
  ["Pronóstico (ml_mean)", "El promedio de tres modelos de aprendizaje automático (LightGBM, XGBoost y Random Forest). Es el número que importa."],
  ["Rango entre modelos", "El menor y el mayor de esos tres modelos. Si se separan mucho, el pronóstico es menos seguro. No es un intervalo estadístico."],
  ["Acierto", "100% menos el error total dividido por el consumo real. Si el real fue 100 TO y el pronóstico 90 TO, el acierto es 90%. Se mide por SKU, por planta y para el total."],
  ["Sesgo", "Si el pronóstico tiende a quedar por debajo (positivo) o por arriba (negativo) del real."],
  ["Modelo simple", "Repite el consumo del mes anterior. Sirve de referencia: el modelo debe ganarle."],
  ["Mes en curso (parcial)", "Un mes del que solo hay una parte de los días. No entra en los resúmenes de consumo, pero el pronóstico sí usa los días que lleva para pronosticar el mes siguiente."],
  ["Estrategia de pronóstico", "La forma de calcular el pronóstico. Con el mes completo se pronostica el mes siguiente; a mitad de mes se usa el mes en curso para pronosticar el que sigue. Se elige en Ajustes."],
  ["Día de corte", "Cuántos días del mes en curso conoce el pronóstico. La app lo toma del último registro de los datos. Desde el día 10 esos días mejoran el resultado."],
  ["Prueba histórica", "Cómo habría acertado el modelo en los últimos 12 meses, usando solo lo que se sabía en cada momento."],
  ["Hiperparámetros", "La configuración interna de los modelos. Reafinarlos es buscar la mejor con los datos más nuevos; se repite cada 6 meses. No es lo mismo que Ajustes, que elige la estrategia del pronóstico."],
];
const FAQ = [
  ["No me deja generar el pronóstico", "<p>La pantalla Predicción dice el motivo. Lo más común es que no haya datos cargados, o que el día de corte guardado en <b>Ajustes</b> sea posterior al último registro de los datos (por ejemplo si cargaste un exporte más corto). Cámbialo en Ajustes o vuelve al corte automático.</p>"],
  ["¿Qué mes está pronosticando la app?", "<p>Si el último mes de los datos está completo, el siguiente. Si el mes sigue en curso, el que viene después de ese: por ejemplo, con datos hasta mediados de octubre pronostica noviembre. La pantalla Predicción lo dice arriba, junto con la estrategia.</p>"],
  ["Cargué un archivo y dice que hay valores distintos", "<p>El exporte trae un número diferente para un día que ya estaba cargado. Eso pasa cuando SAP corrige un movimiento. Si el exporte nuevo es el correcto, elige <b>Usar los valores del exporte nuevo</b>. Los datos de antes quedan en un respaldo.</p>"],
  ["Me equivoqué al cargar datos", "<p>En <b>Datos</b> usa <b>Deshacer la última carga</b>: vuelve a los datos que había antes.</p>"],
  ["¿Cómo se mide cada estrategia?", "<p>Un pronóstico se mide cuando cierra el mes que pronostica, contra el consumo real de ese mes. Si para el mismo mes hay uno hecho a mitad del mes anterior y otro hecho con el mes anterior completo, cada uno se mide por separado (Seguimiento tiene un botón para cada uno). La tabla de comparación promedia los meses medidos de cada estrategia y los pone junto a su prueba histórica de los últimos 12 meses. No son del todo comparables: el de mitad de mes ve menos información.</p>"],
  ["¿Puedo cambiar un pronóstico ya generado?", "<p>No se edita; se genera de nuevo. Al generar el mismo mes otra vez, la app pide confirmación y reemplaza el anterior.</p>"],
  ["¿Para qué sirve la clave de Hiperparámetros?", "<p>Reafinar los modelos cambia los pronósticos futuros, y la clave evita hacerlo por accidente. Es una protección sencilla, no seguridad real: quien tenga acceso a los archivos de la app puede saltársela.</p>"],
  ["Algunas plantas acertaron poco", "<p>Si una planta cambió de nivel de consumo de un mes a otro, el modelo tarda en seguirla. Aparece como aviso en Seguimiento. Conviene confirmar el cambio con quien maneja los datos.</p>"],
];

async function renderHelp(view) {
  const about = await api("acerca");
  view.innerHTML = `<div class="stack">
    <div class="card"><h2>Cada mes, en tres pasos</h2><div class="steps">
      <div class="step"><b>Cargar los datos</b><p>Cuando termina el mes, carga el exporte de SAP con los días nuevos.</p><button class="btn" data-go="datos">Ir a Datos</button></div>
      <div class="step"><b>Generar el pronóstico</b><p>Genera el que corresponde a los datos que hay (el mes siguiente, o el que sigue al mes en curso). Tarda menos de un minuto.</p><button class="btn" data-go="prediccion">Ir a Predicción</button></div>
      <div class="step"><b>Medir el acierto</b><p>Cuando cierra el mes pronosticado, mide qué tanto acertó.</p><button class="btn" data-go="seguimiento">Ir a Seguimiento</button></div></div></div>
    <div class="card"><h2>Qué significa cada palabra</h2><div class="tablewrap" style="margin-top:12px;max-height:none"><table class="data left3"><tbody>${GLOSSARY.map(([k, v]) => `<tr><td style="white-space:nowrap"><b>${esc(k)}</b></td><td style="white-space:normal">${esc(v)}</td></tr>`).join("")}</tbody></table></div></div>
    <div class="card"><h2>Preguntas frecuentes</h2>${FAQ.map(([q, a]) => `<details class="faq"><summary>${esc(q)}</summary>${a}</details>`).join("")}</div>
    <div class="card"><h2>Sobre esta versión</h2><p class="hint" style="margin-top:6px;line-height:1.6">Versión ${esc(about.version)}. Es una versión de prueba: los pronósticos de un mismo mes se reemplazan al generarlos de nuevo.<br>
      Carpeta de la app: ${esc(about.carpeta)}<br>Datos: ${esc(about.archivos.datos)}<br>Pronósticos y seguimiento: ${esc(about.archivos.pronosticos)}<br>Respaldos de datos: ${esc(about.archivos.respaldos)}<br>Registro de errores: ${esc(about.archivos.registro)}</p></div></div>`;
  view.querySelectorAll("[data-go]").forEach((b) => (b.onclick = () => (location.hash = `#/${b.dataset.go}`)));
}
