"use strict";
/* Hiperparámetros (the section that retunes the models). The section is locked with a password; the session token lives only in memory (reloading the app locks it again). */

let tuningToken = "";
const authHeader = () => ({ "X-Ajuste-Token": tuningToken });
const STATE_TEXT = { al_dia: ["ok", "check", "Al día"], por_vencer: ["warn", "alert", "Por vencer"], vencido: ["bad", "cross", "Vencido"], sin_registro: ["neutral", "info", "Sin registro"] };
const TYPE_TEXT = { completo: "Completo", rapido: "Rápido" };

async function renderTuning(view) {
  clearTimeout(pollTimer);
  const s = await api("ajuste");
  const [cls, ic, label] = STATE_TEXT[s.estado.estado];
  const st = s.estado;
  const facts = `<div class="facts"><div><span>Último ajuste</span><b>${st.ultimo ? dateLong(st.ultimo) : "–"}</b></div><div><span>Próximo ajuste</span><b>${st.proximo ? dateLong(st.proximo) : "–"}</b></div>
    <div><span>Estado</span><b><span class="badge ${cls}">${icon(ic)}${label}${st.dias_restantes !== null && st.estado !== "vencido" ? ` · faltan ${fmt(st.dias_restantes, 0)} días` : ""}</span></b></div></div>`;
  view.innerHTML = `<div class="stack">
    <div class="card"><h2>¿Qué es el ajuste?</h2><p class="hint" style="margin-top:6px;line-height:1.5">Los tres modelos (LightGBM, XGBoost y Random Forest) tienen una configuración interna. El ajuste busca la mejor con los datos más nuevos y la prueba en los últimos 12 meses.
      Conviene repetirlo cada ${st.intervalo_meses} meses o cuando el consumo cambie mucho. Los parámetros nuevos no se usan hasta que tú los adoptes.</p>${facts}</div>
    <div id="tuning-body"></div>
    <div class="card"><h2>Historial de ajustes</h2>${historyHtml(s.historial)}</div></div>`;
  if (!tuningToken) lockedHtml(s); else await unlockedHtml(s);
}

function historyHtml(rows) {
  if (!rows.length) return '<p class="hint" style="margin-top:6px">Todavía no hay ajustes registrados.</p>';
  const body = rows.map((r) => `<tr><td>${dateLong(r.fecha)}</td><td>${esc(r.origen === "app" ? "Hecho en la app" : r.origen)}</td><td>${r.tipo ? TYPE_TEXT[r.tipo] || esc(r.tipo) : "–"}</td><td>${esc(r.ventana || "–")}</td><td>${r.mejora_sku_pts === undefined ? "–" : (r.mejora_sku_pts > 0 ? "+" : "") + fmt(r.mejora_sku_pts) + " pts"}</td></tr>`).join("");
  return `<div class="tablewrap" style="margin-top:12px"><table class="data left3"><thead><tr><th>Fecha</th><th>Origen</th><th>Tipo</th><th>Datos usados</th><th>Mejora por SKU</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

function lockedHtml(s) {
  $("#tuning-body").innerHTML = `<div class="card"><h2>Sección protegida</h2><p class="hint" style="margin-top:6px">Reafinar los modelos cambia los pronósticos futuros. Escribe la clave para continuar.</p>
    <div class="lock"><div class="field"><label for="t-clave">Clave</label><div class="input-wrap"><input id="t-clave" type="password" autocomplete="off"></div></div><button class="btn primary" id="t-open">Desbloquear</button></div>
    <p class="note" id="t-msg"></p>${s.clave_configurada ? "" : `<div class="banner warn" style="margin-top:12px">${icon("alert")}<div>No hay una clave configurada. Ejecuta <b>scripts/set_app_password.py</b>.</div></div>`}</div>`;
  const open = async () => {
    const reply = await apiPost("ajuste/desbloquear", { clave: $("#t-clave").value });
    if (!reply.ok) { $("#t-msg").className = "note error-text"; $("#t-msg").textContent = reply.body.detalle || "No se pudo desbloquear."; $("#t-clave").select(); return; }
    tuningToken = reply.body.token;
    route();
  };
  $("#t-open").onclick = open;
  $("#t-clave").addEventListener("keydown", (event) => { if (event.key === "Enter") open(); });
}

async function unlockedHtml(s) {
  const busy = !!s.tarea;
  const types = s.tipos.map((t, i) => `<label><input type="radio" name="t-tipo" value="${t.id}" ${i === 0 ? "checked" : ""}><b>${esc(t.nombre)}</b> · unos ${t.minutos} minutos<small>${esc(t.texto)}</small></label>`).join("");
  const why = s.mes_parcial ? "El último mes de los datos está incompleto: carga el mes completo en la sección Datos." : s.candidato ? "Primero decide sobre los parámetros nuevos de abajo." : "";
  $("#tuning-body").innerHTML = `<div class="stack">
    <div class="card"><div class="toolbar"><div><h2>Reafinar el modelo</h2><p class="hint">Sección desbloqueada. Se vuelve a bloquear al cerrar la app o tras 30 minutos sin usarla.</p></div><button class="btn" id="t-lock">Bloquear</button></div>
      <div class="choice">${types}</div>
      <div class="rowbtns"><button class="btn primary" id="t-start" ${busy || why ? "disabled" : ""}>Empezar el ajuste</button><span class="note" style="margin:0">${esc(why) || "No cierres la app mientras corre. Mientras tanto no se puede generar otro pronóstico."}</span></div>
      <div id="t-progress"></div></div>
    <div id="t-candidate">${s.candidato ? candidateHtml(s.candidato) : ""}</div></div>`;
  $("#t-lock").onclick = async () => { await apiPost("ajuste/bloquear", {}, authHeader()); tuningToken = ""; route(); };
  $("#t-start").onclick = async () => {
    const tipo = document.querySelector('input[name="t-tipo"]:checked').value;
    $("#t-start").disabled = true;
    const reply = await apiPost("ajuste/reafinar", { tipo }, authHeader());
    if (reply.status === 401) { tuningToken = ""; route(); return; }
    if (!reply.ok) { $("#t-start").disabled = false; $("#t-progress").innerHTML = `<div class="banner warn" style="margin-top:14px">${icon("alert")}<div>${esc(reply.body.detalle || "No se pudo empezar.")}</div></div>`; return; }
    watchTuning(reply.body.trabajo);
  };
  if (s.candidato) bindCandidate();
  if (busy) watchTuning(s.tarea.id);
}

function watchTuning(jobId) {
  const tick = async () => {
    const box = $("#t-progress");
    if (!box) return;
    const job = await api(`trabajos/${jobId}`);
    if ($("#t-start")) $("#t-start").disabled = job.estado === "corriendo";
    if (job.estado === "corriendo") {
      box.innerHTML = `<div class="progress"><div class="bar" style="width:${Math.max(3, Math.round(job.progreso * 100))}%"></div></div><p class="note">${esc(job.mensaje)} · ${fmt(Math.floor(job.segundos / 60), 0)} min ${fmt(Math.round(job.segundos % 60), 0)} s</p>`;
      pollTimer = setTimeout(tick, 1500);
    } else if (job.estado === "listo") route();
    else box.innerHTML = `<div class="banner warn" style="margin-top:14px">${icon("alert")}<div><b>No se pudo terminar el ajuste.</b><br>${esc(job.error)}</div></div>`;
  };
  tick();
}

function candidateHtml(c) {
  const gain = c.mejora_sku_pts, better = c.modelos.filter((m) => m.mejora_sku_pts > 0).length, n = c.modelos.length;
  const clear = better === n || (gain > 0.2 && better >= Math.ceil(n / 2));
  const verdict = clear
    ? `<span class="badge ok">${icon("check")}Mejora el acierto por SKU en ${better} de ${n} modelos (promedio ${gain > 0 ? "+" : ""}${fmt(gain)} puntos)</span>`
    : `<span class="badge warn">${icon("alert")}Mejora solo en ${better} de ${n} modelos (promedio ${gain > 0 ? "+" : ""}${fmt(gain)} puntos): no es una mejora clara</span>`;
  const rows = c.modelos.map((m) => `<tr><td>${esc(m.nombre)}</td><td>${m.pruebas}</td><td>${fmt(m.sku_base)}% → <b>${fmt(m.sku_ajustado)}%</b></td><td>${fmt(m.planta_base)}% → ${fmt(m.planta_ajustado)}%</td><td>${fmt(m.total_base)}% → ${fmt(m.total_ajustado)}%</td><td>${m.mejora_sku_pts > 0 ? "+" : ""}${fmt(m.mejora_sku_pts)}</td></tr>`).join("");
  const quick = c.tipo === "rapido" ? `<div class="banner warn" style="margin-top:12px">${icon("alert")}<div>Es un ajuste rápido de prueba. No se recomienda adoptarlo; haz el completo.</div></div>` : "";
  return `<div class="card"><h2>Parámetros nuevos esperando una decisión</h2><p class="hint">Hechos el ${dateLong(c.generado_en)} con datos de ${esc(c.ventana_ajuste)}. Probados en ${esc(c.ventana_prueba)}, meses que no se usaron para elegirlos.</p>
    <div style="margin-top:12px">${verdict}</div>${quick}
    <div class="tablewrap" style="margin-top:12px"><table class="data"><thead><tr><th>Modelo</th><th>Pruebas</th><th>SKU (antes → ahora)</th><th>Planta</th><th>Total</th><th>Mejora SKU (pts)</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="rowbtns"><button class="btn primary" id="t-adopt">Adoptar estos parámetros</button><button class="btn" id="t-discard">Descartar</button></div><div id="t-confirm"></div></div>`;
}

function bindCandidate() {
  $("#t-adopt").onclick = () => {
    $("#t-confirm").innerHTML = `<div class="confirm">Los pronósticos nuevos usarán estos parámetros. Los actuales se guardan como respaldo. ¿Adoptarlos?<button class="btn primary" id="t-yes">Sí, adoptar</button><button class="btn" id="t-no">Cancelar</button></div>`;
    $("#t-no").onclick = () => ($("#t-confirm").innerHTML = "");
    $("#t-yes").onclick = async () => {
      const reply = await apiPost("ajuste/adoptar", {}, authHeader());
      if (reply.status === 401) tuningToken = "";
      route();
    };
  };
  $("#t-discard").onclick = async () => {
    const reply = await apiPost("ajuste/descartar", {}, authHeader());
    if (reply.status === 401) tuningToken = "";
    route();
  };
}
