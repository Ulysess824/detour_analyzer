# detour: guía de instalación

detour es un programa de Windows para ver el consumo real, generar el pronóstico del mes siguiente y medir su acierto. No necesita instalar Python ni tener internet.

## Qué recibes

Un archivo `detour-windows.zip`. Al descomprimirlo queda una carpeta `detour` con:

| Qué | Para qué |
|---|---|
| `detour.exe` | El programa: se abre una ventana |
| `detour-tools.exe` | Herramientas para soporte (comprobación y cambio de clave) |
| `data/` | Los datos de consumo (`consumos_long.csv`) |
| `results/` | Los pronósticos, el seguimiento y los parámetros de los modelos |
| `config/` | La clave de la sección Ajuste (solo su hash) |
| `_internal/` | Lo que el programa necesita para funcionar. No lo toques |

**Cuidado:** la carpeta incluye datos de consumo de la empresa. No la compartas fuera de la empresa.

## Instalar

1. Descomprime el zip en una carpeta donde puedas escribir, por ejemplo `C:\detour` o `Documentos\detour`. No lo ejecutes desde dentro del zip, ni desde `Archivos de programa`.
2. Abre `detour.exe` con doble clic.
3. La primera vez Windows puede mostrar **"Windows protegió su PC"** porque el programa no tiene firma digital. Pulsa **Más información** y luego **Ejecutar de todas formas**. Algunos antivirus también avisan de programas hechos con PyInstaller: si pasa, pide a TI que autorice la carpeta.
4. Se abre la ventana de detour. Si Windows pide instalar **WebView2**, acéptalo (viene con Windows 10 y 11 actualizados).

Para desinstalar, borra la carpeta.

## Uso mensual

1. **Datos**: carga el exporte de SAP con los días nuevos.
2. **Predicción**: genera el pronóstico del mes siguiente.
3. **Seguimiento**: cuando el mes pronosticado cierre, mide qué tanto acertó.

La sección **Ayuda** del programa explica cada palabra.

## Actualizar a una versión nueva

1. Descomprime la versión nueva en otra carpeta.
2. Copia a la carpeta nueva las carpetas `data`, `results` y `config` de la anterior (reemplazando las de la nueva).
3. Abre `detour.exe` de la carpeta nueva.

## Si algo falla

- Abre una consola en la carpeta (`cmd`) y ejecuta `detour-tools.exe self-test`: comprueba que todo está bien. Con `--forecast` también genera un pronóstico real.
- El registro de errores está en `logs\detour.log`. Mándalo a soporte junto con una captura.
- Cambiar la clave de Ajuste: `detour-tools.exe set-password`.

## Notas

- Todo ocurre en tu computador: el programa solo escucha en la propia máquina y no envía datos a ninguna parte.
- La clave de la sección Ajuste evita usos accidentales; no es seguridad real.
- Si abres dos veces el programa, las dos ventanas usan los mismos archivos: cierra una antes de cargar datos en la otra.
