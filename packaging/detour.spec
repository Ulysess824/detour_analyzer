# PyInstaller spec of the desktop app: two programs in one folder (detour: the window; detour-tools: console maintenance and self-test).
# Build from the repository root:  pyinstaller packaging/detour.spec --noconfirm
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
datas = [(str(ROOT / "src" / "app" / "static"), "src/app/static")]
binaries, hiddenimports = [], []

def gather(package):
    r"""Data files, native libraries and submodules of a package, leaving out its tests (some import pytest or hypothesis)."""
    keep = lambda name: not any(part in ("testing", "tests", "test", "_testing") for part in name.split("."))
    return collect_data_files(package), collect_dynamic_libs(package), collect_submodules(package, filter=keep)


# libraries with native files or modules that are imported by name at run time
packages = ["lightgbm", "xgboost", "optuna", "sklearn", "scipy", "uvicorn", "anyio", "python_multipart", "multipart", "fastapi", "starlette", "pydantic", "openpyxl", "webview"]
if sys.platform == "win32":  # the window: pywebview drives WebView2 through pythonnet
    packages += ["pythonnet", "clr_loader"]
for package in packages:
    try:
        d, b, h = gather(package)
    except Exception as error:  # a package that is not installed (pywebview on a build machine without a window system)
        print(f"spec: skipping {package}: {error}")
        continue
    datas += d
    binaries += b
    hiddenimports += h
# so the manifest of every forecast can name the library versions
for package in ("pandas", "numpy", "lightgbm", "xgboost", "scikit-learn"):
    datas += copy_metadata(package)

excludes = ["matplotlib", "IPython", "ipykernel", "jupyter_client", "jupyter_core", "notebook", "tkinter", "pytest", "coverage", "PIL.ImageTk"]
analysis_options = dict(pathex=[str(ROOT)], binaries=binaries, datas=datas, hiddenimports=hiddenimports, excludes=excludes, noarchive=False)

app = Analysis([str(ROOT / "packaging" / "app_main.py")], **analysis_options)
tools = Analysis([str(ROOT / "packaging" / "tools_main.py")], **analysis_options)
for analysis in (app, tools):  # CUDA libraries that the Linux wheels of xgboost pull in (2 GB); the Windows build never has them
    analysis.binaries = [item for item in analysis.binaries if not item[0].startswith(("nvidia", "cuda"))]
MERGE((app, "detour", "detour"), (tools, "detour-tools", "detour-tools"))

icon = str(ROOT / "packaging" / "detour.ico")
app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name="detour", console=False, icon=icon)
tools_exe = EXE(PYZ(tools.pure), tools.scripts, [], exclude_binaries=True, name="detour-tools", console=True, icon=icon)
COLLECT(app_exe, app.binaries, app.datas, tools_exe, tools.binaries, tools.datas, name="detour")
