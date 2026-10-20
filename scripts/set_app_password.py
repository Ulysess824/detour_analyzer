"""
Set the password of the Ajuste section of the app. Only the hash is written (config/seguridad.json), never the text.

A password in a local app prevents accidental use; it is not real security: whoever has the files of the app can get around it.

Usage:
    python scripts/set_app_password.py
"""

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.app.paths import default_paths
from src.app.security import hash_password, save_record


def main() -> None:
    first = getpass.getpass("Nueva clave: ")
    if len(first) < 4:
        raise SystemExit("La clave debe tener al menos 4 caracteres.")
    if getpass.getpass("Repite la clave: ") != first:
        raise SystemExit("Las dos claves no coinciden. No se cambió nada.")
    path = default_paths().security_json
    save_record(path, hash_password(first))
    print(f"Listo: se guardó solo el hash en {path}")


if __name__ == "__main__":
    main()
