"""The password of the Ajuste section: only a hash is stored, sessions live in memory.

A password in a local app prevents accidental use; it is not real security: whoever has the files of the app can get around it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import time
from pathlib import Path
from typing import Callable

SCRYPT = {"n": 2**14, "r": 8, "p": 1}
PBKDF2_ROUNDS = 600_000
IDLE_SECONDS = 30 * 60
MAX_FAILS = 5
LOCK_SECONDS = 60


class Unauthorized(Exception):
    r"""No valid session: the section is locked."""


class WrongPassword(Exception):
    def __init__(self, remaining: int):
        super().__init__(f"Clave incorrecta. Quedan {remaining} intentos." if remaining else "Clave incorrecta.")
        self.remaining = remaining


class Locked(Exception):
    def __init__(self, seconds: int):
        super().__init__(f"Demasiados intentos. Espera {seconds} segundos.")
        self.seconds = seconds


class NoPassword(Exception):
    pass


def _derive(password: str, salt: bytes, algorithm: str) -> bytes:
    if algorithm == "scrypt":
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **SCRYPT)
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS, dklen=32)


def hash_password(password: str, salt: bytes | None = None, algorithm: str | None = None) -> dict:
    r"""The record to store: algorithm, salt and hash (scrypt when this Python has it, else PBKDF2). The password itself is never kept."""
    algorithm = algorithm or ("scrypt" if hasattr(hashlib, "scrypt") else "pbkdf2")
    salt = salt or os.urandom(16)
    return {"algoritmo": algorithm, "sal": salt.hex(), "hash": _derive(password, salt, algorithm).hex()}


def verify_password(password: str, record: dict) -> bool:
    expected = bytes.fromhex(record["hash"])
    return hmac.compare_digest(_derive(password, bytes.fromhex(record["sal"]), record["algoritmo"]), expected)


def load_record(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")).get("clave") if path.exists() else None


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"clave": record}, indent=2) + "\n", encoding="utf-8")


class Sessions:
    r"""Sessions of the unlocked section: a random token, valid while it is used (30 minutes idle); five wrong passwords lock the screen for a minute."""

    def __init__(self, load: Callable[[], dict | None], clock: Callable[[], float] = time.monotonic):
        self._load, self._clock = load, clock
        self._tokens: dict[str, float] = {}
        self._fails, self._locked_until = 0, 0.0

    def configured(self) -> bool:
        return self._load() is not None

    def login(self, password: str) -> str:
        record = self._load()
        if record is None:
            raise NoPassword("No hay una clave configurada. Ejecuta scripts/set_app_password.py.")
        now = self._clock()
        if now < self._locked_until:
            raise Locked(math.ceil(self._locked_until - now))
        if not verify_password(password, record):
            self._fails += 1
            if self._fails >= MAX_FAILS:
                self._fails, self._locked_until = 0, now + LOCK_SECONDS
                raise Locked(LOCK_SECONDS)
            raise WrongPassword(MAX_FAILS - self._fails)
        self._fails = 0
        token = secrets.token_urlsafe(24)
        self._tokens[token] = now + IDLE_SECONDS
        return token

    def check(self, token: str | None) -> None:
        r"""Raise Unauthorized unless `token` is a live session; using it extends it."""
        now = self._clock()
        if not token or self._tokens.get(token, 0.0) < now:
            self._tokens.pop(token, None)
            raise Unauthorized("La sección está bloqueada. Escribe la clave para desbloquearla.")
        self._tokens[token] = now + IDLE_SECONDS

    def logout(self, token: str | None) -> None:
        self._tokens.pop(token, None)
