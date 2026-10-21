"""Download the papers listed in bibliografia/ (only open-access copies; the script never bypasses a paywall)."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MIN_PDF_BYTES = 50_000  # an error page or a landing page is far smaller than a paper


@dataclass(frozen=True)
class Reference:
    key: str
    citation: str
    filename: str
    urls: tuple[str, ...]  # open-access copies, tried in order


REFERENCES = [
    Reference(
        key="giannone2008",
        citation="Giannone, Reichlin y Small (2008), Nowcasting: the real-time informational content of macroeconomic data, Journal of Monetary Economics 55(4), 665-676.",
        filename="giannone_reichlin_small_2008_nowcasting.pdf",
        urls=(
            "https://dipot.ulb.ac.be/dspace/bitstream/2013/6409/1/dg-0006a.pdf",  # ULB repository (working paper)
            "https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp633.pdf",  # ECB working paper 633 (earlier version)
        ),
    ),
    Reference(
        key="shenstone2005",
        citation="Shenstone y Hyndman (2005), Stochastic models underlying Croston's method for intermittent demand forecasting, Journal of Forecasting 24(6), 389-402.",
        filename="hyndman_croston.pdf",
        urls=("https://robjhyndman.com/papers/croston.pdf",),  # author's page
    ),
]


def looks_like_a_paper(data: bytes) -> bool:
    r"""A PDF file (magic number) that is big enough to be the paper and not an error page."""
    return data[:5] == b"%PDF-" and len(data) >= MIN_PDF_BYTES


def fetch(url: str, timeout: float = 60.0) -> bytes:
    r"""Bytes of a URL (urllib honours HTTPS_PROXY and the CA bundle of the environment)."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bibliografia downloader)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(reference: Reference, out_dir: Path, timeout: float = 60.0, overwrite: bool = False) -> tuple[str, str]:
    r"""
    Save the first URL of `reference` that returns a real PDF. Returns (status, detail):
    "exists" (already there), "ok" (downloaded from `detail`) or "failed" (`detail` lists every attempt).
    """
    target = out_dir / reference.filename
    if target.exists() and not overwrite:
        return "exists", str(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for url in reference.urls:
        try:
            data = fetch(url, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            attempts.append(f"{url}: {error}")
            continue
        if not looks_like_a_paper(data):
            attempts.append(f"{url}: the answer is not a PDF of a paper ({len(data):,} bytes)")
            continue
        partial = target.with_suffix(".part")
        partial.write_bytes(data)
        shutil.move(partial, target)  # a half-written file never keeps the final name
        return "ok", url
    return "failed", " | ".join(attempts)
