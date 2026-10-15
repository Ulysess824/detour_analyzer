"""Bibliography downloader: a real PDF is saved, an error page or an unreachable host is reported and leaves no file."""

import functools
import http.server
import threading

import pytest

from src.utils.bibliography_utils import MIN_PDF_BYTES, Reference, download, looks_like_a_paper

PDF = b"%PDF-1.4\n" + b"x" * (MIN_PDF_BYTES + 10)


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "www").mkdir()
    (tmp_path / "www" / "paper.pdf").write_bytes(PDF)
    (tmp_path / "www" / "landing.html").write_bytes(b"<html>" + b"y" * (MIN_PDF_BYTES + 10) + b"</html>")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "www"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_a_pdf_must_have_the_magic_number_and_a_paper_size():
    assert looks_like_a_paper(PDF)
    assert not looks_like_a_paper(b"%PDF-1.4 tiny")
    assert not looks_like_a_paper(b"<html>" + b"y" * (MIN_PDF_BYTES + 10))


def test_the_first_working_url_is_used_and_the_file_is_saved(server, tmp_path):
    ref = Reference("k", "c", "paper.pdf", (f"{server}/missing.pdf", f"{server}/landing.html", f"{server}/paper.pdf"))
    status, detail = download(ref, tmp_path / "out")
    assert status == "ok" and detail.endswith("/paper.pdf")
    assert (tmp_path / "out" / "paper.pdf").read_bytes() == PDF
    assert not list((tmp_path / "out").glob("*.part"))
    assert download(ref, tmp_path / "out")[0] == "exists"  # not downloaded twice


def test_failure_lists_every_attempt_and_keeps_no_file(server, tmp_path):
    ref = Reference("k", "c", "paper.pdf", (f"{server}/landing.html", "http://127.0.0.1:9/none.pdf"))
    status, detail = download(ref, tmp_path / "out", timeout=2.0)
    assert status == "failed" and "landing.html" in detail and "127.0.0.1:9" in detail
    assert not (tmp_path / "out" / "paper.pdf").exists()
