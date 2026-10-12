"""The resident embedder (#211): a neo run borrows the observer's model, and
every way that can fail ends with the run loading its own."""

import json
import shutil
import socket
import stat
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import neo.memory.store as store
from neo.memory import embed_service

pytestmark = pytest.mark.skipif(not embed_service.supported(), reason="POSIX only")

MODEL = store.DEFAULT_EMBED_MODEL


class Fake:
    """Deterministic 4-dim 'model': the vector is the text's length and bytes."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for t in texts:
            b = t.encode()[:3].ljust(3, b"\0")
            yield np.array([len(t), b[0], b[1], b[2]], dtype=np.float32)


@pytest.fixture
def sock():
    # AF_UNIX paths are capped near 104 bytes on macOS; pytest's tmp_path is not.
    d = Path(tempfile.mkdtemp(prefix="ne", dir="/tmp"))
    yield d / "run" / "e.sock"
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def server(sock):
    fake = Fake()
    srv = embed_service.serve(fake, MODEL, sock)
    assert srv is not None
    yield srv, fake
    srv.close()


def _local_never():
    raise AssertionError("the local model must not load while the server is up")


def test_a_run_borrows_the_resident_model(server, sock):
    _, fake = server
    remote = embed_service.connect(MODEL, _local_never, sock)
    assert remote is not None and remote.dim == 4
    got = list(remote.embed(["hello", "ab"]))
    assert [v.tolist() for v in got] == [v.tolist() for v in Fake().embed(["hello", "ab"])]


def test_build_resilient_embedder_uses_the_server_and_never_builds(server, sock, monkeypatch):
    monkeypatch.setattr(store, "FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(store, "_EMBEDDERS", {})
    monkeypatch.setattr(embed_service, "socket_path", lambda: sock)
    monkeypatch.setattr(store, "_build_embedder", lambda *a: _local_never())
    emb = store.build_resilient_embedder(cache_dir=sock.parent)
    assert isinstance(emb, embed_service.RemoteEmbedder)
    assert list(emb.embed(["x"]))[0][0] == 1.0


def test_the_socket_is_private(server, sock):
    assert stat.S_IMODE(sock.stat().st_mode) == 0o600
    assert stat.S_IMODE(sock.parent.stat().st_mode) == 0o700


def test_no_socket_means_no_remote(sock):
    assert embed_service.connect(MODEL, _local_never, sock) is None


def test_a_server_for_another_model_is_not_used(sock):
    srv = embed_service.serve(Fake(), "other/model", sock)
    try:
        assert embed_service.connect(MODEL, _local_never, sock) is None
    finally:
        srv.close()


def test_a_stale_socket_is_not_used_and_a_new_server_replaces_it(sock):
    sock.parent.mkdir(parents=True)
    dead = socket.socket(socket.AF_UNIX)
    dead.bind(str(sock))
    dead.close()  # file left behind, nobody listening
    assert embed_service.connect(MODEL, _local_never, sock) is None
    srv = embed_service.serve(Fake(), MODEL, sock)
    assert srv is not None
    srv.close()
    assert not sock.exists()


def test_a_hung_server_is_abandoned_quickly(sock):
    sock.parent.mkdir(parents=True)
    hung = socket.socket(socket.AF_UNIX)
    hung.bind(str(sock))
    hung.listen(1)  # accepts, never answers
    try:
        start = time.time()
        assert embed_service.connect(MODEL, _local_never, sock) is None
        assert time.time() - start < embed_service.HELLO_TIMEOUT_S + 0.5
    finally:
        hung.close()


def test_a_second_server_does_not_steal_a_live_socket(server, sock):
    assert embed_service.serve(Fake(), MODEL, sock) is None
    assert embed_service.connect(MODEL, _local_never, sock) is not None


def test_a_server_gone_mid_run_falls_back_to_a_local_model_once(server, sock):
    srv, _ = server
    built = []

    def local():
        built.append(1)
        return Fake()

    remote = embed_service.connect(MODEL, local, sock)
    srv.close()  # the observer recycled
    assert list(remote.embed(["hi"]))[0][0] == 2.0
    assert list(remote.embed(["hey"]))[0][0] == 3.0
    assert built == [1]


def test_a_malformed_reply_falls_back(sock):
    sock.parent.mkdir(parents=True)
    liar = socket.socket(socket.AF_UNIX)
    liar.bind(str(sock))
    liar.listen(4)

    def answer():
        for reply in (b'{"v": 1, "model": "%s", "dim": 4}\n' % MODEL.encode(),
                      b'{"v": 1, "dim": 4, "n": 1, "vectors": "AAAA"}\n'):
            conn, _ = liar.accept()
            conn.recv(65536)
            conn.sendall(reply)
            conn.close()

    t = threading.Thread(target=answer, daemon=True)
    t.start()
    remote = embed_service.connect(MODEL, Fake, sock)
    assert remote is not None
    assert list(remote.embed(["hi"]))[0][0] == 2.0  # local, after a short payload
    t.join(2)
    liar.close()


def test_a_server_never_borrows_from_itself(monkeypatch, sock, server):
    monkeypatch.setattr(store, "FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(store, "_EMBEDDERS", {})
    monkeypatch.setattr(embed_service, "socket_path", lambda: sock)
    monkeypatch.setattr(embed_service, "_server_process", True)
    monkeypatch.setattr(store, "_build_embedder", lambda *a: Fake())
    assert isinstance(store.build_resilient_embedder(cache_dir=sock.parent), store._Serialized)


def test_wire_errors_are_reported_not_raised(server, sock):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(sock))
        s.sendall(json.dumps({"v": 1, "op": "embed", "model": "nope", "texts": []}).encode() + b"\n")
        reply = json.loads(s.makefile().readline())
    assert "error" in reply and "embeds with" in reply["error"]


def test_tests_cannot_reach_the_real_daemon():
    from tests.conftest import REAL_HOME

    assert not embed_service.socket_path().is_relative_to(REAL_HOME / ".neo")
