"""The resident embedder (#211): a neo run borrows the observer's model, and
every way that can fail ends with the run loading its own, without paying
that 1.3 GB where waiting would have worked."""

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
from neo.memory import embed_service as es

pytestmark = pytest.mark.skipif(not es.supported(), reason="POSIX only")

MODEL = store.DEFAULT_EMBED_MODEL


class Fake:
    """Deterministic 4-dim 'model': the vector is the text's length and bytes."""

    def __init__(self, delay: float = 0.0):
        self.batches: list[int] = []
        self.delay = delay

    def embed(self, texts):
        texts = list(texts)
        self.batches.append(len(texts))
        time.sleep(self.delay)
        for t in texts:
            b = t.encode()[:3].ljust(3, b"\0")
            yield np.array([len(t), b[0], b[1], b[2]], dtype=np.float32)


def expected(texts):
    return [v.tolist() for v in Fake().embed(texts)]


@pytest.fixture
def sock():
    # AF_UNIX paths are capped near 104 bytes on macOS; pytest's tmp_path is not.
    d = Path(tempfile.mkdtemp(prefix="ne", dir="/tmp"))
    yield d / "run" / "e.sock"
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def server(sock):
    fake = Fake()
    srv = es.serve(fake, MODEL, sock)
    assert srv is not None
    yield srv, fake
    srv.close()


def _local_never():
    raise AssertionError("the local model must not load while the server is up")


def _fake_server(sock, replies, backlog=8):
    """A hand-rolled peer answering each connection with the next reply (bytes,
    or a callable given the connection)."""
    sock.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(sock))
    listener.listen(backlog)

    def run():
        for reply in replies:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            try:
                with conn:
                    conn.recv(65536)
                    reply(conn) if callable(reply) else conn.sendall(reply)
            except OSError:
                pass  # the client gave up first, which is what some tests check

    threading.Thread(target=run, daemon=True).start()
    return listener


def _hello(**override) -> bytes:
    body = {"v": 1, "model": MODEL, "dim": 4, "libs": es.library_versions(), **override}
    return json.dumps({k: v for k, v in body.items() if v is not ...}).encode() + b"\n"


# ------------------------------------------------------------ the happy path


def test_a_run_borrows_the_resident_model(server, sock):
    remote = es.connect(MODEL, _local_never, sock)
    assert remote is not None and remote.dim == 4
    assert [v.tolist() for v in remote.embed(["hello", "ab"])] == expected(["hello", "ab"])


def test_build_resilient_embedder_uses_the_server_and_never_builds(server, sock, monkeypatch):
    monkeypatch.setattr(store, "FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(store, "_EMBEDDERS", {})
    monkeypatch.setattr(es, "socket_path", lambda: sock)
    monkeypatch.setattr(store, "_build_embedder", lambda *a: _local_never())
    emb = store.build_resilient_embedder(cache_dir=sock.parent)
    assert isinstance(emb, es.RemoteEmbedder)
    assert list(emb.embed(["x"]))[0][0] == 1.0


def test_large_requests_are_batched(server, sock, monkeypatch):
    sent = []
    real_call = es._call

    def recording(path, request, timeout):
        if request.get("op") == "embed":
            sent.append(request["texts"])
        return real_call(path, request, timeout)

    monkeypatch.setattr(es, "_call", recording)
    texts = [f"t{i}" for i in range(200)] + ["x" * 9000] * 3
    remote = es.connect(MODEL, _local_never, sock)
    assert [v.tolist() for v in remote.embed(texts)] == expected(texts)
    assert max(len(b) for b in sent) <= es.BATCH_TEXTS
    assert all(len(b) == 1 or sum(map(len, b)) <= es.BATCH_CHARS for b in sent)


def test_a_short_query_is_not_stuck_behind_another_runs_batch(sock):
    """The server takes the model per text, so one run's long batch cannot
    hold it for the whole batch while another run's query times out."""
    fake = Fake(delay=0.2)
    srv = es.serve(store._Serialized(fake), MODEL, sock)
    try:
        a = es.connect(MODEL, _local_never, sock)
        b = es.connect(MODEL, _local_never, sock)
        threading.Thread(target=lambda: list(a.embed([f"t{i}" for i in range(20)]))).start()
        time.sleep(0.3)
        start = time.time()
        assert list(b.embed(["q"]))[0][0] == 1.0
        assert time.time() - start < 1.0  # one text's wait, not twenty
    finally:
        time.sleep(4.5)
        srv.close()


def test_rows_are_writable_like_fastembeds(server, sock):
    row = next(es.connect(MODEL, _local_never, sock).embed(["abc"]))
    row /= 2  # a caller normalizing in place


def test_the_socket_is_private(server, sock):
    assert stat.S_IMODE(sock.stat().st_mode) == 0o600
    assert stat.S_IMODE(sock.parent.stat().st_mode) == 0o700


def test_a_burst_of_runs_all_connect(server, sock, monkeypatch):
    monkeypatch.setattr(es, "RECOVER_S", 0)  # the backlog alone, no retry to mask it
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        es.connect(MODEL, _local_never, sock) is not None)) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == [True] * 40


def test_a_recycle_mid_run_is_waited_out_not_paid_for(server, sock):
    srv, _ = server
    remote = es.connect(MODEL, _local_never, sock)
    srv.close()  # the observer re-execs...
    replacement = []
    threading.Timer(1.0, lambda: replacement.append(es.serve(Fake(), MODEL, sock))).start()
    try:
        assert list(remote.embed(["hi"]))[0][0] == 2.0  # ...and is back in ~1 s
    finally:
        time.sleep(1.2)
        for s in replacement:
            s and s.close()


# ------------------------------------------------------------ no server to use


def test_no_socket_means_no_remote(sock):
    assert es.connect(MODEL, _local_never, sock) is None


def test_a_server_for_another_model_is_not_used(sock):
    srv = es.serve(Fake(), "other/model", sock)
    try:
        assert es.connect(MODEL, _local_never, sock) is None
    finally:
        srv.close()


@pytest.mark.parametrize("hello", [
    _hello(dim=...), _hello(dim="x"), _hello(dim=0), _hello(dim=True),
    _hello(libs={"fastembed": "0.0.1", "onnxruntime": "0.0.1"}),
    b'{"v": 1}\n', b"not json\n", b'{"v": 2, "model": "m", "dim": 4}\n',
])
def test_an_unusable_hello_means_no_remote_never_an_exception(sock, hello):
    listener = _fake_server(sock, [hello])
    try:
        assert es.connect(MODEL, _local_never, sock) is None
    finally:
        listener.close()


def _stale(sock):
    sock.parent.mkdir(parents=True, exist_ok=True)
    dead = socket.socket(socket.AF_UNIX)
    dead.bind(str(sock))
    dead.close()  # file left behind, nobody listening: a SIGKILLed observer


def test_a_stale_socket_with_no_observer_alive_is_abandoned_at_once(sock):
    """A killed observer leaves its socket; waiting RECOVER_S on it would cost
    every neo run that much until an observer came back."""
    _stale(sock)
    start = time.time()
    assert es.connect(MODEL, _local_never, sock) is None
    assert time.time() - start < 1.0


def test_a_refusal_while_the_observer_lives_is_waited_out(sock, monkeypatch):
    from neo.memory import observer

    monkeypatch.setattr(es, "RECOVER_S", 1.0)
    _stale(sock)
    lock = observer._SingleInstanceLock(observer._LOCK_PATH)
    assert lock.acquire()
    try:
        start = time.time()
        assert es.connect(MODEL, _local_never, sock) is None
        assert time.time() - start >= 1.0
    finally:
        lock.release()


def test_a_stale_socket_is_replaced_by_a_new_server(sock):
    _stale(sock)
    srv = es.serve(Fake(), MODEL, sock)
    assert srv is not None
    srv.close()
    assert not sock.exists()


def test_a_silent_server_is_abandoned_after_one_hello_timeout(sock):
    listener = _fake_server(sock, [lambda conn: time.sleep(es.HELLO_TIMEOUT_S + 1)])
    try:
        start = time.time()
        assert es.connect(MODEL, _local_never, sock) is None
        assert time.time() - start < es.HELLO_TIMEOUT_S + 0.5
    finally:
        listener.close()


def test_a_trickling_server_cannot_stretch_the_hello_deadline(sock):
    def trickle(conn):
        for _ in range(20):
            conn.sendall(b" ")
            time.sleep(0.3)

    listener = _fake_server(sock, [trickle])
    try:
        start = time.time()
        assert es.connect(MODEL, _local_never, sock) is None
        assert time.time() - start < es.HELLO_TIMEOUT_S + 0.5
    finally:
        listener.close()


def test_a_second_server_does_not_steal_a_live_socket(server, sock):
    assert es.serve(Fake(), MODEL, sock) is None
    assert es.connect(MODEL, _local_never, sock) is not None


# ------------------------------------------------------------ falling back


def test_a_server_gone_for_good_falls_back_to_a_local_model_once(server, sock, monkeypatch):
    monkeypatch.setattr(es, "RECOVER_S", 0.3)
    srv, _ = server
    built = []
    remote = es.connect(MODEL, lambda: built.append(1) or Fake(), sock)
    srv.close()
    assert list(remote.embed(["hi"]))[0][0] == 2.0
    assert list(remote.embed(["hey"]))[0][0] == 3.0
    assert built == [1]


def test_a_failed_local_load_is_tried_once_not_per_call(server, sock, monkeypatch):
    """A stale cache is healed by deleting and re-downloading it; doing that on
    every embed call (the first cut did) wipes the observer's cache too."""
    monkeypatch.setattr(es, "RECOVER_S", 0.3)
    srv, _ = server
    attempts = []
    remote = es.connect(MODEL, lambda: attempts.append(1), sock)
    srv.close()
    for text in ("a", "b", "c"):
        assert list(remote.embed([text])) == []
    assert attempts == [1]


def test_a_slow_embed_times_out_and_falls_back(sock, monkeypatch):
    monkeypatch.setattr(es, "EMBED_TIMEOUT_S", 0.5)
    srv = es.serve(Fake(), MODEL, sock)
    try:
        remote = es.connect(MODEL, Fake, sock)
        srv._server.embedder.delay = 1.5
        start = time.time()
        assert list(remote.embed(["hi"]))[0][0] == 2.0
        assert time.time() - start < 1.5
    finally:
        srv.close()


@pytest.mark.parametrize("reply", [
    b'{"v": 1, "dim": 4, "n": 1, "vectors": "AAAA"}\n',  # short payload
    b'{"v": 1, "dim": 4, "n": 1, "vect',                  # half a reply, then close
    b"",                                                   # closed without a word
])
def test_a_broken_embed_reply_falls_back(sock, monkeypatch, reply):
    monkeypatch.setattr(es, "RECOVER_S", 0.3)
    listener = _fake_server(sock, [_hello()] + [reply] * 10)
    try:
        remote = es.connect(MODEL, Fake, sock)
        assert remote is not None
        assert list(remote.embed(["hi"]))[0][0] == 2.0
    finally:
        listener.close()


def test_a_server_never_borrows_from_itself(monkeypatch, sock, server):
    monkeypatch.setattr(store, "FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(store, "_EMBEDDERS", {})
    monkeypatch.setattr(es, "socket_path", lambda: sock)
    monkeypatch.setattr(es, "_server_process", True)
    monkeypatch.setattr(store, "_build_embedder", lambda *a: Fake())
    assert isinstance(store.build_resilient_embedder(cache_dir=sock.parent), store._Serialized)


# ------------------------------------------------------------ the server's own guards


def _raw(sock, request: dict) -> dict:
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(sock))
        s.sendall(json.dumps(request).encode() + b"\n")
        return json.loads(s.makefile().readline())


def test_wire_errors_are_reported_not_raised(server, sock):
    assert "embeds with" in _raw(sock, {"v": 1, "op": "embed", "model": "nope", "texts": []})["error"]


def test_one_huge_text_is_cut_to_the_models_window_not_refused(server, sock):
    remote = es.connect(MODEL, _local_never, sock)
    huge = "\U0001F600" * 700_000  # 8.4 MB of JSON, past the server's cap
    assert next(remote.embed([huge]))[0] == es.MAX_TEXT_CHARS


def test_an_oversized_request_is_refused(server, sock):
    reply = _raw(sock, {"v": 1, "op": "embed", "model": MODEL,
                        "texts": ["x"] * (es.MAX_TEXTS + 1)})
    assert "over" in reply["error"]


def test_tests_cannot_reach_the_real_daemon():
    from tests.conftest import REAL_HOME

    assert not es.socket_path().is_relative_to(REAL_HOME / ".neo")
