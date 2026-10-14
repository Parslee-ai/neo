"""A resident embedding model, shared across neo runs (#211).

The embedding model is 614 MB on disk and ~1.3 GB resident once loaded, and it
was the whole of a neo run's memory floor: the FactStore itself is ~126 MB. A
run only needs a handful of query embeddings, so loading the model for each
one is the wrong shape. The observer daemon is already resident, so it holds
the model and serves embeddings over a Unix socket at
``~/.neo/run/embed.sock``; a neo run that finds it borrows the model instead
of loading its own.

The fallback is the product, not the daemon. No socket, a stale socket, a
server for another model, protocol or library version, a hung server, a
malformed reply: every one of these ends with the run loading the model
itself, exactly as it did before this module existed. Loading locally costs
1.3 GB, so the client prefers waiting to falling back where waiting can work:

- requests are split into batches (`BATCH_TEXTS`), so no request holds the
  server's model for long and no reply is large;
- a deadline covers a whole request: a queueing allowance plus a term per
  character, because long texts are expensive (one of 32,000 characters takes
  33-75 s on CPU) and a request may queue behind others on the one model;
- the server embeds one text at a time, so a short query interleaves with a
  long batch instead of waiting behind all of it;
- a transport failure (refused, gone, reset: an observer recycle, ~2 s of exec
  plus model load) is retried until `RECOVER_S` has passed before the run
  gives up on the server.

A run that starts while the socket is missing (mid-recycle, ~2-6 s every
15 minutes) loads its own model rather than waiting: an observer that never
serves (fastembed broken) would otherwise cost every run the wait. And a
server's error reply is final for the process like any other failure; error
replies come only from client or server bugs, not from load.

Once a process has loaded its own model it keeps it: dropping an ONNX session
does not reliably return memory to the OS, so switching back would cost a
reconnect and save nothing. A failed local load is remembered too, so a broken
cache is healed (deleted and re-downloaded) once per process, not per call.

Wire format: one JSON object per line, one request per connection.

- ``{"v": 1, "op": "hello"}`` ->
  ``{"v": 1, "model": <name>, "dim": <int>, "libs": {<package>: <version>}}``
- ``{"v": 1, "op": "embed", "model": <name>, "texts": [...]}`` ->
  ``{"v": 1, "dim": <int>, "n": <int>, "vectors": <base64 float32, row-major>}``
- any failure -> ``{"v": 1, "error": <message>}``

`libs` must match the client's installed versions exactly: a different
fastembed or onnxruntime may preprocess or pool differently, and query vectors
that drift from the stored ones fail retrieval with no error anywhere.

POSIX only: on Windows there is no server and every run loads locally.
"""

from __future__ import annotations

import base64
import importlib.metadata
import json
import logging
import os
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
HELLO_TIMEOUT_S = 2.0
# Measured on the real model (Apple Silicon CPU): a short query 0.16 s, 64 texts
# of 800 chars 14.6 s, one 8,000-char text 4.8 s, one 32,000-char text 33-75 s
# (attention is quadratic in length). So the deadline is a queueing allowance
# plus a per-character term that covers the worst case; the same compute would
# cost a local run just as long, plus 1.3 GB.
# The base sits above the worst single text measured: the server interleaves
# per text, so a short query can wait for one long text (the observer's own
# sweep embeds up to 32,000 chars), and waiting beats loading 1.3 GB to do the
# same work.
EMBED_TIMEOUT_S = 90.0
EMBED_TIMEOUT_PER_CHAR_S = 0.003
# How long a transport failure is retried before the run loads its own model.
# A recycle is exec, imports, the lock and a model load: ~2 s on an idle
# machine, measured past 6 s on a loaded one. Refused connections with no
# observer alive are not retried at all (see `_worth_waiting`), so this only
# bounds waiting through a real recycle.
RECOVER_S = 15.0
RETRY_PAUSE_S = 0.25
BATCH_TEXTS = 64
BATCH_CHARS = 16 * 1024
# Past the model's 8,192-token window anyway; cut on the client so one huge
# text can never exceed the server's request cap.
MAX_TEXT_CHARS = 64 * 1024
# Server-side caps: a client batches well below them, so hitting one means a
# client that does not, and it is refused rather than allowed to hold the model.
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_TEXTS = 256
MAX_REPLY_BYTES = 64 * 1024 * 1024
SERVER_READ_TIMEOUT_S = 10.0
LIB_PACKAGES = ("fastembed", "onnxruntime")

_server_process = False


def socket_path() -> Path:
    """Resolved at call time, so a test's patched home never reaches the real
    daemon (mirrors ``neo.memory.metrics._metrics_path``)."""
    return Path.home() / ".neo" / "run" / "embed.sock"


def supported() -> bool:
    return os.name != "nt" and hasattr(socket, "AF_UNIX")


def mark_server_process() -> None:
    """The process that serves the model must not try to borrow it from
    itself (or from a stale socket left by its previous incarnation)."""
    global _server_process
    _server_process = True


def is_server_process() -> bool:
    return _server_process


def library_versions() -> dict:
    versions = {}
    for package in LIB_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _record(source: str, **fields) -> None:
    """Where this process's embedder came from, so the memory win is a
    recorded fact rather than an inference."""
    try:
        from neo.memory.metrics import record
        record("embedder_source", source=source, **fields)
    except Exception:
        pass


# ------------------------------------------------------------------ client


class _Transport(Exception):
    """The server could not be reached or went away mid-request. Worth
    retrying: this is what an observer recycle looks like from outside."""

    def __init__(self, message: str, refused: bool = False):
        super().__init__(message)
        self.refused = refused


def _worth_waiting(error: _Transport) -> bool:
    """A refused connection with no observer holding its lock is a socket left
    by a killed observer (SIGKILL, jetsam, a native crash skip every cleanup):
    nothing will ever answer, so waiting would only cost every run
    `RECOVER_S`. A missing socket is a recycle in progress (the observer
    unlinks it just before exec), and a refusal with the lock held is a full
    backlog: both are worth the wait."""
    if not error.refused:
        return True
    try:
        # The probe takes the observer's lock for an instant. A starting
        # observer that collides with it retries its acquire (ten tries, 1 s
        # apart, in `_daemon_main`); keep that a retry, never single-shot.
        from neo.memory.observer import _observer_lock_held
        return _observer_lock_held()
    except Exception:
        return True


def _call(path: Path, request: dict, timeout: float) -> dict:
    """One request, bounded by a deadline over the whole exchange (a peer that
    trickles bytes cannot stretch a per-recv timeout indefinitely)."""
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("deadline passed")
        return left

    payload = json.dumps(request).encode() + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(remaining())
            sock.connect(str(path))
            sock.settimeout(remaining())
            sock.sendall(payload)
            chunks, size = [], 0
            while True:
                sock.settimeout(remaining())
                chunk = sock.recv(1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_REPLY_BYTES:
                    raise ValueError("reply too large")
                if chunk.endswith(b"\n"):
                    break
    except (FileNotFoundError, ConnectionError) as e:
        raise _Transport(f"{type(e).__name__}: {e}",
                         refused=isinstance(e, ConnectionRefusedError)) from e
    if not chunks:
        raise _Transport("connection closed before a reply")
    reply = json.loads(b"".join(chunks))
    if not isinstance(reply, dict) or reply.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"unexpected reply {str(reply)[:80]!r}")
    if "error" in reply:
        raise RuntimeError(str(reply["error"]))
    return reply


def _vectors(reply: dict, expected: int, dim: int) -> list:
    n = int(reply["n"])
    if n != expected or int(reply["dim"]) != dim:
        raise ValueError(f"asked for {expected}x{dim} vectors, got {n}x{reply['dim']}")
    data = np.frombuffer(base64.b64decode(reply["vectors"]), dtype=np.float32)
    if data.size != n * dim:
        raise ValueError("vector payload does not match its shape")
    # Writable rows, like fastembed's own: a caller normalizing in place must
    # not trip over a read-only buffer.
    return list(data.reshape(n, dim).copy())


def _batches(texts: list) -> Iterable[list]:
    batch, chars = [], 0
    for text in texts:
        if batch and (len(batch) >= BATCH_TEXTS or chars + len(text) > BATCH_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch


class RemoteEmbedder:
    """Drop-in for a fastembed ``TextEmbedding``: callers use only
    ``embed(texts)``. See the module docstring for when it waits, when it
    falls back, and why a fallback is final for the process."""

    def __init__(self, path: Path, model_name: str, dim: int,
                 local_factory: Callable[[], object]):
        self.path, self.model_name, self.dim = path, model_name, dim
        self._local_factory = local_factory
        self._local: Optional[object] = None
        self._local_failed = False
        self._lock = threading.Lock()

    def embed(self, texts: Iterable[str]):
        texts = [t[:MAX_TEXT_CHARS] for t in texts]
        if self._local is None and not self._local_failed:
            try:
                return iter(self._remote(texts))
            except Exception as e:
                self._fall_back(e)
        if self._local is not None:
            return self._local.embed(texts)
        return iter(())  # no model anywhere: callers treat this as "no embedding"

    def _remote(self, texts: list) -> list:
        vectors: list = []
        for batch in _batches(texts):
            request = {"v": PROTOCOL_VERSION, "op": "embed",
                       "model": self.model_name, "texts": batch}
            give_up = time.monotonic() + RECOVER_S
            while True:
                try:
                    reply = _call(self.path, request, EMBED_TIMEOUT_S
                                  + EMBED_TIMEOUT_PER_CHAR_S * sum(map(len, batch)))
                    break
                except _Transport as e:
                    if time.monotonic() >= give_up or not _worth_waiting(e):
                        raise
                    time.sleep(RETRY_PAUSE_S)
            vectors.extend(_vectors(reply, len(batch), self.dim))
        return vectors

    def _fall_back(self, error: Exception) -> None:
        with self._lock:
            if self._local is not None or self._local_failed:
                return
            logger.warning("Resident embedder at %s failed (%s: %s); loading the "
                           "model in this process", self.path, type(error).__name__, error)
            self._local = self._local_factory()
            self._local_failed = self._local is None
            _record("fallback", reason=type(error).__name__, loaded=not self._local_failed)


def connect(model_name: str, local_factory: Callable[[], object],
            path: Optional[Path] = None) -> Optional[RemoteEmbedder]:
    """A `RemoteEmbedder` when a live, compatible server for `model_name`
    answers at `path`, else None (the caller loads the model itself). Never
    raises."""
    path = path or socket_path()
    if not supported() or not path.exists():
        return None
    give_up = time.monotonic() + RECOVER_S
    while True:
        try:
            hello = _call(path, {"v": PROTOCOL_VERSION, "op": "hello"}, HELLO_TIMEOUT_S)
            break
        except _Transport as e:
            # Refused or gone (a recycle under way): worth a few more tries
            # before paying 1.3 GB. A server that is up but silent is not.
            if time.monotonic() >= give_up or not _worth_waiting(e):
                logger.warning("Resident embedder at %s did not answer (%s); loading "
                               "the model in this process", path, e)
                _record("local", reason="unreachable")
                return None
            time.sleep(RETRY_PAUSE_S)
        except Exception as e:
            logger.warning("Resident embedder at %s answered unusably (%s: %s); "
                           "loading the model in this process", path, type(e).__name__, e)
            _record("local", reason="bad_hello")
            return None
    try:
        dim = hello["dim"]
        if hello.get("model") != model_name:
            reason = f"serves {hello.get('model')}"
        elif not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            reason = f"reported dim {dim!r}"
        elif hello.get("libs") != library_versions():
            reason = f"runs {hello.get('libs')}, this process {library_versions()}"
        else:
            reason = ""
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    if reason:
        # A warning: the user should learn that the memory saving is off
        # (typically a dev venv whose fastembed differs from the observer's).
        logger.warning("Resident embedder at %s not used: %s", path, reason)
        _record("local", reason="incompatible")
        return None
    _record("remote")
    return RemoteEmbedder(path, model_name, dim, local_factory)



def embed_query(text: str, model_name: str, deadline_s: float,
                path: Optional[Path] = None) -> Optional[np.ndarray]:
    """One short text from a live, compatible server within `deadline_s`, or
    None. For callers on a latency budget (`neo recall`, on every prompt):
    no retries, no local fallback, no metrics line, never raises."""
    vectors = embed_queries([text], model_name, deadline_s, path)
    return vectors[0] if vectors else None


def embed_queries(texts: list[str], model_name: str, deadline_s: float,
                  path: Optional[Path] = None) -> Optional[list]:
    """`embed_query` for a few texts in ONE request, so a caller needing two
    vectors pays one round trip and one deadline. None on any failure."""
    path = path or socket_path()
    if not texts or not supported() or not path.exists():
        return None
    end = time.monotonic() + deadline_s
    try:
        hello = _call(path, {"v": PROTOCOL_VERSION, "op": "hello"},
                      max(0.0, end - time.monotonic()))
        if hello.get("model") != model_name or hello.get("libs") != library_versions():
            return None
        dim = int(hello["dim"])
        reply = _call(path, {"v": PROTOCOL_VERSION, "op": "embed", "model": model_name,
                             "texts": [text[:MAX_TEXT_CHARS] for text in texts]},
                      max(0.0, end - time.monotonic()))
        return _vectors(reply, len(texts), dim)
    except Exception:
        return None

# ------------------------------------------------------------------ server


class _Handler(socketserver.StreamRequestHandler):
    timeout = SERVER_READ_TIMEOUT_S

    def handle(self) -> None:
        server: "_Server" = self.server  # type: ignore[assignment]
        try:
            line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if len(line) > MAX_REQUEST_BYTES:
                raise ValueError(f"request over {MAX_REQUEST_BYTES} bytes")
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("v") != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol version")
            if request.get("op") == "hello":
                reply = {"v": PROTOCOL_VERSION, "model": server.model_name,
                         "dim": server.dim, "libs": server.libs}
            elif request.get("op") == "embed":
                if request.get("model") != server.model_name:
                    raise ValueError(f"this server embeds with {server.model_name}")
                texts = request.get("texts")
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    raise ValueError("texts must be a list of strings")
                if len(texts) > MAX_TEXTS:
                    raise ValueError(f"over {MAX_TEXTS} texts in one request")
                # One text at a time: each call takes the model's lock, so a
                # short query from another run waits for one text, never for a
                # whole batch (64 texts can hold the model for 15 s).
                matrix = np.asarray([np.asarray(next(iter(server.embedder.embed([t]))),
                                                dtype=np.float32) for t in texts],
                                    dtype=np.float32).reshape(len(texts), server.dim)
                reply = {"v": PROTOCOL_VERSION, "dim": server.dim, "n": len(texts),
                         "vectors": base64.b64encode(matrix.tobytes()).decode()}
            else:
                raise ValueError(f"unknown op {request.get('op')!r}")
        except Exception as e:
            reply = {"v": PROTOCOL_VERSION, "error": f"{type(e).__name__}: {e}"}
        try:
            self.wfile.write(json.dumps(reply).encode() + b"\n")
        except OSError:
            pass  # the client gave up; nothing to tell it


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    # Runs start together (hooks, parallel agents); a backlog of 5 refused
    # the sixth concurrent hello on macOS, and each refusal cost 1.3 GB.
    request_queue_size = socket.SOMAXCONN
    embedder: object
    model_name: str
    dim: int
    libs: dict


class EmbedServer:
    """A running server; `close()` stops it and removes its socket."""

    def __init__(self, server: _Server, path: Path):
        self._server, self.path = server, path
        self._thread = threading.Thread(target=server.serve_forever,
                                        name="neo-embed-server", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def clear_stale(path: Optional[Path] = None) -> None:
    """Remove a socket nothing answers on. The observer calls this first,
    before anything that can fail: a socket left by a killed predecessor,
    with a live observer holding the lock but unable to serve, would make
    every neo run wait out `RECOVER_S` on it. Never raises."""
    path = path or socket_path()
    try:
        if path.exists() or path.is_symlink():
            try:
                _call(path, {"v": PROTOCOL_VERSION, "op": "hello"}, HELLO_TIMEOUT_S)
            except Exception:
                path.unlink()
    except Exception as e:
        logger.warning("Could not clear a stale embedder socket at %s: %s", path, e)


def serve(embedder, model_name: str, path: Optional[Path] = None) -> Optional[EmbedServer]:
    """Serve `embedder` at `path`. Returns None, having logged why, when it
    cannot: unsupported platform, another live server, or any bind failure.
    Never raises: a failure here must not take the observer down with it."""
    path = path or socket_path()
    if not supported():
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        if path.exists() or path.is_symlink():
            try:
                _call(path, {"v": PROTOCOL_VERSION, "op": "hello"}, HELLO_TIMEOUT_S)
                logger.info("A resident embedder is already serving at %s", path)
                return None
            except Exception:
                path.unlink()  # stale: its server is gone
        dim = len(next(iter(embedder.embed(["dimension probe"]))))
        server = _Server(str(path), _Handler)
        os.chmod(path, 0o600)
        server.embedder, server.model_name, server.dim = embedder, model_name, dim
        server.libs = library_versions()
        logger.info("Serving %s embeddings at %s", model_name, path)
        return EmbedServer(server, path)
    except Exception as e:
        logger.warning("Resident embedder not started at %s: %s: %s",
                       path, type(e).__name__, e)
        return None
