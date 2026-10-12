"""A resident embedding model, shared across neo runs (#211).

The embedding model is 614 MB on disk and ~1.3 GB resident once loaded, and it
was the whole of a neo run's memory floor: the FactStore itself is ~126 MB. A
run only needs a handful of query embeddings, so loading the model for each
one is the wrong shape. The observer daemon is already resident, so it holds
the model and serves embeddings over a Unix socket at
``~/.neo/run/embed.sock``; a neo run that finds it borrows the model instead
of loading its own.

The fallback is the product, not the daemon. No socket, a stale socket, a
server for another model or protocol, a hung server, a malformed reply, a
server that went away mid-run (the observer re-execs to bound its RSS): every
one of these ends with the run loading the model itself, exactly as it did
before this module existed. Only a socket that exists and then fails is worth
a warning; no daemon at all is the ordinary case and stays quiet.

Wire format: one JSON object per line, one request per connection.

- ``{"v": 1, "op": "hello"}`` -> ``{"v": 1, "model": <name>, "dim": <int>}``
- ``{"v": 1, "op": "embed", "model": <name>, "texts": [...]}`` ->
  ``{"v": 1, "dim": <int>, "n": <int>, "vectors": <base64 float32, row-major>}``
- any failure -> ``{"v": 1, "error": <message>}``

POSIX only: on Windows there is no server and every run loads locally.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import socketserver
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
# A hello must answer at once; anything slower is a server not worth waiting on.
HELLO_TIMEOUT_S = 0.5
# Embedding is ~20 ms a query; the allowance grows with the batch.
EMBED_TIMEOUT_BASE_S = 5.0
EMBED_TIMEOUT_PER_TEXT_S = 0.05
MAX_LINE_BYTES = 64 * 1024 * 1024

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


# ------------------------------------------------------------------ client


def _call(path: Path, request: dict, timeout: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        sock.sendall(json.dumps(request).encode() + b"\n")
        chunks, size = [], 0
        while True:
            chunk = sock.recv(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_LINE_BYTES:
                raise ValueError("reply too large")
            if chunk.endswith(b"\n"):
                break
    reply = json.loads(b"".join(chunks))
    if not isinstance(reply, dict) or reply.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"unexpected reply {str(reply)[:80]!r}")
    if "error" in reply:
        raise RuntimeError(str(reply["error"]))
    return reply


def _vectors(reply: dict, expected: int) -> np.ndarray:
    dim, n = int(reply["dim"]), int(reply["n"])
    if n != expected:
        raise ValueError(f"asked for {expected} vectors, got {n}")
    data = np.frombuffer(base64.b64decode(reply["vectors"]), dtype=np.float32)
    if data.size != n * dim:
        raise ValueError("vector payload does not match its shape")
    return data.reshape(n, dim)


class RemoteEmbedder:
    """Drop-in for a fastembed ``TextEmbedding``: callers use only
    ``embed(texts)``. Falls back to ``local_factory()`` for the rest of the
    process on the first failure, so a dead server is tried once, not on
    every call."""

    def __init__(self, path: Path, model_name: str, dim: int,
                 local_factory: Callable[[], object]):
        self.path, self.model_name, self.dim = path, model_name, dim
        self._local_factory = local_factory
        self._local: Optional[object] = None
        self._lock = threading.Lock()

    def embed(self, texts: Iterable[str]):
        texts = list(texts)
        if self._local is None:
            try:
                timeout = EMBED_TIMEOUT_BASE_S + EMBED_TIMEOUT_PER_TEXT_S * len(texts)
                reply = _call(self.path, {"v": PROTOCOL_VERSION, "op": "embed",
                                          "model": self.model_name, "texts": texts},
                              timeout)
                return iter(list(_vectors(reply, len(texts))))
            except Exception as e:
                with self._lock:
                    if self._local is None:
                        logger.warning(
                            "Resident embedder at %s failed (%s: %s); loading the "
                            "model in this process", self.path, type(e).__name__, e)
                        self._local = self._local_factory()
        if self._local is None:
            raise RuntimeError("no embedding model available")
        return self._local.embed(texts)


def connect(model_name: str, local_factory: Callable[[], object],
            path: Optional[Path] = None) -> Optional[RemoteEmbedder]:
    """A `RemoteEmbedder` when a live server for `model_name` answers at
    `path`, else None (the caller loads the model itself)."""
    path = path or socket_path()
    if not supported() or not path.exists():
        return None
    try:
        hello = _call(path, {"v": PROTOCOL_VERSION, "op": "hello"}, HELLO_TIMEOUT_S)
    except Exception as e:
        logger.warning("Resident embedder at %s did not answer (%s: %s); loading "
                       "the model in this process", path, type(e).__name__, e)
        return None
    if hello.get("model") != model_name:
        logger.info("Resident embedder serves %s, not %s", hello.get("model"), model_name)
        return None
    return RemoteEmbedder(path, model_name, int(hello["dim"]), local_factory)


# ------------------------------------------------------------------ server


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "_Server" = self.server  # type: ignore[assignment]
        try:
            line = self.rfile.readline(MAX_LINE_BYTES)
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("v") != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol version")
            if request.get("op") == "hello":
                reply = {"v": PROTOCOL_VERSION, "model": server.model_name, "dim": server.dim}
            elif request.get("op") == "embed":
                if request.get("model") != server.model_name:
                    raise ValueError(f"this server embeds with {server.model_name}")
                texts = request.get("texts")
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    raise ValueError("texts must be a list of strings")
                matrix = np.asarray([np.asarray(v, dtype=np.float32)
                                     for v in server.embedder.embed(texts)],
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
    embedder: object
    model_name: str
    dim: int


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
        logger.info("Serving %s embeddings at %s", model_name, path)
        return EmbedServer(server, path)
    except Exception as e:
        logger.warning("Resident embedder not started at %s: %s: %s",
                       path, type(e).__name__, e)
        return None
