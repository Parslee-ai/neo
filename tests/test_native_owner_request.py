"""Exercise guardian request framing over actual OS pipes, without CAR effects."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE = Path(__file__).resolve().parents[1] / "src/neo/lattice_native_owner.py"
LIMIT = 8 * 1024 * 1024
READER = """
import hashlib, importlib.util, json, sys
spec = importlib.util.spec_from_file_location('bridge', sys.argv[1])
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
value = bridge._read_worker_request(sys.stdin.buffer)
canonical = bridge._canonical(value).encode('utf-8')
print(json.dumps({'sha256': hashlib.sha256(canonical).hexdigest(),
                  'pins': len(value.get('config', {}).get('pins', {})) if isinstance(value, dict) else 0}))
"""


def read_pipe(payload):
    return subprocess.run([sys.executable, "-I", "-B", "-c", READER, str(MODULE)],
                          input=payload, capture_output=True, timeout=15)


def test_full_inventory_larger_than_old_limit_survives_pipe():
    value = {"config": {"pins": {
        "/installed/" + "path/" * 32 + str(i): "a" * 64 for i in range(5372)
    }}}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert 1024 * 1024 < len(payload) < LIMIT
    result = read_pipe(payload)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout) == {"sha256": hashlib.sha256(payload).hexdigest(), "pins": 5372}


def test_exact_byte_limit_is_accepted():
    payload = b'"' + b'a' * (LIMIT - 2) + b'"'
    result = read_pipe(payload)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("payload", [b'"' + b'a' * (LIMIT - 1) + b'"',
                                     ('"' + '\u00e9' * (LIMIT // 2) + '"').encode()],
                         ids=["ascii-over-limit", "utf8-over-limit"])
def test_oversize_is_refused_by_bytes_before_json_decode(payload):
    result = read_pipe(payload)
    assert result.returncode != 0
    assert b"Native owner request exceeds its 8 MiB bound" in result.stderr
    assert b"JSONDecodeError" not in result.stderr


def test_truncated_json_is_refused():
    result = read_pipe(b'{"config": {"pins":')
    assert result.returncode != 0
    assert b"JSONDecodeError" in result.stderr


def test_sender_refuses_oversize_without_launching_guardian():
    import importlib.util
    from unittest.mock import patch
    spec = importlib.util.spec_from_file_location("bridge", MODULE)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    config = {"pins": {"p": "x" * LIMIT}}
    with patch.object(bridge.subprocess, "run") as run:
        with pytest.raises(ValueError, match="8 MiB bound"):
            bridge._run(config, "status", Path("/unused"), {"id": "agent"})
        run.assert_not_called()
