"""The A2A handler's error contract.

`_handle_call`'s own docstring promises that "every error path also returns
valid JSON so the daemon response stays well-formed". A peer consuming those
responses needs them to be *uniformly* shaped too — five of nine omitted
`error_type`, so anything branching on it hit a missing key on more paths than
it hit one.

Asserted against the source rather than by driving the handler: reaching most
of these branches needs a live car-server daemon, and a contract test that only
runs when a daemon happens to be up is not a contract test.
"""

import ast
from pathlib import Path

import pytest

CAR_HOST = Path(__file__).resolve().parent.parent / "src" / "neo" / "car_host.py"


def _error_responses() -> list[dict]:
    """Every `json.dumps({...})` literal in the file that carries an "error" key."""
    tree = ast.parse(CAR_HOST.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "error" in keys:
            found.append({"keys": keys, "lineno": node.lineno})
    return found


def test_the_file_has_the_error_paths_we_think_it_does():
    """Guards the assertions below from silently passing on an empty set."""
    assert len(_error_responses()) >= 8


@pytest.mark.parametrize("response", _error_responses(), ids=lambda r: f"line{r['lineno']}")
def test_every_error_response_carries_error_type(response):
    assert "error_type" in response["keys"], (
        f"car_host.py:{response['lineno']} returns an error without error_type; "
        "a peer branching on the machine-readable type hits a missing key"
    )


@pytest.mark.parametrize("response", _error_responses(), ids=lambda r: f"line{r['lineno']}")
def test_every_error_response_carries_a_message(response):
    assert "message" in response["keys"], (
        f"car_host.py:{response['lineno']} returns an error with no message"
    )


def _response_named(error_name: str) -> dict:
    tree = ast.parse(CAR_HOST.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {
            k.value: v for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
        }
        value = pairs.get("error")
        if isinstance(value, ast.Constant) and value.value == error_name:
            return pairs
    raise AssertionError(f"no error response named {error_name!r}")


def test_engine_busy_is_marked_retryable():
    """A peer that reads a generic failure assumes its own request was
    malformed and stops retrying. Busy is the one condition where retrying is
    exactly right, so it must be distinguishable."""
    busy = _response_named("EngineBusy")
    retryable = busy.get("retryable")
    assert isinstance(retryable, ast.Constant) and retryable.value is True


def test_only_engine_busy_claims_retryability():
    """Marking a genuine failure retryable would send peers into a loop."""
    tree = ast.parse(CAR_HOST.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {
            k.value: v for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
        }
        if "retryable" not in pairs:
            continue
        error = pairs.get("error")
        assert isinstance(error, ast.Constant) and error.value == "EngineBusy", (
            f"line {node.lineno} claims retryable but is not EngineBusy"
        )
