import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neo.engine import NeoEngine
from neo.lattice import Answerer, Task
from neo.operating_mode import OperatingMode, ModeValidationError
from neo.scoped_consumer import ScopedConsumer
from tests.test_scoped_consumer import scope, lifecycle_node


PATCH = '--- a/sample.py\n+++ b/sample.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n'


def test_scoped_review_checks_patch_and_only_processes_advice(tmp_path):
    binding, _, _ = scope(tmp_path)
    answerer = Answerer(str(tmp_path), scoped_consumer=binding)
    requests = []
    answerer._engine = SimpleNamespace(
        process=lambda request: requests.append(request) or object(),
        _extract_input_constraints=lambda request: ['constraint'],
        _check_constraints_static=lambda suggestions, constraints: None,
    )
    checked = []
    def checks(suggestions):
        checked.extend(suggestions)
        return [SimpleNamespace(tool_name='fixture-check', status='passed', summary='checked')]
    with patch('neo.static_analysis.run_static_checks', side_effect=checks), \
         patch('neo.lattice.render_output', return_value='advisory review'):
        result = answerer.review(Task('review', prompt='review patch', diff=PATCH))
    assert len(checked) == 1
    assert checked[0].file_path == str(tmp_path / 'sample.py')
    assert checked[0].unified_diff.endswith('+x = 2\n')
    assert len(requests) == 1 and requests[0].operating_mode is OperatingMode.ADVISE
    assert 'fixture-check: passed' in result and 'advisory review' in result


def test_revocation_during_deterministic_review_blocks_advisory_disclosure(tmp_path):
    binding, _, _ = scope(tmp_path)
    answerer = Answerer(str(tmp_path), scoped_consumer=binding)
    answerer._engine = SimpleNamespace(
        process=lambda request: pytest.fail('inference after revocation'),
        _extract_input_constraints=lambda request: [],
        _check_constraints_static=lambda suggestions, constraints: None,
    )
    def checks(suggestions):
        binding.runtime.refused = True
        return []
    with patch('neo.static_analysis.run_static_checks', side_effect=checks), \
         pytest.raises(PermissionError):
        answerer.review(Task('review', diff=PATCH))


def test_scoped_review_refuses_workspace_escape_before_file_check(tmp_path):
    binding, _, _ = scope(tmp_path)
    answerer = Answerer(str(tmp_path), scoped_consumer=binding)
    with patch('neo.static_analysis.run_static_checks', side_effect=AssertionError('private file read')), \
         pytest.raises(PermissionError, match='escapes'):
        answerer.review(Task('review', diff=PATCH.replace('sample.py', '../private.py')))


def test_missing_mode_is_not_implicitly_advisory(tmp_path):
    binding, _, _ = scope(tmp_path)
    engine = object.__new__(NeoEngine)
    engine.scoped_consumer = binding
    engine._process_lock = threading.Lock()
    engine._process_guarded = lambda request: pytest.fail('private read without explicit mode')
    with pytest.raises(ModeValidationError):
        engine.process(object())
    assert not engine._process_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['sdk', 'issuance', 'expired', 'renewal'])
async def test_running_node_latches_issuance_refusal_without_reconnecting(tmp_path, monkeypatch, failure):
    from neo import lattice
    node, _, calls, runtimes, clients = lifecycle_node(tmp_path, monkeypatch)
    monkeypatch.setattr(lattice, 'POLL_SECS', 0.001)
    original_factory = ScopedConsumer.from_issued
    def refused_factory(*args):
        if failure != 'renewal' or runtimes:
            raise PermissionError('original authority refused')
        return original_factory(*args)
    if failure in {'sdk', 'renewal'}:
        monkeypatch.setattr(ScopedConsumer, 'from_issued', refused_factory)
    original_client_factory = node._client_factory
    def client_factory(url):
        client = original_client_factory(url)
        original_call = client.call
        async def call(method, params, timeout=None):
            result = await original_call(method, params, timeout)
            if method == 'session.consumer.issue':
                if failure == 'issuance':
                    raise PermissionError('parent cannot issue')
                if failure == 'expired':
                    result['expires_at_ms'] = result['issued_at_ms'] - 1
            return result
        client.call = call
        return client
    node._client_factory = client_factory
    running = asyncio.create_task(node.run())
    try:
        if failure == 'renewal':
            await asyncio.wait_for(node._connected.wait(), 5)
            node._consumer_renew_at = 0
        await asyncio.wait_for(running, 5)
        assert node._consumer_refused and node._stop.is_set()
        assert len(clients) == 1 and not clients[0].connected
        assert [m for _, m, _ in calls].count('session.consumer.issue') == (2 if failure == 'renewal' else 1)
        assert all(runtime.closes == 1 for runtime in runtimes)
        assert node._scoped_consumer is None
    finally:
        node._stop.set()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('message', [
    'send scoped consumer authentication failed',
    'scoped consumer authentication timed out',
    'daemon closed during scoped consumer authentication',
    'receive scoped consumer authentication failed',
])
async def test_exact_sdk_attach_transport_loss_keeps_original_recovery(tmp_path, monkeypatch, message):
    node, _, calls, runtimes, clients = lifecycle_node(tmp_path, monkeypatch)
    original_factory = ScopedConsumer.from_issued
    def unavailable_factory(*args):
        raise RuntimeError(message)
    monkeypatch.setattr(ScopedConsumer, 'from_issued', unavailable_factory)
    try:
        with pytest.raises(ConnectionError):
            await node.connect_once()
        assert not node._consumer_refused and node._scoped_consumer is None
        assert [p for _, m, p in calls if m == 'session.consumer.revoke'] == [{'consumer_id': 'child-0'}]
        await clients[0].close()
        monkeypatch.setattr(ScopedConsumer, 'from_issued', original_factory)
        await node.connect_once()
        assert len(clients) == 2 and node._scoped_consumer is not None
        assert node.client.connected and not node._consumer_refused
    finally:
        await node._drop_consumer(node.client)
        if node.client is not None:
            await node.client.close()
        node._executor.shutdown(wait=False, cancel_futures=True)
