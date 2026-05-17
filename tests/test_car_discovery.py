import subprocess

from neo import car_discovery
from neo.car_discovery import CarInstallInfo, discover_car


def test_car_install_info_summary_not_found():
    info = CarInstallInfo()

    assert info.summary() == "not found"
    assert not info.available


def test_discover_car_reports_cli_server_binding_and_daemon(monkeypatch):
    monkeypatch.setattr(car_discovery, "_candidate_executable", lambda name: f"/bin/{name}")
    monkeypatch.setattr(car_discovery, "_version_for", lambda path: "car 1.2.3")
    monkeypatch.setattr(car_discovery, "_python_binding", lambda: "car_runtime")
    monkeypatch.setattr(car_discovery, "_daemon_running", lambda: True)

    info = discover_car()

    assert info.cli_path == "/bin/car"
    assert info.server_path == "/bin/car-server"
    assert info.cli_version == "car 1.2.3"
    assert info.python_binding == "car_runtime"
    assert info.daemon_running is True
    assert info.has_python_runtime


def test_version_for_handles_timeout(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("car", timeout=3)

    monkeypatch.setattr(car_discovery.subprocess, "run", raise_timeout)

    assert car_discovery._version_for("/bin/car") is None
