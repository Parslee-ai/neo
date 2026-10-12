import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib

path = Path(__file__).resolve().parents[1] / "src/neo/lattice_native_owner.py"
spec = importlib.util.spec_from_file_location("owner_limits", path)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class NativeOwnerLimits(unittest.TestCase):
    def test_full_artifact_closure_above_old_limit_still_checks_each_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "neo"
            package.mkdir()
            entry = root / "supervised-owner.py"
            entry.write_text("# pinned entrypoint")
            pins = {str(entry): hashlib.sha256(entry.read_bytes()).hexdigest()}
            for index in range(8193):
                source = package / (str(index) + ".py")
                os.link(entry, source)
                pins[str(source)] = pins[str(entry)]
            python = str(Path(sys.executable).resolve())
            pins[python] = hashlib.sha256(Path(python).read_bytes()).hexdigest()
            config = dict(guardian_python=python, guardian_scripts=str(root),
                          child_python=python, child_sdk=str(entry),
                          child_package=str(package), ownership_root=str(root / "owner"), pins=pins)
            self.assertEqual(owner._config(config), config)
            config["pins"][str(entry)] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Native owner artifact changed"):
                owner._config(config)

