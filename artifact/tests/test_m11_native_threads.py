"""Native format runtimes must obey the command's thread contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_experiment


def test_native_thread_environment_is_set_before_adapter_probe(monkeypatch):
    for name in run_experiment.NATIVE_THREAD_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert run_experiment.configure_native_threads("3") == [3]
    assert all(os.environ[name] == "3" for name in run_experiment.NATIVE_THREAD_ENVS)


@pytest.mark.parametrize("spec", ["", "0", "-1", "1,8"])
def test_invalid_or_multi_value_native_thread_scope_is_rejected(spec):
    with pytest.raises(SystemExit):
        run_experiment.configure_native_threads(spec)
