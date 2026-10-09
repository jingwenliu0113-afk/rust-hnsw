"""正式 run 的批次與等待腳本（scripts/formal_batch.sh、formal_launcher.sh）：以假的 python 驗證流程，不跑任何 benchmark。"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FAKE_PYTHON = r"""#!/usr/bin/env bash
# -m benchmarks.idle_load：寫出紀錄並在最後一行印出路徑；-m benchmarks.run：記下 config，名稱含 fail 的以 1 結束；-c：交給真的 python。
if [ "$1" = "-m" ] && [ "$2" = "benchmarks.idle_load" ]; then
  echo "{\"status\": \"$FAKE_IDLE_STATUS\"}" > "$FAKE_DIR/idle.json"
  echo "$FAKE_IDLE_STATUS: p50 1.0"
  echo "$FAKE_DIR/idle.json"
  exit "${FAKE_IDLE_EXIT:-0}"
fi
if [ "$1" = "-m" ] && [ "$2" = "benchmarks.run" ]; then
  echo "$3 $4 $5" >> "$FAKE_DIR/calls"
  case "$3" in *fail*) exit 1;; esac
  exit 0
fi
if [ "$1" = "-c" ]; then
  exec python3 "$@"
fi
exit 99
"""


@pytest.fixture
def fake(tmp_path):
    python = tmp_path / "python"
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    return tmp_path


def batch(fake, *args, idle_status="ok", idle_exit=0):
    env = {"PATH": "/usr/bin:/bin", "FORMAL_PYTHON": str(fake / "python"), "FAKE_DIR": str(fake),
           "FAKE_IDLE_STATUS": idle_status, "FAKE_IDLE_EXIT": str(idle_exit)}
    log = fake / "batch.log"
    result = subprocess.run([str(REPO / "scripts" / "formal_batch.sh"), str(log), *args], env=env,
                            capture_output=True, text=True, timeout=60)
    markers = [line.split()[0] for line in log.read_text().splitlines()
               if line.split() and line.split()[0] in ("IDLE_START", "IDLE_END", "RUN_START", "RUN_END", "STOP", "ALL_DONE")] \
        if log.exists() else []
    calls = (fake / "calls").read_text().splitlines() if (fake / "calls").exists() else []
    return result, markers, calls, log.read_text() if log.exists() else ""


def test_an_ok_idle_load_runs_every_config_in_order(fake):
    result, markers, calls, text = batch(fake, "a.toml", "b.toml")
    assert result.returncode == 0
    assert markers == ["IDLE_START", "IDLE_END", "RUN_START", "RUN_END", "RUN_START", "RUN_END", "ALL_DONE"]
    assert calls == ["a.toml --mode formal", "b.toml --mode formal"]
    assert "IDLE_END" in text and "status=ok" in text


@pytest.mark.parametrize("status, code", [("not_idle", 0), ("exceeds", 0), ("invalid_power", 0), ("ok", 3)])
def test_no_config_runs_unless_the_idle_load_is_ok(fake, status, code):
    result, markers, calls, text = batch(fake, "a.toml", idle_status=status, idle_exit=code)
    assert result.returncode == 1 and calls == []
    assert markers == ["IDLE_START", "IDLE_END", "STOP"] and f"status={status}" in text


def test_the_batch_stops_at_the_first_run_that_is_not_complete(fake):
    result, markers, calls, text = batch(fake, "--skip-idle", "a.toml", "b-fail.toml", "c.toml")
    assert result.returncode == 1
    assert markers == ["RUN_START", "RUN_END", "RUN_START", "RUN_END", "STOP"]
    assert calls == ["a.toml --mode formal", "b-fail.toml --mode formal"]
    assert "STOP b-fail.toml exit=1" in text


def test_scripts_reject_missing_arguments(fake):
    assert batch(fake, "--skip-idle")[0].returncode == 0  # 沒有 config：只有 ALL_DONE
    usage = subprocess.run([str(REPO / "scripts" / "formal_batch.sh"), str(fake / "x.log")], capture_output=True)
    assert usage.returncode == 64
    launcher = subprocess.run([str(REPO / "scripts" / "formal_launcher.sh"), str(fake / "x.log"), "10"], capture_output=True)
    assert launcher.returncode == 64


def test_the_launcher_gives_up_after_the_maximum_wait_without_launching(fake):
    result = subprocess.run([str(REPO / "scripts" / "formal_launcher.sh"), str(fake / "batch.log"), "0", "a.toml"],
                            env={"PATH": "/usr/bin:/bin", "FORMAL_PYTHON": str(fake / "python")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and result.stdout.startswith("GAVE_UP")
    assert not (fake / "batch.log").exists() and not (fake / "batch.log.launcher").exists()
