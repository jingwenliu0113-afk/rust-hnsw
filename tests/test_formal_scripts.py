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
  case "$3" in *flip*) echo 0 > "$FAKE_DIR/mode";; esac
  case "$3" in *fail*) exit 1;; esac
  exit 0
fi
if [ "$1" = "-c" ]; then
  exec python3 "$@"
fi
if [ "$1" = "-m" ] && [ "$2" = "benchmarks.machine_quiet" ]; then
  # 只模擬電力模式：--powermode N 與 $FAKE_DIR/mode 不同時不閒置。
  if [ "${3:-}" = "--powermode" ] && [ "$(cat "$FAKE_DIR/mode")" != "$4" ]; then
    echo "no powermode=$(cat "$FAKE_DIR/mode") expected=$4"
  else
    echo yes
  fi
  exit 0
fi
exit 99
"""
# pmset -g：powermode 取自 $FAKE_DIR/mode（formal run 中被改掉的情況由 *flip* 的 config 模擬）。
FAKE_PMSET = r"""#!/usr/bin/env bash
echo "System-wide power settings:"
echo " powermode            $(cat "$FAKE_DIR/mode")"
"""


@pytest.fixture
def fake(tmp_path):
    for name, text in (("python", FAKE_PYTHON), ("pmset", FAKE_PMSET)):
        path = tmp_path / name
        path.write_text(text)
        path.chmod(0o755)
    (tmp_path / "mode").write_text("2\n")
    return tmp_path


def batch(fake, *args, idle_status="ok", idle_exit=0):
    env = {"PATH": "/usr/bin:/bin", "FORMAL_PYTHON": str(fake / "python"), "FORMAL_PMSET": str(fake / "pmset"),
           "FAKE_DIR": str(fake), "FAKE_IDLE_STATUS": idle_status, "FAKE_IDLE_EXIT": str(idle_exit)}
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


def test_the_batch_runs_only_in_the_required_power_mode(fake):
    # decision 016：同一個資料集的 formal run 用同一個電力模式；每份 config 開始前確認。
    result, markers, calls, text = batch(fake, "--powermode", "2", "--skip-idle", "a.toml", "b.toml")
    assert result.returncode == 0 and calls == ["a.toml --mode formal", "b.toml --mode formal"]
    (fake / "calls").unlink()
    (fake / "batch.log").unlink()
    (fake / "mode").write_text("0\n")
    result, markers, calls, text = batch(fake, "--powermode", "2", "a.toml")
    assert result.returncode == 1 and calls == [] and markers == ["STOP"]
    assert "powermode=0 expected=2" in text  # 閒置 load 量測之前就停


def test_a_power_mode_change_between_configs_stops_the_batch(fake):
    result, markers, calls, text = batch(fake, "--powermode", "2", "--skip-idle", "a-flip.toml", "b.toml")
    assert result.returncode == 1
    assert calls == ["a-flip.toml --mode formal"] and markers == ["RUN_START", "RUN_END", "STOP"]
    assert "powermode=0 expected=2" in text


def test_the_launcher_keeps_waiting_while_the_power_mode_differs(fake):
    (fake / "mode").write_text("0\n")
    env = {"PATH": "/usr/bin:/bin", "FORMAL_PYTHON": str(fake / "python"), "FORMAL_PMSET": str(fake / "pmset"),
           "FAKE_DIR": str(fake), "FORMAL_POLL_S": "1"}
    result = subprocess.run([str(REPO / "scripts" / "formal_launcher.sh"), str(fake / "batch.log"), "4", "--powermode", "2",
                             "a.toml"], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and result.stdout.startswith("GAVE_UP")
    assert not (fake / "batch.log.launcher").exists()  # 沒有啟動批次
    assert "powermode=0 expected=2" in (fake / "batch.log.wait").read_text()  # 等待的原因留在 .wait


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
