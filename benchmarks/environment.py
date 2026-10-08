"""formal 模式的前置檢查與 hygiene 紀錄（decision 008 §1、§2、§8、§9、§10）。

每個檢查都接受注入的路徑或數值，測試因此能同時走到「通過」與「拒絕」兩條路徑。
只讀取系統狀態，不修改任何系統設定，也不碰 lock 本身（不建立、不刪除）。
"""

import hashlib
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

from benchmarks import datasets
from benchmarks.protocol import LOCK_DIR as _LOCK_DIR

LOCK_DIR = Path(_LOCK_DIR)


class PreflightError(RuntimeError):
    """formal 模式的前置條件不成立：拒絕執行（或在執行中停止）。"""


def _run(*args: str, cwd=None) -> str:
    return subprocess.run(list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- lock ----


def read_lock_owner(lock_dir=LOCK_DIR) -> tuple[int, str]:
    owner = Path(lock_dir) / "owner.txt"
    try:
        text = owner.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PreflightError(f"{owner} not found: run formal benchmarks under ./tools/with-mac-lock") from None
    pids = [line[len("pid=") :] for line in text.splitlines() if line.startswith("pid=")]
    if len(pids) != 1 or not re.fullmatch(r"[0-9]+", pids[0]):
        raise PreflightError(f"{owner} has no valid pid= line")
    return int(pids[0]), text


def ancestor_pids(pid: int) -> list[int]:
    """`pid` 自己以及所有祖先 process（由 `ps` 的 pid / ppid 對照表往上走）。"""
    parents = {}
    for line in _run("ps", "-Ao", "pid=,ppid=").splitlines():
        fields = line.split()
        if len(fields) == 2:
            parents[int(fields[0])] = int(fields[1])
    chain = [pid]
    while chain[-1] in parents and parents[chain[-1]] not in chain and parents[chain[-1]] > 0:
        chain.append(parents[chain[-1]])
    return chain


def check_lock(lock_dir=LOCK_DIR, pid: int | None = None) -> str:
    """lock 的 owner 必須是自己或任一祖先（with-mac-lock 以子 process 執行命令）。回傳 owner.txt 全文。"""
    owner_pid, text = read_lock_owner(lock_dir)
    me = os.getpid() if pid is None else pid
    if owner_pid not in ancestor_pids(me):
        raise PreflightError(f"lock owner pid {owner_pid} is not an ancestor of this process ({me})")
    return text


# ---- git ----


def git_status(repo) -> str:
    return _run("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=repo)


def git_head(repo) -> tuple[str, int]:
    sha, commit_time = _run("git", "log", "-1", "--format=%H %ct", cwd=repo).split()
    return sha, int(commit_time)


def check_git_clean(repo) -> None:
    status = git_status(repo)
    if status.strip():
        raise PreflightError(f"working tree is dirty (git status --porcelain):\n{status}")


# ---- interpreter、版本與 binding ----


def check_venv(prefix: str, repo) -> None:
    expected = (Path(repo) / ".venv").resolve()
    if Path(prefix).resolve() != expected:
        raise PreflightError(f"interpreter prefix {prefix} is not the repo venv {expected}")


def parse_pins(requirements_text: str) -> dict[str, str]:
    pins = {}
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def check_versions(pins: dict[str, str], installed) -> None:
    """`installed(name)` 回傳已安裝的版本（例如 `importlib.metadata.version`）。"""
    mismatched = {name: (installed(name), version) for name, version in pins.items() if installed(name) != version}
    if mismatched:
        detail = ", ".join(f"{name} {got} != pinned {want}" for name, (got, want) in sorted(mismatched.items()))
        raise PreflightError(f"installed versions differ from requirements-dev.txt: {detail}")


def _codegen_rustflags(rustflags: str) -> list[str]:
    """除了 maturin 為 macOS 擴充模組加上的 `-C link-arg=…` 之外的所有旗標。"""
    tokens = rustflags.split()
    others = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-C" and i + 1 < len(tokens) and tokens[i + 1].startswith("link-arg="):
            i += 2
            continue
        others.append(tokens[i])
        i += 1
    return others


def check_build_info(info: dict) -> None:
    """被量測的 binary 必須是 release 預設設定（008 §11）：opt-level 3、沒有 debug assertions、沒有 codegen 旗標。"""
    if info.get("opt_level") != "3":
        raise PreflightError(f"rust_hnsw opt_level is {info.get('opt_level')!r}, expected '3'")
    if info.get("debug_assertions") is not False:
        raise PreflightError("rust_hnsw was built with debug_assertions")
    extra = _codegen_rustflags(info.get("rustflags", ""))
    if extra:
        raise PreflightError(f"rust_hnsw was built with extra rustflags {extra}")


def check_binding_fresh(so_path, head_commit_time: int) -> None:
    mtime = Path(so_path).stat().st_mtime
    if mtime < head_commit_time:
        raise PreflightError(
            f"{so_path} is older than HEAD: run ./scripts/test.sh (maturin develop --release) on the clean HEAD first"
        )


def check_dataset_registered(dataset) -> dict[str, str]:
    """formal 只接受已登記的真實資料集，檔案 SHA-256 必須相符。回傳各檔案的 SHA-256。"""
    if datasets.is_synthetic(dataset):
        raise PreflightError(f"dataset {dataset.name!r} is synthetic: formal runs need a registered real dataset")
    digests = {}
    for name, path in datasets.files(dataset).items():
        digest = sha256_file(path)
        if dataset.sha256.get(name) != digest:
            raise PreflightError(f"{name}: SHA-256 {digest} does not match the registered {dataset.sha256.get(name)}")
        digests[name] = digest
    return digests


# ---- 記憶體與 hygiene ----


def parse_ps_rss(text: str) -> int:
    """`ps -o rss=` 以 1024 bytes 為單位。"""
    return int(text.strip()) * 1024


def ps_rss_bytes(pid: int | None = None) -> int:
    return parse_ps_rss(_run("ps", "-o", "rss=", "-p", str(os.getpid() if pid is None else pid)))


def ru_maxrss_bytes() -> int:
    """process 生命期間的 RSS 高水位。macOS 的單位是 bytes（Linux 是 KiB）。"""
    assert sys.platform == "darwin", "ru_maxrss is in bytes only on macOS"
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


_LIMIT = re.compile(r"CPU_(?:Speed|Scheduler)_Limit\s*=\s*(\d+)")


def therm_warnings(text: str) -> list[str]:
    warnings = []
    for line in text.splitlines():
        stripped = line.strip()
        if "warning level" in stripped and not stripped.startswith("Note: No "):
            warnings.append(stripped)
        match = _LIMIT.search(stripped)
        if match and int(match.group(1)) < 100:
            warnings.append(stripped)
    return warnings


def parse_swap_used(text: str) -> int:
    match = re.search(r"used = ([0-9.]+)([KMG])", text)
    if match is None:
        raise ValueError(f"cannot parse vm.swapusage: {text!r}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def parse_compressor_pages(text: str) -> int:
    match = re.search(r"Pages occupied by compressor:\s+(\d+)\.", text)
    if match is None:
        raise ValueError("cannot find 'Pages occupied by compressor' in vm_stat output")
    return int(match.group(1))


def snapshot() -> dict:
    """子 process 前後的 hygiene 紀錄（008 §9）。"""
    therm = _run("pmset", "-g", "therm")
    return {
        "time": time.time(),
        "loadavg": list(os.getloadavg()),
        "therm": therm,
        "therm_warnings": therm_warnings(therm),
        "swap_used_bytes": parse_swap_used(_run("sysctl", "vm.swapusage")),
        "compressor_pages": parse_compressor_pages(_run("vm_stat")),
    }


def top_processes(count: int = 15) -> str:
    return "\n".join(_run("ps", "-Ao", "pid,pcpu,rss,comm", "-r").splitlines()[: count + 1])


def wait_for_load(read_load, threshold: float, wait_s: float, poll_s: float, sleep, monotonic):
    """load 超過門檻就等待，最多 `wait_s` 秒（008 §2）。回傳 (是否通過, [(經過秒數, load)])。"""
    start = monotonic()
    samples = []
    while True:
        elapsed = monotonic() - start
        load = read_load()
        samples.append((elapsed, load))
        if load <= threshold:
            return True, samples
        if elapsed >= wait_s:
            return False, samples
        sleep(poll_s)


def _sysctl(name: str) -> str:
    return _run("sysctl", "-n", name).strip()


def machine_metadata() -> dict:
    return {
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "performance_cores": int(_sysctl("hw.perflevel0.physicalcpu")),
        "efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu")),
        "logical_cpus": int(_sysctl("hw.logicalcpu")),
        "memory_bytes": int(_sysctl("hw.memsize")),
        "os": platform.platform(),
        "os_version": _run("sw_vers").strip(),
        "pmset": _run("pmset", "-g"),
        "pmset_batt": _run("pmset", "-g", "batt"),
    }
