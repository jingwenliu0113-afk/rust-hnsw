# MACHINE_POLICY

## 核心原則

**本 Project 以 Mac 為主要開發與 Core benchmark 環境。**

Core 不需要 RTX 5070 Ti。

## Mac 上負責

- Rust implementation
- correctness
- Python binding
- profiling
- 正式 CPU benchmark

## Mac Heavy Lock

正式 latency / throughput benchmark 必須透過：

```bash
./tools/with-mac-lock <正式 benchmark 命令>
```

例如：

```bash
./tools/with-mac-lock python benchmarks/run.py
```

`with-mac-lock` 會：

1. 原子性取得 `/tmp/ai-portfolio-mac-heavy.lock`。
2. 記錄 lock owner 資訊。
3. 執行 benchmark。
4. 結束或中斷時盡可能自動釋放 lock。

若拿不到 lock：

- 不跑正式 performance benchmark。
- 不得刪除 lock 來搶資源。
- 可以繼續寫 Rust、tests、correctness、Python binding、profiling setup。
- 若懷疑 stale lock，回報使用者處理。

一般 correctness test 與開發工作不需要 lock。

## Benchmark Hygiene

正式 benchmark 時避免：

- OTel incident collection
- 大型 Docker workload
- 其他 CPU-heavy 工作

記錄 machine / CPU / RAM / OS / power state / background load / compiler / git commit / config。

## Docker

目前 Core 階段，本 Project **不使用** Mac 上的 Docker Desktop。

禁止：

- 執行 Docker Desktop restart。
- 執行任何全域 prune（例如 `docker system prune`、`docker volume prune`、`docker network prune`、`docker image prune`）。
- 操作其他 Project 的 container、volume 或 network（啟動、停止、刪除、修改都不行）。

若未來真的出現 Docker 的必要性：先停下來向使用者提出理由，經同意後才能導入；不得自行導入。

## 遠端桌機

只有使用者明確要求 x86 vs ARM secondary experiment 時才使用。

遠端關機完全不影響 Core。
