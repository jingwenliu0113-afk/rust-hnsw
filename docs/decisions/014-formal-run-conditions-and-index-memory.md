# 014：正式 run 的電力條件、背景 workload 與 index memory 指標（使用者決定）

日期：2026-10-09
狀態：採用（使用者決定，在任何正式結果之前；協定版本改為 `008-v2`）

## 背景

第一次 formal run 之前，NEXT.md 列了兩個需要使用者決定的問題：

- **B1：index memory 的定義**。008 §8 規定 `index_memory_bytes` = build 後的 rss − baseline rss，recall–memory frontier 一律用它。
  但 macOS 上已釋放的大型配置會以 reusable page 留在 rss，直到系統有記憶體壓力才回收（`scripts/probe_reusable_memory.py` 可重現：
  釋放 451 MiB 後 Δrss ≈ +451–470 MiB、Δphys_footprint ≈ 0–14 MiB）。Rust 的 `HnswIndex.add` 會釋放一份與資料同大小的暫時副本，
  所以完整 GloVe 的 smoke 探測中 Rust 的 rss 增量比 footprint 增量多 448 MiB。
- **B2：正式 run 的電力條件**。008 只要求接 AC；電力模式、變壓器與電池狀態也會改變 CPU 效能。

使用者在 2026-10-09 早上決定（原文要點）：

- Formal Mac benchmark 固定使用**自動電力模式 + 原廠 70W AC**，並取得 Mac heavy lock；**背景不得有其他 heavy workload**。
  若電池在 run 中**持續明顯下降**，該 run **作廢並停止**。
- B1：index memory 以 **`phys_footprint` 作主要指標**，**RSS 同時保留作 supplementary metric**，三個 implementation 使用完全相同測法。

## 決策

### 1. 電力條件（formal run 與閒置 load 量測）

- 條件：`pmset -g batt` 為 AC Power；`pmset -g` 的 `powermode` = 0（自動）；`pmset -g ac` 的製造商為 `Apple Inc.`、名稱為 `70W USB-C Power Adapter`
  （系統回報 68 W）。常數在 `benchmarks/protocol.py`。
- formal 前置檢查不符就拒絕；每個子 process 啟動前與 run 結束時的重新確認（008 §1）不符時，run 記為 `invalid` 並停止，已完成的子 process 照常保留在 run.json。
- 閒置 load 量測（008 §2）也要求同樣的條件，否則拒絕，讓門檻的確認與 formal run 在相同的條件下。

### 2. 電池持續明顯下降

- formal 前置檢查記下 run 開始時的電量。每次重新確認時，電量比開始時低 **3 個百分點以上**，就視為「持續明顯下降」：run 記為 `invalid` 並停止。
- 3 個百分點的理由：電量以整數百分比回報、變化緩慢，1–2 點可能只是讀數抖動；接 70W 變壓器時正常不會下降，連續下降 3 點代表供電持續不足。
- 沒有電池的機器不檢查這一項。

### 3. 背景不得有其他 heavy workload

- 定義：自己與祖先 process 以外，任一 process 的 %CPU（`ps` 的瞬間值，100 = 一個 core）≥ **50**。這與 `benchmarks.idle_load` 判定重工作的單一 process 門檻相同。
- 每個子 process 啟動前的 gate（008 §2）改為「1 分鐘 load ≤ 4.0 **而且**沒有背景 heavy workload」，兩者都成立才啟動；
  每 15 秒重查、最多 10 分鐘，仍不成立就中止整個 run（`aborted`），不放寬。
- 子 process 結束後再檢查一次；有背景 heavy workload 時，該子 process 標記新的 hygiene 標記 **`background`**，
  依 010 §3 不進入主要讀值（被標記的點另外列出）。smoke 只記錄、不等待。
- 這是 gate 與標記，不取代閒置 load 量測。桌面程式（WindowServer、Claude 等）各自低於 50%，不算背景 heavy workload。

### 4. index memory 的主要指標改為 phys_footprint

- **主要指標**：`index_footprint_bytes` = build 後的 phys_footprint − baseline phys_footprint（`proc_pid_rusage`）。
  它包含被壓縮的 page、不包含已釋放但仍算在 rss 的 reusable page。recall–memory frontier 與 scaling 的 index memory 一律用它。
- **輔助指標**：`index_memory_bytes`（rss 的增量）照常記錄，報告另列一欄。
- 兩者都由同一個子 process 在 build 前後、以同一段程式量測，三個實作的測法完全相同（008 §6、§8 的時點不變）。
- run.json 的欄位名稱不變（`index_memory_bytes` 仍是 rss 的增量），以免與 008-v1 的紀錄混淆；主要與輔助的對應寫在 `protocol.py` 的
  `INDEX_MEMORY_PRIMARY` / `INDEX_MEMORY_SUPPLEMENTARY` 與報告的欄位名稱中。
- `memory` 標記（footprint 增量比 rss 增量多出 2% + 1 MiB 以上，代表 build 期間有 page 被壓縮）保留為 hygiene 標記：
  主要指標雖然不再被壓縮低估，但記憶體壓力本身可能影響計時。
- SPEC 的 peak RSS（`ru_maxrss`）不變；它是另一個指標，照實包含 Rust build 期間的暫時副本。

### 5. 協定版本

- `protocol.PROTOCOL_VERSION` 改為 `008-v2`，`protocol_record()` 加入上述常數。之前的 smoke run 都是 `008-v1`，報告不會與 formal run 合併（版本不同就拒絕）。

## 為什麼

- 電力條件、電池與背景 workload 都會改變 CPU 的實際速度；事先固定並在每個子 process 前檢查，正式數字才可以互相比較、可以重現。
- phys_footprint 量的是 process 實際持有的記憶體；rss 會因 allocator 是否已把釋放的 page 還給系統而變動，
  而且在 formal run 要求的閒置機器上最容易高估（沒有記憶體壓力去回收 reusable page）。
- 兩個指標都保留，讀者可以看到差異的大小；只有主要指標用於結論。

## 放棄的替代方案

- **高效能電力模式**：使用者選擇自動模式。
- **量測前呼叫 `malloc_zone_pressure_relief`、繼續用 rss**（B1 的選項 2）：會改變被量測時 allocator 的狀態。
- **維持 rss 並在報告中註明**（B1 的選項 3）：大 n 時 Rust 會被多算約一份資料，讀值有系統性偏差。
- **背景 workload 以「所有其他 process 的 %CPU 合計」判定**：本機桌面程式合計約 140%，formal run 時也存在，會讓 gate 永遠不通過。

## 對實驗的影響

- 第一次 formal run 之前仍需：閒置 load 量測（008 §2 與補記）在上述電力條件下得到 `ok`。
- 主要記憶體讀值與 008 §8 原文不同：報告與 README 的 recall–memory frontier、scaling 以 phys_footprint 為準，rss 為輔助。
- 新的 hygiene 標記 `background` 加入 010 §3 的清單（離開主要讀值、另外列出）。
- 電池或電力條件不符而作廢的 run 是 `invalid`，依 013 §5 整份 config 重跑，原本的 run 以 `--excluded` 列出。

## 補記（2026-10-09，第一次 formal run 之前）：使用者允許高效能模式；實作審查的修正

以三個角度（誤判作廢、錯誤放行與忠實度、測試鑑別力）審查本文件的實作，每個 finding 由驗證者重現或反駁。修正如下：

**電力模式（使用者 10:00 前後指示「允許可以在高效能使用」）**

- 允許自動（powermode 0）或高效能（2）；同一個 run 內模式不得改變（重新確認與子 process 期間的取樣都檢查），
  報告不合併不同電力模式的 formal run，所以每份正式報告都在同一個模式下。低耗電模式（1）或讀不到 powermode 一律拒絕。
- 審查時本機接 AC 的設定檔是高效能（`pmset -g custom` 的 AC Power 為 2），因此第一批 formal run 預期都在高效能模式。

**電池**

- 基準改為 run 期間看到的**最高電量**（前置檢查、每次重新確認、子 process 期間的每次取樣都更新）；原本只比 run 開始時的電量，
  「先充到 100 再降到 96」會被放行。目前電量比最高值低 3 點以上就作廢，與是否在充電無關。
- 子 process 期間每 15 秒取樣一次電力狀態；任一取樣不符時，該子 process 照常記錄（含 `power_violation`），run 記為 `invalid` 並停止。

**背景 heavy workload（§3 的判定改寫）**

- 實測：macOS 的 `ps` %CPU 是約 0.3–0.5 秒尺度的瞬間值；桌面與系統 process 會短暫超過 50%
  （Claude Helper (Renderer) 最高 107%、Codex (Renderer) 157%、fseventsd 84–93%、syspolicyd 58–87%；WindowServer 最高 49.4%）。
  §3 原本「子 process 結束後取一次瞬間值」約有 2–5% 的子 process 會被誤標（12 份 config 共 114 個子 process，幾乎必然誤標），
  而且子 process 期間出現又停止的背景工作完全看不到。§3 原文「桌面程式各自低於 50%」不成立，在此更正。
- 改為：
  - **gate**（子 process 啟動前）：維持 ps 瞬間值 ≥ 50% 就等待；它只會延後啟動 15 秒，不會造成誤標。
  - **子 process 期間**：另一個 thread 每 15 秒讀一次所有 process 的**累積 CPU 時間**（`ps -o time`），子 process 啟動前與結束後各再讀一次；
    相鄰兩次的差除以經過時間得到每個區間的平均 %CPU。排除 runner 自己、祖先與受測子 process 的整個 process 樹。
    任一成立就標記 `background`：同一 pid 整段平均 ≥ 50、同一 command 合計整段平均 ≥ 100、同一 pid 連續 4 個區間平均 ≥ 50。
    整段平均的分母至少是 4 個區間（60 秒），很短的子 process 不會因為一次尖峰被標記。
  - 每個區間的非排除 %CPU 合計與前 5 名寫進 run.json，供事後審查。在一個 15 秒區間內開始又結束的 process 看不到，由既有的 `load` 標記涵蓋。
  - 一次 `ps` 約 20–30 ms CPU，在另一個 core 上執行。

**memory 標記**

- 子 process 在 baseline、after_build、end 另外以 `task_info(TASK_VM_INFO)` 記錄 `reusable`、`compressed`、`compressed_lifetime`
  （三個實作同一段程式；欄位位置以 phys_footprint 與 `proc_pid_rusage` 相同來驗證）。實測釋放 64 MiB 後 `reusable` 正好增加 64 MiB，直接印證 B1 的成因。
- `memory` 標記改為：build 期間這個 process 的累計被壓縮量 > 2% × footprint 增量 + 1 MiB。原本「footprint 增量比 rss 增量多」的比較，
  在 reusable page 讓 rss 偏高時會被抵消而漏標；舊紀錄（沒有 task_vm_info）仍用原本的比較。

**其他**

- 閒置 load 量測結束時再檢查一次電力條件（模式不變、電量沒有比開始時低 3 點以上），不符就記為 `invalid_power`，不得寫成 `ok`。
- 作為 Gate 證據的 profiling（`--dataset` 且祖先持有 heavy lock，009）適用相同的電力條件與背景 gate；只驗證工具的取樣不檢查。
- formal run 進行中，repo 內不得出現任何新檔案（包含 untracked 與 `.claude/`）：每次重新確認都要求 `git status` 為空。
  `.gitignore` 加入 Claude Code 會自動產生的 `/.claude/settings.local.json` 與 `/.claude/worktrees/`。

## 補記（2026-10-11）：電量下降規則已取消

- §2「電池持續明顯下降」與補記的「以最高電量為基準、低 3 點作廢」由使用者取消（decision 017，協定 008-v3）。
  接著 AC 時 macOS 充滿後讓電量回落幾點會觸發這條規則，但不影響 CPU 速度；電量只記錄，不再是有效性條件。
