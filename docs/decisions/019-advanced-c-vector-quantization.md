# 019：Advanced C：vector quantization 的假設、表示與評測協定（事先登記）

日期：2026-10-16
狀態：採用（使用者決定；在任何真實資料上的量化結果之前）。撰寫、審查與修訂時都還沒有任何真實資料上的量化結果。本文件 commit 之前，不得在任何真實資料上執行量化表示，包括 smoke、天花板計算與 profiling。

## 背景

- 2026-10-16（台灣時間），使用者依 SPEC 的特例決定進入 Advanced C。
  - 依 015 事先登記的讀法，兩個資料集的主要目標都不是 `gap`：GloVe 的 0.95 是 `within_quantization`、0.90 是 `uncertain`；文字三個目標都是 `within_quantization`。
  - 因此 SPEC Gate 的條件 3 不成立。本文件**不宣稱** Gate 通過；Core 的正式結果、Gate 判讀與事先登記的比較規則一律不改。
  - **特例的適用是使用者的判斷，不是 015 的讀值**：015 明定 `within_quantization`、`uncertain` 都不代表「很接近 hnswlib」。
    照實並列看到結果之後的觀察（不是判讀）：GloVe 在同 ef、同 recall 時 hnswlib 快 1.22–1.38 倍；文字只快 6–8%。
  - **使用者的授權與指示**（2026-10-16，要點照實記錄）：
    - 依 SPEC 的特例進入 Advanced C，**不宣稱原本的 Advanced Gate 全部通過**；
    - 如實記錄 Core 的結果與本次授權；
    - 先建立明確的假設、Baseline、評測協定與停止條件，再做實驗；
    - 最多三個假設，不自動擴張到 A、B 或 Optional；
    - 保留 Core 所有正式結果，不修改既有的 Gate 或事先登記的比較規則；
    - Engineering 可以並行改善測試、API 易用性、文件與 Demo，但不能污染研究結果（§10）；
    - 只有重大研究決策或真正的 Blocker 才詢問使用者；
    - 開跑前也要檢查 CPU、GPU、RAM 的負載（§5 的執行條件）。
  - 使用者對本設計的決定見 §13。
- SPEC C 的要求：
  - FP32 對壓縮表示，研究 memory、recall、latency 三者的 frontier，並搭配 profiler / flamegraph 解釋。
  - Advanced 最多三個假設驅動的實驗。
  - EXPERIMENT_RULES：optimization 必須先寫 hypothesis；做滿三個主要實驗即封版。
  - NEXT 要求先登記假設、baseline、評測協定與停止條件。
- **讀程式確認、會直接影響設計的事實**：
  1. 距離計算：
     - `distance::dot` 是 `iter().zip().map().sum::<f32>()`，依序累加；Rust 不重排浮點加法，也不做 FMA 縮併。
     - 編譯用 release 預設，沒有 `target-cpu=native`（008 §11）。
     - 距離被 inline 進 `search_layer`（009 的已知限制）。
  2. 搜尋結構：
     - `search_layer` 每次搜尋新建 `HashSet`（SipHash）與 `BinaryHeap`；graph 是每個節點各自的 `Vec<Vec<NodeId>>`。
     - `Node`、`Candidate`、`push_padded_row` 是 `hnsw` 模組私有的；`hnsw/persist.rs` 已示範子模組可以直接使用它們（`use super::{HnswIndex, Node}`）。
  3. `HnswIndex::add` 會經 `unit_rows` 先建一份完整的正規化副本，`add` 結束時才釋放。
  4. `interface.check_search_result` 的 `DISTANCE_RANGE_TOL = 1e-5`：
     - 實際條件是有效距離超出 [−1e-5, 2 + 1e-5] 時拋出 `ValueError`。
     - `run_child` 只攔 `IncompleteResultsError`，所以 `ValueError` 會讓整個子 process 失敗，該 ef 與之後的 ef 點全部遺失。
  5. `run_child.py`：
     - 結尾無條件呼叫 `adapter.save_native`。
     - null call 以 n = 1 建 index，library warm-up 用 16 列，所以 SQ8 會遇到 vmin = vmax 的退化維度。
     - 每個 ef 點的 points 檔（`ef<ef>.npz`）只存 `ids_k10`、`ids_k1` 與計時陣列，**不存 distances**。
  6. `report.py`：
     - `FORMAL_CONFIGS` 以資料集為 key；`check_formal_set` 只掃描登記清單中的 config，所以 Core 報告重產時不會看到其他 config 的 run。
     - `_check_compatible` 檢查協定版本、mode、資料集；`_check_mergeable` 檢查 run 狀態、重複的 build、`MEASUREMENT_PATHS`、BUILD_INFO、套件版本、電力模式與變壓器，不符就拒絕合併。
  7. faiss-cpu 1.15.1：
     - 有 `IndexHNSWSQ`、`ScalarQuantizer.QT_8bit` 與 `RS_minmax`。
     - `IndexHNSW.hnsw` 有 SWIG setter，指定時做 C++ 結構複製。
     - `HNSWStats` 的 SWIG docstring 錯位：`ndis` 的說明其實屬於另一個欄位。
  8. 本機 `sysctl` 與資料大小：
     - P-core 叢集的 L2 是 16 MiB（`hw.perflevel0.l2cachesize`），cache line 128 B。
     - GloVe 100k 的 FP32 向量 38.1 MiB > L2，SQ8 codes 9.5 MiB < L2。
     - GloVe 的 graph 由序列化大小推算約 10 MiB（48.4 MiB − 向量 38.1 MiB；這是推算，記憶體中的 `Vec<Vec<NodeId>>` 另有每個 Vec 的 header 與多餘的 capacity，可能更大）。
       所以 SQ8 時的工作集（codes、graph、visited set）仍可能 > 16 MiB。
     - 文字的 FP32 向量 293 MiB 與 SQ8 codes 73 MiB 都遠大於 L2。
     - 每列向量（row-major、連續存放）：GloVe 的 FP32 400 B 跨 4 條 cache line，SQ8 100 B 跨 1–2 條（依該列在 128 B 邊界上的位置，平均約 1.75 條）；
       文字的 FP32 3,072 B 約 24 條，SQ8 768 B 約 6 條。
- **用到的 Core 正式結果**（都已公開在 README / NEXT；對 Core 而言是看過結果之後，對 C 是事前資訊）只用於以下用途：
  - **判讀用的固定 ef e\***：取 Core 正式報告中 (100k, 16, 200) 的 rust-hnsw，跨 seed median Recall@10 第一次 ≥ 0.95 的 ef。
    - GloVe：304（median 0.9567；前一個 ef 256 約 0.947）。
    - 文字：48（約 0.954；前一個 ef 32 約 0.924）。
    - C 的 FP32 對照臂與 Core 逐位元相同（§4），所以現在就能登記。
  - **ef 梯子、timeout 與成本**：(16, 200) seeds 1–3 的子 process 時間。
    - GloVe：rust-hnsw 261–307 s（build 26.5–32.8 s）、faiss-hnsw 154–178 s；最大 ef 384 的點 rust 55–64 s、faiss 35–41 s。
    - 文字：rust-hnsw 395–445 s（build 154.6–178.4 s）、faiss-hnsw 154–167 s。
  - **記憶體量法的設計**（(16, 200) seeds 1–3）：
    - GloVe rust-hnsw：Δfootprint 103.4–105.3 MiB、Δrss 104.8–106.9 MiB、after_build 的 Δreusable 只有 1.5–1.6 MiB、serialized 48.4 MiB。
    - 文字 rust-hnsw：Δfootprint 318.5–318.8 MiB、Δreusable 294.4–294.6 MiB。
    - 也就是說，100 維時 `add` 的暫時副本留在 phys_footprint 內，768 維時則變成 reusable。NEXT 把這點記為未驗證的推測。
  - **H1 ±10% 寬度的理由**：Gate profiling 的 rust-hnsw search 占比。
    - GloVe：`search_layer`（含被 inline 的距離）47.9%、visited set 43.6%。
    - 文字：`search_layer` 84.8%。
  - 撰寫者沒有看過任何量化表示的結果；這類結果目前也不存在。
- **設計方式**：三個獨立設計（成本最小、統計與協定嚴謹、解釋機制），加上三位評審（合規、成本、研究價值）。評分、取捨與 must_fix 的落實見文末「設計過程」。
  草稿另經一位獨立審查者核對程式事實，findings 的落實見「設計過程」的「草稿的審查」。

## 決策

### 1. 研究問題與範圍

- **研究問題**（SPEC C 的字面範圍）：在 Core 的 HNSW 上，把 FP32 向量換成 per-dim 8-bit scalar quantization（SQ8）時：
  - index memory、Recall@10 與 latency / QPS 的 frontier 怎麼移動；
  - profiler 能否解釋這些移動的原因。
- **規模**：只做**一種壓縮表示**（SQ8，§3），搭配兩個假設（§2）。C 使用 Advanced 三個名額中的 2 個；使用者決定 C 做完就封版，第 3 個名額**不保留**（§9、§13）。
- **不做**：
  - A、B 與 Optional：SIMD distance kernel、手寫 intrinsics、`target-cpu=native`、為了自動向量化而設計的整數 kernel。
  - 其他壓縮與加速：rerank、PQ / OPQ / RaBitQ、fp16 / bf16、4 / 6-bit。
  - 格點與變異：M / efC 掃描、scaling、insertion order variance。
  - 其他：新的檔案格式、hnswlib。
- **結論的適用範圍**：
  - Rust 的延遲結論只適用於「依序純量累加」的距離 kernel（`run.DISTANCE_KERNELS`）。FAISS 用的是 SIMD kernel，只作相對效果的描述。
  - SQ8 在圖建完之後才轉換（§3），所以結論只涵蓋搜尋期間的表示；build time 與 peak RSS 不會下降。

### 2. 假設、預測與否證條件

**共同定義**（每個資料集固定 n = 100,000、(M, efC) = (16, 200)；以下 s 為 seed，e 為 ef）：

- **各臂代號**：

  | 代號 | 實作名稱 | 意義 |
  | --- | --- | --- |
  | R | `rust-hnsw` | Core |
  | F | `rust-hnsw-fp32g` | FP32，走新的搜尋路徑 |
  | S | `rust-hnsw-sq8` | SQ8 |
  | P | `faiss-hnsw` | FAISS FP32 |
  | Q | `faiss-hnsw-sq8` | FAISS SQ8 |

  詳細內容見 §3。
- **QPS_X(s, e)**：C formal run 的 QPS（每個點 5 次重跑的中位數，008 §6）。
- **D_X(s, e)**：每個 query 平均呼叫距離 kernel 的次數（所有層，含 entry point）。由不計時的診斷工具取得（§7）；只有當該工具的 k = 10 ids 與 formal run 的 `ids_k10` 逐位元相同時才有效。
- **ρ_QPS** = QPS_S / QPS_F。
- **ρ_eval** = ρ_QPS × D_S / D_F，即兩臂每秒完成的距離評估數之比。ρ_eval > 1 表示 SQ8 的每次評估，連同它分攤的走訪工作，比較便宜。
- **e\***：GloVe 304、文字 48（見背景）。
- **乾淨的配對**：F 與 S 的子 process、build 與該點都沒有 hygiene 標記（010 §3）。
- **範圍規則**（全文共用）：
  - 判定量在所有有效 seed 上的 [min, max] 整段落在成立區 → **成立**。
  - 整段落在否證區 → **不成立**。
  - 跨過邊界 → **不確定**。
  - 有效 seed 不足 → **無法判定**。

#### H1（latency 機制：減少 bytes 只在相依鏈短時有用）

**陳述**：rust-hnsw 距離 kernel 的加總，是一條長度為 d 的依序 f32 相依鏈。SQ8（非對稱距離、同樣依序以 f32 加總，§3）只把每個分量從 4 bytes 減為 1 byte，並不縮短這條相依鏈。因此：

- **768 維（文字）**：
  - 相依鏈長，向量讀取可以與運算重疊。
  - FP32 與 SQ8 的向量集合都遠大於 L2。
  - 每次距離評估的成本由相依鏈決定，SQ8 **不改變**它。
- **100 維（GloVe）**：
  - 相依鏈短，每次評估要付的向量讀取延遲占比較大。
  - 每次評估要讀的 cache line 從 4 條（400 B）減為 1–2 條（100 B；平均約 1.75 條，背景事實 8）。
  - SQ8 **縮短**每次評估的成本，而且縮短的部分在 kernel，不在 visited set 或 heap。
  - 不以「SQ8 放得進 L2」作為主要機制：SQ8 codes（9.5 MiB）雖然小於 P-core L2（16 MiB），但工作集加上 graph（推算約 10 MiB）與 visited set 仍可能 > 16 MiB。

**前提**（formal 之前的靜態檢查，§8）：

- 以 `otool -tV` 反組譯 binding 中的 `rhnsw_kernel_fp32` 與 `rhnsw_kernel_sq8`。兩者的**加總**都必須是依序的純量相依鏈。
- 機械判準：每次累加都是 `fadd sN, sN, sM`，而且 sN 是同一個累加暫存器；不得出現 `faddp`、向量 `fadd`（例如 `fadd.4s`）或 `fadda` 作為累加。
- 乘法與 u8→f32 轉換可以是向量指令（例如 `fmul.4s`、`ucvtf`），與 007 §8 在 hnswlib 看到的情況相同。
- 不符時 H1 記為「前提不成立」，計時讀值只作描述；不為了符合前提而改寫 kernel。

**預測與判定**（在 e\* 判定；H1a–H1c 的乾淨配對少於 3 個時，該子項記為「無法判定」）：

| 子項 | 成立（H1d：一致） | 不成立（H1d：不一致） | 不成立時的解讀（事先登記） |
| --- | --- | --- | --- |
| H1a 文字 | ρ_eval 的 [min, max] 整段在 [0.90, 1.10] 內 | min > 1.10，或 max < 0.90 | > 1.10：768 維的 kernel 有相當部分受向量讀取限制，減少 bytes 有用。< 0.90：u8 載入與轉換的額外指令成了瓶頸，kernel 的成本不是由相依鏈決定 |
| H1b GloVe | min ≥ 1.05 | max < 1.05 | 100 維時每次評估也由相依鏈決定（或向量本來就在 cache 中），或者 graph 與 visited set 的 cache miss 主導；Core 中兩個維度的差異與向量的 bytes 無關。解讀不只這幾種，報告照實寫明無法區分的部分 |
| H1c 維度對比 | GloVe 的 min > 文字的 max | GloVe 的 max < 文字的 min | 維度依賴的方向與預測相反 |
| H1d profiler（seed 1，§8） | **一致**：κ_文字 在 [0.90, 1.10] 內，而且 κ_GloVe ≤ 0.95 | **不一致**：其他情形 | 計時上的差異不在 kernel，例如只來自走訪的 cache spillover |

**profiler 量的定義**：

- κ = t_k(S) / t_k(F)，其中 t_k(X) = f_k(X) / (QPS_X(1, e\*) · D_X(1, e\*))，f_k 是 kernel 類別的 self-time 占比（§8）。
- H1d 只有一次取樣，因此只判「一致 / 不一致」，不判「成立 / 不成立」。
- 任一資料集的 seed 1 在 e\* 的 F 或 S 點有 hygiene 標記、該 seed 的 D 無效（§7 的 ids 核對不通過），或該次取樣沒有通過 §8 的 check 時，H1d 記為**無法判定**。

**診斷 ν**（不作判定）：

- 定義：ν = [(1 − f_k(S)) / (QPS_S · D_S)] ÷ [(1 − f_k(F)) / (QPS_F · D_F)]，即非 kernel 工作在每次評估上所花時間的比值。
- 事先登記的解讀：
  - ν < 0.90：cache spillover，向量變小之後，graph 與 visited set 也跟著變快。
  - ν > 1.10：原本藏在長相依鏈後面的非 kernel 工作露出來了。
- 兩種情形都照實報告，不改變判定。

**寬度的理由**（事先寫明）：

- 若 kernel 的時間由頻寬決定（與 bytes 成正比），依 Gate profiling 的占比，bytes 縮減 4 倍會讓：
  - 文字的 ρ_eval ≈ 1 / (0.15 + 0.85/4) ≈ 2.8；
  - GloVe 的 ρ_eval ≈ 1 / (0.52 + 0.48/4) ≈ 1.6。若以 cache line 而不是 bytes 計（GloVe 每列從 4 條減為平均約 1.75 條，背景事實 8），約 1 / (0.52 + 0.48/2.3) ≈ 1.4；文字的約 24 → 6 條仍約為 4 倍。
- ±10% 遠小於這兩個值，足以區分「相依鏈模型」與「頻寬模型」。
- GloVe 的 1.05 是「有實質增益」的最低門檻，不是中央預期。中央預期約 1.1–1.3，不用於判定。

**H1 整體判定**（各列依序取第一個符合的列）：

| 順序 | 條件 | 判定 |
| --- | --- | --- |
| 1 | 前提不成立 | **前提不成立**（計時讀值只作描述） |
| 2 | 前提成立，且 H1a、H1b、H1c 任一不成立 | **不成立** |
| 3 | H1a–H1c 都成立、H1d 一致 | **成立** |
| 4 | H1a–H1c 都成立、H1d 不一致 | **「計時成立、profile 不一致」**（不算成立） |
| 5 | H1a–H1c 任一無法判定，或 H1a–H1c 都成立但 H1d 無法判定 | **無法判定** |
| 6 | 其他（沒有子項不成立、至少一個子項不確定） | **不確定** |

- 整體判定附上各子項的狀態與數字。
- C7a 不一致時不判定（§6）。

**其他讀值**：

- 其他 ef 的 ρ_QPS 與 ρ_eval 全部列出，不判定；方向與 e\* 不一致時照實寫明。
- FAISS 的 Q ÷ P 的 ρ_QPS 只作描述：FAISS 的 SQ kernel 是 NEON SIMD，不屬於 H1 的機制。

#### H2（recall 損失：來自排序，而不是走訪）

**陳述**：在同一張圖上，SQ8 的 Recall@10 損失主要來自最終排序：表示本身的距離誤差，把候選中真正的近鄰擠出了前 10 名。損失不是主要來自走訪，也就是 SQ8 距離改變了哪些點被找進 ef 個候選。

**定義**（由 §7 的不計時診斷取得；只做 Rust 的兩臂，所有 seed 都算，判定只取有效 seed；recall 不受 hygiene 標記影響，所以用全部 build）：

- **W_X(s, e)**：以 k = ef 搜尋得到的 ef 個候選，也就是 layer 0 的完整結果表。`search_unit` 使用 `max(ef, k)`，所以 W_X 的前 10 個就是 k = 10 的結果；診斷工具會逐位元核對。
- **T_X(s, e)**：把 W_X 以精確距離重排後，前 10 個的 Recall@10。精確距離以 f64 在登記的 unit vectors 上計算，tie-break 依 001 的 (distance, id)。
- **R_X(s, e)**：formal run 的 id-based Recall@10。
- **損失的分解**：
  - L_total = R_F − R_S（總損失）
  - L_trav = T_F − T_S（走訪造成的損失）
  - L_rank = T_S − R_S（排序造成的損失）
  - δ_F = T_F − R_F（FP32 自身重排的差）
  - 恆等式：L_total = L_trav + L_rank − δ_F。
- **門檻**：thr(s) = 0.25 × L_total(s, e\*) + 0.001。其中 0.001 用來吸收 recall 的計數粒度：GloVe 一個命中約 1e-5、文字約 5e-5。
- **有效 seed**：F 與 S 在該 seed、e\* 的 formal recall 都存在（該點不是 incomplete 或缺失），而且兩臂在該 seed 都通過 §7 的 ids 核對。以下的 median 與「所有 seed」都只取有效 seed。

**判定**（每個資料集，在 e\*；各列依序取第一個符合的列）：

| 順序 | 條件 | 判定 |
| --- | --- | --- |
| 1 | 有效 seed 少於 3 個；或一致性檢查 C2 不一致；或 median_s L_total(s, e\*) < −0.003（SQ8 系統性地比 FP32 好） | **無法判定**。有效 seed 不足時註明「有效 seed 不足」；其餘註明「比較有問題」，先調查，不判定 |
| 2 | median_s L_total(s, e\*) < 0.004（SQ8 幾乎無損，分解低於解析度） | **不適用**（事先預期文字可能是這一種） |
| 3 | 所有有效 seed 的 L_trav(s, e\*) ≤ thr(s) | **成立** |
| 4 | 所有有效 seed 的 L_trav(s, e\*) > thr(s) | **不成立**。事先登記的解讀：SQ8 的距離誤差帶偏了走訪，只重排 ef 個候選救不回損失 |
| 5 | 其他 | **不確定** |

**H2 整體判定**（**適用** = 判定不是「不適用」的資料集；各列依序取第一個符合的列）：

| 順序 | 條件 | 判定 |
| --- | --- | --- |
| 1 | 兩個資料集都不適用 | **不適用** |
| 2 | 任一適用的資料集不成立 | **不成立** |
| 3 | 每個適用的資料集都成立 | **成立** |
| 4 | 任一適用的資料集無法判定 | **無法判定** |
| 5 | 其他（任一適用的資料集不確定） | **不確定** |

- 任一適用的資料集為無法判定或不確定時，整體不得判成立（由列序保證）。
- C7a 不一致時不判定（§6）。

**其他讀值**：

- 其他 ef 的 L_total、L_trav、L_rank、δ_F 全部列出，不判定。
- FAISS 不做分解：k = ef 時 FAISS 的前 10 名是否等於 k = 10 的結果，目前沒有測試釘住（007 §4）。

#### 一致性檢查（不是假設，也不算任何假設成立的證據）

| 代號 | 內容 | 規則 |
| --- | --- | --- |
| C1 記憶體 | ΔM_relief = M_relief(F) − M_relief(S)，對照 Δ_theory = 3·n·d − 8·d（GloVe 29,999,200 B = 28.61 MiB；文字 230,393,856 B = 219.72 MiB）。Q 對 P 做同樣的檢查。M_relief 只是診斷（§6） | 每個 seed 的差的絕對值 ≤ max(5% × Δ_theory, 2 MiB) 為一致。ΔM_014 與 allocator 保留量（M_014 − M_relief）照實列出，不判定。M_struct 的差等於 Δ_theory，由單元測試保證 |
| C2 FP32 重排 | δ_F(s, e) | 所有有效 seed、所有 ef 的 δ_F 絕對值 ≤ 0.002 為一致 |
| C3 天花板 | R_S(s, e) 對照 c（以 S 的同一個距離公式做 exact search 得到的 Recall@10，§7） | 所有 seed、所有 ef 都滿足 R_S ≤ c + 0.002 |
| C4 路徑失真 | QPS_F / QPS_R（seed 1、e\*） | 在 [0.90, 1.10] 內為一致。否則標記「path-distorted」：H1 仍在新路徑內判定，但不得把結論寫成 Core 路徑上的效果 |
| C5 Gate 重現 | R 在 e\* 的取樣占比，對照 Gate profiling（traversal、visited set、heap 三類） | 各類相差 ≤ 3 個百分點為一致（只作描述） |
| C6 漂移 | C 中 R 的 QPS ÷ Core rust-hnsw seed 1 的 QPS（共同的 ef） | 只描述。GloVe 的 Core 是 70W / 008-v2，文字是 140W / 008-v4 |
| C7a 逐位元（Rust） | F（seeds 1–5）、R（seed 1）在每個共同 ef 的 `ids_k10`，對照 Core 正式 run 的 points 檔；另比對 C 的 seed 1 中 F 與 R（所有 ef） | 必須完全相同。**不一致就停止，H1、H2 都不判定**，先查明原因（§9） |
| C7b 逐位元（FAISS） | P（seeds 1–3）在每個共同 ef 的 `ids_k10`，對照 Core 正式 run 的 points 檔 | 必須完全相同。不一致時，Q 對 P 的配對讀值、Q ÷ P 的比值與 quantization frontier 判讀都記為不適用；P、Q 各自的 frontier 照列並標明。**H1、H2 不受影響** |
| C8 FAISS | 移植後 Q 的 graph 是否與移植來源的 Flat 相同；Rust 與 FAISS 的 codes 一致比例 | graph 必須相同，否則 FAISS 的配對讀值記為不適用（§3）。核對時點：每個 FAISS seed（1–3）在 build 完成後（移植完成、`del flat` 之前）各核對一次，由 §7 執行。codes 只記錄比例，不擋流程 |

- C7a、C7b 只保證 ids：Core 的 points 檔不存 distances（背景事實 5）；F 與 Core 的 distances 逐位元相同由 §4 的等價測試保證。
- Core 的 points 檔缺失時，該項對照記為**無法判定**（不等於不一致），流程不停止；C 內 F 與 R 的比對照做。

**C7 對照的 Core 正式 run**：

| 資料集 | seed 1 | seeds 2–5 | commit |
| --- | --- | --- | --- |
| GloVe | `20261009T080239Z-formal` | `20261009T091611Z-formal` | f077de9 |
| 文字 | `20261015T191333Z-formal` | `20261015T210125Z-formal` | 4acf799 |

- 從這兩個 commit 到目前的 HEAD，`crates/`、`python/`、`Cargo.*`、`benchmarks/adapters.py`、`benchmarks/interface.py` 沒有任何差異。
- 兩者用的 rustc 都是 1.99.0。

**附帶預期**（不作判定）：

- c_文字 > c_GloVe。
- 文字的 S 在 ef ≤ 512 內達到 0.99。
- GloVe 的 S 在 ef ≤ 768 內達到 0.95。
- GloVe 的 0.99：所有臂都「未達到」。

### 3. 表示

| 臂 | 名稱 | 內容 | seeds |
| --- | --- | --- | --- |
| R | `rust-hnsw` | Core 的 adapter 與程式，一行不改；kernel inline 在 `search_layer` 內 | 1 |
| F | `rust-hnsw-fp32g` | 以 Core 的 `HnswIndex::new` + `add` 建出同一張 graph，向量原樣移入新的搜尋路徑。kernel 是 `#[inline(never)]` 的 `rhnsw_kernel_fp32`，內部呼叫 `distance::cosine_distance_unit` | 1–5 |
| S | `rust-hnsw-sq8` | 在同一個 FP32 build 完成後轉成 SQ8（規則見下），並釋放 FP32 向量；走與 F 相同的新搜尋路徑。kernel 是 `#[inline(never)]` 的 `rhnsw_kernel_sq8` | 1–5 |
| P | `faiss-hnsw` | Core 的 `IndexHNSWFlat` adapter，一行不改 | 1–3 |
| Q | `faiss-hnsw-sq8` | 以與 P 相同的參數建 `IndexHNSWFlat`，再把 `hnsw` 結構移植到 `IndexHNSWSQ(d, QT_8bit, M, METRIC_INNER_PRODUCT)`（storage 以同一份資料 train 與 add），最後釋放 Flat | 1–3 |

**F 是所有機制比較的對照**：

- F 與 S 共用同一份走訪程式：逐行複製 Core 的 `search_unit` / `greedy_search` / `search_layer`，`HashSet`、`BinaryHeap` 與 `Candidate` 排序全部相同。兩者的差別只有向量表示和對應的 kernel。
- F 與 R 的 ids、distances 逐位元相同（distances 由測試保證，ids 另由測試與 C7a 保證）；兩者的時間差，也就是路徑失真，由 C4 量化。
- 因此不宣稱 S 與 R 之間「唯一的變因是表示」。

**SQ8 codec**（per-dim、min–max，刻意對齊 FAISS 的 `QT_8bit` + `RS_minmax`、`rangestat_arg = 0`）：

- **訓練**：在 S 實際儲存的 n 個 unit vectors（`add` 正規化之後）上逐維計算：
  - vmin_j = min，vmax_j = max，vdiff_j = vmax_j − vmin_j（f32）；
  - 不外擴範圍、不使用 query、不需要 seed。
- **編碼**（f32）：
  - vdiff_j = 0 時，code = 0；
  - 否則 t = (x − vmin_j) / vdiff_j，夾到 [0, 1]，再取 code = `(255.0 * t) as u8`（向零截斷）。
- **解碼**（只作參考）：x̂ = vmin_j + (code + 0.5) / 255 · vdiff_j。
- **儲存**：只存 codes（n·d 個 u8）與 codebook（vmin、vdiff 各 d 個 f32）。

**非對稱距離（S）**：

- query 維持 f32，正規化方式與 Core 相同（`unit_rows`）。
- 每個 query 先算：
  - w_j = q_j · (vdiff_j / 255)，以 f32 計算；
  - b = Σ_j q_j · (vmin_j + 0.5 · vdiff_j / 255)，以 f64 累加後轉 f32。
- 距離 = 1 − (b + Σ_j w_j · code_j)。Σ 的寫法與 `dot` 相同：`iter().zip().map().sum::<f32>()`，依序加總，不手寫 SIMD。
- 內部比較用原始值；**輸出時才把距離夾到 [0, 2]**。夾住是單調的，排序不變。
- F 不夾，以保持與 Core 逐位元相同。

**Q 的距離**：FAISS 回傳的是估計的 similarity。adapter 先轉成 1 − ip，再以 `np.clip(…, 0, 2, out=…)` 原地夾住，最後把 padding 設為 +inf。

**Q 的移植**：

1. `sq.hnsw = flat.hnsw`（SWIG setter，C++ 結構複製），並設定 `ntotal`。
2. 以 `downcast_index` 取得 storage，明確設定 `sq.rangestat = RS_minmax`、`rangestat_arg = 0`，然後 `train(data)`、`add(data)`。
3. 明確設定 `IndexHNSWSQ.is_trained = True`：只 train storage 時，外層 index 的 `is_trained` 仍是 False（審查時以 n = 20 的移植試驗確認）。
4. `del flat`。

可行性由 §4 的測試釘住。

**事先登記的備案**：

- 移植不可行時，Q 改用原生的 `IndexHNSWSQ`：先 train 再 add，graph 以量化距離建出。此時 Q 對 P 的配對讀值與 C1 的 FAISS 部分記為「不適用（圖不同）」，C8 的 graph 核對也記為不適用，Q 只進 frontier 讀值。
- **移植可不可行只由 §4 的 synthetic 測試判定**，必須在 019 commit 之後、smoke 之前定案（寫補記）。smoke 或 formal 中 C8 不一致時只能記為不適用，**不得改用備案**。
- 連原生版本也無法通過 null call（n = 1）或 warm-up（16 列）時，在 formal 之前另寫 decision 拿掉 FAISS 這一對（correctness 理由）。H1、H2 不受影響。

**選擇的理由**：

- **per-dim**：各維的範圍不同；這也是 FAISS 的預設，codec 才能對齊。
- **非對稱**：不引入 query 的量化誤差，累加結構與 FP32 相同，H1 才能只改變 bytes。
- **建完圖才轉換**：F 與 S 的圖逐位元相同，才能在同一個 (seed, ef) 上成對比較、才能分解 recall 損失，也不必碰 Core 的 insert。
- **不 rerank**：保留 FP32 向量會讓記憶體 ≥ FP32；H2 的結果本身就回答了 rerank 能救回多少。

不納入的其他表示見「放棄的替代方案」。

### 4. 實作範圍

原則：Core 的 FP32 路徑在原始碼層級不動；只新增模組、類別與登記項；不新增檔案格式，也不建立通用 framework。

**Rust（`crates/hnsw`）**

- **新檔與既有檔**：
  - 新增 `src/hnsw/quant.rs`，作為 `hnsw` 的子模組，與 `persist.rs` 相同的模式。
  - `hnsw.rs` 只加 `mod quant;` 與 `pub use`；`lib.rs` 只加 re-export。
  - `distance.rs`、`batch.rs`、`exact.rs`、`rng.rs`、`error.rs`、`persist.rs` 與 `FORMAT_VERSION` 都不動。
- **轉換與 codec**：
  - `pub enum Representation { Fp32, Sq8 }`。
  - `impl HnswIndex { pub fn into_quant(self, Representation) -> QuantIndex }`：nodes 與 entry point 原樣移入；Fp32 直接移入向量；Sq8 先訓練、編碼，再釋放向量。
  - `pub fn sq8_train_encode(unit_rows, dim) -> (vmin, vdiff, codes)`：由 `into_quant` 與 codec 測試共用。
- **`QuantIndex`**（只能搜尋，不能 add）：
  - `search_batch`：沿用 `check_k`、`unit_rows`、`push_padded_row`。
  - `search_stats`：不計時；結果與 `search_batch` 相同，另外回傳每個 query 的 kernel 呼叫次數與 visited 插入次數。
  - `exact_search_batch`：以同一個 kernel 做 brute force，只用於 §7 的天花板。
  - `memory_accounting`：各 Vec 的 capacity × 元素大小，加上 header；與 allocator 無關。
  - 其他：`len`、`dim`、`representation`、`codes`、`codebook`、`entry_point`、`level`、`neighbors`。
- **走訪與 kernel**：
  - 私有的走訪函式 `search_unit_q` / `greedy_search_q` / `search_layer_q`，對一個只在本模組內使用的距離 trait 做泛型。這個 trait 只有兩個實作（FP32、SQ8）。
  - 計數版本是另一個單態化；計時路徑上沒有計數器。
  - kernel：`#[inline(never)] fn rhnsw_kernel_fp32` 與 `#[inline(never)] fn rhnsw_kernel_sq8`。名稱以 `rhnsw_kernel_` 開頭，供 §8 分類使用。
- 不做 save / load。

**Binding（`python/src/lib.rs`）**

- 新的 pyclass `QuantHnswIndex`：
  - staticmethod `build(data, m, ef_construction, seed, representation)`：參數必須明確給定；內部以 Core 的 `HnswIndex::new` + `add` 建圖，再 `into_quant`；計算期間釋放 GIL。
  - staticmethod `build_pair(data, m, ef_construction, seed)`：只給 §7 使用；建一次圖，clone 後分別轉成 FP32 與 SQ8。
  - `search(queries, k, ef_search)`：只收 float32 C-contiguous；釋放 GIL；一次 FFI 處理整批 query。
  - 其他：`search_stats`、`exact_search`、`memory_accounting`、`codes`、`codebook`、`representation`、`dim`、`__len__`，以及測試用的 graph 存取方法。
- 新的 module 函式 `sq8_encode(data)`。
- 既有的 `HnswIndex`、`ExactIndex` 兩個類別與 `BUILD_INFO` 不變。

**benchmarks**

- **adapters**（`adapters.py`）：
  - 新增 `RustHnswFp32g`、`RustHnswSq8`（共用一個小基底）與 `FaissHnswSq8`；`RustExact`、`RustHnsw`、`Hnswlib`、`FaissHnsw` 不改。
  - 新 adapter 的建構參數同樣是 `(m, ef_construction, seed)`，build 時同樣呼叫 `_check_data`。
  - Rust 的兩個新 adapter 沒有 `save_native`；`FaissHnswSq8` 有（`write_index`）。
  - `FaissHnswSq8` 在 build 與每次 search 開頭都呼叫 `omp_set_num_threads(1)`（007 §6）。
- **登記項**：
  - `config.IMPLEMENTATIONS` 加三個名稱。子 process 唯一允許 import 的 library 分別是 `rust_hnsw`、`rust_hnsw`、`faiss`。
  - `run.DISTANCE_KERNELS` 加三行事實紀錄。
  - `profile.ADAPTERS` 加 F 與 S。
- **runner 的改動，協定改為 008-v5**（計時常數、gate、hygiene 標記與電力條件全部不變）：
  1. adapter 沒有 `save_native` 時，`serialized_bytes = null`；`run.py` 與報告都要能處理 null。
  2. 在子 process 最後（所有計時點與 `save_native` 之後）：
     - 呼叫 `malloc_zone_pressure_relief(NULL, 0)`，沿用 `scripts/probe_reusable_memory.py` 的 ctypes 寫法；
     - 再以 `_memory()` 記錄 `after_relief`；
     - run.json 的 build 紀錄加上 `index_footprint_relieved_bytes` 與 `index_memory_relieved_bytes`（相對 baseline）。
     - 所有臂（含 R、P）都用同一段程式。
  - `interface.py` 不改，`check_search_result` 的容許值也不變；距離改在 Rust 或 adapter 內夾住。
  - `protocol.PROTOCOL_VERSION = "008-v5"`；`protocol_record()` 記錄第 2 點的時點。
  - `test_run_evaluate.py` 的協定常數測試同步更新版本字串；這是唯一修改的既有測試。
- **`report.py`**：
  - `check_formal_set` 加 `registry` 參數，預設仍是 `FORMAL_CONFIGS`。
  - `IMPL_STYLE` 加三個顏色。
  - Core 報告的輸出不變：既有的 `test_report` 必須全綠；另加兩個測試，確認 Core 報告重產時忽略 C 的 run，以及 C 的登記表拒絕 Core 的 run。
- **新工具**：
  - `benchmarks/quant_diag.py`：§7 的不計時診斷。
  - `benchmarks/quant_report.py`：
    - 重用 `report.py` 的 `point_rows`、`aggregate`、`frontier`、`qps_bracket`、`_check_compatible`、`_check_mergeable`、`check_formal_set(registry=QUANT_CONFIGS)` 與繪圖 helper。
    - 新增 §6 的配對讀值、判定表、C7a / C7b 的逐位元核對（參數 `--core-runs`），以及 §7、§8 的輸入。
    - 不呼叫 Gate 的讀值（C 沒有 hnswlib）。
  - `benchmarks/quant_profile.py`：§8 的 plan / check。重用 `gate_profile` 的核對邏輯；若需要參數化，`gate_profile` 的預設行為不變，`test_gate_profile` 保持綠燈。

**FAISS baseline 的語意測試**（放在 `tests/test_third_party_semantics.py`，比照 007，標 [測試]）

- **`IndexHNSWSQ` / `IndexScalarQuantizer`**：
  - 必須先 train。
  - `qtype == QT_8bit`、`rangestat == RS_minmax`、`rangestat_arg == 0` 確實生效（white-box 斷言）。
  - n = 1 與 n = 16 都能 train 並 search，對應 runner 的 null call 與 warm-up。
  - 釘住 vdiff = 0 時的行為。
  - 回傳的 similarity 可能 > 1；clip 之後，排序仍能通過 `check_search_result`。
  - `efSearch` 的語意與 Flat 相同；同 seed 時結果決定性。
  - 單執行緒：以其他 thread 的 CPU time 驗證；build 的 train / add / 移植與 search 都驗，也驗跨 thread 呼叫。
- **移植**：
  - Q 的 `hnsw.levels`、`offsets`、`neighbors`、`entry_point`、`max_level` 與 Flat 完全相同。
  - 移植之後 `is_trained` 為 True，`ntotal` 等於 n，可以 search 與 `write_index`。
  - 在量化誤差不改變排序的資料上，Q 的 ids 與 P 相同。
  - 這些 synthetic 測試是判定移植可不可行的唯一依據（§3）。
- **codec**：在同一個輸入陣列上，比較 Rust 的 `sq8_encode` 與 FAISS 的 `compute_codes`，照實記錄 codes 的一致比例與 codebook 是否相同（C8）；不擋流程。

**correctness 測試**

- **Rust**：
  - **等價**：在多組資料、(M, efC, seed)、ef 與 k 下（含 004 / 007 的重複向量 fixture、n ≤ M0 + 1、n = 1），F 的 `search_batch` 與 `HnswIndex::search_batch` 的 ids、distances 逐位元相同；`search_stats` 的結果與 `search_batch` 相同。
  - **`into_quant` 保留圖**：entry point、各點的 level、各層的 neighbors，都與 clone 的 FP32 index 完全相同。
  - **人工可驗證的 codec 案例**（2–3 維）：vmin → 0、vmax → 255、vdiff = 0 → 0，重建誤差 ≤ vdiff / 510（加上 f32 rounding）。
  - **kernel 與輸出**：kernel 與 f64 參考值 1 − q·x̂ 的差在下列登記的界內；輸出排序、夾住與 padding 正確；k、維度、NaN 的錯誤處理與 FP32 相同。
  - **SQ8 kernel 的登記誤差界**：
    - 參考值 r = 1 − Σ_j q_j · x̂_j，以 f64 在同一份 f32 的 q、vmin、vdiff 與 codes 上計算。
    - 界：|kernel − r| ≤ γ_{d+5} · (Σ_j |q_j| · (vdiff_j / 255) · code_j + |b_ref| + 1)。
      其中 u = 2⁻²⁴（f32 的 unit roundoff）、γ_m = m·u / (1 − m·u)、b_ref = Σ_j q_j · (vmin_j + 0.5 · vdiff_j / 255)（f64）。
    - 推導：
      - 在實數上，b + Σ_j w_j · code_j 恰好等於 q·x̂（展開 x̂_j = vmin_j + (code_j + 0.5) / 255 · vdiff_j 即得），所以差只來自捨入。
      - 每一項 w_j · code_j 經過三次捨入（vdiff_j / 255、q_j · (…)、乘上 code_j；code_j ≤ 255 轉成 f32 是精確的），相對誤差 ≤ γ_3。
      - d 項依序以 f32 加總：每一項最多再經過 d − 1 次捨入，合計 ≤ γ_{d+2} · Σ（依序加總的標準界），其中 Σ = Σ_j |q_j| · (vdiff_j / 255) · code_j（實數值，就是界中括號內的第一項）。
      - b 以 f64 累加後轉 f32：≤ u · |b_ref|，f64 本身的誤差約 d · 2⁻⁵³ 的量級。
      - b + Σ 與 1 − (…) 各一次捨入：分別 ≤ u · (|b_ref| + Σ) 與 u · (1 + |b_ref| + Σ)。
      - 合計 ≤ (γ_{d+2} + 2u) · Σ + 3u · |b_ref| + u ≤ γ_{d+5} · (Σ + |b_ref| + 1)。
        二階項與 f64 的誤差（kernel 的 b 與參考值 r 各約 d · 2⁻⁵³ · Σ_j |q_j| 的量級）由 γ_{d+5} − γ_{d+2} − 3u ≈ 6d · u² 的餘量吸收。
    - 夾住是 1-Lipschitz，所以夾住之後的輸出與夾住之後的 r 也滿足同一個界。
    - 量級：unit vectors 時括號內約為個位數，d = 768 的界約 1e-4–5e-4、d = 100 約 1e-5–6e-5。這是最壞情形的界，測試對每個 (query, 點) 以實際的數值計算它。
    - F 的 kernel 就是 Core 的 `cosine_distance_unit`，以逐位元相同檢查，不用這個界。
  - **其他**：`exact_search_batch` 等於小型 brute force；計數在手算案例上正確；`memory_accounting` 的 F − S 等於 3·n·d − 8·d。
- **pytest**：
  - 新 adapter 的合約測試：輸出格式、build 的 unit-norm 檢查、單執行緒、決定性、batch 與 single 一致、每次 search 只有一次 FFI 呼叫、n = 1 / 16。不放進「與 FP32 語意等價」的測試。
  - runner：以 stub adapter 驗證沒有 `save_native` 時寫出 null，以及 relief 在 `save_native` 之後、不在計時區間內（呼叫序列）。
  - config：10 份 C config 都能解析，`config.execution_order` 的順序等於 §5 的表。
  - `quant_report`：以手寫紀錄逐條驗證本節與 §6 的規則，並比照 015 以 mutation 檢查測試鑑別力。
  - `quant_diag`：在 synthetic 資料上驗證分解算術與 ids 核對。
  - `quant_profile`：驗證分類規則。
- **golden test**：在動任何 C 的程式之前，先 commit 一個測試，固定 Core FP32 在一組 synthetic fixture 上 `search_batch` 結果的 SHA-256，以及 v1 存檔 bytes 的 SHA-256。之後必須一直保持綠燈。

**Core 不變的證據鏈**

1. 上述原始碼範圍的 git diff 只有新增：`hnsw.rs` 只多 `mod` 與 `pub use`；既有的 adapter 與 pyclass 不變。
2. golden test 與既有測試全部通過（只有協定常數測試更新了版本字串）。
3. C7a、C7b：formal run 中 F、R（C7a）與 P（C7b）的 ids 與 Core 逐位元相同。
4. Core 報告的重產不受影響（由測試保證）。README 的 Core 結果、Gate 判讀與 frontier 一律不改；C 另開一節，不宣稱 Gate 通過。

### 5. 格點、ef 與 formal config

- **資料集**：
  - `glove-100-angular`：100 維、10,000 queries。
  - `simplewiki-embeddinggemma2-100k`：768 維、2,000 queries。
  - 兩者都只用 n = 100,000 與 frozen ground truth。兩個維度是刻意的對比（H1）。
- **格點**：只有 (M, efC) = (16, 200)。
  - 它是兩個資料集共同的中心格點、完整 variance 格點，也是 Gate profiling 用的格點（013 §1、015 §2）；每個 build 的成本也最低。
  - 原始插入順序，不做 shuffle。
- **seed**：
  - Rust 的 F、S：seeds 1–5，與 Core variance 格點的 seed 相同，C7a 可以全部核對。
  - FAISS 的 P、Q：seeds 1–3，只作描述。
  - R：只在 seed 1，作為錨點。
- **ef 清單**（同一資料集的所有臂、所有 config 逐字相同）：
  - GloVe：64, 96, 128, 152, 192, 256, 304, 384, 512, 768。即 Core 該格的清單，加上梯子上的 512 與 768。
  - 文字：24, 32, 48, 64, 96, 128, 192, 256, 384, 512。與 Core 該格相同。
  - **規則**（沿用 013 §2 第 4 點）：對要涵蓋的最高目標，取 SQ8 交點悲觀上緣之上的第一個梯子點。悲觀上緣假設 SQ8 要達到同一個 recall，需要 FP32 交點 2 倍的 ef：
    - GloVe 0.95：2 × 304 = 608，取 768。
    - 文字 0.99：2 × 192 = 384，取 512（已在清單中）。
    - GloVe 的 0.99：FP32 約在 ef 1,300 才達到，不涵蓋。所有臂事先預期「未達到」，只描述。
  - 與 Core 只在共同的 ef 上核對（C7）。013 §2 第 6 點「同一格的清單逐字相同」只在 C 內成立。
- **formal config**：共 10 份，每份一個 build。
  - 單一 build 的 config 不會被 008 §5 輪換，執行順序就是 `implementations` 的順序。
  - 成對臂的先後在 config 之間以手動位移平衡。
  - 全文見附錄。

| config（`benchmarks/configs/`） | seed | implementations（= 執行順序） | Rust 配對 F / S | FAISS 配對 P / Q | child_timeout_s |
| --- | --- | --- | --- | --- | --- |
| `quant-glove-seed-1.toml` | 1 | S, P, F, Q, R | S 先，相隔 2 | P 先，相隔 2 | 6,000 |
| `quant-glove-seed-2.toml` | 2 | P, F, Q, S | F 先，相隔 2 | P 先，相隔 2 | 6,000 |
| `quant-glove-seed-3.toml` | 3 | F, Q, S, P | F 先，相隔 2 | Q 先，相隔 2 | 6,000 |
| `quant-glove-seed-4.toml` | 4 | S, F | S 先，相鄰 | — | 4,200 |
| `quant-glove-seed-5.toml` | 5 | F, S | F 先，相鄰 | — | 4,200 |
| `quant-text-seed-1.toml` … `quant-text-seed-5.toml` | 1–5 | 與 GloVe 同一個 seed 的順序相同 | 同左 | 同左 | 3,600 |

  - seed 2、3 的順序就是 008 §5 對 [P, F, Q, S] 做第 0、1 次輪換的結果。seed 1 中 R 與 F 相隔 2 個位置。
  - Rust 配對：F 先的是 seeds 2、3、5；S 先的是 seeds 1、4。
  - FAISS 配對：P 先的是 seeds 1、2；Q 先的是 seed 3。
- **timeout**：013 §6 的公式原本是為 3 個實作寫的，這裡重述。
  - child_timeout_s = 該 config 最慢的臂在負載下的上限 × 5，進位到 600 的倍數，最少 1,800。
  - 各臂的上限：
    1. 從 Core 實測該格最慢的子 process 時間開始，加上延長 ef 的外推（每點時間與 ef 成正比）；
    2. 乘以表示的不確定倍數：Rust 兩臂 ×1.2；Q 的 ef 點 ×3（FAISS SQ kernel 在 ARM 上的效能未知）；
    3. 再乘負載倍數 1.25（013）。
  - **GloVe**：
    - Rust：307 s + 213 s（512、768 兩點）= 520 s，×1.2 ×1.25 = 780 s。
    - Q：build 16 s + ef 點 299 s × 3 ≈ 916 s，×1.25 = 1,145 s。
    - 有 Q 的 config：1,145 × 5 → 6,000；只有 Rust 的 config：780 × 5 → 4,200。
  - **文字**：
    - Rust：445 s，×1.2 ×1.25 = 668 s。
    - Q：62 s + 105 s × 3 ≈ 380 s，×1.25 = 475 s。
    - 668 × 5 → 3,600。
  - smoke 之後只能依同一規則**上調**，不得下調。timeout 不影響任何讀值的定義。
- **執行條件**：
  - 協定 008-v5（§4；常數與 008-v4 相同）。
  - **電力**（已由 016–018 決定，不需要再問使用者；適用於所有 C 的 formal run 與 profiling，smoke 也用高效能模式）：
    - 所有 C 的 run 都用高效能電力模式（016）。
    - 所有 C 的 run 都用同一個 Apple 原廠變壓器：018 允許 70W 或 140W，但 C 報告只能合併同一個。以**第一份 C formal run 實際使用的變壓器**為準。
    - 之後換了變壓器的 run，依 018 不得合併進 C 報告（要處理必須另寫 decision）。
  - **機器負載**（使用者要求開跑前也檢查 CPU、GPU、RAM；formal、smoke、`quant_diag` 與 profiling 都適用）：取得 heavy lock 之前以 `benchmarks/machine_quiet.py` 確認，全部成立才開始：
    - heavy lock 沒人持有；
    - 背景沒有 heavy workload（014：自己與祖先以外，瞬間 %CPU ≥ 50 的 process），1 分鐘 load < 4.0（008 §2）；
    - GPU 的 Device Utilization ≤ 20%；
    - 系統記憶體 free percentage ≥ 30%（`memory_pressure -Q`）；
    - powermode 是 2（`--powermode 2`，016；用於有電力條件的步驟）。
    - `machine_quiet` 決定的是開始的時機；run 的有效性仍由 runner 的協定判定（gate、hygiene 標記）。
  - 之後在 heavy lock 下執行（`./tools/with-mac-lock`）。
  - formal：先跑 `./scripts/test.sh`，再執行 `./scripts/formal_launcher.sh results/batches/quant-<…>.log <最長等待秒數> --powermode 2 --skip-idle <config…>`（launcher 每 60 秒呼叫一次 `machine_quiet`，連續 3 次成立才在 heavy lock 下啟動）。閒置 load 的量測依 017 §4 仍然有效；一個批次可以排多份 config。
  - formal run、診斷與 profiling 進行中 repo 不得變 dirty（NEXT 的執行手冊、§10）。
- **建議順序**（排程，不是協定）：GloVe seed 1 → 5、文字 seed 1 → 5，可以交錯。
- **跑完與失敗的處理**：
  - 10 份都必須跑完。aborted / invalid / error 整份重跑，原本的 run 以 `--excluded` 列出；incomplete 照實保留、不重跑（010 §3、013 §5）。
  - C 的第一份 formal 開始之後，`MEASUREMENT_PATHS` 凍結。之後任何修改都會讓已跑的 C run 無法合併，必須整組重跑並寫補記。工程改善的凍結範圍更大，見 §10。
- **與 Core 正式結果的關係**：
  - **不能當 C 的計時或記憶體基準**，原因有四：
    - C 改了 `MEASUREMENT_PATHS`，`_check_mergeable` 會拒絕合併（013 §5）；
    - 協定版本不同，`_check_compatible` 會拒絕；
    - GloVe 的 Core 用的是 70W 變壓器（018 不得混用）；
    - 相隔數日的漂移與 H1 的 ±10% 是同一個量級。
    - 因此 FP32 對照（F、P、R）在 C 的同一個 run 內重新量。
  - **直接重用**：資料集與 frozen ground truth、(16, 200) 與 ef 梯子、e\*、成本模型、hnswlib 的全部結果（C 不跑 hnswlib），以及 Gate profiling（作為 C5 的對照）。
  - **只用來核對與描述**：C7、C6。

### 6. 讀值與判定規則

**指標**：

- Recall：Recall@10（id-based，主要指標，003、010 §1）、tie-aware Recall@10、Recall@1。
- 速度：QPS、P50、P95。
- 建置：build time（S 含轉換、Q 含移植）、peak RSS。

**記憶體**：

- **主要**：`index_footprint_bytes`（M_014，014）。frontier 與 C 的結論一律用它。
- **診斷**（放在同一張表，明確標示，不作 frontier 讀值）：
  - `index_footprint_relieved_bytes`（M_relief，§4）；
  - `memory_accounting` 的 M_struct（§7）；
  - rss 增量（014 的輔助指標）；
  - `serialized_bytes`：只有 P、Q；R 是 Core 的格式；F、S 為 null。
- M_relief 在所有計時點之後才量，可能含有搜尋期間的少量殘留，時點也與 M_014 不同，照實註明。
- **使用者的決定**（2026-10-16，本文件 commit 之前；§13）：C 的記憶體主要指標維持 014 的 phys_footprint（M_014）；M_relief 與 M_struct 只是診斷。
  本文件 commit 之後（包括 smoke 之後、看到任何 C 的結果之後），這個選擇不得更改。

**讀值 1：frontier**（沿用 010 §2、§3 與 013 §3，不修改）

- 每個資料集、每個臂（F、S、P、Q；R 只有 1 個 build，另列為錨點）：
  - 所有 ef 點取跨 seed 的 median；主要讀值排除有 hygiene 標記的 build。
  - 目標 0.90、0.95 為主，0.99 只描述。
  - 在 median Recall@10 ≥ 目標的點中，取 QPS 最高、P50 / P95 最低、M_014 最低的點。
  - 不內插、不外推；沒有點達標就記「未達到」。
  - 排除標記點會改變讀值時，兩種讀值都列；每個讀值附 build 數與跨 seed 的 min–max。
- **比值**：S ÷ F、Q ÷ P，只在兩者都達標時計算（Q ÷ P 在 C7b 不一致時記為不適用）。不比較 S 與 Q 的絕對值；實作之間的差距 Core 已經報告過。
- **quantization frontier 的判讀**：
  - 方法：015 §1 的 `qps_bracket` 與 seed 分開規則，套用到 S 對 F、Q 對 P。
  - 條件：只有單一格點 (16, 200)；每側至少 3 個未標記的 build。
  - 結果類別：`sq8_faster` / `fp32_faster` / `within_quantization` / `uncertain` / `undetermined`，意義與 015 相同。
  - 這是事先登記的描述性判讀，不是假設。
- **圖**：
  - 每個資料集：recall–QPS（4 個臂，圖例附 M_014）、recall vs ef。
  - 每個目標一張 quantization frontier：x = M_014、y = 該目標的 QPS；另畫一版 x = M_relief，標為診斷。

**讀值 2：同 (seed, ef) 的配對讀值**（010 §2 的例外，見 §11 的補記）

- **只對圖逐位元相同的配對開放**：
  - S 對 F：由構造保證，並有測試；
  - Q 對 P：只在移植可行（§3）、C8 的 graph 核對一致，而且 C7b 一致時。
- **每個 (seed, ef) 計算**：ρ_QPS、ρ_eval（只有 Rust）、ρ_P50、ρ_P95、ΔRecall@10、ΔRecall@1，以及 ΔM（M_014、M_relief、M_struct 三種）。
- **使用哪些 build**：
  - 時間類：只用乾淨的配對；
  - recall：用全部 build；
  - 記憶體類：排除有 `memory` 或 `swap` 標記的 build。
- 報告 median 與 min–max，另列包含標記點的版本。
- H1、H2 都只在 e\* 判定；其他 ef 全部列出。

**完整判定表**

- `quant_report` 輸出一張表，涵蓋：
  - H1：a、b、c、d、前提、整體；
  - H2：GloVe、文字、整體；
  - quantization frontier 的判讀：2 個資料集 × 2 組配對 × 0.90 / 0.95；
  - C1–C6、C7a、C7b、C8。
- 每一列只能是下表中該列允許的狀態之一，並附上判定所用的數字。表外的狀態一律不得出現（由 `quant_report` 的測試檢查）：

  | 列 | 允許的狀態 |
  | --- | --- |
  | H1 前提 | 成立 / 前提不成立 |
  | H1a、H1b、H1c | 成立 / 不成立 / 不確定 / 無法判定 |
  | H1d | 一致 / 不一致 / 無法判定 |
  | H1 整體 | 成立 / 不成立 / 計時成立、profile 不一致 / 前提不成立 / 不確定 / 無法判定 |
  | H2 GloVe、H2 文字 | 成立 / 不成立 / 不確定 / 無法判定 / 不適用 |
  | H2 整體 | 成立 / 不成立 / 不確定 / 無法判定 / 不適用 |
  | quantization frontier：S 對 F | `sq8_faster` / `fp32_faster` / `within_quantization` / `uncertain` / `undetermined` |
  | quantization frontier：Q 對 P | `sq8_faster` / `fp32_faster` / `within_quantization` / `uncertain` / `undetermined` / 不適用（只在 C7b 不一致時） |
  | C1 | 一致 / 不一致 / 無法判定 / 不適用（FAISS 部分，只在採用 §3 的備案時） |
  | C2、C3、C5 | 一致 / 不一致 / 無法判定 |
  | C4 | 一致 / path-distorted / 無法判定 |
  | C6 | 只描述（不判定） |
  | C7a、C7b | 一致 / 不一致 / 無法判定（只在 Core 的 points 檔缺失時） |
  | C8 graph | 一致 / 不一致 / 無法判定（只在 §7 重建的 Q 與 formal 的 ids 不符時） / 不適用（只在採用 §3 的備案時）；codes 一致比例只描述 |

  - 「不判定（C7a 不一致）」：C7a 不一致時，H1a–H1d、H1 整體與 H2 的所有列都只能是這個狀態，報告拒絕輸出其他判定（H1 前提是靜態檢查，照列）。
  - 「未達到」只出現在讀值 1 的各臂 frontier 讀值（不在判定表中）；quantization frontier 的對應類別是 `undetermined`。
  - 「計時成立、profile 不一致」只出現在 H1 整體；「path-distorted」只出現在 C4。
- 不得只報成立的部分。

### 7. 不計時的診斷（`quant_diag.py`）

**執行時機與條件**：

- 在全部 10 份 C formal config 完成之後，先通過 `machine_quiet`（§5），再在 heavy lock 下執行。
- 單執行緒：以 `VECLIB_MAXIMUM_THREADS=1` 啟動（由工具檢查）；FAISS 設 `omp_set_num_threads(1)`。
- numpy 的重排分塊進行，避免 10k × 100k 的大矩陣。

**每個資料集、seeds 1–5**：

1. 以 `QuantHnswIndex.build_pair` 建一次圖。
2. F 與 S 各自在 C 的每個 ef 執行：
   - `search_stats`（k = 10），得到 D、V；
   - k = ef 的搜尋，得到 W。
3. 核對：兩者的前 10 個 ids 必須與 formal run 中同一個 (seed, 臂, ef) 的 `ids_k10` 逐位元相同。不符的 (seed, 臂) 在 H1、H2 中無效（H2 的有效 seed 見 §2），並照實列出；這代表有非決定性，要調查。
4. 計算 T_F、T_S、L_*、δ_F（§2），以及 codes 完全相同的列所占的比例（只描述）。

**FAISS（每個資料集、seeds 1–3）**：

- 以 Q adapter 的同一段建圖程式建 Flat 並移植；在移植完成、`del flat` 之前核對 Q 的 `hnsw.levels`、`offsets`、`neighbors`、`entry_point`、`max_level` 與 Flat 完全相同（C8）。
- 重建的 Q 在 e\* 的 k = 10 ids 必須與 formal run 中同一個 (seed, Q, e\*) 的 `ids_k10` 逐位元相同，診斷才與 formal 綁定（與 Rust 的做法相同）。
  不符時該 seed 的 C8 graph 核對記為無法判定，Q 對 P 的配對讀值記為不適用，照實列出。

**天花板**（每個資料集算一次，與 seed 無關）：

- c：S 的 `exact_search`（同一個 kernel、(distance, id) tie-break）對 frozen GT 的 Recall@10 與 Recall@1。
- c_FAISS：以 FAISS `IndexScalarQuantizer(QT_8bit, IP, RS_minmax)` 做 brute force，只描述。
- 天花板是不計時的分析基準，不作效能比較。CLAUDE.md 規定 brute force 只作 ground truth 與 correctness baseline。

**結構大小**：

- F、S 的 `memory_accounting`（seeds 1–5）。
- P、Q 的 FAISS 陣列大小：storage codes、`hnsw.neighbors`、`offsets`、`levels`（seeds 1–3）。

**證據條件**（比照 015 §2）：

- git 乾淨、binding 晚於 HEAD。
- `gate_profile.PROFILED_PATHS` 與 C formal run 的 commit 沒有差異。唯一的例外是 §9 的「只影響診斷函式的 Rust bug」：差異必須只在 `search_stats`、`exact_search_batch`、`build_pair`，並寫補記。
- BUILD_INFO 與套件版本相同。
- 由祖先 process 持有 heavy lock。
- 輸出寫到 `results/quant/diag-<資料集>.json`（附 commit、BUILD_INFO、codes 的 SHA-256），拒絕覆寫。

**為什麼不在 `run_child` 內計數**：這樣可以避免擴大量測程式的改動。建圖是確定的，所以診斷以逐位元的 ids 核對與 formal run 綁定。

### 8. profiling

**工具與證據條件**（沿用 009 與 015 §2）：

- macOS `sample`，每 1 ms 取樣；只取樣 search 階段，每次 30 秒。
- 先通過 `machine_quiet`（§5），再在 heavy lock 下執行；formal 的電力與背景 gate。
- git 乾淨、binding 晚於 HEAD；`PROFILED_PATHS` 與 C formal run 沒有差異；BUILD_INFO 與套件版本相同。
- 取樣時的 Recall@10 必須與 C formal run 中同一個 build、同一個 ef 的點完全相同。
- 計畫寫在 `results/profiles/quant-<資料集>-*`，拒絕覆寫上一次的輸出。

**取樣內容**（每個資料集：seed 1、(100k, 16, 200)、原始順序）：

1. R 在 e\*：對照 Gate profiling（C5）。
2. F 在 e\*。
3. S 在 e\*：同一張圖、同一個 ef，是 H1d 的受控比較。
4. S 在它自己的同 recall ef：取 C 的 seed 1 中 S 第一次 Recall@10 ≥ 0.95 的 ef。只描述；沒有達到就改用 0.90；與 e\* 相同就略過。

- 不取樣 build 階段：建圖程式與 Core 相同。
- 不取樣 FAISS：沒有符號表，依 009 補記只能近似歸屬。

**分類規則**（取樣之前固定；依下列順序取第一個符合的類別）：

- 比對的對象是 `profile_report.top_of_stack` 回傳的每一列：「符號」指 `Row.name`（`pretty_symbol` 的輸出，例如 `hnsw::hnsw::HnswIndex::search_layer`），「映像檔」指 `Row.image`。
- `profile_report` 已修正：沒有符號的列會保留（以映像檔內的位移或位址表示），無法解析的列一律報錯（009 補記，2026-10-16）。這些列同樣依下列規則分類（除非映像檔符合第 5 條，否則歸入「其他」），不得略過。

1. **kernel**：`Row.name` 含 `rhnsw_kernel_`，或 `Row.symbol`（`sample` 的原始符號）含 `rhnsw_kernel_`。只有 F、S 有這一類；R 的距離 inline 在 traversal 內。
2. **visited set**：符號以 `hashbrown::` 開頭，或含 `core::hash::sip`。
3. **heap**：符號以 `alloc::collections::binary_heap::` 開頭。
4. **traversal**：符號含 `search_layer`、`greedy_search` 或 `search_unit`。
5. **記憶體配置**：映像檔是 `libsystem_malloc.dylib`，或符號以 `_platform_mem` 開頭。
6. **其他**：其餘全部，包括 Python / FFI、沒有符號的列與 `hnsw::batch::unit_rows`。

- 「其他」超過 5% 時，報告列出其中的前 10 名。
- 看過取樣之後的重新歸類只能當註記；判定一律用本規則。原始排行才是證據。

**由取樣導出的量**：

- f_k 與各類別的占比。
- 每 query 的時間分解：占比 × 1 / QPS，QPS 取自 C formal run 的同一個點。
- 每次評估的時間（再除以 D）。
- κ 與 ν（§2）。
- `profile.py` 仍然不輸出 QPS；以上都是在報告中，對事先登記的量做的算術。seed 1 的計時點有 hygiene 標記時，照算並標明；這時 H1d 記為無法判定（§2），κ、ν 只作描述。

**kernel 拆分的方式**：

- 只在新臂（F、S）用 `#[inline(never)]` 讓 kernel 成為獨立函式；R 保持 inline。
- 這正是 009「拆開距離需要另寫 decision」的情形，以 009 的補記登記（§11）。
- 不改 BUILD_INFO，不用 xctrace（B3 的 Xcode 風險）。
- 拆分的代價是函式呼叫的開銷，由 C4 量化；F 與 S 承擔的代價對稱。

**靜態佐證**：§2 中 H1 的前提檢查（`otool -tV`），同時記錄 S 的 kernel 中出現的轉換指令。

**已知限制**：

- `sample` 只看得到函式層級，分不出記憶體停頓與運算。
- 在 OoO 執行下，與長相依鏈重疊的工作可能被算進 kernel。
- 沒有硬體 counter：kperf 需要 root；xctrace 的 CPU Counters 需要完整的 Xcode。
- 因此 cache 的論證只寫到「延遲 / 容量」的層次，並明示這是推論。

**呈現**：

- 每個資料集一張 F 對 S 的每 query 時間分解長條圖，作為 README 的「profiler 前後」。
- 不另外做 flamegraph（見放棄的替代方案）。

### 9. 執行順序、停止與封版

**執行順序**（每一步都在前一步完成之後）：

1. **事先登記**：本文件經獨立審查、由使用者決定採用（§13）之後，019（含附錄的 10 份 config 全文）在分支 `prereg-quant` 上 commit；commit 的時間就是事先登記的證據（比照 015）。這之前不得在任何真實資料上執行量化表示，包括 smoke、天花板與 profiling。
2. **golden test**：先 commit golden test，再開始寫 C 的程式。實作期間只用 synthetic 資料。
3. **獨立審查**：實作完成後做一輪獨立審查（多角度加 mutation，比照 008 / 015），結果寫成 019 的補記。
4. **測試與前提檢查**：
   - `./scripts/test.sh` 全部通過；H1 前提檢查（反組譯）的結果寫進補記。
   - FAISS 移植可不可行，只依 §4 的 synthetic 測試定案（§3），結果寫進補記。這必須在 smoke 之前完成；之後不得再改用備案。
5. **加入 config**：10 份 config 以附錄的內容逐字加入 `benchmarks/configs/`，與 `QUANT_CONFIGS` 在同一個 commit；由測試比對逐字相同。
6. **真實資料的 correctness smoke**：
   - 時機與條件：019 commit 之後；一律先通過 `machine_quiet`，再在 heavy lock 下執行（§5）；使用 smoke 模式。
   - 內容：GloVe 100k 與文字 100k 的 (16, 200) seed 1，ef 為 GloVe {64, 304, 768}、文字 {24, 48, 512}，五個臂都跑。
   - 只檢查：結果格式、010 §6、null call 與 warm-up、F / R / P 與 Core 的 ids 逐位元相同、Q 的移植，以及每個子 process 的時間（只用來依 §5 上調 timeout）。
   - 限制：build 是確定的，smoke 的 recall 就等於 formal 的 recall。所以 smoke 之後不得修改假設、門檻、格點、ef、seed 或讀法；只能修 correctness bug（寫補記）或上調 timeout。
   - **smoke 之後的「correctness bug」只限於實作偏離 §3 的文字定義**，而且必須以測試對照 §3 的文字證明。§3 的 codec、距離公式與夾住（clamp）本身不得修改。
   - smoke 中 C8 不一致時只能記為不適用，不得改用 §3 的備案。
   - correctness smoke 必須在第一份 formal 之前全部完成。
7. **formal**：依 §5 的執行條件跑 10 份 formal（電力條件已由 016–018 決定，不需要再問使用者）。
8. **收尾**：`quant_diag`（§7）→ profiling 的 plan、取樣與 check（§8）→ `quant_report` → README 的 C 一節 → 更新 NEXT。

**停止條件**：

- **formal 之前**任何 correctness、等價或 golden test 失敗：先修正，寫補記，重跑 smoke；假設、格點與門檻都不變。smoke 之後的修正範圍受第 6 步的限制。
- **formal 中 C7a 不一致**：停止，不判定 H1、H2，先查明原因。需要改程式時，整組 C config 重跑並寫補記。
- **formal 中 C7b 不一致**：不停止；FAISS 的配對讀值與 Q ÷ P 記為不適用（§2），照實記錄原因。H1、H2 不受影響。
- **formal 中 C8 不一致**：只能記為不適用，不得改用 §3 的備案。
- **formal 之後發現只影響診斷函式的 Rust bug**（`search_stats`、`exact_search_batch`、`build_pair`）：
  - 計時路徑（`search_batch` 及其走訪與 kernel）不得修改。
  - 修正之後，以 §7 的逐位元 ids 核對把診斷綁定到 formal，並寫補記說明 `PROFILED_PATHS` 與 formal run 的差異只在這些函式（§7 的證據條件）。
  - 若計時路徑也需要修改，整組 C config 重跑並寫補記。
- **aborted / invalid / error**：整份 config 重跑。**incomplete**：照實保留（010 §3）。
- **乾淨配對不足而「無法判定」**：不得重跑 complete 的 run。是否補 seed 由使用者決定，並另寫 decision，標明是在看過 hygiene 狀態之後。
- **看到 C 的結果之後**才增加 ef、seed、格點或臂：必須另寫 decision，標明是在看到結果之後，而且不得取代本文件的判定。

**C 完成的定義**：

- 10 份 config 各有一個 complete 或 incomplete 的 run；失敗的 run 以 `--excluded` 列出。
- `quant_diag` 與 profiling 的 check 完成；失敗的照實列出。
- `quant_report` 產生完整的判定表（包括無法判定等狀態）。
- README 的 C 一節完成，NEXT 已更新，§11 的補記都已 commit。

**封版**（使用者決定，2026-10-16，§13）：

- **C 做完就封版**：C 完成之後，Advanced 封版。
- C 用了 3 個假設名額中的 2 個；第 3 個名額**不保留**。
- 封版之後若使用者要再做實驗，必須另寫 decision，標明是在看到 C 的結果之後才提出，而且不得取代本文件的判定。

### 10. 工程改善與研究的隔離

使用者允許工程改善（測試、API 易用性、文件、Demo）與 C 並行，但不能污染研究結果。規則：

- **(a) 凍結的程式**：C 的第一份 formal 開始之後，到 C 完成（§9）之前，工程改善不得修改：
  - `report.MEASUREMENT_PATHS` 中的任何檔案；
  - `gate_profile.PROFILED_PATHS` 中的任何檔案；
  - C 的 config（附錄的 10 份）；
  - `quant_report`、`quant_diag`、`quant_profile` 的判定邏輯。
- **(b) repo 不得 dirty**：formal run、診斷與 profiling 進行中 repo 不得 dirty。工程工作在另外的 worktree 或分支上進行，之後才併回。
  併回會讓 HEAD 前進；之後的診斷與 profiling 仍須符合各自的證據條件（git 乾淨、binding 晚於 HEAD、`PROFILED_PATHS` 沒有差異、BUILD_INFO 相同）。
- **(c) Demo 與研究結果分開**：Demo 不得出現在任何正式結果或判定中；Demo 的數字不得當作研究結果引用。
- **(d) 等到 C 完成**：工程改動如果碰到 (a) 的任何路徑，必須等 C 完成之後才能併回。
- (a)–(d) 限制的是工程改善。研究本身需要的修正只能依 §9 的停止條件進行，並寫補記。
- C 的第一份 formal 開始之前，工程改動同樣受 §4「Core 不變的證據鏈」約束：golden test 與既有測試必須保持綠燈。

### 11. 受影響的既有 decision（需要時寫補記）

| decision | 補記內容 |
| --- | --- |
| 002 | 量化臂（S、Q）回傳的是表示本身估計的 cosine distance，在 Rust 或 adapter 內夾到 [0, 2]。介面與 `check_search_result` 不變；id-based recall 與 tie-aware recall（harness 以 f64 重算）都不受影響 |
| 005 | 不新增檔案格式。S 與 F 沒有 save / load，`serialized_bytes` 為 null。本表只記錄這點，005 本身不需修改 |
| 007 | `IndexHNSWSQ` 的實測語意（§4 的測試項目）、Q 的移植方式與備案，以及 Q 在計時路徑上多出的 `np.clip` |
| 008 | 008-v5：新的實作名稱、`serialized_bytes` 可以為 null、子 process 最後的 relief 診斷。計時常數不變；只用於 C |
| 009 | 新臂以 `#[inline(never)]` 拆出 kernel；分類規則在取樣前固定（§8）；FAISS 不取樣 |
| 010 | §2：只對圖逐位元相同的配對開放同 (seed, ef) 的比較。§4：C 不做 shuffle，Rust 5 個 seed、FAISS 3 個 seed（使用者決定，理由見「為什麼」）。只適用於 019 的 C run，不改變 Core 的讀值與判讀 |
| 013 | C 的 config 不在 013 的登記表中；「ef 清單逐字相同」只在 C 內成立；timeout 公式對超過 3 個實作的重述（§5）。只適用於 019 的 C run，不改變 Core 的讀值與判讀 |
| 014 | 主要記憶體指標不變（使用者決定，§13）。relief 診斷在所有量測之後才做，不影響 014 的讀值；014 放棄的是「在量測之前」呼叫 relief。M_relief、M_struct 只是診斷，019 commit 之後不得改為主要指標 |
| EXPERIMENT_RULES | C 的報告不含 insertion order variance（Core 已報告 (16, 200) 的 shuffles 101–105），照實寫明 |

015 不修改；C 只借用 015 §1 的 QPS 範圍與 §2 的 profiling 條件。

### 12. 成本

**實作（agent 工時，約 28–38 h）**

| 項目 | 工時 |
| --- | --- |
| Rust `quant.rs`（轉換、codec、兩個 kernel、複製的走訪、stats、exact、accounting）、Rust 測試與 golden test | 6–8 h |
| binding | 1.5–2 h |
| 三個 adapter、FAISS 語意與移植測試、codec 比對 | 4–5 h |
| runner（008-v5）、config、protocol、DISTANCE_KERNELS 與對應測試 | 2–3 h |
| `quant_diag` | 2–3 h |
| `quant_report`（判定表、配對讀值、C7、圖）、規則測試與 mutation | 5–7 h |
| `quant_profile`、分類規則、反組譯檢查 | 2.5–3.5 h |
| 各 decision 的補記、config、NEXT / README | 2–3 h |
| 獨立審查與修正 | 3–4 h |

**機器時間**（heavy lock、高效能模式、同一個變壓器）

各臂每個子 process 的中央估計，以 Core 實測加上延長的 ef 推得；Rust 的 SQ8 取 ×0.9–1.0，Q 的 ef 點取 ×1.3：

| 資料集 | F / R | S | P | Q |
| --- | --- | --- | --- | --- |
| GloVe | 約 500 s | 約 475 s | 約 295 s | 約 400 s |
| 文字 | 約 420 s | 約 420 s | 約 160 s | 約 200 s |

| 項目 | 中央估計 | 悲觀估計（負載 ×1.25、Q ×3） |
| --- | --- | --- |
| GloVe 5 份（seed 1 約 36 分；seeds 2–3 各約 28 分；seeds 4–5 各約 16 分） | 2.1 h | 3.6 h（單份最長約 64 分） |
| 文字 5 份（約 27 / 20 / 20 / 14 / 14 分） | 1.6 h | 2.6 h |
| **formal 合計** | **3.7 h** | **6.2 h** |
| `quant_diag`（GloVe 約 20 分、文字約 28 分、天花板約 5 分） | 0.9 h | 1.1 h |
| profiling（最多 8 次取樣，含每次重建 index） | 0.35 h | 0.5 h |
| 真實資料的 correctness smoke | 0.6 h | 0.8 h |
| `./scripts/test.sh`（2–3 次） | 0.3 h | 0.5 h |
| 重跑預備（預留最長的一份 config） | — | 1.1 h |
| **lock 合計** | **約 5.8 h** | **約 10 h** |

- 重跑預備的依據是 Core 的實績：GloVe 9 個 run 中有 1 個失敗；文字 10 個 run 中有 3 個失敗、3 個作廢。
- 表中沒有列入的等待：每個子 process 前的 load / 背景 gate（最多 10 分鐘，超過就中止），以及 `machine_quiet`（GPU ≤ 20%、記憶體 free ≥ 30%）造成的開跑等待。
- formal 的中央估計約為 Core 正式格點（013 估計 19.2 h）的 19%；連同診斷與 profiling 約為 30%。
- 不需要遠端機器、付費用量或任何 credential。

### 13. 使用者的決定（2026-10-16）

使用者在 2026-10-16（台灣時間）、本文件 commit 之前、任何真實資料上的量化結果之前決定（已記到 NEXT）：

1. **採用本設計**：2 個假設 H1、H2；SQ8；建完 FP32 圖才轉換；5 個臂；各項門檻。
2. **seed 數**：Rust 5 個、FAISS 3 個、不做 shuffle（偏離 010 §4，§11 的補記）。
   - 撰寫時提供的取捨：若把 Rust 改成 3 個 seed，約省 1 h（formal 的 27%），但「乾淨配對 ≥ 3 個」的機率會從約 0.8 降到約 0.3（見「為什麼」）。
3. **記憶體主要指標維持 014 的 phys_footprint（M_014）**：
   - M_relief 與 M_struct 只是診斷（§6）。本文件 commit 之後（包括 smoke 之後），這個選擇不得更改。
   - 揭露：M_relief 的提議參考了 Core 看到結果之後的觀察（GloVe 的 Δreusable 只有 1.5 MiB）。
4. **C 做完就封版**：不保留第三個名額（§9）。
5. **電力條件**：所有 C 的 run 用高效能模式（016），而且全部用同一個變壓器（018 允許 70W 或 140W，但 C 報告只能合併同一個）；以第一份 C formal run 實際使用的變壓器為準，之後換變壓器的 run 依 018 不得合併（§5）。
   - 這已由 016–018 決定，不需要再問使用者。使用者先前「自動電力模式 + 70W + 電量規則」的條件已被 016–018 取代。
6. **使用者的其他明示**：照實記錄在「背景」（使用者的授權與指示）；工程改善的並行見 §10；開跑前的 CPU、GPU、RAM 負載檢查見 §5。

**一般說明**（不需要使用者決定）：

- **與 Optional 相鄰的部分**：本設計不實作任何 SIMD、整數 kernel 或 `target-cpu=native`；`#[inline(never)]` 只改變函式邊界，不改變算術。
- **lock 時間**：中央估計約 5.8 h（悲觀約 10 h），切成 10 份每份約 1 h 以內的 config（§12）。
- 之後只有重大研究決策或真正的 Blocker 才詢問使用者（例如 §9 的「乾淨配對不足」是否補 seed、封版之後是否再做實驗）。

## 為什麼

- **以設計 2 的協定骨架為主**：
  - 設計 1 與設計 2 的平均分數都是 7.0，三位評審中有兩位以設計 2 為基礎。
  - 設計 2 的以下做法，直接對應評審的 must_fix：
    - 事先登記：先 commit 分支，smoke 只能在登記之後跑；
    - 有同一路徑的 FP32 對照臂；
    - 以距離評估次數正規化的 ρ_eval；
    - 固定判定用的 ef；
    - 5 個 seed；
    - 所有子判讀都列在一張表；
    - FAISS 只比相對效果；
    - 分類規則在取樣前固定；
    - 需要使用者決定的事項交給使用者。
- **改用設計 1 的「建完圖才轉換」，不用設計 2 的量化建圖**：
  - 三位評審都指出設計 2 的實作面最大：泛型重寫 insert / connect / select_neighbors、新的存檔格式、修改 interface、008-v5 的大幅改動。
  - 同一張圖讓 F 與 S 能在同一個 (seed, ef) 上成對比較，也才能把 recall 損失拆成走訪與排序兩部分（評審 3 的 must_fix 1）；Core 的 insert 也因此一行不動。
  - 代價：結論不涵蓋 build time 與 peak RSS，也不是「以量化距離建圖」的部署形態，照實寫明。
- **FAISS 用移植，不用原生建圖**：
  - 評審 3 要求兩邊的建圖語意一致。移植讓 Q 與 P 用同一張圖，FAISS 的相對效果才能與 Rust 的對照。
  - codec 對齊 QT_8bit，SQ 的距離計算仍是 FAISS 原生的成熟實作。
  - 原生的 IndexHNSWSQ 留作事先登記的備案。
- **記憶體不當成假設**：在同一張圖上，表示本身的記憶體差必然是 3nd − 8d，這是會計恆等式（評審 3）；唯一可能否證它的是 allocator 的行為。所以把它列為一致性檢查，frontier 依 014。
- **H1 採用依維度而不同的預測**（評審 3）：100 維時相依鏈短，每次評估要讀的 cache line 從 4 條減為 1–2 條，讀取延遲的占比較大；兩個維度的相依鏈長度相差 7.7 倍。設計 2「兩個維度都不變」的預測，在 GloVe 上的理論偏弱。判定以 ρ_eval 為準，profiler 的 κ、ν 作機制上的核對。
  - 審查後不再以「SQ8 放得進 L2」作為主要機制：工作集加上 graph（推算約 10 MiB）與 visited set 仍可能 > 16 MiB（背景事實 8）。
- **用 ρ_eval，不用 ρ_QPS**：同一個 ef 下，SQ8 的距離不同，走訪路徑與距離評估次數也跟著不同；單看 QPS 會把兩者混在一起（評審 3 對設計 1 的指正）。
- **固定 e\***：
  - 不在所有 ef 上取 median，因為那會掩蓋 ef 的影響。
  - F 與 Core 逐位元相同，所以 e\* 可以由 Core 的 FP32 結果現在就固定。
- **拆分 kernel**：
  - SPEC 要求以 profiler 解釋；Core 的距離被 inline，profiler 只能指到 `search_layer`（評審 3 對設計 1、2 的指正）。
  - 只在新臂上拆，F 與 S 對稱；R 保留 inline，作為錨點量化路徑失真（評審 3 對設計 3 的指正）。
- **Rust 用 5 個 seed**：
  - Core 的有效 run 中，約 15–20% 的子 process 被標記；一個配對兩邊都乾淨的機率約 0.64–0.72，取 0.68。
  - 3 個 seed 全部乾淨的機率約 0.31；5 個 seed 中至少 3 個乾淨的機率約 0.81。
  - FAISS 只作描述，用 3 個 seed 約省 0.6 h。
- **GloVe 的 ef 延長到 768**：
  - Core 的 rust (16, 200) 在 ef 384 只有 0.968；SQ8 的損失只要超過 0.018，0.95 就會「未達到」（評審 2、3）。
  - 依 013 §2 的悲觀上緣規則取 768。所有臂共用同一份清單，才能在每個 ef 上成對。
- **每份 config 只有一個 build**：
  - 每份約 1 h 以內（使用者要求越短越好），被中止時只損失一份。
  - 成對的臂在同一個 run 內，相隔不超過 2 個子 process；先後以手動位移平衡（013 §5 已記錄單一 build 的 config 沒有輪換）。
- **診斷放在 runner 之外**：
  - runner 只改兩處（`serialized_bytes` 可為 null、relief），量測程式的改動最小。
  - 建圖是確定的，以 ids 逐位元核對與 formal run 綁定，滿足評審 3 要求的「距離評估次數必須與計時路徑的 ids 相同」。
  - 代價是重建 index，約多 30 分鐘的 lock 時間。
- **relief 診斷放在所有量測之後**：
  - 不改變計時與 014 讀值所處的 allocator 狀態（評審 1 對設計 3 的指正）。
  - 同時提供一個不受 allocator 保留影響的讀值（評審 3 的 must_fix 4）；使用者決定不用它下結論，只作診斷（§13）。
- **不保留第 3 個名額**：避免看到 C 的結果之後再追加假設（評審 1）；使用者決定 C 做完就封版（§13）。
- **工程改善與研究隔離**（§10）：使用者允許工程改善並行；凍結量測與被取樣的程式、判定邏輯與 config，並要求 repo 在量測時乾淨，研究結果才不會被並行的改動污染。

## 放棄的替代方案

- **設計 1 的原案**：
  - 內容：3 個 seed、不延長 ef、沒有 F 對照、保留第 3 個名額、看過取樣才分類、以 H1(a) 的會計恆等式當假設、以天花板作 H1(b) 的上界。
  - 成本最低（formal 約 2–2.7 h）。本文件採用它的表示、Core 的改動範圍與 014 的記憶體讀法，其餘依評審意見修正。
- **設計 2 的量化建圖**：
  - 內容：泛型的 `GraphIndex<S>` 與 `trait VectorStore`、新的存檔格式 RHNSWQ、interface 依 adapter 放寬容許值、以 M_relief 作為 H2 的估計量、在 run_child 內計數。
  - 放棄理由：實作約 35–50 h，lock 約 6–7 h；會改動 002 的驗證邏輯；H2 會卡在使用者的決定上。
- **設計 3 的 R2 整數 kernel**（int8×int8→i32）：本機 rustc 預設已啟用 dotprod，LLVM 很可能產生 sdot，實質上就是 Optional 的 SIMD distance kernel。設計 3 的下列部分也一併不採用：
  - 以 steady footprint 取代主要記憶體指標；
  - 在子 process 開始時呼叫 relief；
  - 6 個臂各自重建同一張 FP32 圖；
  - hot-cache 微基準與頻寬 U。
- **只用 ρ_QPS**（不計距離評估次數）：會把每次評估的成本與評估次數混在一起。
- **以所有 ef 成對比值的 median 判定**：會掩蓋 ef 的影響。
- **以 f64 在重建向量上計算天花板**：與 index 實際使用的 f32 非對稱公式不是同一個距離（評審 3）。
- **其他壓縮表示**：
  - 對稱 int8（query 也量化）；
  - global SQ8（`QT_8bit_uniform`）；
  - SQ4 / SQ6；
  - fp16 / bf16：Rust stable 沒有 f16 運算；做劑量反應時，每多一種表示，每個 build 就多 2 個子 process；
  - PQ / OPQ：k-means 多一個 variance 來源；
  - RaBitQ；
  - SQ8 + FP32 rerank：記憶體 ≥ FP32。
- **以 FAISS 原生的 IndexHNSWSQ 作為主要的 Q**：它的 graph 以量化距離建出，與 S 的建圖語意不同，Q ÷ P 與 S ÷ F 就不能對照。保留為備案。
- **FAISS 也跑 5 個 seed**：約多 0.6 h，而 FAISS 只作描述。
- **Rust 只跑 3 個 seed**：約省 1 h，但乾淨配對不足的機率高（見「為什麼」）。
- **以 Core 的 FP32 結果當計時基準**：`MEASUREMENT_PATHS`、協定版本與變壓器都不同，漂移也與判準同一個量級。
- **以 xctrace 加 debug line info 拆出距離計算**：要改 BUILD_INFO、有 B3 的 Xcode 風險，而且需要另寫 decision。
- **在 Core 的 commit 另開 worktree 編譯，比對 Core 函式的反組譯**（設計 2）：屬於重工作，可能與其他專案的正式 run 衝突。改以 C7 與 C4 取代。
- **把距離評估計數與 recall 分解放進 run_child**：可以少重建一次 index（約省 30 分鐘），但量測程式的改動變大，而且 k = ef 的大陣列會進入子 process 的記憶體紀錄。
- **flamegraph**（folded stacks 轉換）：只是呈現方式，判定用的是 top-of-stack 排行。使用者需要時可以另外做，不改變證據。
- **一份 config 包含多個 build**（例如 seeds 2–3 合為一份）：每份變長（悲觀約 1.7 h），被中止時的損失較大。
- **hnswlib**：沒有量化；它的 FP32 frontier 在 Core 已有。
- **M / efC 掃描、scaling、shuffle**：與兩個假設無關；order variance 是建構的性質，Core 已經報告過。

## 對實驗的影響

- **呈現位置**：
  - C 的結果只出現在 C 報告（`results/reports/quant-<資料集>/`）與 README 新增的一節。
  - Core 的報告、Gate 判讀、frontier 與 README 的 Core 段落都不修改。
- **協定**：008-v5 只用於 C。Core 報告永遠不會合併 C 的 run，反之亦然（由登記表與合併檢查保證）。
- **SPEC 最終成果**：quantization frontier 由 C 提供；「profiler 前後」以 F 對 S 的時間分解呈現。
- **事先預期的結果**：
  - GloVe 的 0.99：所有臂都預期「未達到」。
  - H2 在文字上可能「不適用」。
  - GloVe 的 M_014 很可能因 allocator 保留，顯示 S 沒有省到記憶體。這會依 014 照實報告，並列出 M_relief 與 M_struct 來解釋。
  - FAISS 的 quantization frontier 判讀（Q 對 P）依構造約有 2–5 成機率是 `uncertain`：只有 3 個 seed，而每側需要 3 個未標記的 build，任一個子 process 被標記，seed 範圍就算沒有分開。
- **結論範圍**：只限 Rust 的依序純量 kernel、搜尋期間的表示、n = 100k、(16, 200)。build time 與 peak RSS 不會下降。
- **已知風險與對應的處理**：

  | 風險 | 處理 |
  | --- | --- |
  | allocator 保留 | 診斷讀值並列，判定不依賴它 |
  | 複製的走訪與 `#[inline(never)]` 造成的 codegen 差異 | C4 |
  | 估計距離超出範圍（`ValueError` 會讓整個子 process 失敗） | 在 Rust / adapter 內夾住，並以 smoke 驗證 |
  | FAISS 移植與 SQ 的未知行為 | 語意測試，加上事先登記的備案 |
  | 其他專案的負載 | 5 個 seed、單一 build 的 config、「無法判定」的規則 |
  | codes 相同的向量在 SQ8 距離上變成 tie | 以 (distance, id) 決定；照實報告 codes 的重複比例 |
  | B3 的 Xcode 問題復發 | 以 `DEVELOPER_DIR` 繞過 |
  | lock 長時間被占用 | 只能等待，該步驟標記為 blocked |
  | 並行的工程改善污染研究結果 | §10 的隔離規則 |

- **NEXT**：§13 的使用者決定已記到 NEXT；每次 session 結束前更新 NEXT。

## 設計過程

| 設計 | 合規 | 成本 | 研究價值 | 平均 |
| --- | --- | --- | --- | --- |
| 1 成本最小（建完圖才轉換、3 個 seed、2 份 config） | 7 | 8 | 6 | 7.0 |
| 2 統計與協定嚴謹（量化建圖、fp32g 對照、5 個 seed、008-v5） | 8 | 5.5 | 7.5 | 7.0 |
| 3 解釋機制（同一張圖、bytes 與算術分解、6 個臂） | 5 | 5 | 6.5 | 5.5 |

**基礎與移植**：

- 以設計 2 為基礎（三位評審中有兩位選它）。
- 從設計 1 移植：建完圖才轉換、Core 的最小改動範圍、不新增檔案格式、不改 interface、014 的記憶體讀法。
- 從設計 3 移植：kernel 拆分與 R 錨點、ν 診斷、事前固定的分類規則、formal 之前的反組譯關卡、golden test、單一 build 的 config 加手動輪換。

**must_fix 的落實**：

- **登記時點與程式凍結**（評審 1 #1、評審 2 #1、評審 3 #8、#9）：§9 第 1、5、6 步；§5 的凍結規則。
- **受影響 decision 的補記**（評審 1 #2、評審 3 #8）：§11。
- **不默改使用者決定的記憶體指標**（評審 1 #3、評審 2 #11、評審 3 #4）：§6、§13。
- **停止與封版**（評審 1 #4）：§9。
- **Core 不變的證據**（評審 1 #5、評審 2 #5）：§4。
- **同一路徑的 FP32 對照**（評審 1 #6）：F 臂與 C4。
- **分類規則在取樣前固定、gate_profile 預設不變**（評審 1 #7、評審 2 #13）：§8、§4。
- **範圍邊界與 Optional**（評審 1 #8、評審 2 #12、評審 3 #6）：§1、§3。
- **執行環境與單執行緒的診斷**（評審 1 #9、評審 2 #6、#10）：§5、§7、§13。
- **完整判定表、FAISS 的拿掉條件、brute force 的定位**（評審 1 #10、評審 3 #10）：§6、§3、§7。
- **NEXT 的 Blocked**（評審 1 #11）：§13（使用者已決定，記到 NEXT）。
- **距離範圍**（評審 2 #2）：§3、§4。
- **`serialized_bytes` 為 null**（評審 2 #3）：§4。
- **FAISS 的語意測試、codec 對齊與 SIMD 紀錄**（評審 2 #4、評審 3 #5）：§4、§3（`faiss.get_compile_options()` 已寫在 metadata）。
- **偏離的登記**（評審 2 #7）：§5、§11。
- **timeout 與時程**（評審 2 #8、#9）：§5、§12。
- **recall 分解與天花板的公式**（評審 3 #1）：H2、§7。
- **距離評估次數與 cache spillover**（評審 3 #2）：H1、§7。
- **延長 ef 梯子**（評審 3 #3）：§5。
- **假設與一致性檢查分開**（評審 3 #7）：§2。

### 草稿的審查（2026-10-16，任何真實資料上的量化結果之前）

一位獨立審查者逐條核對草稿引用的程式事實（`report.py`、`interface.py`、`run_child.py`、`profile_report.py`、正式報告的數字），
並以 n = 20 的 synthetic 資料試驗 FAISS 的移植。17 個 findings（高 3、中 8、低 6）全部採用，主要修正：

- **判定表有重疊（高）**：H1 整體、H2 各資料集與 H2 整體的條件可以同時成立。三張表都改為「各列依序取第一個符合的列」；H1d 改判「一致 / 不一致」；
  H2 先判「無法判定」、再判「不適用」，並定義「適用」與 H2 的有效 seed（§2）。
- **M_relief 的時點矛盾（高）**：原文同時說「由使用者決定」與「必須在 formal 結果之前決定」。改為使用者在 commit 之前已決定維持 M_014，commit 之後不得更改（§6、§13）。
- **狀態清單不完整**：§6 列出每一列允許的狀態，含「計時成立、profile 不一致」、5 個 frontier 類別、C4 的 path-distorted、C7a / C7b。
- **C7 拆成 C7a（Rust，不一致就停止）與 C7b（FAISS，只讓 FAISS 的讀值不適用）**；Core 的 points 檔只存 ids，distances 由測試保證。
- **H1 的機制**：「SQ8 放得進 L2」不足以作為主要機制（工作集含 graph 仍可能 > 16 MiB），改以每次評估讀的 cache line 數說明；H1b 的不成立解讀補上 graph 與 visited set 的 cache miss。
- **smoke 之後與 formal 之後能修什麼**：correctness bug 限於偏離 §3 的文字定義；移植可不可行只由 synthetic 測試在 smoke 之前定案；只影響診斷函式的 bug 有明確的處理方式（§9）。
- **程式事實的更正**：`_check_compatible` 與 `_check_mergeable` 的分工、`check_search_result` 的實際條件與 `ValueError` 的後果、`profile_report` 的比對對象（`Row.name` / `Row.symbol`）、
  移植後 `is_trained` 仍是 False（n = 20 試驗）、GloVe ef 304 的 median 0.9567。
- **其他**：反組譯前提的機械判準、kernel 誤差界的數值與推導、smoke 一律在 heavy lock 下、H1d 遇到 hygiene 標記時無法判定、FAISS frontier 判讀的事先預期、§11 補記的適用範圍。

修訂時另外補上一點（不是審查者提出的）：100 B 的 SQ8 列依起點在 128 B 邊界上的位置跨 1–2 條 cache line（平均約 1.75 條，不是 1 條），
所以 H1 的陳述寫成「4 條減為 1–2 條」，寬度的理由也補上以 cache line 計的頻寬模型估計（GloVe 約 1.4）。判定門檻不變。

同時寫入使用者 2026-10-16 的決定（§13）與工程改善的隔離規則（§10）；原 §10–§12 依序改為 §11–§13。

## 補記（2026-10-17，實作與審查之後、任何真實資料上的量化結果之前）

本補記涵蓋 §9 的第 3–5 步。到目前為止，量化表示只在 synthetic 資料上執行過；smoke、formal、天花板與 profiling 都還沒有開始。
**假設、預測、門檻、格點、ef、seed、e\*、讀值、判定表、停止條件與主要記憶體指標都沒有改變。**以下只記錄實作層級的事實與 019 沒有寫明的細節。

### 1. 實作與審查（§9 第 2、3 步）

- **commit**：
  - golden test：9cfc35f，在任何 C 的程式之前；
  - 實作：dca22c9；
  - 機器碼比對的 probe：e30784d（見第 5 節）。
- **實作方式**：依序分 5 個階段：Rust、binding、adapter、runner 008-v5、工具與 config。每個階段結束時 `./scripts/test.sh` 都通過，全程只用 synthetic 資料。
- **獨立審查**：三位審查者分別從三個角度檢查：對 019 的忠實度、Core 隔離、測試鑑別力（mutation）。
  - 共 20 個 findings：高 1、中 5、中低 3、低 11。
  - 18 個修正並補上測試；2 個（Core 的機器碼改變）屬實但不改程式，記在第 5 節。
  - 修正後，把審查列出的 mutation 加上新增檢查共 44 個套回工作目錄重跑，全部被測試抓到。
  - 各階段實作者自己做的 mutation：Rust 16 個、binding 15 個、adapter 25 個、runner 18 個、工具 115 個。存活的都已逐一說明是等價 mutation。
- **最高嚴重度的 finding（高）**：
  - 問題：沒有測試釘住「W_X 是 k = ef 的完整候選表」。若誤用 k = 10 的結果，L_rank 與 δ_F 會恆為 0，C2 永遠一致，H2 只要 L_total > 0.00133 就會一律判「不成立」，而且不會有警告。
  - 修正：測試獨立重算 T_X，用 spy 確認 search 以 k = ef 呼叫；診斷輸出加 `w_width`。
- **最後的 `./scripts/test.sh`**（dca22c9）：
  - rustfmt、clippy `-D warnings`、cargo test 都通過；
  - pytest 1024 passed、3 skipped；.venv-embed 16 passed；
  - `tests/test_core_golden.py` 的登記值沒有改，全程綠燈。
- **Core 不變的證據鏈（§4）**：
  1. `git diff 9cfc35f dca22c9` 在 Core 範圍內，只有 `hnsw.rs` 多 `mod quant;` 與一行 `pub use`、`lib.rs` 擴充 re-export。
     Core 範圍指 distance、batch、exact、rng、error、persist、crc32、`Cargo.*`、`interface.py`、`datasets.py`、`metrics.py`，以及 golden test、`test_report.py`、`test_gate_profile.py`。
  2. golden test 與既有測試全部通過。
  3. 隔離審查者以新舊 `report.py` 重產兩份 Core 報告，report.json、report.md 與圖逐位元相同。
  4. 機器碼層級**不是**不變的，見第 5 節。

### 2. FAISS 移植的定案（§3、§9 第 4 步）

- **定案**：移植可行，不採用備案（原生 `IndexHNSWSQ`）。依 §9 第 4 步，之後不得再改用備案；smoke 或 formal 中 C8 不一致時只能記為不適用。
- **依據**：只看 synthetic 測試，全部通過（`tests/test_third_party_semantics.py`，比照 007 標 [測試]）。
  - 6 組資料：多層圖、M = 4、重複向量、n = 17、16、1。移植後的 `levels`、`offsets`、`neighbors`、`entry_point`、`max_level` 都與 Flat 完全相同。
  - 是深複製：改動或釋放 Flat 都不影響 Q。
  - `is_trained`：只 train storage 時，外層 `is_trained` 仍是 False；移植程式設為 True 之後才能 search。
  - `write_index` / `read_index` 之後，graph 與結果逐位元相同。
  - 在量化不改變排序的 sign-cube 資料上，Q 與 Flat 的 ids 相同（前提在測試內驗證）。
- **其他實測語意**（007 的補記）：
  - storage 是 QT_8bit、RS_minmax、`rangestat_arg = 0`；`trained` 等於逐維的 min 與 max − min。
  - vdiff = 0 的維度 code 為 0，解碼為 vmin。
  - similarity 實測可到約 1.0039，不夾住就會被 `check_search_result` 拒絕。
  - 距離 > 2 的情形以 q = −x 的 fixture 確認：先確認原生 similarity < −1，再確認輸出恰好夾到 2.0。
- **codes 一致比例**：
  - synthetic 的 7 組輸入上都是 1.0。
  - 真實資料上可能略低於 1：S 在 Core `add` 重新正規化之後的向量上訓練，Q 在 harness 傳入的向量上訓練。
  - 依 C8，這只描述，不擋流程。

### 3. H1 前提檢查（§2、§9 第 4 步）

- **結果**：`results/quant/premise.json`，**成立**。
  - 檢查對象：binding SHA-256 `a208618a2b0ecf574235809e15739270f04f3e4c46bdcd66c8449aecc73931ad`，git 乾淨（e30784d）。e30784d 的 `crates/`、`python/` 與 dca22c9 相同。
  - `rhnsw_kernel_fp32`：97 條指令。累加暫存器 s0，21 次依序的 `fadd s0, s0, sN`，沒有違反。
  - `rhnsw_kernel_sq8`：241 條指令。累加暫存器 s1，69 次依序累加，加上恰好一次 b + Σ（`fadd s0, s0, s1`），沒有違反。
  - SQ8 kernel 的轉換指令（只作紀錄）：`tbl.16b` 16、`ucvtf.4s` 17、`ucvtf` 1、`ushll.8h` 1、`ushll.4s` 1。
- **機械判準的具體化**（019 只寫了原則）：
  - 累加暫存器是 `fadd sN, sN, sM` 中最常出現的 N；平手算違反。
  - SQ8 允許恰好一次「以累加暫存器為來源、寫入另一個暫存器」的加法（b + Σ），FP32 為零次。
  - 下列都算違反：運算元不全是 s 暫存器的 `fadd`、`faddp`、`fadda`、`fadd.<T>`，以及融合乘加（`fmadd`、`fmla` 等）。
- **與 formal binary 的綁定**：
  - `quant_report` 要求 premise 的 binding SHA-256 等於 C formal run 記錄的 `rust_hnsw` 擴充模組 SHA-256。
  - smoke 或 formal 之前若重新編譯而 bytes 不同，要在那份 binary 上重跑 premise：舊的輸出改名保留，工具拒絕覆寫。

### 4. 019 沒有寫明、由實作決定的細節

**Rust（`crates/hnsw/src/hnsw/quant.rs`）**

- **走訪函式也加 `#[inline(never)]`**（§4 只寫了 kernel），對應審查 finding「中」。
  - 原因：`search_unit_q`、`greedy_search_q`、`search_layer_q` 每個單態化只有一個呼叫點，原本會被整個 inline 進 `QuantIndex::search_batch`，依 §8 第 4 條，F、S 的走訪會全部落到「其他」。
  - 效果：只改函式邊界，讓 F、S 的走訪成為獨立符號（Core 的三個函式本來就是），共 12 個走訪符號，歸類為 traversal。
  - 代價：函式呼叫的開銷，F、S 對稱，由 C4 量化。加上之後 Core 函式的機器碼沒有再改變，前提檢查仍成立。
- **多了 `is_empty()`**：clippy 的 `len_without_is_empty` 在 `-D warnings` 下會讓 test.sh 失敗。
- **計數**：用私有的 `Counter` trait（`NoCount` / `Count`）做成另一個單態化；計時路徑用 `NoCount`，沒有任何計數器。距離 trait 仍然只有 FP32、SQ8 兩個實作。
- **V 的定義**：呼叫 `visited.insert` 的次數，含 entry，不論插入的是不是新元素。「新元素的插入數」等於 layer 0 的 kernel 次數 + 1，可以由 D 推得。
- **`sq8_train_encode` 的邊界**：
  - n = 0 時回傳全為 0 的 vmin / vdiff。
  - 下列情形 panic（binding 先檢查並改成 `ValueError`）：dim = 0、長度不是 dim 的整數倍、含 NaN / Inf、某一維的範圍溢位成 Inf。
  - min / max 照 FAISS 的寫法：從第一列開始，以嚴格的 `<` / `>` 更新。結果與 min / max 相同，只差在 ±0 時保留哪一個。
- **S 每個 query 配置一次 `w`**（d 個 f32），F 沒有這筆配置。
  - 這是 S 本身的非 kernel 工作，反映在 ν 中；019 沒有規定，照原樣保留。
  - 量級：每個 query 一次 d × 4 B 的配置，而每個 query 有數千次 kernel 評估。

**binding（`python/src/lib.rs`）**

- 表示名稱用字串 `"fp32"` / `"sq8"`。沒有建構子，只能用 `build` / `build_pair` 建立。
- **`build_pair` 的 F 與 S 都從 clone 轉換**：`Vec::clone` 會把 capacity 收緊成長度，混用原 index 與 clone 時，F − S 就不等於 3nd − 8d。
  - 先轉 SQ8 再轉 FP32，峰值記憶體比較低。
  - 副作用：`build_pair` 的 M_struct 小於 formal index 的實際結構大小。
- **回傳格式**：
  - `search_stats`：`(ids, distances, kernel_calls, visited_inserts)`，後兩者是 uint64、長度等於 query 數。
  - `memory_accounting`：dict，5 個欄位加 `total_bytes`。
  - `codes()` / `codebook()` 回傳複製；FP32 時回傳 `None`。
- **`sq8_encode`**：不正規化，直接在輸入上訓練，供 §4 對 FAISS `compute_codes` 的比對。

**adapter（`benchmarks/adapters.py`）**

- **兩個新的 module 函式**：`faiss_hnsw_flat`、`faiss_sq8_from_flat`。§7 的診斷要用「同一段建圖程式」重建 Q。
  - `FaissHnsw` 不能改，所以建 Flat 的四行重寫了一次；兩者 graph 相同由測試保證。
- **`omp_set_num_threads(1)`**：放在純 numpy 的檢查之後、任何 FAISS 呼叫之前，與 `FaissHnsw` 的順序相同。
- **合約測試**：「F 與 R 的 adapter 輸出逐位元相同」保留，作為 C7a 的前提（配對正確性），不屬於 019 排除的 FP32 語意等價測試。

**runner 008-v5（`benchmarks/run_child.py`、`run.py`、`protocol.py`）**

- **紀錄格式**：
  - protocol 的新欄位 `memory_relief` 記錄 relief 的時點，寫法比照既有的 `gc` 欄位。
  - build 紀錄保留 `after_relief` 的原始讀數，比照 `after_build`。
- **不記錄**：relief 的回傳值不記錄（synthetic 試驗中它回傳 0，但 rss 確實下降了，回傳值不可靠）。
- **relief 之前不做 `gc.collect`，也不釋放最後一個 ef 點的陣列**。
  - 殘留量：約 1.11 MiB（GloVe）、0.22 MiB（文字）。
  - 所有臂相同，在 F − S 中抵消。
- **`PROTOCOL_VERSION` 是全域常數**：之後任何 run 都記為 008-v5。Core 不再跑新的 run，既有報告讀的是 run.json 裡記錄的版本，不受影響。

**報告與診斷工具（`quant_report`、`quant_diag`、`quant_profile`、`quant_smoke`）**

- **輸出路徑**：
  - 報告：`results/reports/quant-glove/`、`quant-text/`，兩份列出同一張判定表；
  - 診斷：`results/quant/diag-<glove|text>.json`；
  - 前提檢查：`results/quant/premise.json`；
  - 取樣：`results/profiles/quant-<glove|text>-<實作>-ef<ef>`（S 有兩次取樣，所以名稱帶 ef）。
- **門檻比較一律用精確分數**：recall 用命中數 / (nq × k)；QPS 與 D 用浮點值本身的精確分數；門檻用十進位精確值。
  - 例：0.9007 − 0.8967 以浮點計算會判成「不適用」，以精確分數則是「成立」。
- **合併條件比 019 寫明的嚴格**：
  - 只讀 formal run，兩個資料集都要有；
  - 每個 run 的協定是 008-v5、電力模式是 2；
  - 兩個資料集之間也比對 metadata 與量測程式（§5「整組 C、同一個變壓器」）；
  - 每個資料集內照 `_check_mergeable` 檢查。
- **完整性**：`quant_diag` 與 `quant_profile plan/check` 都要列出全部 10 份 C formal run，再以 `--dataset` 指定資料集；兩個資料集各 5 份必須都是完整的 C formal set（§7「在全部 10 份完成之後」）。
- **`--core-runs`** 必須恰好列出上表登記的 4 個 Core run。
- **條件式狀態的細部**（都在 §6 允許的狀態內）：
  - C7a / C7b：C 自己的點缺失（`c_missing`）也記為無法判定；019 只寫了 Core 的 points 檔缺失。
  - C8：無法判定的原因分三種，`ids_mismatch`（019 寫明的情形）、`formal_ids_missing`、`diag_missing`。validator 要求各 seed 的狀態與原因一致。
  - C4：遇到 hygiene 標記或點缺失時記為無法判定，比值照列。
  - C1：排除有 memory / swap 標記的 build，F − S 與 P − Q 分列；所有 C 列都依資料集分列。
  - FAISS 的配對讀值要求 C7b 為「一致」。
- **有效性的細部**：
  - 「formal recall 存在」排除結果不足 k 的點。
  - (seed, 臂) 有效，要求沒有任何不符的點，而且至少有一個相符的點。
  - D 只在該 ef 的 formal 點相符時有效。
- **方向**：ρ = SQ8 ÷ FP32，Δ = FP32 − SQ8。frontier 的比值除了 QPS，也列出 P50、P95 與 M_014。
- **報告新增的描述性輸出**：
  - 每 build 記憶體表加「M_struct（診斷）」一欄，並註明來源是 `build_pair` 的 clone；
  - 各 ef 的 Recall@10、tie-aware Recall@10、Recall@1 表；
  - 其他 ef 的 ρ_QPS、ρ_eval 與 e\* 分屬 1 的兩側時加粗標出（§2「方向不一致時照實寫明」）。
- **診斷的細部**：
  - c 的 Recall@1 取 k = 10 exact 結果的第一欄。
  - 天花板只用 seed 1 的 S；各 seed 的 codes 相同，由 SHA-256 核對。
  - Q 的結構大小另外加上 codebook。
  - 診斷不計時，所以不檢查電力條件；`machine_quiet` 由操作者在取 lock 之前執行。
- **S 的同 recall 取樣**（§8 第 4 項）：0.95 與 0.90 都沒達到時略過。profiling 的 check 另外要求高效能模式，以及與 formal run 同一個變壓器（§5）。
- **smoke（§9 第 6 步）**：新增 `benchmarks/configs/glove-quant-smoke.toml`、`text-quant-smoke.toml` 與 `benchmarks/quant_smoke.py`。
  - config 照 §9 第 6 步：seed 1、五個臂；ef 是 GloVe {64, 304, 768}、文字 {24, 48, 512}，由測試對照 019 的原文。
  - 臂的順序與 timeout 沿用 seed 1 的 formal config（019 沒有寫明）。
  - `quant_smoke` 沿用 `c7_core`，核對 F、R、P 與 Core 的 ids，以及 C 內 F 對 R。
  - `--port` 在 heavy lock 下重建 Q，核對 graph 與 e\* 的 ids。
- **「只影響診斷函式的 Rust bug」例外**（§9）：沒有預先實作，需要時另寫補記再加參數。

### 5. Core 的機器碼改變（審查 finding「中」與「低」各一，不改程式）

**事實**：

- Core 的原始碼沒有改，但加入 quant 模組改變了 hnsw crate 的 codegen：CGU 的內容變了，LLVM 的 inline 決策也跟著變。
- 紀錄：`results/quant/core-codegen.json`，由 `scripts/probe_core_codegen.py 9cfc35f dca22c9` 產生。
  - 9cfc35f 的 `crates/`、`python/`、`Cargo.*` 與 Core formal（f077de9、4acf799）及 Gate profiling（5af774d）的 commit 相同。
  - 只編譯零相依的 hnsw crate，各約數秒；同一個 commit 自己比對時，101 個函式全部相同（工具的對照）。
- 結果：quant 模組以外的函式，86 個相同、7 個不同；另有 Core 端 8 個函式消失、C 端出現 25 個新函式（多為 drop glue、iterator 與 heap 的 helper）。不同的 7 個是：

  | 函式 | 指令數（Core → C） |
  | --- | --- |
  | `HnswIndex::search_layer` | 418 → 569 |
  | `HnswIndex::search_batch` | 245 → 185 |
  | `HnswIndex::add` | 1485 → 1222 |
  | `BinaryHeap<Candidate>::into_sorted_vec` | 84 → 31 |
  | `quicksort<Candidate>` | 299 → 2071 |
  | `exact` 的 `Map<IntoIter<Candidate>>` | 92 → 93 |
  | `persist::temp_path` | 124 → 128 |

- 在連結後的 binding 上（`otool -tV`），Core 的 `search_layer` 只剩一次 `bl HashMap<u32, ()>::insert`（迴圈之前的 entry）：
  - 迴圈內的 `visited.insert` 已 inline，連帶出現 `make_hasher`；
  - `into_sorted_vec` 也 inline 了（改為呼叫 `sift_down_range`）；
  - `push_padded_row` 反而不再 inline 進 `search_batch`。
- 不變的部分：
  - `search_unit`、`greedy_search`、`unit_rows`、`ExactIndex` 的函式完全相同。
  - 輸出不受影響：依序的 f32 加總沒有被重排，golden test 與 F ≡ R 的等價測試都通過；真實資料上由 C7a 核對。

**評估過恢復成與 Core 相同，結論是不可行**：

- 把 quant 模組全部標成 `#[inline]` 之後，Core 仍有 6 個函式不同：`push_padded_row`、`unit_rows`、`check_k` 被 downstream 引用之後，linkage 改變了。
- 要完全恢復，就得在 quant.rs 複製 `check_k`、`unit_rows`、`normalize`、`push_padded_row`，連 kernel 都要改寫。這違反 §3「kernel 內部呼叫 `cosine_distance_unit`」與 §4「沿用」的規定。
- CGU 依大小合併，這種做法很脆弱。
- 修改 `hnsw.rs` 或 Cargo profile（例如 codegen-units）會違反 §4，也會再改變 Core 的 codegen 與 BUILD_INFO，所以也不採用。

**對讀值的影響（判定規則都不變）**：

- **H1、H2 不受影響**：兩者比的是 F 與 S，在同一個 binary 中、走同一份複製的走訪程式。
- **R 的描述**：C 的 R 與 Core formal、Gate profiling 用的 binary「原始碼相同，機器碼不同（inline 決策改變）」。§3「Core 的 adapter 與程式，一行不改」只對原始碼成立。README 的 C 一節照這樣寫。
- **C5（Gate 重現，只描述）**：
  - Gate profiling 中，`HashMap::insert` 是獨立符號：GloVe 占 35.8%（visited set 合計 43.6%），文字占 4.6%（合計 6.4%）。
  - C 的 binary 中，它 inline 進 `search_layer`，依 §8 的分類規則會從 visited set 改歸 traversal。
  - 所以預期 C5 依構造會判成「不一致」：GloVe 的 visited set 預期會遠低於 Gate 的 43.6%，文字約少 4.6 個百分點，兩者都超出 3 個百分點的容許差。
  - C5 照登記的規則判定；報告必須註明已知原因是 inline 決策改變，不得寫成「Gate 無法重現」。
- **C6（漂移，只描述）**：讀為「時間漂移加上 codegen 差異」，兩者無法分開。R 的 build time 也照同樣方式註明。
- **C4（路徑失真）**：QPS_F / QPS_R 中的 R 是 C binary 裡的 Core 函式；規則不變，比值同樣包含 codegen 的差異。
- **「放棄的替代方案」的更正**：
  - 原文以「屬於重工作」為由，不做 Core 函式的反組譯比對。實際上只編譯零相依的 hnsw crate，各約數秒，不是重工作。
  - 現在把它加為描述性紀錄（本節），不取代 C7 與 C4，也不新增任何判定。
- **之後的核對**：
  - C formal 開始之前，若 `crates/` 有任何改動，要以 formal 的 commit 重跑 `scripts/probe_core_codegen.py`，舊的紀錄改名保留。
  - formal 的 commit 與 dca22c9 的 `crates/` 相同時，本紀錄直接適用。

## 補記（2026-10-17，correctness smoke 之後、第一份 C formal 之前）：§9 第 6 步

- **執行**：
  - 等待腳本連續 3 次確認 `machine_quiet` 成立（`--powermode 2`）之後，在 heavy lock 下以 smoke 模式執行：
    `./scripts/formal_launcher.sh … --powermode 2 --skip-idle --mode smoke`。
  - run 的條件：commit 3b69923、git 乾淨、協定 008-v5；binding SHA-256 `a208618a…`，與 premise 相同（補記第 3 節）。
- **run**：兩份都是 complete，10 個子 process 全部 ok、沒有任何 hygiene 標記。

  | 資料集 | run | 時段（UTC） |
  | --- | --- | --- |
  | GloVe | `20261016T195505Z-smoke` | 2026-10-16 19:55–20:10 |
  | 文字 | `20261016T201035Z-smoke` | 20:10–20:25 |

- **核對**（`benchmarks.quant_smoke`，只檢查 §9 第 6 步列出的項目）：
  - **F、R、P 與 Core 的 ids**：F、R、P 與 §2 登記的 Core run，在 seed 1 的共同 ef 上比對；另比對 C 內 F 與 R 的所有 ef。
    全部逐位元相同：GloVe 9 組、文字 12 組。
  - **Q 的移植**（`--port`，再次確認 `machine_quiet` 之後在 heavy lock 下，20:31–20:33 UTC）：
    以 Q adapter 的同一段建圖程式重建 seed 1；在 `del flat` 之前，graph 與 Flat 完全相同（兩個資料集都沒有差異），重建的 Q 在 e\* 的 ids 與 smoke run 相同。
  - **結果格式、010 §6、null call 與 warm-up**：由 runner 在 smoke 模式本身檢查，兩份都通過（status complete）。
- **timeout**（§5 的規則：最慢的臂在負載下的上限 × 5，進位到 600 的倍數）：
  - 外推方式：以 smoke 每個 ef 點的時間，依「每點時間與 ef 成正比」外推到 formal 的 10 個 ef 點，再加上 build 與其他時間。
  - 最慢的都是 F：
    - GloVe 約 530 s → 530 × 1.25 × 5 ≈ 3,313 → 3,600。登記值是 4,200（只有 Rust 的 config）與 6,000（含 FAISS）。
    - 文字約 396 s → 396 × 1.25 × 5 ≈ 2,473 → 3,000。登記值是 3,600。
    - 再乘上 §5 的 Rust 不確定倍數 1.2，GloVe 約 3,976 → 4,200、文字約 2,968 → 3,000，仍然不超過登記值。
  - 結論：都不需要上調，10 份 config 維持登記的內容。
- **看到 smoke 之後的觀察（不是結論；smoke 的數字不得用於任何結論）**：
  - GloVe 三個 ef 點的搜尋時間，F 合計約 197 s、R 約 163 s，F 比 R 長約兩成；文字兩者差不多（約 81 s 與 82 s）。
  - 這正是 C4（路徑失真）要量的東西，可能的來源是 kernel 的函式呼叫（`#[inline(never)]`，§4）。
  - smoke 之後依第 6 步不得修改程式（這不是 correctness bug）。formal 的 C4 若判 path-distorted，H1 照登記的規則在新路徑內判定，不得寫成 Core 路徑上的效果。
  - H1、H2 比的是 F 對 S，兩者的函式邊界相同，所以不受影響。
- **變壓器**：smoke 時接的是 Apple 140W。依 §5，C 報告以第一份 C formal run 實際使用的變壓器為準。

## 附錄：formal config（逐字登記）

在實作 commit 中加入 `benchmarks/configs/` 的檔案，內容必須與下列範本代入後完全相同；唯一的例外是依 §5 上調的 `child_timeout_s`。`benchmarks/quant_report.py` 的 `QUANT_CONFIGS` 恰好登記這 10 份，每個資料集 5 份。

範本：

```toml
# Advanced C（decision 019）：<資料集> n = 100,000、(M, efC) = (16, 200)、seed <s>、原始順序。
# 單一 build：implementations 的順序就是執行順序（019 §5 的手動輪換）。
# 本檔內容必須與 decision 019 附錄逐字相同（child_timeout_s 只能依 019 §5 上調）。
schema_version = 1
dataset = "<資料集>"
implementations = <implementations>
ef_search = <ef_search>
child_timeout_s = <timeout>

[[sweep]]
n = [100000]
m = [16]
ef_construction = [200]
seeds = [<s>]
shuffles = []
```

`<ef_search>` 的值：

- `glove-100-angular`：`[64, 96, 128, 152, 192, 256, 304, 384, 512, 768]`
- `simplewiki-embeddinggemma2-100k`：`[24, 32, 48, 64, 96, 128, 192, 256, 384, 512]`

| 檔案 | `<資料集>` | `<s>` | `<implementations>` | `<timeout>` |
| --- | --- | --- | --- | --- |
| `quant-glove-seed-1.toml` | glove-100-angular | 1 | `["rust-hnsw-sq8", "faiss-hnsw", "rust-hnsw-fp32g", "faiss-hnsw-sq8", "rust-hnsw"]` | 6000 |
| `quant-glove-seed-2.toml` | glove-100-angular | 2 | `["faiss-hnsw", "rust-hnsw-fp32g", "faiss-hnsw-sq8", "rust-hnsw-sq8"]` | 6000 |
| `quant-glove-seed-3.toml` | glove-100-angular | 3 | `["rust-hnsw-fp32g", "faiss-hnsw-sq8", "rust-hnsw-sq8", "faiss-hnsw"]` | 6000 |
| `quant-glove-seed-4.toml` | glove-100-angular | 4 | `["rust-hnsw-sq8", "rust-hnsw-fp32g"]` | 4200 |
| `quant-glove-seed-5.toml` | glove-100-angular | 5 | `["rust-hnsw-fp32g", "rust-hnsw-sq8"]` | 4200 |
| `quant-text-seed-1.toml` | simplewiki-embeddinggemma2-100k | 1 | `["rust-hnsw-sq8", "faiss-hnsw", "rust-hnsw-fp32g", "faiss-hnsw-sq8", "rust-hnsw"]` | 3600 |
| `quant-text-seed-2.toml` | simplewiki-embeddinggemma2-100k | 2 | `["faiss-hnsw", "rust-hnsw-fp32g", "faiss-hnsw-sq8", "rust-hnsw-sq8"]` | 3600 |
| `quant-text-seed-3.toml` | simplewiki-embeddinggemma2-100k | 3 | `["rust-hnsw-fp32g", "faiss-hnsw-sq8", "rust-hnsw-sq8", "faiss-hnsw"]` | 3600 |
| `quant-text-seed-4.toml` | simplewiki-embeddinggemma2-100k | 4 | `["rust-hnsw-sq8", "rust-hnsw-fp32g"]` | 3600 |
| `quant-text-seed-5.toml` | simplewiki-embeddinggemma2-100k | 5 | `["rust-hnsw-fp32g", "rust-hnsw-sq8"]` | 3600 |