# Rust HNSW SPEC

## 一句話問題

哪些演算法與低階實作決策，真正決定 HNSW 的 recall、延遲與記憶體取捨？

---

# Core

## 1. Rust 從零實作

至少：

- insert
- search
- save
- load

距離固定 cosine。

## 2. Python Binding

Python binding 放在 Core。

目的：

- Rust 實作
- hnswlib
- FAISS HNSW

三者都從同一層 Python batch interface 被測。

避免不同 FFI / 呼叫開銷污染比較。

## 3. Correctness

Ground truth 使用 exact search。

另外建立小型、人工可驗證案例。

efSearch 增加時：

- Recall 整體應改善或飽和。
- 不要求每個點嚴格單調。
- 不應出現系統性惡化。

## 4. Baselines

- brute force：ground truth
- hnswlib：成熟 ANN baseline
- FAISS HNSW：成熟 ANN baseline

正式 performance comparison 要在相近 Recall@K 下比較。

## 5. Dataset

至少：

- 一組標準 ANN 真實資料集
- 一組真實文本經現代 embedding model 得到的向量

Synthetic random vectors 只用於：

- correctness
- stress test

不作主要 transferable conclusion。

## 6. Parameter Sweep

至少：

- M
- efConstruction
- efSearch

## 7. Metrics

- Recall@1
- Recall@10
- P50
- P95
- QPS
- build time
- peak RSS
- index size
- scaling with N
- seed / insertion order variance

---

# Advanced

前提：

- correctness 已通過。
- 同 recall 下與 hnswlib 有差距。
- profiling 已顯示主要瓶頸。

Advanced 最多做三個假設驅動實驗。

## A. Memory Representation

比較：

- per-node allocation
- arena / contiguous layout
- vector / graph memory layout

## B. Optimization Ablation

兩個方向都做：

- 從 baseline 逐項加入 optimization
- 從 optimized version 逐項拿掉 optimization

不要用乘法假設各因素獨立。

差異本身就是 interaction effect。

## C. Vector Quantization

例如 FP32 vs 壓縮表示。

研究：

- memory
- recall
- latency

三者 frontier。

搭配 profiler / flamegraph 解釋。

---

# Optional

- SIMD distance kernel
- parallel index construction
- concurrent queries
- ARM M4 Pro vs x86 Ryzen
- 更完整 Python API

---

# 不做什麼

- 不做 Pinecone。
- 不做 distributed vector DB。
- 不做 metadata engine。
- 不要求超越成熟 library。

---

# GitHub 最終成果

README 要有：

- recall-latency frontier
- recall-memory frontier
- quantization frontier
- scaling
- profiler 前後
- 與 hnswlib / FAISS 的公平比較

最有價值的結論可能是：

> 自己的實作仍慢 X 倍，但 profiling 顯示主要差距來自某些 representation / distance kernel / cache 行為。

---

# Repo 建議結構

```text
rust-hnsw/
├── CLAUDE.md
├── SPEC.md
├── NEXT.md
├── crates/
│   └── hnsw/
├── python/
├── benchmarks/
├── datasets/
├── scripts/
├── tests/
├── tools/
│   └── with-mac-lock
└── docs/
    ├── EXPERIMENT_RULES.md
    └── decisions/
```

---

# Day 1

1. 建 Rust workspace。
2. 建 cosine distance 與 exact brute-force search。
3. 寫 correctness tests。
4. 建 Python binding skeleton。
5. 固定 benchmark interface。

---

# Week 1

1. exact baseline。
2. HNSW skeleton。
3. 小型 hand-verifiable test。
4. Python binding 能批次 query。
5. hnswlib / FAISS 同介面。
6. 跑第一個小型真實 dataset。
7. 產生第一張 recall vs efSearch 圖。

正式 latency benchmark 前，確認機器處於乾淨狀態。

---

# Gate

## 進 Advanced

- correctness 通過。
- Recall 隨 efSearch 整體改善 / 飽和，沒有系統性惡化。
- 同 recall 下與 hnswlib 有差距。
- profiler 已顯示時間主要花在哪。

## 特例

若 Core 已很接近 hnswlib：

- 不必硬做 optimization attribution。
- Advanced 改聚焦 vector quantization frontier。
