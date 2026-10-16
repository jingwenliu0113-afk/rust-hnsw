//! Advanced C 的向量表示（decision 019 §3、§4）：在 Core 建完的 FP32 圖上轉換表示，只供搜尋。
//!
//! - `HnswIndex::into_quant` 把 nodes 與 entry point 原樣移入 `QuantIndex`；FP32 直接移入向量，
//!   SQ8 先以 `sq8_train_encode` 訓練、編碼，再釋放 FP32 向量。Core 的 insert 一行不動。
//! - 走訪（`search_unit_q` / `greedy_search_q` / `search_layer_q`）是 Core 的 `search_unit` /
//!   `greedy_search` / `search_layer` 的逐行複製：`HashSet`、`BinaryHeap` 與 `Candidate` 排序全部相同，
//!   只把距離換成本模組私有的 `Kernel` trait（只有 FP32、SQ8 兩個實作）。
//! - kernel 是 `#[inline(never)]` 的 `rhnsw_kernel_fp32` 與 `rhnsw_kernel_sq8`，讓 profiler 能把時間
//!   歸屬到 kernel（019 §8）；`#[inline(never)]` 只改變函式邊界，不改變算術。
//! - 三個走訪函式也是 `#[inline(never)]`：否則它們的每個單態化都只有一個呼叫點，會被整個 inline 進
//!   `QuantIndex::search_batch`，profiler 看不到 019 §8 第 4 條的 `search_layer` / `greedy_search` / `search_unit`
//!   （走訪會落到「其他」）。Core 的三個函式本來就是獨立的符號，這樣 F、S 的函式邊界與 Core 相同；
//!   代價（函式呼叫的開銷）F、S 對稱，由 C4 量化。019 §4 沒有寫明這一點，實作審查後加上（記入 019 的補記）。
//! - 計數（`search_stats`）是另一個單態化（`Count`）；計時路徑用的 `NoCount` 沒有任何計數器。
//! - 不做 save / load，也不能 add。

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::{Candidate, HnswIndex, Node, push_padded_row};
use crate::batch::{check_k, unit_rows};
use crate::distance::cosine_distance_unit;
use crate::{BatchResult, HnswError, Neighbor, NodeId};

/// 搜尋期間的向量表示（decision 019 §3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// FP32 unit vectors 原樣移入（臂 F，`rust-hnsw-fp32g`）。
    Fp32,
    /// per-dim、min–max 的 8-bit scalar quantization（臂 S，`rust-hnsw-sq8`）。
    Sq8,
}

/// 只能搜尋的 index：Core 建出的 graph，加上 `Representation` 指定的向量表示（decision 019 §4）。
#[derive(Debug, Clone)]
pub struct QuantIndex {
    dim: usize,
    nodes: Vec<Node>,
    entry_point: Option<NodeId>,
    store: Store,
}

#[derive(Debug, Clone)]
enum Store {
    Fp32(Fp32Store),
    Sq8(Sq8Store),
}

/// row-major 的 FP32 unit vectors，就是 `HnswIndex` 的 `vectors`。
#[derive(Debug, Clone)]
struct Fp32Store {
    dim: usize,
    vectors: Vec<f32>,
}

/// SQ8：只存 codes（`n·d` 個 u8，row-major）與 codebook（vmin、vdiff 各 d 個 f32）。
#[derive(Debug, Clone)]
struct Sq8Store {
    dim: usize,
    vmin: Vec<f32>,
    vdiff: Vec<f32>,
    codes: Vec<u8>,
}

/// `search_stats` 的每個 query 計數（decision 019 §4、§7），順序與 query 的 row 相同。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchStats {
    /// 呼叫距離 kernel 的次數：所有層，含 entry point（019 §2 的 D）。
    pub kernel_calls: Vec<u64>,
    /// 呼叫 visited set `insert` 的次數：含 entry，不論插入的是不是新元素（019 §7 的 V）。
    pub visited_inserts: Vec<u64>,
}

/// `QuantIndex::memory_accounting` 的結果（decision 019 §7 的 M_struct）。
///
/// 每個 Vec 以 capacity × 元素大小計算，巢狀 Vec 的 header 算在外層的元素大小內；
/// 與 allocator 無關（不含 allocator 的 metadata、對齊與保留量）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccounting {
    /// `QuantIndex` 本身（含頂層各 Vec 的 header）。
    pub struct_bytes: usize,
    /// graph：nodes，以及每個節點各層的鄰居清單。
    pub graph_bytes: usize,
    /// FP32 向量；SQ8 為 0。
    pub vectors_bytes: usize,
    /// SQ8 codes；FP32 為 0。
    pub codes_bytes: usize,
    /// SQ8 codebook（vmin 與 vdiff）；FP32 為 0。
    pub codebook_bytes: usize,
}

impl MemoryAccounting {
    pub fn total_bytes(&self) -> usize {
        self.struct_bytes
            + self.graph_bytes
            + self.vectors_bytes
            + self.codes_bytes
            + self.codebook_bytes
    }
}

/// SQ8 codec 的訓練與編碼（decision 019 §3）。刻意對齊 FAISS 的 `QT_8bit` + `RS_minmax`、`rangestat_arg = 0`。
///
/// - 訓練：逐維計算 `vmin_j = min`、`vmax_j = max`（從第一列開始，以嚴格的 `<`、`>` 更新，與 FAISS 相同），
///   `vdiff_j = vmax_j − vmin_j`（f32）；不外擴範圍、不使用 query、不需要 seed。
/// - 編碼（f32）：`vdiff_j = 0` 時 code = 0；否則 `t = (x − vmin_j) / vdiff_j` 夾到 `[0, 1]`，
///   code = `(255.0 * t) as u8`（向零截斷）。
/// - 回傳 `(vmin, vdiff, codes)`；codes 是 row-major 的 `n·d` 個 u8。`n = 0` 時 vmin、vdiff 全為 0（019 沒有定義這個情形）。
///
/// `into_quant` 傳入的是 `add` 正規化之後的 unit vectors；codec 本身不要求單位長度。
///
/// # Panics
///
/// `dim = 0`、長度不是 `dim` 的整數倍、含 NaN / Inf，或某一維的 `vmax − vmin` 溢位成 Inf 時 panic：
/// 這些都是呼叫端的錯誤（`into_quant` 的輸入已由 `add` 驗證）。
pub fn sq8_train_encode(unit_rows: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    assert!(dim > 0, "sq8_train_encode: dimension must be positive");
    assert!(
        unit_rows.len().is_multiple_of(dim),
        "sq8_train_encode: buffer length {} is not a multiple of dimension {dim}",
        unit_rows.len()
    );
    assert!(
        unit_rows.iter().all(|x| x.is_finite()),
        "sq8_train_encode: values must be finite"
    );

    let mut rows = unit_rows.chunks_exact(dim);
    let Some(first) = rows.next() else {
        return (vec![0.0; dim], vec![0.0; dim], Vec::new());
    };
    let mut vmin = first.to_vec();
    let mut vmax = first.to_vec();
    for row in rows {
        for ((&x, lo), hi) in row.iter().zip(&mut vmin).zip(&mut vmax) {
            if x < *lo {
                *lo = x;
            }
            if x > *hi {
                *hi = x;
            }
        }
    }
    let vdiff: Vec<f32> = vmax.iter().zip(&vmin).map(|(&hi, &lo)| hi - lo).collect();
    assert!(
        vdiff.iter().all(|d| d.is_finite()),
        "sq8_train_encode: the range of a dimension overflows f32"
    );

    let mut codes = Vec::with_capacity(unit_rows.len());
    for row in unit_rows.chunks_exact(dim) {
        codes.extend(
            row.iter()
                .zip(&vmin)
                .zip(&vdiff)
                .map(|((&x, &lo), &diff)| encode_component(x, lo, diff)),
        );
    }
    (vmin, vdiff, codes)
}

/// 單一分量的編碼，全部以 f32 計算（019 §3）。
fn encode_component(x: f32, vmin: f32, vdiff: f32) -> u8 {
    if vdiff == 0.0 {
        return 0;
    }
    let t = ((x - vmin) / vdiff).clamp(0.0, 1.0);
    (255.0 * t) as u8
}

/// FP32 的距離 kernel：就是 Core 的 `cosine_distance_unit`（`1 − dot`，依序以 f32 累加）。
/// 算術與 Core 相同，所以距離逐位元相同；只有函式邊界不同（019 §3、§8）。
#[inline(never)]
fn rhnsw_kernel_fp32(query: &[f32], vector: &[f32]) -> f32 {
    cosine_distance_unit(query, vector)
}

/// SQ8 的非對稱距離 kernel（019 §3）：`1 − (b + Σ_j w_j · code_j)`。
/// Σ 的寫法與 `distance::dot` 相同（`iter().zip().map().sum::<f32>()`，依序加總，不手寫 SIMD）。
/// 回傳原始值，不夾住；夾住只在輸出時做（`Sq8Store::output`）。
#[inline(never)]
fn rhnsw_kernel_sq8(w: &[f32], b: f32, code: &[u8]) -> f32 {
    debug_assert_eq!(w.len(), code.len());
    let sum: f32 = w.iter().zip(code).map(|(&w, &c)| w * f32::from(c)).sum();
    1.0 - (b + sum)
}

/// 走訪用的距離（019 §4）：只在本模組內使用，只有 FP32 與 SQ8 兩個實作。
trait Kernel {
    /// 每個 query 在搜尋前先算好的資料：FP32 是 unit query 本身，SQ8 是 w 與 b。
    type Query<'q>;

    fn prepare<'q>(&self, unit_query: &'q [f32]) -> Self::Query<'q>;

    /// 走訪與排序用的原始距離。
    fn distance(&self, query: &Self::Query<'_>, id: NodeId) -> f32;

    /// 輸出時的距離：FP32 不變（與 Core 逐位元相同），SQ8 夾到 `[0, 2]`。夾住是單調的，排序不變。
    fn output(distance: f32) -> f32;
}

impl Kernel for Fp32Store {
    type Query<'q> = &'q [f32];

    fn prepare<'q>(&self, unit_query: &'q [f32]) -> &'q [f32] {
        unit_query
    }

    #[inline]
    fn distance(&self, query: &&[f32], id: NodeId) -> f32 {
        let start = id as usize * self.dim;
        rhnsw_kernel_fp32(query, &self.vectors[start..start + self.dim])
    }

    #[inline]
    fn output(distance: f32) -> f32 {
        distance
    }
}

/// SQ8 的 query 前置資料（019 §3）。
struct Sq8Query {
    /// `w_j = q_j · (vdiff_j / 255)`，以 f32 計算。
    w: Vec<f32>,
    /// `b = Σ_j q_j · (vmin_j + 0.5 · vdiff_j / 255)`，以 f64 累加後轉 f32。
    b: f32,
}

impl Kernel for Sq8Store {
    type Query<'q> = Sq8Query;

    fn prepare(&self, unit_query: &[f32]) -> Sq8Query {
        let w = unit_query
            .iter()
            .zip(&self.vdiff)
            .map(|(&q, &vdiff)| q * (vdiff / 255.0))
            .collect();
        let b = unit_query
            .iter()
            .zip(&self.vmin)
            .zip(&self.vdiff)
            .map(|((&q, &vmin), &vdiff)| {
                f64::from(q) * (f64::from(vmin) + 0.5 * f64::from(vdiff) / 255.0)
            })
            .sum::<f64>() as f32;
        Sq8Query { w, b }
    }

    #[inline]
    fn distance(&self, query: &Sq8Query, id: NodeId) -> f32 {
        let start = id as usize * self.dim;
        rhnsw_kernel_sq8(&query.w, query.b, &self.codes[start..start + self.dim])
    }

    #[inline]
    fn output(distance: f32) -> f32 {
        distance.clamp(0.0, 2.0)
    }
}

/// 走訪的計數：`NoCount` 給計時路徑（編譯後沒有任何計數器），`Count` 給 `search_stats`。
trait Counter {
    fn kernel_call(&mut self);
    fn visited_insert(&mut self);
}

struct NoCount;

impl Counter for NoCount {
    #[inline(always)]
    fn kernel_call(&mut self) {}

    #[inline(always)]
    fn visited_insert(&mut self) {}
}

#[derive(Default)]
struct Count {
    kernel_calls: u64,
    visited_inserts: u64,
}

impl Counter for Count {
    fn kernel_call(&mut self) {
        self.kernel_calls += 1;
    }

    fn visited_insert(&mut self) {
        self.visited_inserts += 1;
    }
}

/// Core 的 `distance_to`：每次呼叫 kernel 都經過這裡，計數才不會漏。
#[inline(always)]
fn distance_to<K: Kernel, C: Counter>(
    kernel: &K,
    query: &K::Query<'_>,
    id: NodeId,
    counter: &mut C,
) -> f32 {
    counter.kernel_call();
    kernel.distance(query, id)
}

impl HnswIndex {
    /// 把建好的 index 轉成只能搜尋的 `QuantIndex`（decision 019 §4）。
    ///
    /// nodes 與 entry point 原樣移入，graph 不變。`Fp32` 直接移入向量；
    /// `Sq8` 先以這些 unit vectors 訓練並編碼（`sq8_train_encode`），再釋放 FP32 向量。
    pub fn into_quant(self, representation: Representation) -> QuantIndex {
        let HnswIndex {
            dim,
            vectors,
            nodes,
            entry_point,
            ..
        } = self;
        let store = match representation {
            Representation::Fp32 => Store::Fp32(Fp32Store { dim, vectors }),
            Representation::Sq8 => {
                let (vmin, vdiff, codes) = sq8_train_encode(&vectors, dim);
                drop(vectors);
                Store::Sq8(Sq8Store {
                    dim,
                    vmin,
                    vdiff,
                    codes,
                })
            }
        };
        QuantIndex {
            dim,
            nodes,
            entry_point,
            store,
        }
    }
}

impl QuantIndex {
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn representation(&self) -> Representation {
        match self.store {
            Store::Fp32(_) => Representation::Fp32,
            Store::Sq8(_) => Representation::Sq8,
        }
    }

    /// SQ8 的 codes（row-major，`n·d` 個 u8）；FP32 為 `None`。
    pub fn codes(&self) -> Option<&[u8]> {
        match &self.store {
            Store::Fp32(_) => None,
            Store::Sq8(store) => Some(&store.codes),
        }
    }

    /// SQ8 的 codebook `(vmin, vdiff)`，各 d 個 f32；FP32 為 `None`。
    pub fn codebook(&self) -> Option<(&[f32], &[f32])> {
        match &self.store {
            Store::Fp32(_) => None,
            Store::Sq8(store) => Some((&store.vmin, &store.vdiff)),
        }
    }

    /// 目前的 entry point；空 index 為 `None`。
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// 節點的最高層；id 不存在為 `None`。
    pub fn level(&self, id: NodeId) -> Option<usize> {
        self.nodes.get(id as usize).map(|n| n.neighbors.len() - 1)
    }

    /// 節點在某一層的鄰居；id 不存在或高於該節點的 level 為 `None`。
    pub fn neighbors(&self, id: NodeId, level: usize) -> Option<&[NodeId]> {
        self.nodes
            .get(id as usize)?
            .neighbors
            .get(level)
            .map(Vec::as_slice)
    }

    /// 與 `HnswIndex::search_batch` 相同的語意與輸出格式：每個 query 最多 k 個鄰居，依 `(distance, id)` 升冪；
    /// layer 0 使用 `ef = max(ef_search, k)`；不足 k 的 row 以 `PADDING_ID` / `+inf` 補齊。
    /// FP32 的 ids 與 distances 與 Core 逐位元相同；SQ8 的距離在輸出時夾到 `[0, 2]`。
    pub fn search_batch(
        &self,
        queries: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<BatchResult, HnswError> {
        check_k(self.len(), k)?;
        let (num_queries, unit) = unit_rows(queries, self.dim)?;
        Ok(match &self.store {
            Store::Fp32(store) => self.search_rows(store, &unit, num_queries, k, ef_search),
            Store::Sq8(store) => self.search_rows(store, &unit, num_queries, k, ef_search),
        })
    }

    /// 不計時的診斷（decision 019 §7）：結果與 `search_batch` 相同，另外回傳每個 query 的計數。
    pub fn search_stats(
        &self,
        queries: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<(BatchResult, SearchStats), HnswError> {
        check_k(self.len(), k)?;
        let (num_queries, unit) = unit_rows(queries, self.dim)?;
        Ok(match &self.store {
            Store::Fp32(store) => self.search_rows_counted(store, &unit, num_queries, k, ef_search),
            Store::Sq8(store) => self.search_rows_counted(store, &unit, num_queries, k, ef_search),
        })
    }

    /// 以同一個 kernel 對所有點做 brute force，只用於 decision 019 §7 的天花板（CLAUDE.md：brute force 只作 baseline）。
    /// 依原始距離的 `(distance, id)` 取前 k 個；輸出的距離處理與 `search_batch` 相同（SQ8 夾到 `[0, 2]`）。
    pub fn exact_search_batch(&self, queries: &[f32], k: usize) -> Result<BatchResult, HnswError> {
        check_k(self.len(), k)?;
        let (num_queries, unit) = unit_rows(queries, self.dim)?;
        Ok(match &self.store {
            Store::Fp32(store) => self.exact_rows(store, &unit, num_queries, k),
            Store::Sq8(store) => self.exact_rows(store, &unit, num_queries, k),
        })
    }

    /// 結構大小（decision 019 §7 的 M_struct）：各 Vec 的 capacity × 元素大小，加上 header。
    pub fn memory_accounting(&self) -> MemoryAccounting {
        let graph_bytes = self.nodes.capacity() * size_of::<Node>()
            + self
                .nodes
                .iter()
                .map(|node| {
                    node.neighbors.capacity() * size_of::<Vec<NodeId>>()
                        + node
                            .neighbors
                            .iter()
                            .map(|list| list.capacity() * size_of::<NodeId>())
                            .sum::<usize>()
                })
                .sum::<usize>();
        let (vectors_bytes, codes_bytes, codebook_bytes) = match &self.store {
            Store::Fp32(store) => (store.vectors.capacity() * size_of::<f32>(), 0, 0),
            Store::Sq8(store) => (
                0,
                store.codes.capacity() * size_of::<u8>(),
                (store.vmin.capacity() + store.vdiff.capacity()) * size_of::<f32>(),
            ),
        };
        MemoryAccounting {
            struct_bytes: size_of::<Self>(),
            graph_bytes,
            vectors_bytes,
            codes_bytes,
            codebook_bytes,
        }
    }

    // ---- 內部 ----

    fn node_level(&self, id: NodeId) -> usize {
        self.nodes[id as usize].neighbors.len() - 1
    }

    /// 計時路徑：Core 的 `search_batch` 迴圈，沒有計數器。
    fn search_rows<K: Kernel>(
        &self,
        kernel: &K,
        unit: &[f32],
        num_queries: usize,
        k: usize,
        ef_search: usize,
    ) -> BatchResult {
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);
        for query in unit.chunks_exact(self.dim) {
            let query = kernel.prepare(query);
            let result = self.search_unit_q(kernel, &query, k, ef_search, &mut NoCount);
            push_padded_row(&result, k, &mut ids, &mut distances);
        }
        BatchResult {
            num_queries,
            k,
            ids,
            distances,
        }
    }

    /// 計數版本：與 `search_rows` 相同的走訪，另一個單態化。
    fn search_rows_counted<K: Kernel>(
        &self,
        kernel: &K,
        unit: &[f32],
        num_queries: usize,
        k: usize,
        ef_search: usize,
    ) -> (BatchResult, SearchStats) {
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);
        let mut stats = SearchStats {
            kernel_calls: Vec::with_capacity(num_queries),
            visited_inserts: Vec::with_capacity(num_queries),
        };
        for query in unit.chunks_exact(self.dim) {
            let query = kernel.prepare(query);
            let mut count = Count::default();
            let result = self.search_unit_q(kernel, &query, k, ef_search, &mut count);
            push_padded_row(&result, k, &mut ids, &mut distances);
            stats.kernel_calls.push(count.kernel_calls);
            stats.visited_inserts.push(count.visited_inserts);
        }
        let batch = BatchResult {
            num_queries,
            k,
            ids,
            distances,
        };
        (batch, stats)
    }

    fn exact_rows<K: Kernel>(
        &self,
        kernel: &K,
        unit: &[f32],
        num_queries: usize,
        k: usize,
    ) -> BatchResult {
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);
        for query in unit.chunks_exact(self.dim) {
            let query = kernel.prepare(query);
            let result = self.exact_unit(kernel, &query, k);
            push_padded_row(&result, k, &mut ids, &mut distances);
        }
        BatchResult {
            num_queries,
            k,
            ids,
            distances,
        }
    }

    /// size-k max-heap 的 brute force（`ExactIndex::search_unit` 的寫法），距離用 kernel 的原始值。
    fn exact_unit<K: Kernel>(&self, kernel: &K, query: &K::Query<'_>, k: usize) -> Vec<Neighbor> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for id in 0..self.len() as NodeId {
            let candidate = Candidate {
                distance: kernel.distance(query, id),
                id,
            };
            if heap.len() < k {
                heap.push(candidate);
            } else if let Some(mut worst) = heap.peek_mut()
                && candidate < *worst
            {
                *worst = candidate;
            }
        }
        heap.into_sorted_vec()
            .into_iter()
            .map(|c| Neighbor {
                id: c.id,
                distance: K::output(c.distance),
            })
            .collect()
    }

    /// Core 的 `greedy_search`（逐行複製）：上層的 greedy search（ef = 1）。
    #[inline(never)]
    fn greedy_search_q<K: Kernel, C: Counter>(
        &self,
        kernel: &K,
        query: &K::Query<'_>,
        entry: Candidate,
        level: usize,
        counter: &mut C,
    ) -> Candidate {
        let mut best = entry;
        let mut changed = true;
        while changed {
            changed = false;
            for &neighbor in &self.nodes[best.id as usize].neighbors[level] {
                let candidate = Candidate {
                    distance: distance_to(kernel, query, neighbor, counter),
                    id: neighbor,
                };
                if candidate < best {
                    best = candidate;
                    changed = true;
                }
            }
        }
        best
    }

    /// Core 的 `search_layer`（逐行複製）：在單一層找最多 ef 個最近點，依 `(distance, id)` 升冪回傳。
    #[inline(never)]
    fn search_layer_q<K: Kernel, C: Counter>(
        &self,
        kernel: &K,
        query: &K::Query<'_>,
        entry: Candidate,
        ef: usize,
        level: usize,
        counter: &mut C,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        counter.visited_insert();
        visited.insert(entry.id);
        let mut candidates = BinaryHeap::new();
        candidates.push(Reverse(entry));
        let mut results = BinaryHeap::new();
        results.push(entry);

        while let Some(Reverse(current)) = candidates.pop() {
            let furthest = *results.peek().expect("results is never empty");
            if current > furthest {
                break;
            }
            for &neighbor in &self.nodes[current.id as usize].neighbors[level] {
                counter.visited_insert();
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = Candidate {
                    distance: distance_to(kernel, query, neighbor, counter),
                    id: neighbor,
                };
                let improves = results.len() < ef
                    || candidate < *results.peek().expect("results is never empty");
                if improves {
                    candidates.push(Reverse(candidate));
                    results.push(candidate);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results.into_sorted_vec()
    }

    /// Core 的 `search_unit`（逐行複製）：上層 greedy，layer 0 以 `ef = max(ef_search, k)` 搜尋，取前 k 個。
    /// 排序一律用原始距離；只在最後輸出時經過 `K::output`（SQ8 夾住，FP32 不變）。
    #[inline(never)]
    fn search_unit_q<K: Kernel, C: Counter>(
        &self,
        kernel: &K,
        query: &K::Query<'_>,
        k: usize,
        ef_search: usize,
        counter: &mut C,
    ) -> Vec<Neighbor> {
        let entry = self
            .entry_point
            .expect("search_unit_q requires a non-empty index");
        let mut ep = Candidate {
            distance: distance_to(kernel, query, entry, counter),
            id: entry,
        };
        for l in (1..=self.node_level(entry)).rev() {
            ep = self.greedy_search_q(kernel, query, ep, l, counter);
        }
        let mut found = self.search_layer_q(kernel, query, ep, ef_search.max(k), 0, counter);
        found.truncate(k);
        found
            .into_iter()
            .map(|c| Neighbor {
                id: c.id,
                distance: K::output(c.distance),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HnswParams;
    use crate::distance::normalize;
    use crate::rng::SplitMix64;

    /// f32 的 unit roundoff `u = 2⁻²⁴`。
    const U: f64 = 1.0 / (1u64 << 24) as f64;

    /// `γ_m = m·u / (1 − m·u)`（decision 019 §4）。
    fn gamma(m: usize) -> f64 {
        let mu = m as f64 * U;
        mu / (1.0 - mu)
    }

    /// `[-1, 1)` 的決定性 synthetic 資料，逐列以 crate 的規則正規化（與 `add` 存下的 unit vectors 相同）。
    fn unit_data(seed: u64, n: usize, dim: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let raw: Vec<f32> = (0..n * dim)
            .map(|_| (rng.next_unit_open0() * 2.0 - 1.0) as f32)
            .collect();
        raw.chunks_exact(dim)
            .flat_map(|v| normalize(v).unwrap())
            .collect()
    }

    fn sq8_store(rows: &[f32], dim: usize) -> Sq8Store {
        let (vmin, vdiff, codes) = sq8_train_encode(rows, dim);
        Sq8Store {
            dim,
            vmin,
            vdiff,
            codes,
        }
    }

    /// decision 019 §4 的 f64 參考值 `r = 1 − Σ_j q_j · x̂_j` 與登記的誤差界；
    /// `r_without_half` 是少了 `+0.5` 的錯誤解碼，只用來確認這個界有鑑別力。
    struct Reference {
        r: f64,
        bound: f64,
        r_without_half: f64,
    }

    fn reference(store: &Sq8Store, query: &[f32], id: usize) -> Reference {
        let d = store.dim;
        let code = &store.codes[id * d..(id + 1) * d];
        let (mut dot, mut dot_without_half, mut sigma, mut b_ref) = (0.0, 0.0, 0.0, 0.0);
        for (((&q, &vmin), &vdiff), &c) in query.iter().zip(&store.vmin).zip(&store.vdiff).zip(code)
        {
            let (q, vmin, vdiff, c) = (
                f64::from(q),
                f64::from(vmin),
                f64::from(vdiff),
                f64::from(c),
            );
            dot += q * (vmin + (c + 0.5) / 255.0 * vdiff);
            dot_without_half += q * (vmin + c / 255.0 * vdiff);
            sigma += q.abs() * (vdiff / 255.0) * c;
            b_ref += q * (vmin + 0.5 * vdiff / 255.0);
        }
        Reference {
            r: 1.0 - dot,
            bound: gamma(d + 5) * (sigma + b_ref.abs() + 1.0),
            r_without_half: 1.0 - dot_without_half,
        }
    }

    #[test]
    fn sq8_kernel_is_within_the_registered_error_bound() {
        for (seed, dim) in [(1, 2), (2, 3), (3, 16), (4, 100), (5, 768)] {
            let n = 200;
            let data = unit_data(seed, n, dim);
            let store = sq8_store(&data, dim);
            // 隨機 query 之外，也用存進去的向量本身當 query（距離接近 0，夾住之前可能是負的）。
            let mut queries = unit_data(seed + 100, 20, dim);
            queries.extend_from_slice(&data[..10 * dim]);

            let (mut worst_ratio, mut detects_missing_half) = (0.0_f64, false);
            for query in queries.chunks_exact(dim) {
                let prepared = store.prepare(query);
                for id in 0..n {
                    let kernel = store.distance(&prepared, id as NodeId);
                    let reference = reference(&store, query, id);
                    let error = (f64::from(kernel) - reference.r).abs();
                    assert!(
                        error <= reference.bound,
                        "dim {dim} id {id}: |{kernel} − {}| = {error:e} > {:e}",
                        reference.r,
                        reference.bound
                    );
                    // 夾住是 1-Lipschitz：夾住之後也滿足同一個界。
                    let clamped = f64::from(Sq8Store::output(kernel));
                    assert!((clamped - reference.r.clamp(0.0, 2.0)).abs() <= reference.bound);
                    worst_ratio = worst_ratio.max(error / reference.bound);
                    detects_missing_half |=
                        (f64::from(kernel) - reference.r_without_half).abs() > reference.bound;
                }
            }
            eprintln!("dim {dim}: max |kernel − r| / bound = {worst_ratio:.4}");
            assert!(
                detects_missing_half,
                "dim {dim}: the bound cannot detect a missing +0.5"
            );
        }
    }

    #[test]
    fn sq8_query_preparation_follows_decision_019() {
        // w_j = q_j · (vdiff_j / 255)（f32）；b 以 f64 累加後轉 f32。
        let data = unit_data(9, 50, 5);
        let store = sq8_store(&data, 5);
        let query = &unit_data(10, 1, 5)[..];
        let prepared = store.prepare(query);
        for ((&w, &q), &vdiff) in prepared.w.iter().zip(query).zip(&store.vdiff) {
            assert_eq!(w.to_bits(), (q * (vdiff / 255.0)).to_bits());
        }
        let b: f64 = query
            .iter()
            .zip(&store.vmin)
            .zip(&store.vdiff)
            .map(|((&q, &vmin), &vdiff)| {
                f64::from(q) * (f64::from(vmin) + 0.5 * f64::from(vdiff) / 255.0)
            })
            .sum();
        assert_eq!(prepared.b.to_bits(), (b as f32).to_bits());
    }

    #[test]
    fn fp32_kernel_is_bitwise_cosine_distance_unit() {
        let (n, dim) = (40, 7);
        let data = unit_data(11, n, dim);
        let store = Fp32Store {
            dim,
            vectors: data.clone(),
        };
        for query in data.chunks_exact(dim).take(5) {
            for (id, v) in data.chunks_exact(dim).enumerate() {
                let got = store.distance(&store.prepare(query), id as NodeId);
                assert_eq!(got.to_bits(), cosine_distance_unit(query, v).to_bits());
                assert_eq!(Fp32Store::output(got).to_bits(), got.to_bits());
            }
        }
    }

    /// 輸出的距離是原始距離夾住的結果，順序仍是原始距離的 `(distance, id)`（夾住之後不重排）。
    /// 回傳 (是否出現原始距離 < 0, 是否出現「夾住之後同為 0、id 卻遞減」的相鄰兩名)：
    /// 後者只有在排序用的是原始距離時才會發生，用來確認 fixture 真的能分辨「在走訪中就夾住」。
    fn assert_output_follows_raw_order(
        index: &QuantIndex,
        queries: &[f32],
        batch: &BatchResult,
    ) -> (bool, bool) {
        let Store::Sq8(store) = &index.store else {
            panic!("expected SQ8");
        };
        let (mut saw_negative, mut saw_reordered_tie) = (false, false);
        for ((query, ids), distances) in queries
            .chunks_exact(index.dim)
            .zip(batch.ids.chunks_exact(batch.k))
            .zip(batch.distances.chunks_exact(batch.k))
        {
            let unit = normalize(query).unwrap();
            let prepared = store.prepare(&unit);
            let raw: Vec<Candidate> = ids
                .iter()
                .take_while(|&&id| id != crate::PADDING_ID)
                .map(|&id| Candidate {
                    distance: store.distance(&prepared, id),
                    id,
                })
                .collect();
            assert!(raw.windows(2).all(|w| w[0] < w[1]), "not in raw order");
            for (c, &d) in raw.iter().zip(distances) {
                assert_eq!(d.to_bits(), c.distance.clamp(0.0, 2.0).to_bits());
                saw_negative |= c.distance < 0.0;
            }
            saw_reordered_tie |= raw
                .windows(2)
                .any(|w| w[1].distance < 0.0 && w[0].id > w[1].id);
        }
        (saw_negative, saw_reordered_tie)
    }

    /// `memory_accounting` 的期望值，由公開的 graph 存取方法與各 Vec 的 capacity 獨立算出：
    /// nodes 的 capacity × `size_of::<Node>()`，每個節點外層 Vec 的 capacity × Vec header，每份鄰居清單的 capacity × 4。
    fn expected_graph_bytes(index: &QuantIndex) -> usize {
        index.nodes.capacity() * size_of::<Node>()
            + index
                .nodes
                .iter()
                .map(|node| {
                    node.neighbors.capacity() * size_of::<Vec<NodeId>>()
                        + node
                            .neighbors
                            .iter()
                            .map(|list| list.capacity() * size_of::<NodeId>())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    /// capacity 恰好等於長度時（clone 之後），graph 的大小只由 level 與鄰居數決定：
    /// `n·size_of::<Node>() + Σ_v (level_v + 1)·size_of::<Vec<NodeId>>() + Σ len·4`。
    fn tight_graph_bytes(index: &QuantIndex) -> usize {
        let ids = 0..index.len() as NodeId;
        index.len() * size_of::<Node>()
            + ids
                .clone()
                .map(|v| (index.level(v).unwrap() + 1) * size_of::<Vec<NodeId>>())
                .sum::<usize>()
            + ids
                .map(|v| {
                    (0..=index.level(v).unwrap())
                        .map(|l| size_of_val(index.neighbors(v, l).unwrap()))
                        .sum::<usize>()
                })
                .sum::<usize>()
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn memory_accounting_hand_verifiable_three_node_graph() {
        // 手算的 3 點 graph（64 位元：`Node` 與 Vec 的 header 都是 24 B，NodeId 4 B），capacity 刻意大於長度。
        let list = |ids: &[NodeId], capacity: usize| {
            let mut v = Vec::with_capacity(capacity);
            v.extend_from_slice(ids);
            v
        };
        let outer = |lists: Vec<Vec<NodeId>>, capacity: usize| {
            let mut v = Vec::with_capacity(capacity);
            v.extend(lists);
            v
        };
        let mut nodes = Vec::with_capacity(5);
        nodes.push(Node {
            neighbors: outer(vec![list(&[1, 2], 4)], 2),
        });
        nodes.push(Node {
            neighbors: outer(vec![list(&[0], 1), list(&[2], 3)], 2),
        });
        nodes.push(Node {
            neighbors: outer(vec![list(&[0, 1], 2), list(&[1], 1)], 3),
        });
        assert_eq!((size_of::<Node>(), size_of::<Vec<NodeId>>()), (24, 24));
        let mut vectors = Vec::with_capacity(7);
        vectors.extend_from_slice(&[1.0, 0.0, 0.0, 1.0, 0.6, 0.8]);
        let fp32 = QuantIndex {
            dim: 2,
            nodes,
            entry_point: Some(1),
            store: Store::Fp32(Fp32Store { dim: 2, vectors }),
        };
        // nodes：capacity 5 × 24 = 120；外層 Vec：(2 + 2 + 3) × 24 = 168；鄰居清單：(4 + 1 + 3 + 2 + 1) × 4 = 44。
        let accounting = fp32.memory_accounting();
        assert_eq!(accounting.graph_bytes, 120 + 168 + 44);
        assert_eq!(accounting.graph_bytes, expected_graph_bytes(&fp32));
        assert_eq!(accounting.struct_bytes, size_of::<QuantIndex>());
        assert_eq!(accounting.vectors_bytes, 7 * 4);
        assert_eq!((accounting.codes_bytes, accounting.codebook_bytes), (0, 0));
        assert_eq!(accounting.total_bytes(), size_of::<QuantIndex>() + 332 + 28);

        // SQ8：codes 與 codebook 也以 capacity 計算。
        let mut codes = Vec::with_capacity(9);
        codes.extend_from_slice(&[0, 255, 0, 0, 255, 0]);
        let sq8 = QuantIndex {
            store: Store::Sq8(Sq8Store {
                dim: 2,
                vmin: Vec::with_capacity(2),
                vdiff: Vec::with_capacity(3),
                codes,
            }),
            ..fp32.clone()
        };
        let accounting = sq8.memory_accounting();
        assert_eq!(accounting.vectors_bytes, 0);
        assert_eq!(accounting.codes_bytes, 9);
        assert_eq!(accounting.codebook_bytes, (2 + 3) * 4);
        assert_eq!(accounting.struct_bytes, size_of::<QuantIndex>());
        // clone 收緊 capacity：nodes 3 × 24、外層 (1 + 2 + 2) × 24、鄰居 (2 + 1 + 1 + 2 + 1) × 4。
        assert_eq!(sq8.memory_accounting().graph_bytes, 72 + 120 + 28);
        assert_eq!(sq8.memory_accounting().graph_bytes, tight_graph_bytes(&sq8));
    }

    #[test]
    fn memory_accounting_of_built_indexes_uses_capacities() {
        for (seed, n, dim) in [(81, 1, 4), (82, 3, 2), (83, 300, 8)] {
            let data = unit_data(seed, n, dim);
            let mut core = HnswIndex::new(
                dim,
                HnswParams {
                    m: 4,
                    ef_construction: 16,
                    seed,
                },
            )
            .unwrap();
            core.add(&data).unwrap();
            // clone 之後每個 Vec 的 capacity 都等於長度：絕對值由 level 與鄰居數決定。
            let tight = core.clone().into_quant(Representation::Fp32);
            let accounting = tight.memory_accounting();
            assert_eq!(accounting.graph_bytes, tight_graph_bytes(&tight), "n {n}");
            assert_eq!(
                accounting.graph_bytes,
                expected_graph_bytes(&tight),
                "n {n}"
            );
            assert_eq!(accounting.struct_bytes, size_of::<QuantIndex>());
            assert_eq!(accounting.vectors_bytes, 4 * n * dim);
            if n == 1 {
                let level = tight.level(0).unwrap();
                assert_eq!(
                    accounting.graph_bytes,
                    size_of::<Node>() + (level + 1) * size_of::<Vec<NodeId>>()
                );
            }
            // 直接轉換（不 clone）時，鄰居清單的 capacity 可能大於長度：以 capacity 計算，不是長度。
            let direct = core.into_quant(Representation::Sq8);
            let accounting = direct.memory_accounting();
            assert_eq!(
                accounting.graph_bytes,
                expected_graph_bytes(&direct),
                "n {n}"
            );
            assert_eq!(accounting.struct_bytes, size_of::<QuantIndex>());
            if n == 300 {
                assert!(
                    accounting.graph_bytes > tight_graph_bytes(&direct),
                    "the fixture never has a neighbour list with spare capacity"
                );
            }
        }
    }

    #[test]
    fn sq8_output_is_clamped_and_keeps_the_raw_order() {
        // 30 個中心，各帶 15 個微擾（約 1e-3）的近似副本，再加 400 個隨機點。
        // 以中心當 query：許多近似副本的原始距離是負的，而且彼此不同。
        let dim = 6;
        let centers = unit_data(21, 30, dim);
        let mut rng = SplitMix64::new(22);
        let mut raw = Vec::new();
        for center in centers.chunks_exact(dim) {
            for _ in 0..15 {
                raw.extend(
                    center
                        .iter()
                        .map(|&x| x + ((rng.next_unit_open0() - 0.5) * 2e-3) as f32),
                );
            }
        }
        raw.extend(unit_data(23, 400, dim));
        let data: Vec<f32> = raw
            .chunks_exact(dim)
            .flat_map(|v| normalize(v).unwrap())
            .collect();
        let mut core = HnswIndex::new(
            dim,
            HnswParams {
                m: 6,
                ef_construction: 40,
                seed: 21,
            },
        )
        .unwrap();
        core.add(&data).unwrap();
        let index = core.into_quant(Representation::Sq8);

        let mut search_flags = (false, false);
        for (k, ef) in [(1, 10), (10, 10), (10, 64), (20, 100), (50, 100)] {
            let batch = index.search_batch(&centers, k, ef).unwrap();
            let flags = assert_output_follows_raw_order(&index, &centers, &batch);
            search_flags = (search_flags.0 | flags.0, search_flags.1 | flags.1);
        }
        assert_eq!(
            search_flags,
            (true, true),
            "search_batch never exercised the clamp"
        );
        let exact = index.exact_search_batch(&centers, 20).unwrap();
        assert_eq!(
            assert_output_follows_raw_order(&index, &centers, &exact),
            (true, true),
            "exact_search_batch never exercised the clamp"
        );
    }
}
