//! HNSW index：insert 與 search（decision 004）。

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::batch::{PADDING_ID, check_k, unit_query, unit_rows};
use crate::distance::cosine_distance_unit;
use crate::rng::SplitMix64;
use crate::{BatchResult, HnswError, Neighbor, NodeId};

mod persist;

pub use persist::FORMAT_VERSION;

/// `m` 的上限（decision 005）：保證 `2m` 放得進 `NodeId` 與任何平台的 usize。
pub const MAX_M: usize = (NodeId::MAX / 2) as usize;

/// 建 index 用的參數（decision 004）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    /// layer ≥ 1 每個點最多的鄰居數；layer 0 的上限為 `2 * m`。必須在 `2..=MAX_M`。
    pub m: usize,
    /// 建圖時的候選數，必須 ≥ 1；實際使用 `max(ef_construction, m)`。
    pub ef_construction: usize,
    /// level 亂數種子。
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            seed: 100,
        }
    }
}

/// HNSW index。id 依插入順序從 0 開始。
///
/// 表示法刻意採最直接的寫法（decision 004）：向量存成一個連續的 `Vec<f32>`，
/// graph 為每個節點各自的 `Vec<Vec<NodeId>>`，visited set 為每次搜尋新建的 `HashSet`。
#[derive(Debug, Clone)]
pub struct HnswIndex {
    dim: usize,
    params: HnswParams,
    /// level 分佈的 `mL = 1/ln(M)`。
    level_mult: f64,
    /// row-major 的 unit vectors。
    vectors: Vec<f32>,
    nodes: Vec<Node>,
    entry_point: Option<NodeId>,
    rng: SplitMix64,
}

#[derive(Debug, Clone)]
struct Node {
    /// `neighbors[l]` 為第 l 層的鄰居；長度為節點 level + 1。
    neighbors: Vec<Vec<NodeId>>,
}

impl HnswIndex {
    pub fn new(dim: usize, params: HnswParams) -> Result<Self, HnswError> {
        if dim == 0 {
            return Err(HnswError::InvalidDimension(dim));
        }
        if !(2..=MAX_M).contains(&params.m) {
            return Err(HnswError::InvalidParameter {
                name: "m",
                min: 2,
                max: MAX_M,
                value: params.m,
            });
        }
        if params.ef_construction < 1 {
            return Err(HnswError::InvalidParameter {
                name: "ef_construction",
                min: 1,
                max: usize::MAX,
                value: params.ef_construction,
            });
        }
        Ok(Self {
            dim,
            params,
            level_mult: 1.0 / (params.m as f64).ln(),
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry_point: None,
            rng: SplitMix64::new(params.seed),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn params(&self) -> HnswParams {
        self.params
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 依 row 順序逐一插入 row-major 向量。任一 row 不合法則整批不插入
    /// （graph 與亂數狀態都不變），錯誤中的 `row` 相對於這次傳入的 `vectors`。
    pub fn add(&mut self, vectors: &[f32]) -> Result<(), HnswError> {
        let (rows, unit) = unit_rows(vectors, self.dim)?;
        if self.len() + rows > NodeId::MAX as usize {
            return Err(HnswError::CapacityExceeded);
        }
        self.vectors.reserve(unit.len());
        self.nodes.reserve(rows);
        for v in unit.chunks_exact(self.dim) {
            self.insert(v);
        }
        Ok(())
    }

    /// 回傳最多 k 個鄰居，依 `(distance, id)` 升冪；layer 0 使用 `ef = max(ef_search, k)`。
    /// graph 可到達的點不足 k 時，回傳數量會少於 k。
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Neighbor>, HnswError> {
        check_k(self.len(), k)?;
        let query = unit_query(query, self.dim)?;
        Ok(self.search_unit(&query, k, ef_search))
    }

    /// 對 row-major 的多個 query 做 `search`；不足 k 的 row 以 `PADDING_ID` / `+inf` 補齊。
    pub fn search_batch(
        &self,
        queries: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<BatchResult, HnswError> {
        check_k(self.len(), k)?;
        let (num_queries, unit) = unit_rows(queries, self.dim)?;
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);
        for query in unit.chunks_exact(self.dim) {
            let result = self.search_unit(query, k, ef_search);
            push_padded_row(&result, k, &mut ids, &mut distances);
        }
        Ok(BatchResult {
            num_queries,
            k,
            ids,
            distances,
        })
    }

    /// 目前的 entry point；空 index 為 `None`。
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// graph 最高層（即 entry point 的 level）；空 index 為 `None`。
    pub fn max_level(&self) -> Option<usize> {
        self.entry_point.map(|ep| self.node_level(ep))
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

    // ---- 內部 ----

    fn node_level(&self, id: NodeId) -> usize {
        self.nodes[id as usize].neighbors.len() - 1
    }

    fn vector(&self, id: NodeId) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    fn distance_to(&self, query: &[f32], id: NodeId) -> f32 {
        cosine_distance_unit(query, self.vector(id))
    }

    fn distance_between(&self, a: NodeId, b: NodeId) -> f32 {
        cosine_distance_unit(self.vector(a), self.vector(b))
    }

    /// layer 0 的上限為 `2M`，其他層為 `M`。
    fn max_neighbors(&self, level: usize) -> usize {
        if level == 0 {
            2 * self.params.m
        } else {
            self.params.m
        }
    }

    /// `level = floor(−ln(U) · mL)`，`U ∈ (0, 1]`。
    fn random_level(&mut self) -> usize {
        (-self.rng.next_unit_open0().ln() * self.level_mult).floor() as usize
    }

    /// Algorithm 1：插入一個已正規化的向量。
    fn insert(&mut self, query: &[f32]) {
        let id = self.nodes.len() as NodeId;
        let level = self.random_level();
        self.vectors.extend_from_slice(query);
        self.nodes.push(Node {
            neighbors: vec![Vec::new(); level + 1],
        });

        let Some(entry) = self.entry_point else {
            self.entry_point = Some(id);
            return;
        };
        let top = self.node_level(entry);
        let mut ep = Candidate {
            distance: self.distance_to(query, entry),
            id: entry,
        };
        for l in (level + 1..=top).rev() {
            ep = self.greedy_search(query, ep, l);
        }

        let ef = self.params.ef_construction.max(self.params.m);
        for l in (0..=level.min(top)).rev() {
            let found = self.search_layer(query, ep, ef, l);
            let selected =
                select_neighbors(&found, self.params.m, |a, b| self.distance_between(a, b));
            self.nodes[id as usize].neighbors[l] = selected.iter().map(|c| c.id).collect();
            for s in &selected {
                self.connect(s.id, id, l);
            }
            ep = found[0];
        }

        if level > top {
            self.entry_point = Some(id);
        }
    }

    /// 加入反向連結 `from → to`；超過上限時以 heuristic 重新挑選 `from` 的鄰居。
    fn connect(&mut self, from: NodeId, to: NodeId, level: usize) {
        let cap = self.max_neighbors(level);
        if self.nodes[from as usize].neighbors[level].len() < cap {
            self.nodes[from as usize].neighbors[level].push(to);
            return;
        }
        let mut candidates: Vec<Candidate> = self.nodes[from as usize].neighbors[level]
            .iter()
            .chain(std::iter::once(&to))
            .map(|&id| Candidate {
                distance: self.distance_between(from, id),
                id,
            })
            .collect();
        candidates.sort_unstable();
        let selected = select_neighbors(&candidates, cap, |a, b| self.distance_between(a, b));
        self.nodes[from as usize].neighbors[level] = selected.into_iter().map(|c| c.id).collect();
    }

    /// 上層的 greedy search（ef = 1）：持續移往更近的鄰居直到沒有改善。
    fn greedy_search(&self, query: &[f32], entry: Candidate, level: usize) -> Candidate {
        let mut best = entry;
        let mut changed = true;
        while changed {
            changed = false;
            for &neighbor in &self.nodes[best.id as usize].neighbors[level] {
                let candidate = Candidate {
                    distance: self.distance_to(query, neighbor),
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

    /// Algorithm 2：在單一層找最多 ef 個最近點，依 `(distance, id)` 升冪回傳（至少含 entry）。
    fn search_layer(
        &self,
        query: &[f32],
        entry: Candidate,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
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
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = Candidate {
                    distance: self.distance_to(query, neighbor),
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

    /// Algorithm 5：上層 greedy，layer 0 以 `ef = max(ef_search, k)` 搜尋，取前 k 個。
    fn search_unit(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Neighbor> {
        let entry = self
            .entry_point
            .expect("search_unit requires a non-empty index");
        let mut ep = Candidate {
            distance: self.distance_to(query, entry),
            id: entry,
        };
        for l in (1..=self.node_level(entry)).rev() {
            ep = self.greedy_search(query, ep, l);
        }
        let mut found = self.search_layer(query, ep, ef_search.max(k), 0);
        found.truncate(k);
        found
            .into_iter()
            .map(|c| Neighbor {
                id: c.id,
                distance: c.distance,
            })
            .collect()
    }
}

/// 排序 key 為 `(distance, id)`，讓所有比較都是決定性的（decision 004）。
#[derive(Debug, Clone, Copy)]
struct Candidate {
    distance: f32,
    id: NodeId,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

/// 鄰居選擇 heuristic（Algorithm 4 的 hnswlib 變體，decision 004）。
/// `candidates` 必須依到 base 的距離升冪排序；`distance(a, b)` 為兩個節點間的距離。
fn select_neighbors(
    candidates: &[Candidate],
    m: usize,
    distance: impl Fn(NodeId, NodeId) -> f32,
) -> Vec<Candidate> {
    // 候選數 < m 時直接全收（hnswlib / FAISS 的 shortcut）。
    if candidates.len() < m {
        return candidates.to_vec();
    }
    let mut selected: Vec<Candidate> = Vec::with_capacity(m);
    for &candidate in candidates {
        if selected.len() >= m {
            break;
        }
        // 若某個已選點比 base 更靠近 candidate，candidate 的方向已被涵蓋 → 丟掉。
        let covered = selected
            .iter()
            .any(|s| distance(candidate.id, s.id) < candidate.distance);
        if !covered {
            selected.push(candidate);
        }
    }
    selected
}

/// 把一個 query 的結果寫入批次輸出；不足 k 時補 `PADDING_ID` / `+inf`。
fn push_padded_row(result: &[Neighbor], k: usize, ids: &mut Vec<NodeId>, distances: &mut Vec<f32>) {
    debug_assert!(result.len() <= k);
    for neighbor in result {
        ids.push(neighbor.id);
        distances.push(neighbor.distance);
    }
    for _ in result.len()..k {
        ids.push(PADDING_ID);
        distances.push(f32::INFINITY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(deg: f64) -> [f32; 2] {
        let rad = deg.to_radians();
        [rad.cos() as f32, rad.sin() as f32]
    }

    fn candidate_ids(selected: &[Candidate]) -> Vec<NodeId> {
        selected.iter().map(|c| c.id).collect()
    }

    #[test]
    fn heuristic_prefers_diverse_directions() {
        // base（id 0）在 0°；候選 id 1 在 10°、id 2 在 15°、id 3 在 -60°。
        let vectors = [circle(0.0), circle(10.0), circle(15.0), circle(-60.0)];
        let distance =
            |a: NodeId, b: NodeId| cosine_distance_unit(&vectors[a as usize], &vectors[b as usize]);
        let candidates: Vec<Candidate> = [1, 2, 3]
            .iter()
            .map(|&id| Candidate {
                distance: distance(0, id),
                id,
            })
            .collect();

        // 15° 離 10° 只差 5°，比離 base 的 15° 近 → 丟掉；
        // -60° 離 10° 差 70°，比離 base 的 60° 遠 → 保留。
        assert_eq!(
            candidate_ids(&select_neighbors(&candidates, 2, distance)),
            [1, 3]
        );
        // 不做 keepPrunedConnections：m = 3 也只選出兩個。
        assert_eq!(
            candidate_ids(&select_neighbors(&candidates, 3, distance)),
            [1, 3]
        );
        // 候選數 < m：shortcut，全收。
        assert_eq!(
            candidate_ids(&select_neighbors(&candidates, 4, distance)),
            [1, 2, 3]
        );
        // m = 1：只取最近的。
        assert_eq!(
            candidate_ids(&select_neighbors(&candidates, 1, distance)),
            [1]
        );
    }

    #[test]
    fn heuristic_keeps_exact_duplicates_of_base() {
        // 與 base 完全相同的候選距離為 0；`0 < 0` 不成立，所以不會互相 prune。
        let vectors = [circle(0.0), circle(0.0), circle(0.0), circle(90.0)];
        let distance =
            |a: NodeId, b: NodeId| cosine_distance_unit(&vectors[a as usize], &vectors[b as usize]);
        let candidates: Vec<Candidate> = [1, 2, 3]
            .iter()
            .map(|&id| Candidate {
                distance: distance(0, id),
                id,
            })
            .collect();
        assert_eq!(
            candidate_ids(&select_neighbors(&candidates, 3, distance)),
            [1, 2, 3]
        );
    }

    #[test]
    fn rng_state_is_seed_plus_len_times_gamma() {
        // decision 005「v1 語意綁定」：每次成功 insert 恰好抽一次亂數，失敗的 add 不抽。
        // 這個測試紅燈代表亂數抽取方式改了，必須 bump format_version 並寫 decision log。
        const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
        let seed = 0xDEAD_BEEF;
        let mut index = HnswIndex::new(
            3,
            HnswParams {
                m: 3,
                ef_construction: 8,
                seed,
            },
        )
        .unwrap();
        let expect = |index: &HnswIndex| {
            assert_eq!(
                index.rng.state(),
                seed.wrapping_add((index.len() as u64).wrapping_mul(GAMMA))
            );
        };
        expect(&index);
        index.add(&[1.0, 0.0, 0.0]).unwrap();
        expect(&index);
        assert!(index.add(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).is_err());
        assert!(index.add(&[f32::NAN, 1.0, 0.0]).is_err());
        assert!(index.add(&[1.0, 2.0]).is_err());
        expect(&index);
        index.add(&[]).unwrap();
        let more: Vec<f32> = (0..60)
            .map(|i| ((i * 37 % 11) as f32) - 5.0 + 0.5)
            .collect();
        index.add(&more).unwrap();
        expect(&index);
        assert_eq!(index.len(), 21);
    }

    #[test]
    fn short_results_are_padded() {
        let mut ids = vec![];
        let mut distances = vec![];
        let result = [Neighbor {
            id: 4,
            distance: 0.25,
        }];
        push_padded_row(&result, 3, &mut ids, &mut distances);
        push_padded_row(&result, 1, &mut ids, &mut distances);
        assert_eq!(ids, [4, PADDING_ID, PADDING_ID, 4]);
        assert_eq!(distances, [0.25, f32::INFINITY, f32::INFINITY, 0.25]);
    }
}
