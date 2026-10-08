//! Exact brute-force search：只作 ground truth 與 correctness baseline。

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distance::{dot_f64, normalize_into};
use crate::{HnswError, Neighbor, NodeId};

/// 存放已正規化向量的 exact index。
#[derive(Debug, Clone)]
pub struct ExactIndex {
    dim: usize,
    /// row-major 的 unit vectors。
    data: Vec<f32>,
}

/// 批次搜尋結果，row-major，shape 為 `(num_queries, k)`。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResult {
    pub num_queries: usize,
    pub k: usize,
    pub ids: Vec<u32>,
    pub distances: Vec<f32>,
}

/// 排序 key 為 `(distance, id)`：同距離時 id 小者優先（decision 001）。
#[derive(Debug, Clone, Copy)]
struct Candidate {
    distance: f64,
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

/// row-major buffer 的 row 數；長度不是 `dim` 的整數倍時回傳錯誤。
fn row_count(buffer: &[f32], dim: usize) -> Result<usize, HnswError> {
    if !buffer.len().is_multiple_of(dim) {
        return Err(HnswError::RaggedBuffer {
            len: buffer.len(),
            dim,
        });
    }
    Ok(buffer.len() / dim)
}

impl ExactIndex {
    pub fn new(dim: usize) -> Result<Self, HnswError> {
        if dim == 0 {
            return Err(HnswError::InvalidDimension(dim));
        }
        Ok(Self {
            dim,
            data: Vec::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 加入 row-major 向量；id 接續目前的 `len()`。任一 row 不合法則整批不寫入，
    /// 錯誤中的 `row` 相對於這次傳入的 `vectors`。
    pub fn add(&mut self, vectors: &[f32]) -> Result<(), HnswError> {
        let rows = row_count(vectors, self.dim)?;
        if self.len() + rows > NodeId::MAX as usize {
            return Err(HnswError::CapacityExceeded);
        }
        let old_len = self.data.len();
        self.data.reserve(vectors.len());
        for (row, v) in vectors.chunks_exact(self.dim).enumerate() {
            if let Err(err) = normalize_into(v, row, &mut self.data) {
                self.data.truncate(old_len);
                return Err(err);
            }
        }
        Ok(())
    }

    /// 回傳最近的 k 個鄰居，依 `(distance, id)` 升冪。
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, HnswError> {
        self.check_k(k)?;
        if query.len() != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let mut unit_query = Vec::with_capacity(self.dim);
        normalize_into(query, 0, &mut unit_query)?;
        Ok(self.search_unit(&unit_query, k))
    }

    /// 對 row-major 的多個 query 做 `search`。所有 query 先驗證完才開始搜尋，
    /// 錯誤中的 `row` 為 query 的 row 編號。
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Result<BatchResult, HnswError> {
        self.check_k(k)?;
        let num_queries = row_count(queries, self.dim)?;
        let mut unit_queries = Vec::with_capacity(queries.len());
        for (row, q) in queries.chunks_exact(self.dim).enumerate() {
            normalize_into(q, row, &mut unit_queries)?;
        }

        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);
        for unit_query in unit_queries.chunks_exact(self.dim) {
            for neighbor in self.search_unit(unit_query, k) {
                ids.push(neighbor.id);
                distances.push(neighbor.distance);
            }
        }
        Ok(BatchResult {
            num_queries,
            k,
            ids,
            distances,
        })
    }

    fn check_k(&self, k: usize) -> Result<(), HnswError> {
        if self.is_empty() {
            return Err(HnswError::EmptyIndex);
        }
        if k == 0 || k > self.len() {
            return Err(HnswError::InvalidK { k, len: self.len() });
        }
        Ok(())
    }

    /// 已正規化 query 的 top-k：size-k max-heap，堆頂是目前保留中最差的候選。
    fn search_unit(&self, unit_query: &[f32], k: usize) -> Vec<Neighbor> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for (id, v) in self.data.chunks_exact(self.dim).enumerate() {
            let candidate = Candidate {
                distance: 1.0 - dot_f64(v, unit_query),
                id: id as NodeId,
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
                distance: c.distance as f32,
            })
            .collect()
    }
}
