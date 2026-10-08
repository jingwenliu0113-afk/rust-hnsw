//! 從零實作的 HNSW（cosine distance）。
//!
//! 目前只有 exact brute-force search，作為 ground truth 與 correctness baseline。
//! 距離定義與正規化規則見 `docs/decisions/001-cosine-distance-and-normalization.md`。

pub mod distance;
mod error;
mod exact;

pub use error::HnswError;
pub use exact::{BatchResult, ExactIndex};

/// 向量 id：依插入順序從 0 開始連續編號。
pub type NodeId = u32;

/// 一筆搜尋結果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: NodeId,
    /// cosine distance，`1 − cos`，範圍 `[0, 2]`。
    pub distance: f32,
}
