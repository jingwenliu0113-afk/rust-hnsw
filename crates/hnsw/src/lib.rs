//! 從零實作的 HNSW（cosine distance）。
//!
//! - `HnswIndex`：HNSW insert / search（decision 004）與 save / load（decision 005）。
//! - `ExactIndex`：exact brute-force search，只作 ground truth 與 correctness baseline。
//!
//! 距離定義與正規化規則見 `docs/decisions/001-cosine-distance-and-normalization.md`。

mod batch;
mod crc32;
pub mod distance;
mod error;
mod exact;
mod hnsw;
mod rng;

pub use batch::{BatchResult, PADDING_ID};
pub use error::{HnswError, PersistError};
pub use exact::ExactIndex;
pub use hnsw::{FORMAT_VERSION, HnswIndex, HnswParams, MAX_M};

/// 向量 id：依插入順序從 0 開始連續編號。
pub type NodeId = u32;

/// 一筆搜尋結果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: NodeId,
    /// cosine distance，`1 − cos`，約在 `[0, 2]`。f32 rounding 會讓它略微超出；
    /// load 的檔案允許 `|‖v‖ − 1| ≤ 1e-3`（decision 005），手工構造的檔案可能超出約 2e-3。
    pub distance: f32,
}
