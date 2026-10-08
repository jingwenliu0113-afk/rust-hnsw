//! exact 與 HNSW 共用的批次輸入 / 輸出格式。

use crate::distance::normalize_into;
use crate::{HnswError, NodeId};

/// 結果不足 k 時補齊用的 id（decision 004）；`NodeId::MAX` 永遠不是合法 id。
pub const PADDING_ID: NodeId = NodeId::MAX;

/// 批次搜尋結果，row-major，shape 為 `(num_queries, k)`。
/// 結果不足 k 時，row 尾端補 `PADDING_ID` 與 `f32::INFINITY`。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResult {
    pub num_queries: usize,
    pub k: usize,
    pub ids: Vec<NodeId>,
    pub distances: Vec<f32>,
}

/// row-major buffer 的 row 數；長度不是 `dim` 的整數倍時回傳錯誤。
pub(crate) fn row_count(buffer: &[f32], dim: usize) -> Result<usize, HnswError> {
    if !buffer.len().is_multiple_of(dim) {
        return Err(HnswError::RaggedBuffer {
            len: buffer.len(),
            dim,
        });
    }
    Ok(buffer.len() / dim)
}

/// 檢查 index 非空且 `1 ≤ k ≤ len`。
pub(crate) fn check_k(len: usize, k: usize) -> Result<(), HnswError> {
    if len == 0 {
        return Err(HnswError::EmptyIndex);
    }
    if k == 0 || k > len {
        return Err(HnswError::InvalidK { k, len });
    }
    Ok(())
}

/// 驗證維度並正規化單一 query。
pub(crate) fn unit_query(query: &[f32], dim: usize) -> Result<Vec<f32>, HnswError> {
    if query.len() != dim {
        return Err(HnswError::DimensionMismatch {
            expected: dim,
            actual: query.len(),
        });
    }
    let mut unit = Vec::with_capacity(dim);
    normalize_into(query, 0, &mut unit)?;
    Ok(unit)
}

/// 驗證並正規化 row-major 的多個向量，回傳 row 數與 unit vectors。
/// 錯誤中的 `row` 相對於 `buffer`。
pub(crate) fn unit_rows(buffer: &[f32], dim: usize) -> Result<(usize, Vec<f32>), HnswError> {
    let rows = row_count(buffer, dim)?;
    let mut unit = Vec::with_capacity(buffer.len());
    for (row, v) in buffer.chunks_exact(dim).enumerate() {
        normalize_into(v, row, &mut unit)?;
    }
    Ok((rows, unit))
}
