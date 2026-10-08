use std::fmt;

/// 所有公開 API 的錯誤。非法輸入一律回傳錯誤，不默默修正。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswError {
    /// 維度為 0。
    InvalidDimension(usize),
    /// 單一向量的長度與預期維度不符。
    DimensionMismatch { expected: usize, actual: usize },
    /// 扁平 row-major buffer 長度不是維度的整數倍。
    RaggedBuffer { len: usize, dim: usize },
    /// 第 `row` 個向量是零向量，cosine 無定義。
    ZeroVector { row: usize },
    /// 第 `row` 個向量含 NaN 或 Inf。
    NonFiniteValue { row: usize },
    /// k 必須滿足 `1 ≤ k ≤ len`。
    InvalidK { k: usize, len: usize },
    /// 在空 index 上搜尋。
    EmptyIndex,
    /// 向量數超過 `NodeId` 可表示的範圍。
    CapacityExceeded,
}

impl fmt::Display for HnswError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(dim) => write!(f, "dimension must be positive, got {dim}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::RaggedBuffer { len, dim } => {
                write!(
                    f,
                    "buffer length {len} is not a multiple of dimension {dim}"
                )
            }
            Self::ZeroVector { row } => write!(f, "row {row} is a zero vector"),
            Self::NonFiniteValue { row } => write!(f, "row {row} contains NaN or Inf"),
            Self::InvalidK { k, len } => {
                write!(f, "k must satisfy 1 <= k <= {len}, got k = {k}")
            }
            Self::EmptyIndex => write!(f, "index is empty"),
            Self::CapacityExceeded => write!(f, "index capacity exceeded"),
        }
    }
}

impl std::error::Error for HnswError {}
