use std::{fmt, io};

/// 所有公開 API 的錯誤。非法輸入一律回傳錯誤，不默默修正。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswError {
    /// 維度為 0。
    InvalidDimension(usize),
    /// index 參數不在 `min..=max` 範圍內。
    InvalidParameter {
        name: &'static str,
        min: usize,
        max: usize,
        value: usize,
    },
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
            Self::InvalidParameter {
                name,
                min,
                max,
                value,
            } => write!(f, "{name} must be in {min}..={max}, got {value}"),
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

/// save / load 的錯誤（decision 005）。`std::io::Error` 無法 `Clone` / `Eq`，所以獨立於 `HnswError`。
#[derive(Debug)]
pub enum PersistError {
    /// 讀寫時的 I/O 錯誤（不含 EOF；EOF 一律是 `Truncated`）。
    Io(io::Error),
    /// 預期的資料讀完前遇到 EOF。
    Truncated,
    /// 前 8 bytes 不是 `RHNSWIDX`。
    BadMagic,
    /// 不支援的 `format_version`。
    UnsupportedVersion { found: u32 },
    /// header CRC 不符。
    HeaderChecksumMismatch { stored: u32, computed: u32 },
    /// 不支援的 flags（v1 必須為 0）。
    UnsupportedFlags(u32),
    /// 不支援的 metric（v1 只有 1 = cosine）。
    UnsupportedMetric(u32),
    /// CRC 正確但內容不合法。
    Corrupt(String),
    /// 整個檔案的 CRC 不符。
    FileChecksumMismatch { stored: u32, computed: u32 },
    /// trailer 之後還有資料。
    TrailingBytes,
}

impl fmt::Display for PersistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Truncated => write!(f, "index file is truncated"),
            Self::BadMagic => write!(f, "not a rust-hnsw index file (bad magic)"),
            Self::UnsupportedVersion { found } => {
                write!(f, "unsupported index format version {found}")
            }
            Self::HeaderChecksumMismatch { stored, computed } => write!(
                f,
                "header checksum mismatch: stored {stored:#010x}, computed {computed:#010x}"
            ),
            Self::UnsupportedFlags(flags) => write!(f, "unsupported flags {flags:#010x}"),
            Self::UnsupportedMetric(metric) => write!(f, "unsupported metric {metric}"),
            Self::Corrupt(reason) => write!(f, "corrupt index file: {reason}"),
            Self::FileChecksumMismatch { stored, computed } => write!(
                f,
                "file checksum mismatch: stored {stored:#010x}, computed {computed:#010x}"
            ),
            Self::TrailingBytes => write!(f, "unexpected bytes after the end of the index file"),
        }
    }
}

impl std::error::Error for PersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}
