//! Cosine distance 與正規化。定義見 decision 001。

use crate::HnswError;

/// f32 dot product。呼叫端保證兩者長度相同。
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 依序以 f64 累加的 dot product，給 exact search 與 ground truth 用。
pub(crate) fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// 檢查 finite 後以 f64 計算 L2 norm。
/// 任何有限 f32 的平方在 f64 都不會 overflow / underflow 成 0。
fn checked_norm(v: &[f32], row: usize) -> Result<f64, HnswError> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(HnswError::NonFiniteValue { row });
    }
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(HnswError::ZeroVector { row });
    }
    Ok(norm)
}

/// 把 `v` 正規化後 append 到 `out`；錯誤時 `out` 不變。`row` 只用於錯誤訊息。
pub(crate) fn normalize_into(v: &[f32], row: usize, out: &mut Vec<f32>) -> Result<(), HnswError> {
    let norm = checked_norm(v, row)?;
    out.extend(v.iter().map(|&x| (f64::from(x) / norm) as f32));
    Ok(())
}

/// 把向量正規化成 unit vector（f64 計算、存 f32）。
/// 零向量或含 NaN / Inf 時回傳錯誤（`row` 為 0）。
pub fn normalize(v: &[f32]) -> Result<Vec<f32>, HnswError> {
    let mut out = Vec::with_capacity(v.len());
    normalize_into(v, 0, &mut out)?;
    Ok(out)
}

/// 兩個原始向量（不需事先正規化）的 cosine distance，以 f64 計算。
/// 錯誤中的 `row` 0 指 `a`、1 指 `b`。
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, HnswError> {
    if a.len() != b.len() {
        return Err(HnswError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let norm_a = checked_norm(a, 0)?;
    let norm_b = checked_norm(b, 1)?;
    Ok((1.0 - dot_f64(a, b) / (norm_a * norm_b)) as f32)
}

/// 兩個 unit vector 的 cosine distance：`1 − dot`。
pub fn cosine_distance_unit(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}
