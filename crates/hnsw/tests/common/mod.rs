//! 整合測試共用的小工具。
#![allow(dead_code)]

/// 決定性亂數（SplitMix64），避免為了測試引入外部依賴。
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `[-1, 1)` 均勻分佈。
    pub fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }

    /// `n` 個 `dim` 維向量，row-major。
    pub fn vectors(&mut self, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| self.next_f32()).collect()
    }
}

/// 單位圓上角度為 `deg` 度的 2D 向量。
pub fn unit_circle(deg: f64) -> [f32; 2] {
    let rad = deg.to_radians();
    [rad.cos() as f32, rad.sin() as f32]
}

pub fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual {actual}, expected {expected} (tol {tol})"
    );
}
