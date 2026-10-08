//! 決定性亂數（SplitMix64），用於 HNSW level（decision 004）。
//! 自帶實作，避免外部 crate 升級改變亂數序列。

#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `(0, 1]` 均勻分佈。
    pub(crate) fn next_unit_open0(&mut self) -> f64 {
        unit_open0(self.next_u64())
    }
}

/// 取 64-bit 亂數的高 53 bits 映射到 `(0, 1]`：永遠不會是 0，`−ln(U)` 因此有限。
fn unit_open0(bits: u64) -> f64 {
    ((bits >> 11) + 1) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_splitmix64_sequence() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn unit_open0_excludes_zero_and_includes_one() {
        assert_eq!(unit_open0(0), 1.0 / (1u64 << 53) as f64);
        assert_eq!(unit_open0(u64::MAX), 1.0);
    }
}
