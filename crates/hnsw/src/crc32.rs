//! IEEE 802.3 / zlib CRC-32（decision 005）。

const POLY: u32 = 0xEDB8_8320;
const TABLE: [u32; 256] = make_table();

const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// 串流計算的 CRC-32。
#[derive(Debug, Clone)]
pub(crate) struct Crc32 {
    state: u32,
}

impl Crc32 {
    pub(crate) fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state =
                TABLE[((self.state ^ u32::from(byte)) & 0xFF) as usize] ^ (self.state >> 8);
        }
    }

    pub(crate) fn finish(&self) -> u32 {
        !self.state
    }
}

/// 一次計算整段資料的 CRC-32。
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(bytes);
    crc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_standard_check_values() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn streaming_equals_one_shot() {
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let mut crc = Crc32::new();
        for chunk in data.chunks(7) {
            crc.update(chunk);
        }
        assert_eq!(crc.finish(), crc32(&data));
    }
}
