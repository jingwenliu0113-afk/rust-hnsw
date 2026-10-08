//! 依 decision 005 獨立撰寫的 v1 encoder / decoder 與逐 bit CRC-32，作為測試 oracle。
//! 刻意不呼叫 crate 的 save / load 或其內部程式。

pub const MAGIC: &[u8; 8] = b"RHNSWIDX";
pub const HEADER_LEN: usize = 76;
pub const NO_ENTRY: u32 = 0xFFFF_FFFF;
/// SplitMix64 每次 `next_u64` 的 state 增量（decision 005「v1 語意綁定」）。
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// 逐 bit 計算的 CRC-32（IEEE / zlib），與 crate 內的 table 版本互相獨立。
pub fn crc32_bitwise(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// v1 檔案的完整內容。欄位可以是不合法的值，用來產生「CRC 正確但語意不合法」的檔案。
#[derive(Debug, Clone, PartialEq)]
pub struct SpecFile {
    pub version: u32,
    pub flags: u32,
    pub metric: u32,
    pub dim: u64,
    pub m: u64,
    pub ef_construction: u64,
    pub seed: u64,
    pub rng_state: u64,
    pub len: u64,
    pub entry_point: u32,
    /// 向量值的 bit pattern（逐位元比較，`-0.0` 與 `0.0` 不同）。
    pub vector_bits: Vec<u32>,
    /// `graph[id][level]` 為該層鄰居，順序即檔案中的順序。
    pub graph: Vec<Vec<Vec<u32>>>,
}

impl SpecFile {
    /// 依 decision 005 的不變量推導 rng_state 與 entry point，建立合法的檔案內容。
    pub fn valid(
        dim: u64,
        m: u64,
        ef_construction: u64,
        seed: u64,
        vectors: &[f32],
        graph: Vec<Vec<Vec<u32>>>,
    ) -> Self {
        let len = graph.len() as u64;
        let max_level = graph.iter().map(|levels| levels.len() - 1).max();
        let entry_point = match max_level {
            None => NO_ENTRY,
            Some(top) => graph
                .iter()
                .position(|levels| levels.len() - 1 == top)
                .unwrap() as u32,
        };
        Self {
            version: 1,
            flags: 0,
            metric: 1,
            dim,
            m,
            ef_construction,
            seed,
            rng_state: seed.wrapping_add(len.wrapping_mul(GAMMA)),
            len,
            entry_point,
            vector_bits: vectors.iter().map(|x| x.to_bits()).collect(),
            graph,
        }
    }

    pub fn vectors(&self) -> Vec<f32> {
        self.vector_bits
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect()
    }
}

/// 76 bytes 的 v1 header（含 header CRC）。
pub fn encode_header(spec: &SpecFile) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&spec.version.to_le_bytes());
    header.extend_from_slice(&spec.flags.to_le_bytes());
    header.extend_from_slice(&spec.metric.to_le_bytes());
    for v in [
        spec.dim,
        spec.m,
        spec.ef_construction,
        spec.seed,
        spec.rng_state,
        spec.len,
    ] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    header.extend_from_slice(&spec.entry_point.to_le_bytes());
    let crc = crc32_bitwise(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    assert_eq!(header.len(), HEADER_LEN);
    header
}

/// file CRC 涵蓋的 bytes：`[0, 72)` 接上 `[76, len − 4)`，排除 header CRC 欄位與 trailer。
pub fn file_crc_input(file_without_trailer: &[u8]) -> Vec<u8> {
    let mut input = file_without_trailer[..72].to_vec();
    input.extend_from_slice(&file_without_trailer[HEADER_LEN..]);
    input
}

/// 完整的 v1 檔案：header、payload、file CRC。
pub fn encode(spec: &SpecFile) -> Vec<u8> {
    let mut out = encode_header(spec);
    for bits in &spec.vector_bits {
        out.extend_from_slice(&bits.to_le_bytes());
    }
    for levels in &spec.graph {
        out.extend_from_slice(&((levels.len() - 1) as u32).to_le_bytes());
        for neighbors in levels {
            out.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
            for id in neighbors {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    let crc = crc32_bitwise(&file_crc_input(&out));
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// 修改過 bytes 之後，重算 header CRC 與 file CRC，讓檢查走到語意驗證。
pub fn recompute_crcs(bytes: &mut [u8]) {
    let header_crc = crc32_bitwise(&bytes[..72]);
    bytes[72..76].copy_from_slice(&header_crc.to_le_bytes());
    let end = bytes.len() - 4;
    let file_crc = crc32_bitwise(&file_crc_input(&bytes[..end]));
    bytes[end..].copy_from_slice(&file_crc.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        s
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
}

/// 解析格式良好的 v1 檔案；任何不符都直接 panic（只給測試 oracle 用）。
pub fn decode(bytes: &[u8]) -> SpecFile {
    let mut c = Cursor { bytes, pos: 0 };
    assert_eq!(c.take(8), MAGIC);
    let version = c.u32();
    let flags = c.u32();
    let metric = c.u32();
    let dim = c.u64();
    let m = c.u64();
    let ef_construction = c.u64();
    let seed = c.u64();
    let rng_state = c.u64();
    let len = c.u64();
    let entry_point = c.u32();
    let header_crc = c.u32();
    assert_eq!(header_crc, crc32_bitwise(&bytes[..72]), "header crc");

    let vector_bits = (0..len * dim).map(|_| c.u32()).collect();
    let graph = (0..len)
        .map(|_| {
            let level = c.u32() as usize;
            (0..=level)
                .map(|_| {
                    let count = c.u32() as usize;
                    (0..count).map(|_| c.u32()).collect()
                })
                .collect()
        })
        .collect();
    let body_end = c.pos;
    let file_crc = c.u32();
    assert_eq!(
        file_crc,
        crc32_bitwise(&file_crc_input(&bytes[..body_end])),
        "file crc"
    );
    assert_eq!(c.pos, bytes.len(), "trailing bytes");

    SpecFile {
        version,
        flags,
        metric,
        dim,
        m,
        ef_construction,
        seed,
        rng_state,
        len,
        entry_point,
        vector_bits,
        graph,
    }
}
