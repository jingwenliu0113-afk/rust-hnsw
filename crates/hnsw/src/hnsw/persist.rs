//! HNSW index 的 save / load（decision 005）。
//!
//! v1 佈局：76-byte header（含 header CRC）→ vectors → graph → file CRC。
//! file CRC 涵蓋 header CRC 欄位以外的所有 bytes（不含 trailer）。
//! load 以串流方式讀取並同時計算 CRC；配置量只隨實際讀到的資料成長。

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use super::{HnswIndex, Node};
use crate::crc32::{Crc32, crc32};
use crate::rng::{GAMMA, SplitMix64};
use crate::{HnswParams, NodeId, PersistError};

/// 目前寫出的檔案格式版本。
pub const FORMAT_VERSION: u32 = 1;

/// 可以讀取的版本。
const SUPPORTED_VERSIONS: [u32; 1] = [1];
const MAGIC: &[u8; 8] = b"RHNSWIDX";
const HEADER_LEN: usize = 76;
const METRIC_COSINE: u32 = 1;
/// n = 0 時的 entry point。
const NO_ENTRY: u32 = NodeId::MAX;
/// 與 M 無關的 level 上限：`U ≥ 2^-53` 時 level ≤ 53。
const MAX_LEVEL: usize = 64;
const NORM_TOLERANCE: f64 = 1e-3;
/// vectors 分塊讀取的大小（f32 個數），讓配置量不超過已讀資料。
const VECTOR_CHUNK: usize = 16 * 1024;

fn corrupt(reason: impl Into<String>) -> PersistError {
    PersistError::Corrupt(reason.into())
}

/// 讀取錯誤：EOF 一律是 `Truncated`，其他才是 `Io`。
fn read_error(err: io::Error) -> PersistError {
    if err.kind() == ErrorKind::UnexpectedEof {
        PersistError::Truncated
    } else {
        PersistError::Io(err)
    }
}

/// 寫出時同時計算 file CRC。
struct CrcWriter<W> {
    inner: W,
    crc: Crc32,
}

impl<W: Write> CrcWriter<W> {
    fn put(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.crc.update(bytes);
        self.inner.write_all(bytes)
    }

    fn put_u32(&mut self, value: u32) -> io::Result<()> {
        self.put(&value.to_le_bytes())
    }
}

/// 讀取時同時計算 file CRC。
struct CrcReader<R> {
    inner: R,
    crc: Crc32,
}

impl<R: Read> CrcReader<R> {
    /// 讀滿 `buf` 才回傳；EOF 為 `Truncated`。
    fn fill(&mut self, buf: &mut [u8]) -> Result<(), PersistError> {
        self.inner.read_exact(buf).map_err(read_error)?;
        self.crc.update(buf);
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, PersistError> {
        let mut bytes = [0u8; 4];
        self.fill(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("4 bytes"))
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("8 bytes"))
}

fn to_usize(name: &str, value: u64) -> Result<usize, PersistError> {
    usize::try_from(value).map_err(|_| corrupt(format!("{name} = {value} does not fit in usize")))
}

/// trailer 之後是否還有資料。
fn has_more<R: Read>(reader: &mut R) -> Result<bool, PersistError> {
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(false),
            Ok(_) => return Ok(true),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(read_error(err)),
        }
    }
}

/// `path` 同目錄下的暫存檔名。
fn temp_path(path: &Path) -> io::Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "index path has no file name"))?;
    let mut temp = std::ffi::OsString::from(".");
    temp.push(name);
    temp.push(format!(".tmp-{}", std::process::id()));
    Ok(path.with_file_name(temp))
}

impl HnswIndex {
    /// 以 decision 005 的 v1 格式寫出；結尾會 flush `writer`。
    pub fn save<W: Write>(&self, writer: W) -> Result<(), PersistError> {
        let mut buffered = BufWriter::new(writer);
        self.write_v1(&mut buffered).map_err(PersistError::Io)?;
        let mut inner = buffered
            .into_inner()
            .map_err(|err| PersistError::Io(err.into_error()))?;
        inner.flush().map_err(PersistError::Io)
    }

    /// 讀取 decision 005 格式的 index。以小欄位讀取，呼叫端應傳入 buffered reader。
    pub fn load<R: Read>(reader: R) -> Result<Self, PersistError> {
        let mut input = CrcReader {
            inner: reader,
            crc: Crc32::new(),
        };

        // 1–3：固定前綴（magic、version），再讀該版本 header 的其餘部分並驗證 header CRC。
        let mut header = [0u8; HEADER_LEN];
        input.fill(&mut header[..8])?;
        if &header[..8] != MAGIC {
            return Err(PersistError::BadMagic);
        }
        input.fill(&mut header[8..12])?;
        let version = le_u32(&header[8..12]);
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(PersistError::UnsupportedVersion { found: version });
        }
        input.fill(&mut header[12..72])?;
        // header CRC 欄位不算進 file CRC：兩個合法 header 的差異是 CRC codeword，
        // 若算進去，換成另一個合法 header 時 file CRC 不會改變（decision 005）。
        input
            .inner
            .read_exact(&mut header[72..76])
            .map_err(read_error)?;
        let stored = le_u32(&header[72..76]);
        let computed = crc32(&header[..72]);
        if stored != computed {
            return Err(PersistError::HeaderChecksumMismatch { stored, computed });
        }

        // 4：flags、metric。
        let flags = le_u32(&header[12..16]);
        if flags != 0 {
            return Err(PersistError::UnsupportedFlags(flags));
        }
        let metric = le_u32(&header[16..20]);
        if metric != METRIC_COSINE {
            return Err(PersistError::UnsupportedMetric(metric));
        }

        // 5：header 一致性。參數範圍與 mL 一律交給 `new`。
        let dim = le_u64(&header[20..28]);
        let m = le_u64(&header[28..36]);
        let ef_construction = le_u64(&header[36..44]);
        let seed = le_u64(&header[44..52]);
        let rng_state = le_u64(&header[52..60]);
        let len = le_u64(&header[60..68]);
        let entry_point = le_u32(&header[68..72]);

        let params = HnswParams {
            m: to_usize("m", m)?,
            ef_construction: to_usize("ef_construction", ef_construction)?,
            seed,
        };
        let mut index = HnswIndex::new(to_usize("dim", dim)?, params)
            .map_err(|err| corrupt(format!("invalid parameters: {err}")))?;
        if len > u64::from(NodeId::MAX) {
            return Err(corrupt(format!("len {len} exceeds NodeId::MAX")));
        }
        let n = len as usize;
        let total = len
            .checked_mul(dim)
            .filter(|values| values.checked_mul(4).is_some())
            .ok_or_else(|| corrupt(format!("vector size overflows: len {len} × dim {dim}")))?;
        let total = to_usize("len × dim", total)?;
        match (n, entry_point) {
            (0, NO_ENTRY) => {}
            (0, _) => return Err(corrupt("empty index must not have an entry point")),
            (_, NO_ENTRY) => return Err(corrupt("non-empty index has no entry point")),
            (_, ep) if ep as usize >= n => {
                return Err(corrupt(format!("entry point {ep} out of range (len {n})")));
            }
            _ => {}
        }
        let expected_rng = seed.wrapping_add(len.wrapping_mul(GAMMA));
        if rng_state != expected_rng {
            return Err(corrupt(format!(
                "rng_state {rng_state:#018x} != seed + len × γ = {expected_rng:#018x}"
            )));
        }

        // 6：vectors。分塊讀取；只檢查、不重新正規化。
        let dim = index.dim;
        let mut vectors: Vec<f32> = Vec::new();
        let mut chunk = vec![0u8; total.min(VECTOR_CHUNK) * 4];
        let mut remaining = total;
        while remaining > 0 {
            let take = remaining.min(VECTOR_CHUNK);
            let bytes = &mut chunk[..take * 4];
            input.fill(bytes)?;
            let (words, _) = bytes.as_chunks::<4>();
            vectors.extend(words.iter().map(|w| f32::from_bits(u32::from_le_bytes(*w))));
            remaining -= take;
        }
        for (row, v) in vectors.chunks_exact(dim).enumerate() {
            if v.iter().any(|x| !x.is_finite()) {
                return Err(corrupt(format!("vector {row} contains NaN or Inf")));
            }
            let norm = v
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            if (norm - 1.0).abs() > NORM_TOLERANCE {
                return Err(corrupt(format!(
                    "vector {row} is not unit length (norm {norm})"
                )));
            }
        }

        // 7：graph。每份清單在配置前先檢查 count。
        let mut nodes: Vec<Node> = Vec::new();
        for id in 0..n {
            let level = input.u32()? as usize;
            if level > MAX_LEVEL {
                return Err(corrupt(format!("node {id}: level {level} > {MAX_LEVEL}")));
            }
            let mut neighbors = Vec::with_capacity(level + 1);
            for l in 0..=level {
                let count = input.u32()? as usize;
                let cap = index.max_neighbors(l).min(n - 1);
                if count > cap {
                    return Err(corrupt(format!(
                        "node {id} layer {l}: {count} neighbors exceed the limit {cap}"
                    )));
                }
                let mut list = Vec::with_capacity(count);
                for _ in 0..count {
                    let neighbor = input.u32()?;
                    if neighbor as usize >= n {
                        return Err(corrupt(format!(
                            "node {id} layer {l}: neighbor {neighbor} out of range (len {n})"
                        )));
                    }
                    if neighbor as usize == id {
                        return Err(corrupt(format!("node {id} layer {l}: self loop")));
                    }
                    list.push(neighbor);
                }
                let mut sorted = list.clone();
                sorted.sort_unstable();
                if sorted.windows(2).any(|w| w[0] == w[1]) {
                    return Err(corrupt(format!("node {id} layer {l}: duplicate neighbor")));
                }
                neighbors.push(list);
            }
            nodes.push(Node { neighbors });
        }
        for (id, node) in nodes.iter().enumerate() {
            for (l, list) in node.neighbors.iter().enumerate() {
                if let Some(&low) = list
                    .iter()
                    .find(|&&nb| nodes[nb as usize].neighbors.len() <= l)
                {
                    return Err(corrupt(format!(
                        "node {id} layer {l}: neighbor {low} does not exist on that layer"
                    )));
                }
            }
        }
        if n > 0 {
            let top = nodes
                .iter()
                .map(|node| node.neighbors.len())
                .max()
                .unwrap_or(0);
            let expected = nodes
                .iter()
                .position(|node| node.neighbors.len() == top)
                .expect("n > 0") as NodeId;
            if entry_point != expected {
                return Err(corrupt(format!(
                    "entry point {entry_point} is not {expected}, the smallest id on the top layer"
                )));
            }
        }

        // 8–9：file CRC，以及 trailer 之後不得有資料。
        let computed = input.crc.finish();
        let mut trailer = [0u8; 4];
        input.inner.read_exact(&mut trailer).map_err(read_error)?;
        let stored = u32::from_le_bytes(trailer);
        if stored != computed {
            return Err(PersistError::FileChecksumMismatch { stored, computed });
        }
        if has_more(&mut input.inner)? {
            return Err(PersistError::TrailingBytes);
        }

        index.vectors = vectors;
        index.nodes = nodes;
        index.entry_point = (n > 0).then_some(entry_point);
        index.rng = SplitMix64::from_state(rng_state);
        Ok(index)
    }

    /// 先寫同目錄的暫存檔，flush 與 `sync_all` 成功後才 rename；失敗時刪除暫存檔。
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), PersistError> {
        let path = path.as_ref();
        let temp = temp_path(path).map_err(PersistError::Io)?;
        // 暫存檔已存在（例如先前的 save 被中斷）時明確失敗，並指出是哪個檔案；不刪除不屬於這次的檔案。
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|err| {
                PersistError::Io(io::Error::new(
                    err.kind(),
                    format!("cannot create temporary file {}: {err}", temp.display()),
                ))
            })?;
        // 暫存檔是這裡建立的，之後任何失敗都要刪掉它，且絕不 rename。
        let result = self
            .save(&file)
            .and_then(|()| file.sync_all().map_err(PersistError::Io))
            .and_then(|()| fs::rename(&temp, path).map_err(PersistError::Io));
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    /// 以 `BufReader` 包裝後呼叫 `load`。
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, PersistError> {
        let file = File::open(path).map_err(PersistError::Io)?;
        Self::load(BufReader::new(file))
    }

    fn write_v1<W: Write>(&self, writer: W) -> io::Result<()> {
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&METRIC_COSINE.to_le_bytes());
        for value in [
            self.dim as u64,
            self.params.m as u64,
            self.params.ef_construction as u64,
            self.params.seed,
            self.rng.state(),
            self.nodes.len() as u64,
        ] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        header.extend_from_slice(&self.entry_point.unwrap_or(NO_ENTRY).to_le_bytes());
        let header_crc = crc32(&header);

        let mut out = CrcWriter {
            inner: writer,
            crc: Crc32::new(),
        };
        out.put(&header)?;
        // header CRC 欄位不算進 file CRC（見 load）。
        out.inner.write_all(&header_crc.to_le_bytes())?;
        for value in &self.vectors {
            out.put_u32(value.to_bits())?;
        }
        for node in &self.nodes {
            out.put_u32((node.neighbors.len() - 1) as u32)?;
            for list in &node.neighbors {
                out.put_u32(list.len() as u32)?;
                for &neighbor in list {
                    out.put_u32(neighbor)?;
                }
            }
        }
        let file_crc = out.crc.finish();
        out.inner.write_all(&file_crc.to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HnswParams;
    use crate::rng::SplitMix64;

    fn params(m: usize, ef_construction: usize, seed: u64) -> HnswParams {
        HnswParams {
            m,
            ef_construction,
            seed,
        }
    }

    fn data(seed: u64, n: usize, dim: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n * dim)
            .map(|_| (rng.next_unit_open0() * 2.0 - 1.0) as f32)
            .collect()
    }

    /// decision 005 驗證方式第 1 項：逐欄位比較（含私有狀態），浮點一律比較 bit pattern。
    fn assert_identical(a: &HnswIndex, b: &HnswIndex) {
        assert_eq!(a.dim, b.dim);
        assert_eq!(a.params, b.params);
        assert_eq!(a.level_mult.to_bits(), b.level_mult.to_bits());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.vectors), bits(&b.vectors));
        let graph = |index: &HnswIndex| {
            index
                .nodes
                .iter()
                .map(|n| n.neighbors.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(graph(a), graph(b));
        assert_eq!(a.entry_point, b.entry_point);
        assert_eq!(a.rng.state(), b.rng.state());
    }

    fn round_trip(index: &HnswIndex) -> HnswIndex {
        let mut bytes = Vec::new();
        index.save(&mut bytes).unwrap();
        HnswIndex::load(bytes.as_slice()).unwrap()
    }

    /// decision 005 的測試用 index (a)–(d)，外加含 `-0.0` 的向量。
    fn test_indexes() -> Vec<(&'static str, HnswIndex)> {
        let empty = HnswIndex::new(4, params(4, 20, 1)).unwrap();

        let mut single = HnswIndex::new(4, params(4, 20, 2)).unwrap();
        single.add(&[0.3, -0.4, 1.2, 0.5]).unwrap();

        let mut multi = HnswIndex::new(8, params(4, 40, 7)).unwrap();
        multi.add(&data(7, 1200, 8)).unwrap();
        assert!(multi.max_level().unwrap() >= 2);

        let base = data(12, 5, 6);
        let duplicated: Vec<f32> = (0..300)
            .flat_map(|i| {
                let b = i * 7 % 5;
                base[b * 6..(b + 1) * 6].to_vec()
            })
            .collect();
        let mut duplicates = HnswIndex::new(6, params(4, 20, 12)).unwrap();
        duplicates.add(&duplicated).unwrap();

        let mut signed_zero = HnswIndex::new(2, params(2, 4, 3)).unwrap();
        signed_zero
            .add(&[-0.0, 1.0, 1.0, -0.0, -0.0, -1.0])
            .unwrap();
        assert!(
            signed_zero
                .vectors
                .iter()
                .any(|x| x.to_bits() == 0x8000_0000)
        );

        vec![
            ("empty", empty),
            ("single", single),
            ("multi-layer", multi),
            ("duplicates", duplicates),
            ("signed zero", signed_zero),
        ]
    }

    #[test]
    fn round_trip_is_field_by_field_identical() {
        for (name, index) in test_indexes() {
            eprintln!("checking {name}");
            assert_identical(&index, &round_trip(&index));
        }
    }
}
