//! save / load 的 correctness tests（decision 005「Round-trip correctness 的驗證方式」）。
//! 結構的逐欄位（含私有狀態）比較在 crate 內的 unit test；這裡只用公開 API 與 bytes。

mod common;

use std::io::{self, Write};
use std::path::PathBuf;

use common::SplitMix64;
use common::spec_codec::{
    self, GAMMA, HEADER_LEN, MAGIC, NO_ENTRY, SpecFile, crc32_bitwise, recompute_crcs,
};
use hnsw::distance::normalize;
use hnsw::{
    ExactIndex, FORMAT_VERSION, HnswIndex, HnswParams, MAX_M, NodeId, PADDING_ID, PersistError,
};

// ---- 共用工具 ----

fn params(m: usize, ef_construction: usize, seed: u64) -> HnswParams {
    HnswParams {
        m,
        ef_construction,
        seed,
    }
}

fn build(data: &[f32], dim: usize, p: HnswParams) -> HnswIndex {
    let mut index = HnswIndex::new(dim, p).unwrap();
    index.add(data).unwrap();
    index
}

fn to_bytes(index: &HnswIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    index.save(&mut bytes).unwrap();
    bytes
}

fn load(bytes: &[u8]) -> Result<HnswIndex, PersistError> {
    HnswIndex::load(bytes)
}

fn graph_snapshot(index: &HnswIndex) -> Vec<Vec<Vec<NodeId>>> {
    (0..index.len() as NodeId)
        .map(|id| {
            (0..=index.level(id).unwrap())
                .map(|l| index.neighbors(id, l).unwrap().to_vec())
                .collect()
        })
        .collect()
}

fn assert_same_public_structure(a: &HnswIndex, b: &HnswIndex) {
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a.params(), b.params());
    assert_eq!(a.len(), b.len());
    assert_eq!(a.entry_point(), b.entry_point());
    assert_eq!(a.max_level(), b.max_level());
    assert_eq!(graph_snapshot(a), graph_snapshot(b));
}

/// 搜尋結果的 bit 層級表示：id 與 distance 的 bit pattern。
fn batch_bits(index: &HnswIndex, queries: &[f32], k: usize, ef: usize) -> (Vec<u32>, Vec<u32>) {
    let batch = index.search_batch(queries, k, ef).unwrap();
    let bits = batch.distances.iter().map(|d| d.to_bits()).collect();
    (batch.ids, bits)
}

fn duplicate_data(seed: u64, dim: usize, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let base = rng.vectors(5, dim);
    let mut order: Vec<usize> = (0..n).map(|i| i % 5).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
        .iter()
        .flat_map(|&b| base[b * dim..(b + 1) * dim].to_vec())
        .collect()
}

// ---- decision 005 的測試用 index (a)–(d) ----

const MULTI_DIM: usize = 8;

/// (a) n = 0。
fn empty_index() -> HnswIndex {
    HnswIndex::new(4, params(4, 20, 1)).unwrap()
}

/// (b) n = 1。
fn single_index() -> HnswIndex {
    build(&[0.3, -0.4, 1.2, 0.5], 4, params(4, 20, 2))
}

/// (c) 多層 graph：M = 4、n = 1200，max level ≥ 2。
fn multi_layer_index() -> (HnswIndex, Vec<f32>) {
    let mut rng = SplitMix64::new(7);
    let data = rng.vectors(1200, MULTI_DIM);
    let index = build(&data, MULTI_DIM, params(4, 40, 7));
    assert!(index.max_level().unwrap() >= 2, "need a multi-layer graph");
    (index, data)
}

/// (d) 004 的大量重複向量（會出現結果不足 k）。
fn duplicate_index() -> HnswIndex {
    build(&duplicate_data(12, 6, 300), 6, params(4, 20, 12))
}

/// 窮舉損壞測試用的小 index（檔案約 300 bytes）。
fn small_index() -> HnswIndex {
    let mut rng = SplitMix64::new(3);
    build(&rng.vectors(6, 3), 3, params(2, 8, 3))
}

/// ef_construction < m：檔案必須存使用者設定值（3），而不是 max(efC, M) = 8。
fn small_efc_index() -> (HnswIndex, Vec<f32>) {
    let mut rng = SplitMix64::new(5);
    let data = rng.vectors(300, 5);
    (build(&data, 5, params(8, 3, 99)), data)
}

fn all_test_indexes() -> Vec<(&'static str, HnswIndex)> {
    vec![
        ("empty", empty_index()),
        ("single", single_index()),
        ("multi-layer", multi_layer_index().0),
        ("duplicates", duplicate_index()),
        ("small", small_index()),
        ("efC < m", small_efc_index().0),
    ]
}

// ---- 1. 結構與參數一致（公開 API 部分） ----

#[test]
fn round_trip_preserves_public_structure() {
    for (name, index) in all_test_indexes() {
        let loaded = load(&to_bytes(&index)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_same_public_structure(&index, &loaded);
    }
}

// ---- 2. 搜尋等價 ----

#[test]
fn loaded_multi_layer_index_gives_bitwise_identical_results() {
    let (index, data) = multi_layer_index();
    let loaded = load(&to_bytes(&index)).unwrap();
    let mut rng = SplitMix64::new(70);
    let queries = rng.vectors(120, MULTI_DIM);
    let k = 10;

    // efSearch 必須包含尚未飽和的值，否則搜尋等價的比較太容易通過。
    let mut exact = ExactIndex::new(MULTI_DIM).unwrap();
    exact.add(&data).unwrap();
    let truth = exact.search_batch(&queries, k).unwrap();
    let low = index.search_batch(&queries, k, 1).unwrap();
    let hits = low
        .ids
        .chunks_exact(k)
        .zip(truth.ids.chunks_exact(k))
        .map(|(f, t)| f.iter().filter(|id| t.contains(id)).count())
        .sum::<usize>();
    assert!(hits < 120 * k, "ef = 1 should not saturate recall");

    for ef in [1, 4, 16, 64] {
        assert_eq!(
            batch_bits(&index, &queries, k, ef),
            batch_bits(&loaded, &queries, k, ef),
            "ef {ef}"
        );
        for query in queries.as_chunks::<MULTI_DIM>().0.iter().take(20) {
            let a = index.search(query, k, ef).unwrap();
            let b = loaded.search(query, k, ef).unwrap();
            let bits = |r: &[hnsw::Neighbor]| -> Vec<(u32, u32)> {
                r.iter().map(|n| (n.id, n.distance.to_bits())).collect()
            };
            assert_eq!(bits(&a), bits(&b));
        }
    }
}

#[test]
fn loaded_duplicate_index_gives_identical_padded_batches() {
    let index = duplicate_index();
    let loaded = load(&to_bytes(&index)).unwrap();
    let mut rng = SplitMix64::new(13);
    let queries = rng.vectors(10, 6);
    let original = batch_bits(&index, &queries, 80, 300);
    assert!(
        original.0.contains(&PADDING_ID),
        "this dataset should produce padded rows (decision 004)"
    );
    assert_eq!(original, batch_bits(&loaded, &queries, 80, 300));
}

#[test]
fn loaded_single_and_empty_indexes_search_identically() {
    let single = single_index();
    let loaded = load(&to_bytes(&single)).unwrap();
    let query = [1.0, 0.0, 0.0, 0.0];
    let bits = |r: Vec<hnsw::Neighbor>| -> Vec<(u32, u32)> {
        r.iter().map(|n| (n.id, n.distance.to_bits())).collect()
    };
    assert_eq!(
        bits(single.search(&query, 1, 10).unwrap()),
        bits(loaded.search(&query, 1, 10).unwrap())
    );

    let empty = empty_index();
    let loaded = load(&to_bytes(&empty)).unwrap();
    assert_eq!(
        empty.search(&query, 1, 10).unwrap_err(),
        hnsw::HnswError::EmptyIndex
    );
    assert_eq!(
        loaded.search(&query, 1, 10).unwrap_err(),
        hnsw::HnswError::EmptyIndex
    );
}

// ---- 3. 繼續插入 ----

#[test]
fn adding_after_load_equals_adding_without_save() {
    let (mut original, _) = multi_layer_index();
    let mut loaded = load(&to_bytes(&original)).unwrap();
    let before = original.len();
    let mut rng = SplitMix64::new(71);
    let more = rng.vectors(150, MULTI_DIM);
    original.add(&more).unwrap();
    loaded.add(&more).unwrap();

    // M = 4 時 P(level ≥ 1) = 1/4；150 個新點幾乎必定有 level ≥ 1 的點，RNG 狀態錯了就會不同。
    let new_upper = (before..original.len())
        .filter(|&id| original.level(id as NodeId).unwrap() >= 1)
        .count();
    assert!(new_upper > 0);
    assert_eq!(to_bytes(&original), to_bytes(&loaded));
}

#[test]
fn adding_after_loading_an_empty_index_equals_a_fresh_index() {
    let mut rng = SplitMix64::new(72);
    let batch = rng.vectors(200, 4);

    let mut fresh = empty_index();
    let mut loaded = load(&to_bytes(&empty_index())).unwrap();
    fresh.add(&batch).unwrap();
    loaded.add(&batch).unwrap();
    assert_eq!(to_bytes(&fresh), to_bytes(&loaded));
}

// ---- 4. 決定性 ----

#[test]
fn save_is_deterministic_and_idempotent() {
    for (name, index) in all_test_indexes() {
        let first = to_bytes(&index);
        assert_eq!(first, to_bytes(&index), "{name}: save twice");
        let reloaded = load(&first).unwrap();
        assert_eq!(first, to_bytes(&reloaded), "{name}: save(load(save(x)))");
    }
}

// ---- 5. 規格一致 ----

#[test]
fn independent_crc_matches_check_values() {
    assert_eq!(crc32_bitwise(b""), 0);
    assert_eq!(crc32_bitwise(b"123456789"), 0xCBF4_3926);
}

#[test]
fn independent_decoder_agrees_with_saved_files() {
    let (index, data) = small_efc_index();

    let indexes = [
        (index, Some(data)),
        (empty_index(), Some(Vec::new())),
        (multi_layer_index().0, None),
    ];
    for (index, data) in &indexes {
        let bytes = to_bytes(index);
        let spec = spec_codec::decode(&bytes);
        assert_eq!(spec.version, FORMAT_VERSION);
        assert_eq!((spec.version, spec.flags, spec.metric), (1, 0, 1));
        assert_eq!(spec.dim, index.dim() as u64);
        assert_eq!(spec.m, index.params().m as u64);
        assert_eq!(spec.ef_construction, index.params().ef_construction as u64);
        assert_eq!(spec.seed, index.params().seed);
        assert_eq!(spec.len, index.len() as u64);
        assert_eq!(
            spec.rng_state,
            spec.seed.wrapping_add(spec.len.wrapping_mul(GAMMA))
        );
        assert_eq!(spec.entry_point, index.entry_point().unwrap_or(NO_ENTRY));
        assert_eq!(spec.graph, graph_snapshot(index));
        if let Some(data) = data {
            let expected: Vec<u32> = data
                .chunks_exact(index.dim())
                .flat_map(|row| normalize(row).unwrap())
                .map(|x| x.to_bits())
                .collect();
            assert_eq!(spec.vector_bits, expected);
        }
        assert_eq!(spec_codec::encode(&spec), bytes);
    }
    assert_eq!(indexes[0].0.params().ef_construction, 3);
}

#[test]
fn empty_index_file_is_exactly_80_bytes() {
    assert_eq!(to_bytes(&empty_index()).len(), HEADER_LEN + 4);
}

// ---- golden fixtures（依 decision 005 以 Python struct + zlib 手工編碼；永不由 save 產生或覆寫） ----

/// n = 0，dim = 3，m = 2，efC = 5，seed = 42。
const GOLDEN_EMPTY: [u8; 80] = [
    0x52, 0x48, 0x4e, 0x53, 0x57, 0x49, 0x44, 0x58, 0x01, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, //
    0x30, 0x91, 0x8b, 0x74, 0x30, 0x91, 0x8b, 0x74,
];

/// n = 5，dim = 2，m = 2，efC = 10，seed = 7。
/// 向量：id 0 = [1, 0]、1 = [0, 1]、2 = [-1, 0]、3 = [0, -1]、4 = [0.6, 0.8]。
/// level：[0, 1, 0, 1, 0]，entry = 1（最高層中 id 最小者）。
/// layer 0：0 → [1, 4, 3, 2]、1 → [4, 0, 2]、2 → [1, 3]、3 → []、4 → [0, 1]。
/// layer 1：1 → [3]、3 → [1]。
const GOLDEN_SMALL: [u8; 220] = [
    0x52, 0x48, 0x4e, 0x53, 0x57, 0x49, 0x44, 0x58, 0x01, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x70, 0x6c, 0x74, 0x7c, 0x9f, 0x60, 0x15, 0x17, //
    0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, //
    0xc7, 0x4a, 0xe8, 0x94, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf, //
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf, //
    0x9a, 0x99, 0x19, 0x3f, 0xcd, 0xcc, 0x4c, 0x3f, 0x00, 0x00, 0x00, 0x00, //
    0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, //
    0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, //
    0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, //
    0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, //
    0x47, 0x86, 0x29, 0x3c,
];

fn golden_small_spec() -> SpecFile {
    SpecFile::valid(
        2,
        2,
        10,
        7,
        &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.6, 0.8],
        vec![
            vec![vec![1, 4, 3, 2]],
            vec![vec![4, 0, 2], vec![3]],
            vec![vec![1, 3]],
            vec![vec![], vec![1]],
            vec![vec![0, 1]],
        ],
    )
}

#[test]
fn golden_fixtures_match_the_independent_encoder() {
    assert_eq!(
        spec_codec::encode(&SpecFile::valid(3, 2, 5, 42, &[], vec![])),
        GOLDEN_EMPTY
    );
    assert_eq!(spec_codec::encode(&golden_small_spec()), GOLDEN_SMALL);
}

#[test]
fn golden_empty_loads_with_expected_contents() {
    let index = load(&GOLDEN_EMPTY).unwrap();
    assert_eq!(index.dim(), 3);
    assert_eq!(index.params(), params(2, 5, 42));
    assert_eq!(index.len(), 0);
    assert_eq!(index.entry_point(), None);
    assert_eq!(index.max_level(), None);
    if FORMAT_VERSION == 1 {
        assert_eq!(to_bytes(&index), GOLDEN_EMPTY);
    }
}

#[test]
fn golden_small_loads_with_expected_contents() {
    let index = load(&GOLDEN_SMALL).unwrap();
    assert_eq!(index.dim(), 2);
    assert_eq!(index.params(), params(2, 10, 7));
    assert_eq!(index.len(), 5);
    assert_eq!(index.entry_point(), Some(1));
    assert_eq!(index.max_level(), Some(1));
    let levels: Vec<usize> = (0..5).map(|id| index.level(id).unwrap()).collect();
    assert_eq!(levels, [0, 1, 0, 1, 0]);
    assert_eq!(graph_snapshot(&index), golden_small_spec().graph);

    // 向量 bit pattern 與 rng_state 不在公開 API；以 save 的 bytes 逐位元比較。
    if FORMAT_VERSION == 1 {
        assert_eq!(to_bytes(&index), GOLDEN_SMALL);
    }

    // 人工可驗證的搜尋：query [1, 0]，與各點的角度差為 0°、53.13°、90°、90°、180°。
    let result = index.search(&[1.0, 0.0], 5, 5).unwrap();
    let ids: Vec<NodeId> = result.iter().map(|n| n.id).collect();
    assert_eq!(ids, [0, 4, 1, 3, 2]);
    let expected = [0.0, 0.4, 1.0, 1.0, 2.0];
    for (n, e) in result.iter().zip(expected) {
        assert!((n.distance - e).abs() < 1e-6, "{} vs {e}", n.distance);
    }
    let nearest = index.search(&[0.6, 0.8], 1, 1).unwrap();
    assert_eq!(nearest[0].id, 4);
}

#[test]
fn load_checks_but_never_renormalizes_vectors() {
    // ‖v‖ = 1.0005：在 1e-3 容忍範圍內，必須被接受，且 bit pattern 原樣保留（decision 005 第 6 項）。
    let mut spec = golden_small_spec();
    spec.vector_bits[0] = 1.0005f32.to_bits();
    let bytes = spec_codec::encode(&spec);
    let index = load(&bytes).unwrap();
    assert_eq!(to_bytes(&index), bytes);

    spec.vector_bits[0] = 1.002f32.to_bits();
    assert_corrupt(&spec, "norm off by 2e-3");
}

// ---- 6. 損壞與截斷 ----

#[test]
fn every_truncation_is_reported_as_truncated() {
    for bytes in [to_bytes(&small_index()), GOLDEN_SMALL.to_vec()] {
        for len in 0..bytes.len() {
            let result = load(&bytes[..len]);
            assert!(
                matches!(result, Err(PersistError::Truncated)),
                "len {len}: {result:?}"
            );
        }
    }
}

#[test]
fn every_single_bit_flip_is_rejected() {
    for bytes in [to_bytes(&small_index()), GOLDEN_SMALL.to_vec()] {
        for i in 0..bytes.len() {
            for bit in 0..8 {
                let mut corrupted = bytes.clone();
                corrupted[i] ^= 1 << bit;
                assert!(load(&corrupted).is_err(), "byte {i} bit {bit} was accepted");
            }
        }
    }
}

#[test]
fn trailing_bytes_are_rejected() {
    let bytes = to_bytes(&small_index());
    for extra in [vec![0u8], vec![0xFF; 100]] {
        let mut longer = bytes.clone();
        longer.extend_from_slice(&extra);
        assert!(matches!(load(&longer), Err(PersistError::TrailingBytes)));
    }
}

#[test]
fn bad_magic_and_short_files() {
    let mut bytes = to_bytes(&small_index());
    bytes[0] ^= 0x01;
    assert!(matches!(load(&bytes), Err(PersistError::BadMagic)));
    assert!(matches!(load(b"NOTINDEX"), Err(PersistError::BadMagic)));
    assert!(matches!(load(b"RHNSWID"), Err(PersistError::Truncated)));
    assert!(matches!(load(b""), Err(PersistError::Truncated)));
}

#[test]
fn newer_versions_are_unsupported_without_parsing_the_rest() {
    for version in [0u32, 2, u32::MAX] {
        let mut prefix = MAGIC.to_vec();
        prefix.extend_from_slice(&version.to_le_bytes());
        let mut with_garbage = prefix.clone();
        with_garbage.extend_from_slice(&[0xAB; 64]);
        for bytes in [prefix, with_garbage] {
            let result = load(&bytes);
            assert!(
                matches!(result, Err(PersistError::UnsupportedVersion { found }) if found == version),
                "version {version}, {} bytes: {result:?}",
                bytes.len()
            );
        }
    }
}

#[test]
fn header_bit_flips_are_classified_by_position() {
    // decision 005 第 1–3 項的順序：magic → version → header CRC（在 flags、metric 與任何大小欄位之前）。
    let bytes = to_bytes(&small_index());
    for i in 0..HEADER_LEN {
        for bit in 0..8 {
            let mut corrupted = bytes.clone();
            corrupted[i] ^= 1 << bit;
            let result = load(&corrupted);
            let ok = match i {
                0..8 => matches!(result, Err(PersistError::BadMagic)),
                8..12 => matches!(result, Err(PersistError::UnsupportedVersion { .. })),
                _ => matches!(result, Err(PersistError::HeaderChecksumMismatch { .. })),
            };
            assert!(ok, "byte {i} bit {bit}: {result:?}");
        }
    }
}

#[test]
fn header_checksum_mismatch_is_detected() {
    let mut bytes = to_bytes(&small_index());
    bytes[20] ^= 0x01; // dim 欄位，不重算 CRC
    assert!(matches!(
        load(&bytes),
        Err(PersistError::HeaderChecksumMismatch { .. })
    ));
}

#[test]
fn unsupported_flags_and_metric_are_rejected() {
    let base = spec_codec::decode(&to_bytes(&small_index()));
    let mut flags = base.clone();
    flags.flags = 1;
    assert!(matches!(
        load(&spec_codec::encode(&flags)),
        Err(PersistError::UnsupportedFlags(1))
    ));
    let mut metric = base;
    metric.metric = 2;
    assert!(matches!(
        load(&spec_codec::encode(&metric)),
        Err(PersistError::UnsupportedMetric(2))
    ));
}

#[test]
fn swapping_payloads_between_indexes_is_detected() {
    // 同資料、同 seed、同 m，只有 efC 不同：level 序列、entry、rng_state 都相同。
    let mut rng = SplitMix64::new(21);
    let data = rng.vectors(200, 4);
    let a = to_bytes(&build(&data, 4, params(4, 4, 21)));
    let b = to_bytes(&build(&data, 4, params(4, 64, 21)));
    assert_ne!(a[HEADER_LEN..], b[HEADER_LEN..], "graphs should differ");

    let mut spliced = a[..HEADER_LEN].to_vec();
    spliced.extend_from_slice(&b[HEADER_LEN..]);
    assert!(matches!(
        load(&spliced),
        Err(PersistError::FileChecksumMismatch { .. })
    ));
}

/// 對 `SpecFile` 的一個修改（產生 CRC 正確但語意不合法的檔案）。
type Mutation = Box<dyn Fn(&mut SpecFile)>;

fn assert_corrupt(spec: &SpecFile, case: &str) {
    let result = load(&spec_codec::encode(spec));
    assert!(
        matches!(result, Err(PersistError::Corrupt(_))),
        "{case}: {result:?}"
    );
}

#[test]
fn checksum_valid_but_invalid_headers_are_corrupt() {
    let base = spec_codec::decode(&to_bytes(&small_index()));
    load(&spec_codec::encode(&base)).expect("base spec must be valid");

    let cases: Vec<(&str, Mutation)> = vec![
        ("m = 1", Box::new(|s| s.m = 1)),
        ("m = MAX_M + 1", Box::new(|s| s.m = MAX_M as u64 + 1)),
        ("m = 2^63", Box::new(|s| s.m = 1 << 63)),
        ("m = u64::MAX", Box::new(|s| s.m = u64::MAX)),
        ("dim = 0", Box::new(|s| s.dim = 0)),
        ("ef_construction = 0", Box::new(|s| s.ef_construction = 0)),
        ("n * dim overflows", Box::new(|s| s.dim = u64::MAX)),
        // n = 6：n·dim = 3·2^62 不溢位，×4 才溢位。
        ("n * dim * 4 overflows", Box::new(|s| s.dim = 1 << 61)),
        (
            "len > NodeId::MAX",
            Box::new(|s| {
                s.len = u64::from(NodeId::MAX) + 1;
                s.rng_state = s.seed.wrapping_add(s.len.wrapping_mul(GAMMA));
            }),
        ),
        ("entry out of range", Box::new(|s| s.entry_point = 6)),
        (
            "entry sentinel with n > 0",
            Box::new(|s| s.entry_point = NO_ENTRY),
        ),
        ("rng_state changed", Box::new(|s| s.rng_state ^= 1)),
        ("seed changed", Box::new(|s| s.seed ^= 1)),
    ];
    for (case, mutate) in cases {
        let mut spec = base.clone();
        mutate(&mut spec);
        assert_corrupt(&spec, case);
        // header 階段就要擋下（005「回傳最先偵測到的錯誤」）：只有 header 時也是 Corrupt，不是 Truncated。
        let header_only = load(&spec_codec::encode_header(&spec));
        assert!(
            matches!(header_only, Err(PersistError::Corrupt(_))),
            "{case} (header only): {header_only:?}"
        );
    }

    let mut empty = SpecFile::valid(3, 2, 5, 42, &[], vec![]);
    empty.entry_point = 0;
    assert_corrupt(&empty, "n = 0 with an entry point");
    let mut empty = SpecFile::valid(3, 2, 5, 42, &[], vec![]);
    empty.rng_state += 1;
    assert_corrupt(&empty, "n = 0 with rng_state != seed");
}

#[test]
fn checksum_valid_but_invalid_payloads_are_corrupt() {
    let base = golden_small_spec();
    load(&spec_codec::encode(&base)).expect("golden spec must be valid");

    let cases: Vec<(&str, Mutation)> = vec![
        (
            "NaN component",
            Box::new(|s| s.vector_bits[3] = f32::NAN.to_bits()),
        ),
        (
            "Inf component",
            Box::new(|s| s.vector_bits[0] = f32::INFINITY.to_bits()),
        ),
        (
            "non-unit vector",
            Box::new(|s| s.vector_bits[0] = 2.0f32.to_bits()),
        ),
        (
            "zero vector",
            Box::new(|s| s.vector_bits[0] = 0.0f32.to_bits()),
        ),
        (
            "neighbor id out of range",
            Box::new(|s| s.graph[2][0] = vec![1, 5]),
        ),
        ("self loop", Box::new(|s| s.graph[2][0] = vec![2, 1])),
        (
            "duplicate neighbor",
            Box::new(|s| s.graph[2][0] = vec![1, 1]),
        ),
        (
            "neighbor below the layer",
            Box::new(|s| s.graph[1][1] = vec![0]),
        ),
        (
            "entry is not the smallest id on the top layer",
            Box::new(|s| s.entry_point = 3),
        ),
        ("entry below the top layer", Box::new(|s| s.entry_point = 0)),
    ];
    for (case, mutate) in cases {
        let mut spec = base.clone();
        mutate(&mut spec);
        assert_corrupt(&spec, case);
    }
}

#[test]
fn level_64_is_accepted_and_65_is_corrupt() {
    // node 1 本來就是 entry（最高層 id 最小者），升層後仍是唯一的最高層節點，
    // 所以 level 65 只會被 level 上限擋下，不會被 entry 檢查擋下。
    let mut spec = golden_small_spec();
    spec.graph[1].extend(vec![vec![]; 63]);
    let index = load(&spec_codec::encode(&spec)).expect("level 64 must be accepted");
    assert_eq!(index.level(1), Some(64));
    assert_eq!(index.max_level(), Some(64));
    spec.graph[1].push(vec![]);
    assert_corrupt(&spec, "level 65");
}

#[test]
fn per_layer_capacity_is_enforced_below_n_minus_1() {
    // 只違反 cap(l)，不違反 n − 1、id 範圍、重複或 level 規則。
    // (i) layer 0：n = 6、m = 2，count 5 > 2m = 4，且 ≤ n − 1 = 5。
    let mut spec = spec_codec::decode(&to_bytes(&small_index()));
    spec.graph[0][0] = vec![1, 2, 3, 4, 5];
    assert_corrupt(&spec, "layer-0 count > 2m");

    // (ii) layer 1：四個節點都在 level 1；node 0 的 layer-1 count 3 > m = 2，且 ≤ n − 1 = 3。
    let vectors = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
    let rest = || vec![vec![0], vec![0]];
    let graph = |upper: Vec<u32>| vec![vec![vec![1], upper], rest(), rest(), rest()];
    let ok = SpecFile::valid(2, 2, 10, 7, &vectors, graph(vec![1, 2]));
    load(&spec_codec::encode(&ok)).expect("count = m must be accepted");
    let bad = SpecFile::valid(2, 2, 10, 7, &vectors, graph(vec![1, 2, 3]));
    assert_corrupt(&bad, "layer-1 count > m");
}

#[test]
fn huge_count_is_corrupt() {
    // m 合法但很大、n 小。u32::MAX 由 `count ≤ 2m` 擋下；2m（= 2^32 − 2）通過 `count ≤ 2m`，
    // 只能由 `count ≤ n − 1` 擋下。配置前就擋下這件事由 tests/load_alloc.rs 以峰值計數驗證；
    // 這裡只驗證錯誤分類。
    let mut spec = golden_small_spec();
    spec.m = MAX_M as u64;
    let base = spec_codec::encode(&spec);
    load(&base).expect("base spec must be valid");
    // node 0 的 layer-0 count 位於 graph 開頭（vectors 之後）的 level 欄位之後。
    let count_at = HEADER_LEN + 5 * 2 * 4 + 4;
    for count in [u32::MAX, u32::try_from(2 * MAX_M).unwrap()] {
        let mut bytes = base.clone();
        bytes[count_at..count_at + 4].copy_from_slice(&count.to_le_bytes());
        recompute_crcs(&mut bytes);
        let result = load(&bytes);
        assert!(
            matches!(result, Err(PersistError::Corrupt(_))),
            "count {count}: {result:?}"
        );
    }
}

// ---- load 端的 I/O 錯誤 ----

/// 讀到第 `limit` byte 後，每次 read 都回 `error_kind` 的錯誤。
struct FailingReader<'a> {
    data: &'a [u8],
    pos: usize,
    limit: usize,
    error_kind: io::ErrorKind,
}

impl io::Read for FailingReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.limit {
            return Err(io::Error::new(self.error_kind, "injected"));
        }
        let n = buf.len().min(self.limit - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[test]
fn load_reports_read_errors_as_io() {
    // limit = len 涵蓋 trailer 之後的探測讀取：讀取錯誤不得被當成「沒有更多資料」而接受。
    let bytes = to_bytes(&small_index());
    for limit in 0..=bytes.len() {
        let reader = FailingReader {
            data: &bytes,
            pos: 0,
            limit,
            error_kind: io::ErrorKind::Other,
        };
        let result = HnswIndex::load(reader);
        assert!(
            matches!(result, Err(PersistError::Io(_))),
            "limit {limit}: {result:?}"
        );
    }
}

#[test]
fn unexpected_eof_errors_are_truncated_everywhere() {
    // 005：任何讀取遇到 UnexpectedEof 一律回 Truncated，包括 trailer 之後的探測讀取。
    let bytes = to_bytes(&small_index());
    for limit in [0, HEADER_LEN, bytes.len() - 1, bytes.len()] {
        let reader = FailingReader {
            data: &bytes,
            pos: 0,
            limit,
            error_kind: io::ErrorKind::UnexpectedEof,
        };
        let result = HnswIndex::load(reader);
        assert!(
            matches!(result, Err(PersistError::Truncated)),
            "limit {limit}: {result:?}"
        );
    }
}

// ---- save 端的 I/O 錯誤 ----

/// 寫入 `limit` bytes 後回錯誤；`fail_flush` 時 flush 也回錯誤。
struct FailingWriter {
    limit: usize,
    written: usize,
    fail_flush: bool,
}

impl Write for FailingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.written >= self.limit {
            return Err(io::Error::other("disk full"));
        }
        let n = buf.len().min(self.limit - self.written);
        self.written += n;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.fail_flush {
            Err(io::Error::other("flush failed"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn save_reports_write_errors_at_every_byte() {
    let index = small_index();
    let len = to_bytes(&index).len();
    for limit in 0..len {
        let writer = FailingWriter {
            limit,
            written: 0,
            fail_flush: false,
        };
        assert!(
            matches!(index.save(writer), Err(PersistError::Io(_))),
            "limit {limit}"
        );
    }
}

#[test]
fn save_reports_flush_errors() {
    let index = small_index();
    let writer = FailingWriter {
        limit: usize::MAX,
        written: 0,
        fail_flush: true,
    };
    assert!(matches!(index.save(writer), Err(PersistError::Io(_))));
}

// ---- 檔案路徑 API ----

fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dir_entries(dir: &PathBuf) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn path_round_trip_leaves_no_temporary_files() {
    let dir = fresh_dir("persist_path_round_trip");
    let path = dir.join("index.rhnsw");
    let (index, _) = multi_layer_index();
    index.save_to_path(&path).unwrap();
    assert_eq!(dir_entries(&dir), ["index.rhnsw"]);
    assert_eq!(std::fs::read(&path).unwrap(), to_bytes(&index));

    let loaded = HnswIndex::load_from_path(&path).unwrap();
    assert_eq!(to_bytes(&loaded), to_bytes(&index));

    // 覆寫既有檔案。
    single_index().save_to_path(&path).unwrap();
    assert_eq!(dir_entries(&dir), ["index.rhnsw"]);
    assert_eq!(std::fs::read(&path).unwrap(), to_bytes(&single_index()));
}

#[test]
fn failed_rename_removes_temporary_file() {
    // 目標路徑是既有目錄：暫存檔已建立並寫完，rename 才失敗。
    let dir = fresh_dir("persist_failed_rename");
    let target = dir.join("index.rhnsw");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("keep"), b"keep").unwrap();
    assert!(matches!(
        small_index().save_to_path(&target),
        Err(PersistError::Io(_))
    ));
    assert_eq!(dir_entries(&dir), ["index.rhnsw"]);
    assert_eq!(std::fs::read(target.join("keep")).unwrap(), b"keep");
}

#[test]
fn existing_temporary_file_is_reported_and_kept() {
    // 例如先前的 save 被中斷而留下暫存檔：明確失敗並指出路徑，不刪除不屬於這次的檔案。
    let dir = fresh_dir("persist_existing_temp");
    let temp = dir.join(format!(".index.rhnsw.tmp-{}", std::process::id()));
    std::fs::write(&temp, b"junk").unwrap();
    match small_index().save_to_path(dir.join("index.rhnsw")) {
        Err(PersistError::Io(err)) => {
            assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
            assert!(
                err.to_string().contains(&temp.display().to_string()),
                "{err}"
            );
        }
        other => panic!("expected an Io error, got {other:?}"),
    }
    assert_eq!(std::fs::read(&temp).unwrap(), b"junk");
    assert!(!dir.join("index.rhnsw").exists());
}

#[test]
fn path_errors_are_io_errors() {
    let dir = fresh_dir("persist_path_errors");
    let missing_dir = dir.join("missing").join("index.rhnsw");
    assert!(matches!(
        small_index().save_to_path(&missing_dir),
        Err(PersistError::Io(_))
    ));
    assert!(dir_entries(&dir).is_empty());
    assert!(matches!(
        HnswIndex::load_from_path(dir.join("nope.rhnsw")),
        Err(PersistError::Io(_))
    ));
}
