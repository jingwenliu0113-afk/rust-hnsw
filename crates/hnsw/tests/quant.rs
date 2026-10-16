//! Advanced C 的 `QuantIndex` correctness tests（decision 019 §4「correctness 測試 — Rust」）。
//! synthetic random vectors 只用於 correctness，不作 benchmark 結論；本檔不碰任何真實資料。

mod common;

use common::{SplitMix64, unit_circle};
use hnsw::distance::{cosine_distance_unit, normalize};
use hnsw::{
    BatchResult, ExactIndex, HnswError, HnswIndex, HnswParams, NodeId, PADDING_ID, QuantIndex,
    Representation, sq8_train_encode,
};

const REPRESENTATIONS: [Representation; 2] = [Representation::Fp32, Representation::Sq8];

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

/// 逐列以 crate 的規則正規化，與 `add` 存下的 unit vectors 逐位元相同。
fn unit_rows(data: &[f32], dim: usize) -> Vec<f32> {
    data.chunks_exact(dim)
        .flat_map(|v| normalize(v).unwrap())
        .collect()
}

/// 搜尋結果的 bit 層級表示：id 與 distance 的 bit pattern。
fn bits(batch: &BatchResult) -> (usize, usize, Vec<NodeId>, Vec<u32>) {
    let distances = batch.distances.iter().map(|d| d.to_bits()).collect();
    (batch.num_queries, batch.k, batch.ids.clone(), distances)
}

/// 004 的大量重複向量：5 個方向各重複多次並打散（會出現結果不足 k）。
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

/// 等價測試用的 fixture。
struct Fixture {
    name: String,
    dim: usize,
    data: Vec<f32>,
    params: HnswParams,
    queries: Vec<f32>,
}

impl Fixture {
    fn new(name: &str, dim: usize, data: Vec<f32>, params: HnswParams, queries: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            dim,
            data,
            params,
            queries,
        }
    }

    fn n(&self) -> usize {
        self.data.len() / self.dim
    }

    fn core(&self) -> HnswIndex {
        build(&self.data, self.dim, self.params)
    }
}

/// 隨機 query，加上存進去的前幾列向量本身（精確重複的點：距離接近 0，FP32 可能略為負）。
fn queries_with_stored_rows(rng: &mut SplitMix64, data: &[f32], dim: usize, nq: usize) -> Vec<f32> {
    let mut queries = rng.vectors(nq, dim);
    let stored = (data.len() / dim).min(5);
    queries.extend_from_slice(&data[..stored * dim]);
    queries
}

/// 019 §4：多組資料與 (M, efC, seed)，含 004 / 007 的重複向量、n ≤ M0 + 1、n = 1。
fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();

    let mut rng = SplitMix64::new(42);
    let data = rng.vectors(1000, 16);
    let queries = queries_with_stored_rows(&mut rng, &data, 16, 25);
    out.push(Fixture::new(
        "random 16-d",
        16,
        data,
        params(8, 64, 42),
        queries,
    ));

    let mut rng = SplitMix64::new(7);
    let data = rng.vectors(1200, 8);
    let queries = queries_with_stored_rows(&mut rng, &data, 8, 25);
    let multi = Fixture::new("multi-layer", 8, data, params(4, 40, 7), queries);
    assert!(
        multi.core().max_level().unwrap() >= 2,
        "need a multi-layer graph"
    );
    out.push(multi);

    let mut rng = SplitMix64::new(5);
    let data = rng.vectors(300, 5);
    let queries = queries_with_stored_rows(&mut rng, &data, 5, 20);
    out.push(Fixture::new("efC < m", 5, data, params(8, 3, 99), queries));

    // golden test 的形狀：12 維、一列精確重複（tie 由 (distance, id) 決定）。
    let mut rng = SplitMix64::new(1);
    let mut data = rng.vectors(600, 12);
    let row7 = data[7 * 12..8 * 12].to_vec();
    data[100 * 12..101 * 12].copy_from_slice(&row7);
    let mut queries = rng.vectors(20, 12);
    queries.extend_from_slice(&row7);
    out.push(Fixture::new(
        "duplicate row",
        12,
        data,
        params(16, 200, 1),
        queries,
    ));

    // 004 / 007 的大量重複向量（2 組資料 × 2 個 build seed × M ∈ {4, 8}，efC = 20）。
    for data_seed in [12, 13] {
        for build_seed in [12, 13] {
            for m in [4, 8] {
                let data = duplicate_data(data_seed, 6, 300);
                let queries = SplitMix64::new(data_seed + 100).vectors(10, 6);
                out.push(Fixture::new(
                    &format!("duplicates data {data_seed} seed {build_seed} M {m}"),
                    6,
                    data,
                    params(m, 20, build_seed),
                    queries,
                ));
            }
        }
    }

    // 007 的 tie fixture：id 0–4 都是 [1, 0]，id 5–9 都是 [0, 1]。
    let tie: Vec<f32> = (0..10)
        .flat_map(|i| if i < 5 { [1.0, 0.0] } else { [0.0, 1.0] })
        .collect();
    let tie_queries = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.2]].concat();
    out.push(Fixture::new("ties", 2, tie, params(4, 20, 0), tie_queries));

    // n ≤ M0 + 1：layer 0 從不 prune，graph 必定連通。
    for seed in 0..3 {
        let mut rng = SplitMix64::new(100 + seed);
        let data = rng.vectors(17, 8);
        let queries = queries_with_stored_rows(&mut rng, &data, 8, 10);
        out.push(Fixture::new(
            &format!("n <= M0 + 1, seed {seed}"),
            8,
            data,
            params(8, 16, seed),
            queries,
        ));
    }

    let single = vec![0.3, -0.4, 1.2, 0.5];
    let single_queries = [0.3, -0.4, 1.2, 0.5, -1.0, 0.0, 0.0, 0.0].to_vec();
    out.push(Fixture::new(
        "n = 1",
        4,
        single,
        params(4, 20, 2),
        single_queries,
    ));

    let pair: Vec<f32> = [unit_circle(10.0), unit_circle(100.0)].concat();
    let pair_queries = [unit_circle(0.0), unit_circle(90.0), unit_circle(200.0)].concat();
    out.push(Fixture::new(
        "n = 2",
        2,
        pair,
        params(2, 4, 3),
        pair_queries,
    ));

    // decision 005 的 signed zero fixture。
    let signed_zero = vec![-0.0, 1.0, 1.0, -0.0, -0.0, -1.0];
    out.push(Fixture::new(
        "signed zero",
        2,
        signed_zero,
        params(2, 4, 3),
        [1.0, 0.0, -0.0, 1.0].to_vec(),
    ));

    out
}

/// (k, ef) 的組合；k ≤ n。
fn search_grid(n: usize) -> Vec<(usize, usize)> {
    let mut grid = Vec::new();
    for k in [1, 5, 10, 80] {
        if k > n {
            continue;
        }
        for ef in [1, 5, 10, 32, 100, 300] {
            grid.push((k, ef));
        }
    }
    grid.push((n, n));
    grid
}

// ---- FP32（臂 F）與 Core 逐位元等價 ----

#[test]
fn fp32_search_batch_is_bitwise_identical_to_core() {
    let mut saw_negative = false;
    let mut saw_padding = false;
    for fixture in fixtures() {
        let core = fixture.core();
        let fp32 = core.clone().into_quant(Representation::Fp32);
        for (k, ef) in search_grid(fixture.n()) {
            let expected = core.search_batch(&fixture.queries, k, ef).unwrap();
            let got = fp32.search_batch(&fixture.queries, k, ef).unwrap();
            assert_eq!(
                bits(&got),
                bits(&expected),
                "{} k {k} ef {ef}",
                fixture.name
            );
            let (stats_batch, stats) = fp32.search_stats(&fixture.queries, k, ef).unwrap();
            assert_eq!(
                bits(&stats_batch),
                bits(&expected),
                "{} stats",
                fixture.name
            );
            assert_eq!(stats.kernel_calls.len(), expected.num_queries);
            assert_eq!(stats.visited_inserts.len(), expected.num_queries);
            saw_negative |= expected.distances.iter().any(|&d| d < 0.0);
            saw_padding |= expected.ids.contains(&PADDING_ID);
        }
    }
    // F 不夾住：Core 的負距離（f32 rounding）原樣保留；不足 k 的補齊也相同。
    assert!(saw_negative, "no fixture produced a negative FP32 distance");
    assert!(saw_padding, "no fixture produced padded results");
}

#[test]
fn sq8_search_stats_returns_the_same_results_as_search_batch() {
    for fixture in fixtures() {
        let sq8 = fixture.core().into_quant(Representation::Sq8);
        for (k, ef) in search_grid(fixture.n()) {
            let batch = sq8.search_batch(&fixture.queries, k, ef).unwrap();
            let (stats_batch, _) = sq8.search_stats(&fixture.queries, k, ef).unwrap();
            assert_eq!(
                bits(&stats_batch),
                bits(&batch),
                "{} k {k} ef {ef}",
                fixture.name
            );
        }
    }
}

// ---- into_quant 保留圖 ----

fn core_graph(index: &HnswIndex) -> Vec<Vec<Vec<NodeId>>> {
    (0..index.len() as NodeId)
        .map(|id| {
            (0..=index.level(id).unwrap())
                .map(|l| index.neighbors(id, l).unwrap().to_vec())
                .collect()
        })
        .collect()
}

fn quant_graph(index: &QuantIndex) -> Vec<Vec<Vec<NodeId>>> {
    (0..index.len() as NodeId)
        .map(|id| {
            (0..=index.level(id).unwrap())
                .map(|l| index.neighbors(id, l).unwrap().to_vec())
                .collect()
        })
        .collect()
}

#[test]
fn into_quant_preserves_the_graph() {
    for fixture in fixtures() {
        let core = fixture.core();
        for representation in REPRESENTATIONS {
            let quant = core.clone().into_quant(representation);
            assert_eq!(quant.representation(), representation);
            assert_eq!(quant.dim(), core.dim());
            assert_eq!(quant.len(), core.len());
            assert!(!quant.is_empty());
            assert_eq!(quant.entry_point(), core.entry_point(), "{}", fixture.name);
            assert_eq!(quant_graph(&quant), core_graph(&core), "{}", fixture.name);
            let n = core.len() as NodeId;
            assert_eq!(quant.level(n), None);
            assert_eq!(quant.neighbors(n, 0), None);
            for id in 0..n {
                let level = quant.level(id).unwrap();
                assert_eq!(quant.neighbors(id, level + 1), None);
            }
        }
    }
}

#[test]
fn into_quant_on_an_empty_index() {
    for representation in REPRESENTATIONS {
        let quant = HnswIndex::new(3, params(4, 20, 1))
            .unwrap()
            .into_quant(representation);
        assert_eq!((quant.len(), quant.dim()), (0, 3));
        assert!(quant.is_empty());
        assert_eq!(quant.entry_point(), None);
        assert_eq!(quant.level(0), None);
        let query = [1.0, 0.0, 0.0];
        assert_eq!(
            quant.search_batch(&query, 1, 10).unwrap_err(),
            HnswError::EmptyIndex
        );
        assert_eq!(
            quant.search_stats(&query, 1, 10).unwrap_err(),
            HnswError::EmptyIndex
        );
        assert_eq!(
            quant.exact_search_batch(&query, 1).unwrap_err(),
            HnswError::EmptyIndex
        );
    }
    let sq8 = HnswIndex::new(3, params(4, 20, 1))
        .unwrap()
        .into_quant(Representation::Sq8);
    assert_eq!(sq8.codes(), Some(&[][..]));
    assert_eq!(sq8.codebook(), Some((&[0.0; 3][..], &[0.0; 3][..])));
}

#[test]
fn sq8_codes_are_trained_on_the_stored_unit_vectors() {
    for fixture in fixtures() {
        let core = fixture.core();
        let fp32 = core.clone().into_quant(Representation::Fp32);
        assert_eq!(fp32.codes(), None);
        assert_eq!(fp32.codebook(), None);
        let sq8 = core.into_quant(Representation::Sq8);
        let (vmin, vdiff, codes) =
            sq8_train_encode(&unit_rows(&fixture.data, fixture.dim), fixture.dim);
        let (got_vmin, got_vdiff) = sq8.codebook().unwrap();
        let to_bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(to_bits(got_vmin), to_bits(&vmin), "{}", fixture.name);
        assert_eq!(to_bits(got_vdiff), to_bits(&vdiff), "{}", fixture.name);
        assert_eq!(sq8.codes().unwrap(), codes.as_slice(), "{}", fixture.name);
    }
}

// ---- 人工可驗證的 codec 案例 ----

/// 解碼（只作參考，019 §3）：`x̂ = vmin + (code + 0.5) / 255 · vdiff`，以 f64 計算。
fn decode(code: u8, vmin: f32, vdiff: f32) -> f64 {
    f64::from(vmin) + (f64::from(code) + 0.5) / 255.0 * f64::from(vdiff)
}

#[test]
fn codec_hand_verifiable_3d_case() {
    // 第 0 維 [0, 1]、第 1 維 [-1, 1]、第 2 維全為 0.5（vdiff = 0）。
    let rows = [
        0.0, 1.0, 0.5, //
        1.0, -1.0, 0.5, //
        0.5, 0.0, 0.5,
    ];
    let (vmin, vdiff, codes) = sq8_train_encode(&rows, 3);
    assert_eq!(vmin, [0.0, -1.0, 0.5]);
    assert_eq!(vdiff, [1.0, 2.0, 0.0]);
    // vmin → 0、vmax → 255、vdiff = 0 → 0；中點 t = 0.5 → 255 · 0.5 = 127.5 向零截斷為 127。
    assert_eq!(codes, [0, 255, 0, 255, 0, 0, 127, 127, 0]);

    // 重建誤差：端點恰為 vdiff / 510，中點 (127 + 0.5) / 255 = 0.5 恰好無誤差，vdiff = 0 時恰好等於 vmin。
    let expected_error = [
        1.0 / 510.0,
        2.0 / 510.0,
        0.0,
        1.0 / 510.0,
        2.0 / 510.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (j, ((&x, &code), &want)) in rows.iter().zip(&codes).zip(&expected_error).enumerate() {
        let d = j % 3;
        let error = (decode(code, vmin[d], vdiff[d]) - f64::from(x)).abs();
        assert!(
            (error - want).abs() <= 1e-12,
            "component {j}: error {error}, expected {want}"
        );
    }
}

#[test]
fn codec_hand_verifiable_2d_unit_vectors() {
    // (1, 0)、(0, 1)、(-1, 0)：vmin = (-1, 0)、vdiff = (2, 1)。
    let rows = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let (vmin, vdiff, codes) = sq8_train_encode(&rows, 2);
    assert_eq!(vmin, [-1.0, 0.0]);
    assert_eq!(vdiff, [2.0, 1.0]);
    // (0 − (−1)) / 2 = 0.5 → 127。
    assert_eq!(codes, [255, 0, 127, 255, 0, 0]);
}

#[test]
fn codec_reconstruction_error_is_at_most_half_a_step() {
    // 7 維隨機資料，第 3 維固定為常數（vdiff = 0）。
    let (n, dim) = (500, 7);
    let mut rng = SplitMix64::new(31);
    let mut rows = rng.vectors(n, dim);
    for row in rows.chunks_exact_mut(dim) {
        row[3] = 0.25;
    }
    let (vmin, vdiff, codes) = sq8_train_encode(&rows, dim);
    assert_eq!(vdiff[3], 0.0);
    // f32 rounding 的容許量：t 的計算有 3 次捨入，另加輸入大小的幾個 ulp。
    let u = f64::from(f32::EPSILON) / 2.0;
    for (row, code_row) in rows.chunks_exact(dim).zip(codes.chunks_exact(dim)) {
        for j in 0..dim {
            let (x, code) = (row[j], code_row[j]);
            let half_step = f64::from(vdiff[j]) / 510.0;
            let tol = f64::from(vdiff[j]) * 8.0 * u
                + 4.0 * u * (f64::from(x.abs()) + f64::from(vmin[j].abs()));
            let error = (decode(code, vmin[j], vdiff[j]) - f64::from(x)).abs();
            assert!(
                error <= half_step + tol,
                "dim {j}: x {x}, code {code}, error {error}"
            );
            if vdiff[j] == 0.0 || x == vmin[j] {
                assert_eq!(code, 0);
            } else if x == vmin[j] + vdiff[j] {
                assert_eq!(code, 255);
            }
        }
    }
    // 每一維都有 vmin 與 vmax 的列，所以 0 與 255 都會出現在 vdiff > 0 的維度。
    for j in (0..dim).filter(|&j| j != 3) {
        let column: Vec<u8> = codes.chunks_exact(dim).map(|c| c[j]).collect();
        assert!(column.contains(&0) && column.contains(&255), "dim {j}");
    }
    // 編碼在每一維都是單調的（f32 的捨入是單調的）。
    for j in 0..dim {
        let mut pairs: Vec<(f32, u8)> = rows
            .chunks_exact(dim)
            .zip(codes.chunks_exact(dim))
            .map(|(r, c)| (r[j], c[j]))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(
            pairs.windows(2).all(|w| w[0].1 <= w[1].1),
            "dim {j} is not monotone"
        );
    }
}

#[test]
fn codec_of_an_empty_input_is_all_zero() {
    let (vmin, vdiff, codes) = sq8_train_encode(&[], 4);
    assert_eq!(vmin, [0.0; 4]);
    assert_eq!(vdiff, [0.0; 4]);
    assert!(codes.is_empty());
}

#[test]
#[should_panic(expected = "dimension must be positive")]
fn codec_rejects_zero_dimension() {
    sq8_train_encode(&[1.0], 0);
}

#[test]
#[should_panic(expected = "not a multiple of dimension")]
fn codec_rejects_a_ragged_buffer() {
    sq8_train_encode(&[1.0, 2.0, 3.0], 2);
}

#[test]
#[should_panic(expected = "values must be finite")]
fn codec_rejects_non_finite_values() {
    sq8_train_encode(&[1.0, f32::NAN], 1);
}

#[test]
#[should_panic(expected = "overflows f32")]
fn codec_rejects_a_range_that_overflows() {
    sq8_train_encode(&[f32::MAX, -f32::MAX], 1);
}

// ---- SQ8 的距離、夾住與誤差界（公開 API） ----

/// 依 019 §3 的文字定義重新實作的 SQ8 原始距離（不夾住），作為獨立於 crate 的 oracle：
/// `w_j = q_j · (vdiff_j / 255)`（f32），`b` 以 f64 累加後轉 f32，`1 − (b + Σ w_j · code_j)` 依序以 f32 加總。
fn sq8_text_distance(query: &[f32], vmin: &[f32], vdiff: &[f32], code: &[u8]) -> f32 {
    let w: Vec<f32> = query
        .iter()
        .zip(vdiff)
        .map(|(&q, &d)| q * (d / 255.0))
        .collect();
    let b = query
        .iter()
        .zip(vmin)
        .zip(vdiff)
        .map(|((&q, &m), &d)| f64::from(q) * (f64::from(m) + 0.5 * f64::from(d) / 255.0))
        .sum::<f64>() as f32;
    let sum: f32 = w.iter().zip(code).map(|(&w, &c)| w * f32::from(c)).sum();
    1.0 - (b + sum)
}

/// 小型 brute force：依 `(distance, id)` 全排序取前 k 個，輸出時套用 `output`。
fn brute_force(
    distances: impl Fn(&[f32]) -> Vec<f32>,
    output: impl Fn(f32) -> f32,
    unit_queries: &[f32],
    dim: usize,
    k: usize,
) -> BatchResult {
    let mut ids = Vec::new();
    let mut out = Vec::new();
    for query in unit_queries.chunks_exact(dim) {
        let mut all: Vec<(f32, NodeId)> = distances(query)
            .into_iter()
            .enumerate()
            .map(|(id, d)| (d, id as NodeId))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(d, id) in &all[..k] {
            ids.push(id);
            out.push(output(d));
        }
    }
    BatchResult {
        num_queries: unit_queries.len() / dim,
        k,
        ids,
        distances: out,
    }
}

#[test]
fn sq8_hand_verifiable_distances_and_clamping() {
    // (1, 0)、(0, 1)、(-1, 0)：codes (255, 0)、(127, 255)、(0, 0)，x̂ = (1 + 1/255, 1/510)、(0, 1 + 1/510)、(−1 + 1/255, 1/510)。
    // n = 3 ≤ M0 + 1，ef ≥ n：HNSW 的結果必須等於 brute force。
    let rows = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let sq8 = build(&rows, 2, params(2, 10, 1)).into_quant(Representation::Sq8);
    let close =
        |got: f32, want: f64| assert!((f64::from(got) - want).abs() <= 1e-6, "{got} vs {want}");

    // q = (1, 0)：原始距離 1 − 256/255 = −1/255 → 夾成 0；(0, 1) 為 1；(-1, 0) 為 1 + 254/255。
    // q = (-1, 0)：(1, 0) 的原始距離 1 + 256/255 → 夾成 2；(-1, 0) 為 1/255。
    let queries = [1.0, 0.0, -1.0, 0.0];
    for batch in [
        sq8.search_batch(&queries, 3, 3).unwrap(),
        sq8.exact_search_batch(&queries, 3).unwrap(),
    ] {
        assert_eq!(batch.ids, [0, 1, 2, 2, 1, 0]);
        assert_eq!(batch.distances[0], 0.0);
        close(batch.distances[1], 1.0);
        close(batch.distances[2], 1.0 + 254.0 / 255.0);
        close(batch.distances[3], 1.0 / 255.0);
        close(batch.distances[4], 1.0);
        assert_eq!(batch.distances[5], 2.0);
    }
}

#[test]
fn sq8_distances_are_within_the_registered_bound_of_the_f64_reference() {
    // 019 §4：|輸出 − clamp(r)| ≤ γ_{d+5} · (Σ_j |q_j| · (vdiff_j / 255) · code_j + |b_ref| + 1)。
    let u = 1.0 / (1u64 << 24) as f64;
    let gamma = |m: usize| m as f64 * u / (1.0 - m as f64 * u);
    for (seed, dim) in [(41, 3), (42, 32), (43, 100)] {
        let n = 150;
        let mut rng = SplitMix64::new(seed);
        let data = rng.vectors(n, dim);
        let mut queries = rng.vectors(15, dim);
        queries.extend_from_slice(&data[..5 * dim]);
        let sq8 = build(&data, dim, params(6, 40, seed)).into_quant(Representation::Sq8);
        let (vmin, vdiff) = sq8.codebook().unwrap();
        let codes = sq8.codes().unwrap();
        // k = n：每個 (query, 點) 都經過一次 kernel。
        let batch = sq8.exact_search_batch(&queries, n).unwrap();
        let unit_queries = unit_rows(&queries, dim);
        for (qi, query) in unit_queries.chunks_exact(dim).enumerate() {
            for slot in 0..n {
                let id = batch.ids[qi * n + slot] as usize;
                let got = batch.distances[qi * n + slot];
                let code = &codes[id * dim..(id + 1) * dim];
                let (mut dot, mut sigma, mut b_ref) = (0.0, 0.0, 0.0);
                for j in 0..dim {
                    let (q, m, d, c) = (
                        f64::from(query[j]),
                        f64::from(vmin[j]),
                        f64::from(vdiff[j]),
                        f64::from(code[j]),
                    );
                    dot += q * (m + (c + 0.5) / 255.0 * d);
                    sigma += q.abs() * (d / 255.0) * c;
                    b_ref += q * (m + 0.5 * d / 255.0);
                }
                let r = (1.0 - dot).clamp(0.0, 2.0);
                let bound = gamma(dim + 5) * (sigma + b_ref.abs() + 1.0);
                assert!(
                    (f64::from(got) - r).abs() <= bound,
                    "dim {dim} query {qi} id {id}: {got} vs {r} (bound {bound:e})"
                );
                assert!((0.0..=2.0).contains(&got));
            }
        }
    }
}

#[test]
fn exact_search_batch_equals_a_small_brute_force() {
    for (seed, n, dim, k) in [(51, 200, 8, 10), (52, 120, 3, 120), (53, 64, 24, 7)] {
        let mut rng = SplitMix64::new(seed);
        let data = rng.vectors(n, dim);
        let queries = queries_with_stored_rows(&mut rng, &data, dim, 12);
        let core = build(&data, dim, params(4, 20, seed));
        let unit_data = unit_rows(&data, dim);
        let unit_queries = unit_rows(&queries, dim);

        // F：Core 的 f32 kernel，不夾住。
        let fp32 = core.clone().into_quant(Representation::Fp32);
        let expected = brute_force(
            |q| {
                unit_data
                    .chunks_exact(dim)
                    .map(|v| cosine_distance_unit(q, v))
                    .collect()
            },
            |d| d,
            &unit_queries,
            dim,
            k,
        );
        assert_eq!(
            bits(&fp32.exact_search_batch(&queries, k).unwrap()),
            bits(&expected)
        );

        // S：依 019 §3 的文字定義重新計算，夾住之後輸出。
        let sq8 = core.into_quant(Representation::Sq8);
        let (vmin, vdiff) = sq8.codebook().unwrap();
        let codes = sq8.codes().unwrap();
        let expected = brute_force(
            |q| {
                codes
                    .chunks_exact(dim)
                    .map(|c| sq8_text_distance(q, vmin, vdiff, c))
                    .collect()
            },
            |d| d.clamp(0.0, 2.0),
            &unit_queries,
            dim,
            k,
        );
        assert_eq!(
            bits(&sq8.exact_search_batch(&queries, k).unwrap()),
            bits(&expected)
        );
    }
}

#[test]
fn small_index_with_large_ef_equals_exact_search_batch() {
    // n ≤ M0 + 1 時 graph 必定連通；ef ≥ n 時 search_layer 不會提早停止，結果必須與同一個 kernel 的 brute force 逐位元相同。
    let (m, dim) = (8, 8);
    let n = 2 * m + 1;
    for seed in 0..5 {
        let mut rng = SplitMix64::new(200 + seed);
        let data = rng.vectors(n, dim);
        let queries = queries_with_stored_rows(&mut rng, &data, dim, 20);
        let core = build(&data, dim, params(m, 16, seed));
        for representation in REPRESENTATIONS {
            let quant = core.clone().into_quant(representation);
            for k in [1, 5, n] {
                let exact = quant.exact_search_batch(&queries, k).unwrap();
                let hnsw = quant.search_batch(&queries, k, n).unwrap();
                assert_eq!(
                    bits(&hnsw),
                    bits(&exact),
                    "{representation:?} seed {seed} k {k}"
                );
            }
        }
    }
}

#[test]
fn sq8_recall_against_exact_search_is_reasonable() {
    // 粗略的健全性檢查（synthetic，不作結論）：SQ8 的 brute force 對 f64 exact search 的 Recall@10。
    let (n, dim, k) = (1000, 16, 10);
    let mut rng = SplitMix64::new(61);
    let data = rng.vectors(n, dim);
    let queries = rng.vectors(50, dim);
    let mut exact = ExactIndex::new(dim).unwrap();
    exact.add(&data).unwrap();
    let truth = exact.search_batch(&queries, k).unwrap();
    let sq8 = build(&data, dim, params(8, 64, 61)).into_quant(Representation::Sq8);
    let got = sq8.exact_search_batch(&queries, k).unwrap();
    let hits: usize = truth
        .ids
        .chunks_exact(k)
        .zip(got.ids.chunks_exact(k))
        .map(|(t, g)| g.iter().filter(|id| t.contains(id)).count())
        .sum();
    let recall = hits as f64 / (50 * k) as f64;
    eprintln!("SQ8 brute-force Recall@{k} = {recall}");
    assert!(recall >= 0.8, "recall {recall}");
}

// ---- 輸出格式 ----

#[test]
fn results_are_sorted_unique_padded_and_in_range() {
    // 004 的大量重複向量：ef = 300 也只到得了少數點，k = 80 時一定需要補齊。
    let (dim, n, k) = (6, 300, 80);
    let data = duplicate_data(12, dim, n);
    let queries = SplitMix64::new(112).vectors(10, dim);
    let core = build(&data, dim, params(4, 20, 12));
    for representation in REPRESENTATIONS {
        let quant = core.clone().into_quant(representation);
        let batch = quant.search_batch(&queries, k, n).unwrap();
        let mut padded_rows = 0;
        for (ids_row, dist_row) in batch
            .ids
            .chunks_exact(k)
            .zip(batch.distances.chunks_exact(k))
        {
            let valid = ids_row.iter().take_while(|&&id| id != PADDING_ID).count();
            assert!(valid >= 1);
            padded_rows += usize::from(valid < k);
            assert!(ids_row[valid..].iter().all(|&id| id == PADDING_ID));
            assert!(dist_row[valid..].iter().all(|&d| d == f32::INFINITY));
            assert!(dist_row[..valid].iter().all(|d| d.is_finite()));
            assert!(dist_row[..valid].windows(2).all(|w| w[0] <= w[1]));
            if representation == Representation::Sq8 {
                assert!(dist_row[..valid].iter().all(|d| (0.0..=2.0).contains(d)));
            }
            let mut unique = ids_row[..valid].to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), valid);
        }
        assert!(
            padded_rows > 0,
            "{representation:?}: fixture must produce padding"
        );
    }
}

#[test]
fn ef_search_smaller_than_k_is_raised_to_k() {
    let dim = 8;
    let mut rng = SplitMix64::new(13);
    let core = build(&rng.vectors(200, dim), dim, params(6, 40, 13));
    let query = rng.vectors(3, dim);
    for representation in REPRESENTATIONS {
        let quant = core.clone().into_quant(representation);
        let low = quant.search_batch(&query, 10, 1).unwrap();
        assert!(!low.ids.contains(&PADDING_ID));
        assert_eq!(
            bits(&low),
            bits(&quant.search_batch(&query, 10, 10).unwrap())
        );
    }
}

// ---- 錯誤處理與 FP32 相同 ----

#[test]
fn input_validation_matches_core() {
    let rows: Vec<f32> = [0.0, 90.0, 180.0]
        .iter()
        .flat_map(|&a| unit_circle(a))
        .collect();
    let core = build(&rows, 2, params(2, 10, 0));
    let mut exact = ExactIndex::new(2).unwrap();
    exact.add(&rows).unwrap();
    let cases: Vec<(&[f32], usize)> = vec![
        (&[1.0, 0.0], 0),
        (&[1.0, 0.0], 4),
        (&[1.0, 0.0, 1.0], 1),
        (&[1.0, 0.0, 0.0, 0.0], 1),
        (&[1.0, 0.0, f32::NAN, 1.0], 1),
        (&[f32::INFINITY, 0.0], 1),
        (&[], 2),
    ];
    for representation in REPRESENTATIONS {
        let quant = core.clone().into_quant(representation);
        for &(queries, k) in &cases {
            let expected = core
                .search_batch(queries, k, 10)
                .map(|b| (b.num_queries, b.k));
            let got = quant
                .search_batch(queries, k, 10)
                .map(|b| (b.num_queries, b.k));
            assert_eq!(got, expected, "{representation:?} {queries:?} k {k}");
            let stats = quant
                .search_stats(queries, k, 10)
                .map(|(b, _)| (b.num_queries, b.k));
            assert_eq!(stats, expected);
            let expected_exact = exact.search_batch(queries, k).map(|b| (b.num_queries, b.k));
            let got_exact = quant
                .exact_search_batch(queries, k)
                .map(|b| (b.num_queries, b.k));
            assert_eq!(got_exact, expected_exact);
        }
        // 錯誤的種類本身也要釘住（不只是與 Core 相同）。
        assert_eq!(
            quant.search_batch(&[1.0, 0.0], 0, 10).unwrap_err(),
            HnswError::InvalidK { k: 0, len: 3 }
        );
        assert_eq!(
            quant.search_batch(&[1.0, 0.0, 1.0], 1, 10).unwrap_err(),
            HnswError::RaggedBuffer { len: 3, dim: 2 }
        );
        assert_eq!(
            quant
                .search_batch(&[1.0, 0.0, f32::NAN, 1.0], 1, 10)
                .unwrap_err(),
            HnswError::NonFiniteValue { row: 1 }
        );
        assert_eq!(
            quant
                .search_batch(&[1.0, 0.0, 0.0, 0.0], 1, 10)
                .unwrap_err(),
            HnswError::ZeroVector { row: 1 }
        );
    }
}

// ---- 計數 ----

#[test]
fn stats_hand_verifiable_counts() {
    // id 0 = (1, 0)、id 1 = (0, 1)、id 2 = (-1, 0)，M = 2，seed 1：三個點都在 layer 0（下面先斷言）。
    // 建圖：1 → [0]、0 → [1]；2 的候選 [1 (d = 1), 0 (d = 2)]，0 被 1 涵蓋（dist(0, 1) = 1 < 2），
    // 所以 2 → [1]、1 → [0, 2]。graph：0: [1]、1: [0, 2]、2: [1]，entry point 為 0。
    //
    // query 170°，k = 1，ef = 1：
    // - entry 0：kernel 1 次；layer 0 的 visited.insert(0)：insert 1 次。
    // - pop 0：鄰居 1（新，kernel 2；比 0 近，取代 0）。insert 2 次。
    // - pop 1：鄰居 0（已拜訪）、2（新，kernel 3；比 1 近，取代 1）。insert 4 次。
    // - pop 2：鄰居 1（已拜訪）。insert 5 次。
    // 所以 D = 3、V = 5，結果為 id 2。k = 3、ef = 3 時走訪相同（D = 3、V = 5，結果 2、1、0）。
    let rows = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let core = build(&rows, 2, params(2, 10, 1));
    for id in 0..3 {
        assert_eq!(
            core.level(id),
            Some(0),
            "the hand calculation assumes level 0"
        );
    }
    assert_eq!(core.entry_point(), Some(0));
    assert_eq!(core.neighbors(0, 0).unwrap(), [1]);
    assert_eq!(core.neighbors(1, 0).unwrap(), [0, 2]);
    assert_eq!(core.neighbors(2, 0).unwrap(), [1]);

    let query = unit_circle(170.0);
    for representation in REPRESENTATIONS {
        let quant = core.clone().into_quant(representation);
        let (batch, stats) = quant.search_stats(&query, 1, 1).unwrap();
        assert_eq!(batch.ids, [2]);
        assert_eq!(stats.kernel_calls, [3], "{representation:?}");
        assert_eq!(stats.visited_inserts, [5], "{representation:?}");
        let (batch, stats) = quant.search_stats(&query, 3, 3).unwrap();
        assert_eq!(batch.ids, [2, 1, 0]);
        assert_eq!(stats.kernel_calls, [3]);
        assert_eq!(stats.visited_inserts, [5]);
    }

    // 兩層：同樣的三個點，seed 11 讓 level 為 [1, 1, 0]（下面先斷言），entry point 仍是 0。
    // 建圖：layer 1 為 0: [1]、1: [0]；2 插入時先在 layer 1 greedy 移到 1，layer 0 與上面相同（0: [1]、1: [0, 2]、2: [1]）。
    //
    // query 170°，k = 1，ef = 1：
    // - entry 0：kernel 1 次。
    // - layer 1 的 greedy：0 的鄰居 1（kernel 2，比 0 近，移到 1）；1 的鄰居 0（kernel 3，沒有改善），停在 1。
    // - layer 0 從 1 開始：insert 1；pop 1：鄰居 0（新，kernel 4，比 1 遠，不改善）、2（新，kernel 5，取代 1）；
    //   pop 2：鄰居 1（已拜訪）。insert 共 4 次。
    // 所以 D = 5、V = 4：greedy 的 kernel 呼叫也要算進 D（019 §2：所有層，含 entry point）。
    let core = build(&rows, 2, params(2, 10, 11));
    assert_eq!(
        (0..3).map(|id| core.level(id).unwrap()).collect::<Vec<_>>(),
        [1, 1, 0],
        "the hand calculation assumes levels [1, 1, 0]"
    );
    assert_eq!(core.entry_point(), Some(0));
    assert_eq!(core.neighbors(0, 1).unwrap(), [1]);
    assert_eq!(core.neighbors(1, 1).unwrap(), [0]);
    assert_eq!(core.neighbors(0, 0).unwrap(), [1]);
    assert_eq!(core.neighbors(1, 0).unwrap(), [0, 2]);
    assert_eq!(core.neighbors(2, 0).unwrap(), [1]);
    for representation in REPRESENTATIONS {
        let quant = core.clone().into_quant(representation);
        let (batch, stats) = quant.search_stats(&query, 1, 1).unwrap();
        assert_eq!(batch.ids, [2]);
        assert_eq!(stats.kernel_calls, [5], "{representation:?}");
        assert_eq!(stats.visited_inserts, [4], "{representation:?}");
    }

    // n = 1：只有 entry point 的 1 次 kernel 與 1 次 insert。
    let single = build(&[0.3, -0.4, 1.2, 0.5], 4, params(4, 20, 2)).into_quant(Representation::Sq8);
    let (_, stats) = single
        .search_stats(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1, 10)
        .unwrap();
    assert_eq!(stats.kernel_calls, [1, 1]);
    assert_eq!(stats.visited_inserts, [1, 1]);
}

#[test]
fn stats_count_every_visit_when_ef_covers_the_whole_graph() {
    // 全部在 layer 0、n ≤ M0 + 1（連通）、ef ≥ n：每個點恰好被拜訪並展開一次，
    // 所以 D = n（entry 1 次 + 其餘 n − 1 個點各 1 次），V = 1 + Σ_v deg₀(v)。
    let (m, dim) = (8, 8);
    let n = 2 * m + 1;
    let mut rng = SplitMix64::new(300);
    let data = rng.vectors(n, dim);
    let queries = rng.vectors(10, dim);
    let core = build(&data, dim, params(m, 16, 1));
    for id in 0..n as NodeId {
        assert_eq!(
            core.level(id),
            Some(0),
            "seed 1 must put every node on layer 0"
        );
    }
    let degree_sum: usize = (0..n as NodeId)
        .map(|id| core.neighbors(id, 0).unwrap().len())
        .sum();
    for representation in REPRESENTATIONS {
        let quant = core.clone().into_quant(representation);
        for ef in [n, 3 * n] {
            let (_, stats) = quant.search_stats(&queries, 1, ef).unwrap();
            assert!(
                stats.kernel_calls.iter().all(|&d| d == n as u64),
                "{stats:?}"
            );
            assert!(
                stats
                    .visited_inserts
                    .iter()
                    .all(|&v| v == 1 + degree_sum as u64),
                "{stats:?}"
            );
        }
    }
}

// ---- 結構大小 ----

#[test]
fn memory_accounting_difference_is_3nd_minus_8d() {
    // 兩個 index 以相同的資料與參數分別建出（graph 的 capacity 也相同），各轉成 F 與 S。
    // 單次 `add` 時 FP32 向量的 capacity 恰為 n·d，所以 M_struct(F) − M_struct(S) = 4nd − (nd + 8d) = 3nd − 8d。
    for (seed, n, dim) in [
        (71, 300, 8),
        (72, 1000, 16),
        (73, 17, 8),
        (74, 1, 4),
        (75, 50, 100),
    ] {
        let mut rng = SplitMix64::new(seed);
        let data = rng.vectors(n, dim);
        let p = params(6, 40, seed);
        let fp32 = build(&data, dim, p)
            .into_quant(Representation::Fp32)
            .memory_accounting();
        let sq8 = build(&data, dim, p)
            .into_quant(Representation::Sq8)
            .memory_accounting();
        assert_eq!(fp32.vectors_bytes, 4 * n * dim);
        assert_eq!((fp32.codes_bytes, fp32.codebook_bytes), (0, 0));
        assert_eq!(sq8.vectors_bytes, 0);
        assert_eq!(sq8.codes_bytes, n * dim);
        assert_eq!(sq8.codebook_bytes, 8 * dim);
        assert_eq!(fp32.graph_bytes, sq8.graph_bytes);
        assert_eq!(fp32.struct_bytes, sq8.struct_bytes);
        assert!(fp32.graph_bytes > 0);
        // n·d 很小時差值為負（例如 n = 1、d = 4：12 − 32），以有號整數比較。
        let difference = fp32.total_bytes() as i64 - sq8.total_bytes() as i64;
        assert_eq!(
            difference,
            3 * (n * dim) as i64 - 8 * dim as i64,
            "n {n} d {dim}"
        );
    }
}
