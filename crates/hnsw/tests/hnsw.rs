//! HNSW insert / search 的 correctness tests（decision 004）。
//! synthetic random vectors 只用於 correctness，不作 benchmark 結論。

mod common;

use common::{SplitMix64, assert_close, unit_circle};
use hnsw::{ExactIndex, HnswError, HnswIndex, HnswParams, MAX_M, Neighbor, NodeId, PADDING_ID};

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

fn ids(result: &[Neighbor]) -> Vec<NodeId> {
    result.iter().map(|n| n.id).collect()
}

/// 每個節點每一層的鄰居清單，用來比較兩個 graph 是否完全相同。
fn graph_snapshot(index: &HnswIndex) -> Vec<Vec<Vec<NodeId>>> {
    (0..index.len() as NodeId)
        .map(|id| {
            (0..=index.level(id).unwrap())
                .map(|l| index.neighbors(id, l).unwrap().to_vec())
                .collect()
        })
        .collect()
}

fn mean_recall(index: &HnswIndex, exact: &ExactIndex, queries: &[f32], k: usize, ef: usize) -> f64 {
    let dim = index.dim();
    let mut hits = 0;
    for query in queries.chunks_exact(dim) {
        let truth = ids(&exact.search(query, k).unwrap());
        let found = ids(&index.search(query, k, ef).unwrap());
        hits += found.iter().filter(|id| truth.contains(id)).count();
    }
    hits as f64 / (queries.len() / dim * k) as f64
}

// ---- 建構與參數 ----

#[test]
fn new_validates_parameters() {
    assert_eq!(
        HnswIndex::new(0, HnswParams::default()).unwrap_err(),
        HnswError::InvalidDimension(0)
    );
    assert_eq!(
        HnswIndex::new(4, params(1, 10, 0)).unwrap_err(),
        HnswError::InvalidParameter {
            name: "m",
            min: 2,
            max: MAX_M,
            value: 1
        }
    );
    assert_eq!(
        HnswIndex::new(4, params(4, 0, 0)).unwrap_err(),
        HnswError::InvalidParameter {
            name: "ef_construction",
            min: 1,
            max: usize::MAX,
            value: 0
        }
    );
}

#[test]
fn m_has_an_upper_bound_so_2m_cannot_overflow() {
    // decision 005：m ≤ 2^31 − 1，保證 2m 放得進 NodeId。原本沒有上限，2m 會溢位。
    assert_eq!(MAX_M, (1usize << 31) - 1);
    assert!(HnswIndex::new(4, params(MAX_M, 10, 0)).is_ok());
    for m in [MAX_M + 1, usize::MAX / 2 + 1, usize::MAX] {
        assert_eq!(
            HnswIndex::new(4, params(m, 10, 0)).unwrap_err(),
            HnswError::InvalidParameter {
                name: "m",
                min: 2,
                max: MAX_M,
                value: m
            }
        );
    }
}

#[test]
fn new_index_is_empty() {
    let p = params(4, 10, 7);
    let index = HnswIndex::new(3, p).unwrap();
    assert_eq!(index.dim(), 3);
    assert_eq!(index.params(), p);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.entry_point(), None);
    assert_eq!(index.max_level(), None);
    assert_eq!(index.level(0), None);
    assert_eq!(index.neighbors(0, 0), None);
}

#[test]
fn search_on_empty_index_is_rejected() {
    let index = HnswIndex::new(2, HnswParams::default()).unwrap();
    assert_eq!(
        index.search(&[1.0, 0.0], 1, 10).unwrap_err(),
        HnswError::EmptyIndex
    );
}

#[test]
fn single_element_index() {
    let index = build(&[3.0, 4.0], 2, params(4, 10, 0));
    assert_eq!(index.len(), 1);
    assert_eq!(index.entry_point(), Some(0));
    assert_eq!(index.max_level(), index.level(0));
    assert_eq!(index.neighbors(0, 0), Some(&[][..]));

    let result = index.search(&[0.6, 0.8], 1, 10).unwrap();
    assert_eq!(ids(&result), vec![0]);
    assert_close(result[0].distance, 0.0, 1e-6);
}

// ---- 與 exact 完全一致的小案例 ----

#[test]
fn hand_verifiable_unit_circle_matches_exact() {
    // 與 exact 測試相同的單位圓。M = 2 → M0 = 4，n = 5 ≤ M0 + 1。
    // id:     0   1    2    3    4
    // 角度:   0° 30°  60°  90°  180°
    let rows: Vec<f32> = [0.0, 30.0, 60.0, 90.0, 180.0]
        .iter()
        .flat_map(|&a| unit_circle(a))
        .collect();
    let index = build(&rows, 2, params(2, 10, 1));

    let result = index.search(&unit_circle(10.0), 5, 5).unwrap();
    assert_eq!(ids(&result), vec![0, 1, 2, 3, 4]);
    for (n, diff) in result.iter().zip([10.0_f64, 20.0, 50.0, 80.0, 170.0]) {
        assert_close(n.distance, (1.0 - diff.to_radians().cos()) as f32, 1e-6);
    }
    let result = index.search(&unit_circle(100.0), 5, 5).unwrap();
    assert_eq!(ids(&result), vec![3, 2, 1, 4, 0]);
}

#[test]
fn small_index_with_large_ef_equals_exact_search() {
    // n ≤ M0 + 1 時 layer 0 從不 prune，每條邊都是雙向且不會被刪，graph 必定連通；
    // ef ≥ n 時 search_layer 不會提早停止，所以結果必須與 exact 完全一致。
    let (m, dim) = (8, 8);
    let n = 2 * m + 1;
    for seed in 0..5 {
        let mut rng = SplitMix64::new(100 + seed);
        let data = rng.vectors(n, dim);
        let queries = rng.vectors(20, dim);
        let index = build(&data, dim, params(m, 16, seed));
        let mut exact = ExactIndex::new(dim).unwrap();
        exact.add(&data).unwrap();

        for query in queries.chunks_exact(dim) {
            let expected = exact.search(query, n).unwrap();
            let got = index.search(query, n, n).unwrap();
            assert_eq!(ids(&got), ids(&expected), "seed {seed}");
            for (g, e) in got.iter().zip(&expected) {
                assert_close(g.distance, e.distance, 1e-5);
            }
        }
    }
}

// ---- graph 結構 ----

/// graph 結構不變量：鄰居數上限、無自環、無重複鄰居、鄰居的 level 足夠、
/// 每個點在 layer 0 至少有一個鄰居、entry point 位於最高層。
fn assert_graph_invariants(index: &HnswIndex) {
    let n = index.len();
    let m = index.params().m;
    let mut highest = 0;
    for id in 0..n as NodeId {
        let level = index.level(id).unwrap();
        highest = highest.max(level);
        assert_eq!(index.neighbors(id, level + 1), None);
        if n > 1 {
            assert!(
                !index.neighbors(id, 0).unwrap().is_empty(),
                "node {id} has no layer-0 neighbor"
            );
        }
        for l in 0..=level {
            let neighbors = index.neighbors(id, l).unwrap();
            let cap = if l == 0 { 2 * m } else { m };
            assert!(
                neighbors.len() <= cap,
                "node {id} layer {l}: {} > {cap}",
                neighbors.len()
            );
            let mut sorted = neighbors.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), neighbors.len(), "duplicate neighbor");
            for &nb in neighbors {
                assert_ne!(nb, id, "self loop");
                assert!(index.level(nb).unwrap() >= l, "neighbor below layer {l}");
            }
        }
    }
    assert_eq!(index.max_level(), Some(highest));
    assert_eq!(index.level(index.entry_point().unwrap()), Some(highest));
    assert_eq!(index.level(n as NodeId), None);
}

#[test]
fn graph_invariants_hold() {
    let (n, dim) = (1000, 16);
    let mut rng = SplitMix64::new(3);
    let m = 6;
    let index = build(&rng.vectors(n, dim), dim, params(m, 50, 3));
    assert_graph_invariants(&index);

    // layer 0 的上限是 2M 而非 M：反向連結累積後，應有節點的 layer-0 鄰居數超過 M。
    let max_degree = (0..n as NodeId)
        .map(|id| index.neighbors(id, 0).unwrap().len())
        .max()
        .unwrap();
    assert!(max_degree > m, "max layer-0 degree {max_degree} <= M = {m}");
}

#[test]
fn level_distribution_matches_ml() {
    // P(level ≥ 1) = 1/M、P(level ≥ 2) = 1/M²。n = 4000、M = 8：
    // 期望 500（σ ≈ 21）與 62.5（σ ≈ 7.8），容許 ±5σ。
    let (n, dim, m) = (4000, 4, 8);
    let mut rng = SplitMix64::new(5);
    let index = build(&rng.vectors(n, dim), dim, params(m, 8, 5));
    let at_least = |l: usize| {
        (0..n as NodeId)
            .filter(|&id| index.level(id).unwrap() >= l)
            .count()
    };
    let (l1, l2) = (at_least(1), at_least(2));
    assert!((395..=605).contains(&l1), "level >= 1: {l1}");
    assert!((23..=102).contains(&l2), "level >= 2: {l2}");
}

// ---- 決定性 ----

#[test]
fn same_seed_and_insertion_order_give_identical_graph() {
    let dim = 8;
    let mut rng = SplitMix64::new(9);
    let data = rng.vectors(500, dim);
    let a = build(&data, dim, params(6, 40, 77));
    let b = build(&data, dim, params(6, 40, 77));
    assert_eq!(graph_snapshot(&a), graph_snapshot(&b));
    assert_eq!(a.entry_point(), b.entry_point());

    let c = build(&data, dim, params(6, 40, 78));
    let levels =
        |idx: &HnswIndex| -> Vec<usize> { (0..500).map(|id| idx.level(id).unwrap()).collect() };
    assert_ne!(
        levels(&a),
        levels(&c),
        "different seed should change levels"
    );
}

#[test]
fn adding_in_several_calls_equals_one_call() {
    let dim = 8;
    let mut rng = SplitMix64::new(10);
    let data = rng.vectors(300, dim);
    let one = build(&data, dim, params(6, 40, 1));

    let mut split = HnswIndex::new(dim, params(6, 40, 1)).unwrap();
    split.add(&data[..100 * dim]).unwrap();
    split.add(&data[100 * dim..]).unwrap();
    assert_eq!(graph_snapshot(&one), graph_snapshot(&split));
}

#[test]
fn failed_add_is_atomic_and_keeps_rng_state() {
    let dim = 2;
    let good_1 = [1.0, 0.0, 0.0, 1.0, -1.0, 0.5];
    let good_2 = [0.3, 0.7, -0.2, -0.9];
    let reference = {
        let mut index = HnswIndex::new(dim, params(2, 10, 4)).unwrap();
        index.add(&good_1).unwrap();
        index.add(&good_2).unwrap();
        index
    };

    let mut index = HnswIndex::new(dim, params(2, 10, 4)).unwrap();
    index.add(&good_1).unwrap();
    let before = graph_snapshot(&index);
    assert_eq!(
        index.add(&[1.0, 1.0, 0.0, 0.0]).unwrap_err(),
        HnswError::ZeroVector { row: 1 }
    );
    assert_eq!(
        index.add(&[1.0, 1.0, f32::NAN, 1.0]).unwrap_err(),
        HnswError::NonFiniteValue { row: 1 }
    );
    assert_eq!(
        index.add(&[1.0, 1.0, 1.0]).unwrap_err(),
        HnswError::RaggedBuffer { len: 3, dim: 2 }
    );
    assert_eq!(index.len(), 3);
    assert_eq!(graph_snapshot(&index), before);

    // 失敗的 add 不能消耗亂數，否則之後的 level 會與參考不同。
    index.add(&good_2).unwrap();
    assert_eq!(graph_snapshot(&index), graph_snapshot(&reference));
}

// ---- 搜尋品質（correctness 等級） ----

#[test]
fn recall_improves_or_saturates_with_ef_search() {
    let (n, dim, nq, k) = (3000, 16, 200, 10);
    let mut rng = SplitMix64::new(42);
    let data = rng.vectors(n, dim);
    let queries = rng.vectors(nq, dim);
    let index = build(&data, dim, params(12, 100, 42));
    let mut exact = ExactIndex::new(dim).unwrap();
    exact.add(&data).unwrap();

    let efs = [10, 20, 40, 80, 160, 320];
    let recalls: Vec<f64> = efs
        .iter()
        .map(|&ef| mean_recall(&index, &exact, &queries, k, ef))
        .collect();
    eprintln!("recall@{k} for ef {efs:?}: {recalls:?}");

    // SPEC：整體改善或飽和，不要求逐點嚴格單調，但不得系統性惡化。
    for w in recalls.windows(2) {
        assert!(w[1] >= w[0] - 0.01, "recall dropped: {recalls:?}");
    }
    assert!(recalls[efs.len() - 1] >= recalls[0], "{recalls:?}");
    assert!(recalls[efs.len() - 1] >= 0.95, "{recalls:?}");
}

#[test]
fn finds_inserted_vectors_as_nearest() {
    let (n, dim) = (1000, 16);
    let mut rng = SplitMix64::new(8);
    let data = rng.vectors(n, dim);
    let index = build(&data, dim, params(8, 64, 8));
    for id in (0..n).step_by(20) {
        let result = index
            .search(&data[id * dim..(id + 1) * dim], 1, 50)
            .unwrap();
        assert_eq!(result[0].id, id as NodeId);
        assert_close(result[0].distance, 0.0, 1e-5);
    }
}

#[test]
fn many_exact_duplicates_keep_graph_valid_and_results_well_formed() {
    // 5 個方向各重複 60 次並打散。
    //
    // 注意：大量完全重複時，HNSW 不保證連通。heuristic 對距離 0 的副本永不 prune（`0 < 0` 不成立），
    // 鄰居清單會被同方向副本佔滿、跨方向連結被 prune 掉，可到達的點可能遠少於 n
    // （此設定下 ef = 300 只到得了 9 個點；見 decision 004）。
    // 因此這裡只斷言有保證的性質：graph 不變量、結果格式、不足 k 時正確補齊。
    let (dim, n, k) = (6, 300, 80);
    let mut rng = SplitMix64::new(12);
    let base = rng.vectors(5, dim);
    let mut order: Vec<usize> = (0..n).map(|i| i % 5).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let data: Vec<f32> = order
        .iter()
        .flat_map(|&b| base[b * dim..(b + 1) * dim].to_vec())
        .collect();
    let index = build(&data, dim, params(4, 20, 12));
    assert_graph_invariants(&index);

    let queries = rng.vectors(10, dim);
    let batch = index.search_batch(&queries, k, n).unwrap();
    for (ids_row, dist_row) in batch
        .ids
        .chunks_exact(k)
        .zip(batch.distances.chunks_exact(k))
    {
        let valid = ids_row.iter().take_while(|&&id| id != PADDING_ID).count();
        assert!(valid >= 1);
        assert!(ids_row[valid..].iter().all(|&id| id == PADDING_ID));
        assert!(dist_row[valid..].iter().all(|&d| d == f32::INFINITY));
        assert!(dist_row[..valid].iter().all(|d| d.is_finite()));
        assert!(dist_row[..valid].windows(2).all(|w| w[0] <= w[1]));
        let mut unique = ids_row[..valid].to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), valid);
    }
}

// ---- 輸入驗證與批次 ----

#[test]
fn search_validates_input() {
    let rows: Vec<f32> = [0.0, 90.0].iter().flat_map(|&a| unit_circle(a)).collect();
    let index = build(&rows, 2, params(2, 10, 0));
    let query = unit_circle(10.0);
    assert_eq!(
        index.search(&query, 0, 10).unwrap_err(),
        HnswError::InvalidK { k: 0, len: 2 }
    );
    assert_eq!(
        index.search(&query, 3, 10).unwrap_err(),
        HnswError::InvalidK { k: 3, len: 2 }
    );
    assert_eq!(
        index.search(&[1.0, 0.0, 0.0], 1, 10).unwrap_err(),
        HnswError::DimensionMismatch {
            expected: 2,
            actual: 3
        }
    );
    assert_eq!(
        index.search(&[0.0, 0.0], 1, 10).unwrap_err(),
        HnswError::ZeroVector { row: 0 }
    );
    assert_eq!(
        index.search(&[f32::NAN, 0.0], 1, 10).unwrap_err(),
        HnswError::NonFiniteValue { row: 0 }
    );
}

#[test]
fn ef_search_smaller_than_k_is_raised_to_k() {
    let dim = 8;
    let mut rng = SplitMix64::new(13);
    let index = build(&rng.vectors(200, dim), dim, params(6, 40, 13));
    let query = rng.vectors(1, dim);
    let low = index.search(&query, 10, 1).unwrap();
    assert_eq!(low.len(), 10);
    assert_eq!(low, index.search(&query, 10, 10).unwrap());
}

#[test]
fn results_are_sorted_and_unique() {
    let dim = 8;
    let mut rng = SplitMix64::new(14);
    let index = build(&rng.vectors(500, dim), dim, params(6, 40, 14));
    for query in rng.vectors(20, dim).chunks_exact(dim) {
        let result = index.search(query, 20, 30).unwrap();
        assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
        let mut unique = ids(&result);
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), result.len());
    }
}

#[test]
fn batch_layout_is_row_major_and_matches_single_search() {
    let (dim, nq, k, ef) = (8, 13, 7, 20);
    let mut rng = SplitMix64::new(15);
    let index = build(&rng.vectors(300, dim), dim, params(6, 40, 15));
    let queries = rng.vectors(nq, dim);

    let batch = index.search_batch(&queries, k, ef).unwrap();
    assert_eq!((batch.num_queries, batch.k), (nq, k));
    assert_eq!(batch.ids.len(), nq * k);
    assert_eq!(batch.distances.len(), nq * k);
    for (qi, query) in queries.chunks_exact(dim).enumerate() {
        let single = index.search(query, k, ef).unwrap();
        let row = qi * k..(qi + 1) * k;
        assert_eq!(batch.ids[row.clone()], ids(&single));
        let single_distances: Vec<f32> = single.iter().map(|n| n.distance).collect();
        assert_eq!(batch.distances[row], single_distances);
    }
}

#[test]
fn batch_validates_input() {
    let rows: Vec<f32> = [0.0, 90.0, 180.0]
        .iter()
        .flat_map(|&a| unit_circle(a))
        .collect();
    let index = build(&rows, 2, params(2, 10, 0));
    assert_eq!(
        index
            .search_batch(&[1.0, 0.0, 0.0, 0.0], 1, 10)
            .unwrap_err(),
        HnswError::ZeroVector { row: 1 }
    );
    assert_eq!(
        index.search_batch(&[1.0, 0.0, 1.0], 1, 10).unwrap_err(),
        HnswError::RaggedBuffer { len: 3, dim: 2 }
    );
    assert_eq!(
        index.search_batch(&[1.0, 0.0], 4, 10).unwrap_err(),
        HnswError::InvalidK { k: 4, len: 3 }
    );
    let empty = index.search_batch(&[], 2, 10).unwrap();
    assert_eq!((empty.num_queries, empty.k), (0, 2));
    assert!(empty.ids.is_empty() && empty.distances.is_empty());
}
