//! Exact brute-force search 的 correctness tests。
//! 包含人工可驗證的小案例，以及與獨立參考實作（完整排序）的比對。

mod common;

use common::{SplitMix64, assert_close, unit_circle};
use hnsw::distance::{cosine_distance, normalize};
use hnsw::{ExactIndex, HnswError, Neighbor};

fn index_from_rows(dim: usize, rows: &[&[f32]]) -> ExactIndex {
    let mut index = ExactIndex::new(dim).unwrap();
    index.add(&rows.concat()).unwrap();
    index
}

fn circle_index(angles: &[f64]) -> ExactIndex {
    let rows: Vec<[f32; 2]> = angles.iter().map(|&a| unit_circle(a)).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    index_from_rows(2, &refs)
}

fn ids(result: &[Neighbor]) -> Vec<u32> {
    result.iter().map(|n| n.id).collect()
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

#[test]
fn new_rejects_zero_dimension() {
    assert_eq!(
        ExactIndex::new(0).unwrap_err(),
        HnswError::InvalidDimension(0)
    );
}

#[test]
fn new_index_is_empty() {
    let index = ExactIndex::new(3).unwrap();
    assert_eq!(index.dim(), 3);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

#[test]
fn hand_verifiable_unit_circle_ordering() {
    // id:     0   1    2    3    4
    // 角度:   0° 30°  60°  90°  180°
    let index = circle_index(&[0.0, 30.0, 60.0, 90.0, 180.0]);

    // query 10°：角度差 10、20、50、80、170
    let result = index.search(&unit_circle(10.0), 5).unwrap();
    assert_eq!(ids(&result), vec![0, 1, 2, 3, 4]);
    for (n, diff) in result.iter().zip([10.0_f64, 20.0, 50.0, 80.0, 170.0]) {
        assert_close(n.distance, (1.0 - diff.to_radians().cos()) as f32, 1e-6);
    }

    // query 100°：角度差 100、70、40、10、80 → 順序 3、2、1、4、0
    let result = index.search(&unit_circle(100.0), 5).unwrap();
    assert_eq!(ids(&result), vec![3, 2, 1, 4, 0]);
}

#[test]
fn top_k_is_prefix_of_full_ranking() {
    let index = circle_index(&[0.0, 30.0, 60.0, 90.0, 180.0]);
    let query = unit_circle(100.0);
    let full = index.search(&query, 5).unwrap();
    for k in 1..=5 {
        assert_eq!(index.search(&query, k).unwrap(), full[..k]);
    }
}

#[test]
fn distances_are_non_decreasing() {
    let index = circle_index(&[0.0, 30.0, 60.0, 90.0, 180.0, 270.0]);
    let result = index.search(&unit_circle(200.0), 6).unwrap();
    assert!(result.windows(2).all(|w| w[0].distance <= w[1].distance));
}

#[test]
fn magnitude_does_not_affect_ranking() {
    // 與 hand_verifiable 相同的方向，但每個向量長度不同。
    let angles = [0.0, 30.0, 60.0, 90.0, 180.0];
    let scales = [0.001_f32, 1.0, 50.0, 3.0, 1e4];
    let rows: Vec<Vec<f32>> = angles
        .iter()
        .zip(scales)
        .map(|(&a, s)| unit_circle(a).iter().map(|x| x * s).collect())
        .collect();
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let index = index_from_rows(2, &refs);

    let query: Vec<f32> = unit_circle(10.0).iter().map(|x| x * 123.0).collect();
    assert_eq!(ids(&index.search(&query, 5).unwrap()), vec![0, 1, 2, 3, 4]);
}

#[test]
fn exact_match_is_first_with_zero_distance() {
    let index = circle_index(&[0.0, 45.0, 90.0]);
    let result = index.search(&unit_circle(45.0), 1).unwrap();
    assert_eq!(result[0].id, 1);
    assert_close(result[0].distance, 0.0, 1e-6);
}

#[test]
fn ties_are_broken_by_smaller_id() {
    // id 0、2、3 方向完全相同（長度不同，正規化後都是 [1, 0]），id 1 正交。
    let index = index_from_rows(2, &[&[1.0, 0.0], &[0.0, 1.0], &[2.0, 0.0], &[0.5, 0.0]]);
    assert_eq!(ids(&index.search(&[1.0, 0.0], 2).unwrap()), vec![0, 2]);
    assert_eq!(ids(&index.search(&[1.0, 0.0], 3).unwrap()), vec![0, 2, 3]);
    assert_eq!(
        ids(&index.search(&[1.0, 0.0], 4).unwrap()),
        vec![0, 2, 3, 1]
    );

    // 同距離的點出現在較近的點之後，仍要依 id 選。
    let index = index_from_rows(
        2,
        &[
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
        ],
    );
    assert_eq!(ids(&index.search(&[1.0, 0.0], 2).unwrap()), vec![2, 3]);
}

#[test]
fn ids_continue_across_add_calls() {
    let mut index = ExactIndex::new(2).unwrap();
    index.add(&[1.0, 0.0, 0.0, 1.0]).unwrap(); // id 0、1
    index.add(&[-1.0, 0.0, 0.0, -1.0, 1.0, 1.0]).unwrap(); // id 2、3、4
    assert_eq!(index.len(), 5);
    assert_eq!(index.search(&[0.0, -2.0], 1).unwrap()[0].id, 3);
    assert_eq!(index.search(&[3.0, 3.0], 1).unwrap()[0].id, 4);
}

#[test]
fn add_empty_buffer_is_noop() {
    let mut index = ExactIndex::new(2).unwrap();
    index.add(&[]).unwrap();
    assert!(index.is_empty());
}

#[test]
fn add_rejects_ragged_buffer() {
    let mut index = ExactIndex::new(3).unwrap();
    assert_eq!(
        index.add(&[1.0, 2.0, 3.0, 4.0]).unwrap_err(),
        HnswError::RaggedBuffer { len: 4, dim: 3 }
    );
    assert!(index.is_empty());
}

#[test]
fn add_reports_offending_row_and_is_atomic() {
    let mut index = ExactIndex::new(2).unwrap();
    index.add(&[1.0, 0.0]).unwrap();

    // row 編號相對於這次傳入的 buffer。
    let err = index.add(&[1.0, 1.0, 0.0, 0.0, 2.0, 1.0]).unwrap_err();
    assert_eq!(err, HnswError::ZeroVector { row: 1 });
    assert_eq!(index.len(), 1);

    let err = index.add(&[1.0, 1.0, 2.0, f32::NAN]).unwrap_err();
    assert_eq!(err, HnswError::NonFiniteValue { row: 1 });
    assert_eq!(index.len(), 1);
}

#[test]
fn search_rejects_invalid_k() {
    let index = circle_index(&[0.0, 90.0]);
    let query = unit_circle(10.0);
    assert_eq!(
        index.search(&query, 0).unwrap_err(),
        HnswError::InvalidK { k: 0, len: 2 }
    );
    assert_eq!(
        index.search(&query, 3).unwrap_err(),
        HnswError::InvalidK { k: 3, len: 2 }
    );
}

#[test]
fn search_on_empty_index_is_rejected() {
    let index = ExactIndex::new(2).unwrap();
    assert_eq!(
        index.search(&[1.0, 0.0], 1).unwrap_err(),
        HnswError::EmptyIndex
    );
}

#[test]
fn search_rejects_bad_query() {
    let index = circle_index(&[0.0, 90.0]);
    assert_eq!(
        index.search(&[1.0, 0.0, 0.0], 1).unwrap_err(),
        HnswError::DimensionMismatch {
            expected: 2,
            actual: 3
        }
    );
    assert_eq!(
        index.search(&[0.0, 0.0], 1).unwrap_err(),
        HnswError::ZeroVector { row: 0 }
    );
    assert_eq!(
        index.search(&[f32::NAN, 0.0], 1).unwrap_err(),
        HnswError::NonFiniteValue { row: 0 }
    );
}

#[test]
fn batch_layout_is_row_major_and_matches_single_search() {
    let mut rng = SplitMix64::new(42);
    let (n, dim, nq, k) = (200, 8, 13, 7);
    let mut index = ExactIndex::new(dim).unwrap();
    index.add(&rng.vectors(n, dim)).unwrap();
    let queries = rng.vectors(nq, dim);

    let batch = index.search_batch(&queries, k).unwrap();
    assert_eq!(batch.num_queries, nq);
    assert_eq!(batch.k, k);
    assert_eq!(batch.ids.len(), nq * k);
    assert_eq!(batch.distances.len(), nq * k);

    for (qi, query) in queries.chunks_exact(dim).enumerate() {
        let single = index.search(query, k).unwrap();
        let row = qi * k..(qi + 1) * k;
        assert_eq!(batch.ids[row.clone()], ids(&single));
        let single_distances: Vec<f32> = single.iter().map(|n| n.distance).collect();
        assert_eq!(batch.distances[row], single_distances);
    }
}

#[test]
fn batch_reports_offending_query_row() {
    let index = circle_index(&[0.0, 90.0, 180.0]);
    assert_eq!(
        index
            .search_batch(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1)
            .unwrap_err(),
        HnswError::ZeroVector { row: 1 }
    );
    assert_eq!(
        index.search_batch(&[1.0, 0.0, 1.0], 1).unwrap_err(),
        HnswError::RaggedBuffer { len: 3, dim: 2 }
    );
    assert_eq!(
        index.search_batch(&[1.0, 0.0], 4).unwrap_err(),
        HnswError::InvalidK { k: 4, len: 3 }
    );
}

#[test]
fn empty_query_batch_returns_empty_result() {
    let index = circle_index(&[0.0, 90.0, 180.0]);
    let batch = index.search_batch(&[], 2).unwrap();
    assert_eq!(batch.num_queries, 0);
    assert_eq!(batch.k, 2);
    assert!(batch.ids.is_empty());
    assert!(batch.distances.is_empty());
}

/// 獨立參考實作：算出所有距離後完整排序。
/// 使用與 decision 001 相同的 f64 依序累加，所以距離應逐位元相同、id 必須完全一致。
fn reference_search(unit_data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(f64, u32)> {
    let uq = normalize(query).unwrap();
    let mut all: Vec<(f64, u32)> = unit_data
        .iter()
        .enumerate()
        .map(|(i, u)| (1.0 - dot_f64(u, &uq), i as u32))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.truncate(k);
    all
}

fn assert_matches_reference(data: &[f32], queries: &[f32], dim: usize, k: usize) {
    let mut index = ExactIndex::new(dim).unwrap();
    index.add(data).unwrap();
    let unit_data: Vec<Vec<f32>> = data
        .chunks_exact(dim)
        .map(|r| normalize(r).unwrap())
        .collect();

    for query in queries.chunks_exact(dim) {
        let expected = reference_search(&unit_data, query, k);
        let got = index.search(query, k).unwrap();
        assert_eq!(ids(&got), expected.iter().map(|e| e.1).collect::<Vec<_>>());
        for (n, (d, _)) in got.iter().zip(&expected) {
            assert_eq!(n.distance, *d as f32);
        }
        // 與原始（未正規化）向量的 cosine distance 一致，容忍 f32 正規化誤差。
        for n in &got {
            let row = &data[n.id as usize * dim..(n.id as usize + 1) * dim];
            assert_close(n.distance, cosine_distance(row, query).unwrap(), 1e-5);
        }
    }
}

#[test]
fn matches_full_sort_reference_on_random_data() {
    let mut rng = SplitMix64::new(2026);
    let (n, dim, nq, k) = (500, 16, 50, 10);
    let data = rng.vectors(n, dim);
    let queries = rng.vectors(nq, dim);
    assert_matches_reference(&data, &queries, dim, k);
}

#[test]
fn matches_full_sort_reference_with_many_exact_duplicates() {
    // 只有 5 個不同方向，各重複 40 次並打散，製造大量完全相同的距離。
    let mut rng = SplitMix64::new(11);
    let dim = 6;
    let base = rng.vectors(5, dim);
    let mut order: Vec<usize> = (0..200).map(|i| i % 5).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let data: Vec<f32> = order
        .iter()
        .flat_map(|&b| base[b * dim..(b + 1) * dim].to_vec())
        .collect();
    let queries = rng.vectors(20, dim);
    assert_matches_reference(&data, &queries, dim, 50);
}
