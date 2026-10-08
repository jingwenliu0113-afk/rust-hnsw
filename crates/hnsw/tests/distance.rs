//! cosine distance 與正規化的 correctness tests（decision 001）。

mod common;

use common::{SplitMix64, assert_close, unit_circle};
use hnsw::HnswError;
use hnsw::distance::{cosine_distance, cosine_distance_unit, dot, normalize};

#[test]
fn dot_matches_hand_computation() {
    assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
}

#[test]
fn normalize_3_4_gives_0_6_0_8() {
    let u = normalize(&[3.0, 4.0]).unwrap();
    assert_close(u[0], 0.6, 1e-7);
    assert_close(u[1], 0.8, 1e-7);
}

#[test]
fn normalize_produces_unit_norm() {
    let u = normalize(&[1.0, -2.0, 3.0, -4.0, 5.0]).unwrap();
    assert_close(dot(&u, &u).sqrt(), 1.0, 1e-6);
}

#[test]
fn normalize_handles_magnitudes_that_overflow_or_underflow_in_f32() {
    // 1e-30² 在 f32 會 underflow 成 0、1e30² 會 overflow 成 inf；f64 計算 norm 仍應正確。
    let s = std::f32::consts::FRAC_1_SQRT_2;
    for v in [[1e-30, 1e-30], [1e30, 1e30]] {
        let u = normalize(&v).unwrap();
        assert_close(u[0], s, 1e-7);
        assert_close(u[1], s, 1e-7);
    }
}

#[test]
fn normalize_rejects_zero_vector() {
    assert_eq!(
        normalize(&[0.0, 0.0, 0.0]),
        Err(HnswError::ZeroVector { row: 0 })
    );
}

#[test]
fn normalize_rejects_non_finite_values() {
    assert_eq!(
        normalize(&[1.0, f32::NAN]),
        Err(HnswError::NonFiniteValue { row: 0 })
    );
    assert_eq!(
        normalize(&[f32::INFINITY, 1.0]),
        Err(HnswError::NonFiniteValue { row: 0 })
    );
}

#[test]
fn identical_vectors_have_zero_distance() {
    assert_close(
        cosine_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap(),
        0.0,
        1e-7,
    );
}

#[test]
fn orthogonal_vectors_have_distance_one() {
    assert_close(
        cosine_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap(),
        1.0,
        1e-7,
    );
}

#[test]
fn opposite_vectors_have_distance_two() {
    assert_close(
        cosine_distance(&[1.0, 2.0], &[-1.0, -2.0]).unwrap(),
        2.0,
        1e-7,
    );
}

#[test]
fn hand_verifiable_angles() {
    // d = 1 − cos(θ)
    for (deg, expected) in [(60.0, 0.5), (90.0, 1.0), (120.0, 1.5), (180.0, 2.0)] {
        let d = cosine_distance(&unit_circle(0.0), &unit_circle(deg)).unwrap();
        assert_close(d, expected, 1e-6);
    }
}

#[test]
fn distance_is_scale_invariant() {
    let a = [1.0, 2.0, -3.0];
    let b = [0.5, -1.0, 4.0];
    let b_scaled: Vec<f32> = b.iter().map(|x| x * 7.5).collect();
    assert_close(
        cosine_distance(&a, &b).unwrap(),
        cosine_distance(&a, &b_scaled).unwrap(),
        1e-6,
    );
}

#[test]
fn distance_is_symmetric() {
    let a = [0.3, -1.2, 2.5, 0.0];
    let b = [1.1, 0.4, -0.7, 2.0];
    assert_eq!(
        cosine_distance(&a, &b).unwrap(),
        cosine_distance(&b, &a).unwrap()
    );
}

#[test]
fn cosine_distance_rejects_bad_input() {
    assert_eq!(
        cosine_distance(&[1.0, 2.0], &[1.0, 2.0, 3.0]),
        Err(HnswError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    );
    // row 0 指第一個參數、row 1 指第二個參數。
    assert_eq!(
        cosine_distance(&[0.0, 0.0], &[1.0, 2.0]),
        Err(HnswError::ZeroVector { row: 0 })
    );
    assert_eq!(
        cosine_distance(&[1.0, 2.0], &[0.0, 0.0]),
        Err(HnswError::ZeroVector { row: 1 })
    );
    assert_eq!(
        cosine_distance(&[1.0, f32::NAN], &[1.0, 2.0]),
        Err(HnswError::NonFiniteValue { row: 0 })
    );
}

#[test]
fn unit_distance_matches_raw_distance_after_normalization() {
    let mut rng = SplitMix64::new(7);
    for _ in 0..100 {
        let a = rng.vectors(1, 32);
        let b = rng.vectors(1, 32);
        let expected = cosine_distance(&a, &b).unwrap();
        let actual = cosine_distance_unit(&normalize(&a).unwrap(), &normalize(&b).unwrap());
        assert_close(actual, expected, 1e-5);
    }
}
