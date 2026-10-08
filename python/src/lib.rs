//! `rust_hnsw` Python module：hnsw crate 的 batch interface（decision 002）。
//!
//! 只收 float32、C-contiguous 的 2-D array，不做隱性轉型或複製；
//! 每次 `search` 處理整批 queries，搜尋期間釋放 GIL。

use hnsw::HnswError;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `search` 的回傳：`(ids, distances)`。
type SearchOutput<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>);

fn to_py_err(err: HnswError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// 取出 row-major slice；要求 C-contiguous 且列寬等於 `dim`。
fn rows<'a>(array: &'a PyReadonlyArray2<'_, f32>, dim: usize) -> PyResult<&'a [f32]> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "array must be a C-contiguous float32 array",
        ));
    }
    let cols = array.shape()[1];
    if cols != dim {
        return Err(to_py_err(HnswError::DimensionMismatch {
            expected: dim,
            actual: cols,
        }));
    }
    array
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Exact brute-force cosine index：ground truth 與 correctness baseline。
#[pyclass(name = "ExactIndex", module = "rust_hnsw")]
struct PyExactIndex {
    inner: hnsw::ExactIndex,
}

#[pymethods]
impl PyExactIndex {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        hnsw::ExactIndex::new(dim)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// 加入 shape 為 `(n, dim)` 的向量；id 接續目前的長度。任一 row 不合法則整批不寫入。
    fn add(&mut self, py: Python<'_>, data: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let data = rows(&data, self.inner.dim())?;
        let inner = &mut self.inner;
        py.detach(|| inner.add(data)).map_err(to_py_err)
    }

    /// 對 shape 為 `(num_queries, dim)` 的 queries 做一次批次搜尋。
    /// 回傳 `(ids, distances)`：int64 與 float32，shape 都是 `(num_queries, k)`。
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
    ) -> PyResult<SearchOutput<'py>> {
        let queries = rows(&queries, self.inner.dim())?;
        let inner = &self.inner;
        let (ids, distances) = py
            .detach(|| {
                inner.search_batch(queries, k).map(|result| {
                    let shape = (result.num_queries, result.k);
                    let ids = result.ids.into_iter().map(i64::from).collect();
                    (
                        Array2::from_shape_vec(shape, ids),
                        Array2::from_shape_vec(shape, result.distances),
                    )
                })
            })
            .map_err(to_py_err)?;
        let shape_err = |err: numpy::ndarray::ShapeError| PyValueError::new_err(err.to_string());
        Ok((
            ids.map_err(shape_err)?.into_pyarray(py),
            distances.map_err(shape_err)?.into_pyarray(py),
        ))
    }
}

#[pymodule]
fn rust_hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExactIndex>()?;
    Ok(())
}
