//! `rust_hnsw` Python module：hnsw crate 的 batch interface（decision 002、006）。
//!
//! 只收 float32、C-contiguous 的 2-D array，不做隱性轉型或複製；
//! 每次 `search` 處理整批 queries，計算期間釋放 GIL。

use std::path::PathBuf;

use hnsw::{BatchResult, HnswError, HnswParams, PADDING_ID, PersistError};
use numpy::ndarray::{Array2, ShapeError};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `search` 的回傳：`(ids, distances)`。
type SearchOutput<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>);

fn to_py_err(err: HnswError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// I/O 錯誤依 `io::ErrorKind` 轉成對應的 `OSError` 子類；格式與損壞錯誤為 `ValueError`。
fn persist_err(err: PersistError) -> PyErr {
    match err {
        PersistError::Io(io) => PyErr::from(io),
        other => PyValueError::new_err(other.to_string()),
    }
}

type Arrays = (Array2<i64>, Array2<f32>);

/// 批次結果轉成 `(num_queries, k)` 的陣列；`PADDING_ID` 轉成 `-1`（decision 002）。
fn to_arrays(result: BatchResult) -> Result<Arrays, ShapeError> {
    let shape = (result.num_queries, result.k);
    let ids = result
        .ids
        .into_iter()
        .map(|id| if id == PADDING_ID { -1 } else { i64::from(id) })
        .collect();
    Ok((
        Array2::from_shape_vec(shape, ids)?,
        Array2::from_shape_vec(shape, result.distances)?,
    ))
}

fn into_output(py: Python<'_>, arrays: Result<Arrays, ShapeError>) -> PyResult<SearchOutput<'_>> {
    let (ids, distances) = arrays.map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((ids.into_pyarray(py), distances.into_pyarray(py)))
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
        let arrays = py
            .detach(|| inner.search_batch(queries, k).map(to_arrays))
            .map_err(to_py_err)?;
        into_output(py, arrays)
    }
}

/// HNSW index（decision 004–006）。
#[pyclass(name = "HnswIndex", module = "rust_hnsw")]
struct PyHnswIndex {
    inner: hnsw::HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    /// 未指定的參數使用 `HnswParams::default()`（m = 16、ef_construction = 200、seed = 100）。
    #[new]
    #[pyo3(signature = (dim, m = None, ef_construction = None, seed = None))]
    fn new(
        dim: usize,
        m: Option<usize>,
        ef_construction: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let defaults = HnswParams::default();
        let params = HnswParams {
            m: m.unwrap_or(defaults.m),
            ef_construction: ef_construction.unwrap_or(defaults.ef_construction),
            seed: seed.unwrap_or(defaults.seed),
        };
        hnsw::HnswIndex::new(dim, params)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.params().m
    }

    #[getter]
    fn ef_construction(&self) -> usize {
        self.inner.params().ef_construction
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.params().seed
    }

    /// graph 最高層；空 index 為 `None`。
    #[getter]
    fn max_level(&self) -> Option<usize> {
        self.inner.max_level()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// 依 row 順序插入 shape 為 `(n, dim)` 的向量；任一 row 不合法則整批不插入。
    fn add(&mut self, py: Python<'_>, data: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let data = rows(&data, self.inner.dim())?;
        let inner = &mut self.inner;
        py.detach(|| inner.add(data)).map_err(to_py_err)
    }

    /// 一次批次搜尋，layer 0 使用 `max(ef_search, k)`。
    /// 回傳 `(ids, distances)`：int64 與 float32，shape 都是 `(num_queries, k)`；不足 k 時補 `-1` / `+inf`。
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<SearchOutput<'py>> {
        let queries = rows(&queries, self.inner.dim())?;
        let inner = &self.inner;
        let arrays = py
            .detach(|| inner.search_batch(queries, k, ef_search).map(to_arrays))
            .map_err(to_py_err)?;
        into_output(py, arrays)
    }

    /// 以 decision 005 的格式存檔（先寫暫存檔再 rename）。
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.save_to_path(&path)).map_err(persist_err)
    }

    /// 讀取 decision 005 格式的檔案；損壞、截斷或格式不符時為 `ValueError`。
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        py.detach(|| hnsw::HnswIndex::load_from_path(&path))
            .map(|inner| Self { inner })
            .map_err(persist_err)
    }
}

/// 編譯設定（`build.rs` 在編譯時記錄，decision 008 §10）。formal benchmark 以此確認被量測的是 release build。
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("profile", env!("RUST_HNSW_BUILD_PROFILE"))?;
    info.set_item("opt_level", env!("RUST_HNSW_BUILD_OPT_LEVEL"))?;
    info.set_item("debug", env!("RUST_HNSW_BUILD_DEBUG"))?;
    info.set_item("debug_assertions", cfg!(debug_assertions))?;
    info.set_item("rustflags", env!("RUST_HNSW_BUILD_RUSTFLAGS"))?;
    info.set_item("profile_env", env!("RUST_HNSW_BUILD_PROFILE_ENV"))?;
    info.set_item("rustc", env!("RUST_HNSW_BUILD_RUSTC"))?;
    Ok(info)
}

#[pymodule]
fn rust_hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExactIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    m.add("BUILD_INFO", build_info(m.py())?)?;
    Ok(())
}
