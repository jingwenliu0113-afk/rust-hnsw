//! `rust_hnsw` Python module：hnsw crate 的 batch interface（decision 002、006）。
//!
//! 只收 float32、C-contiguous 的 2-D array，不做隱性轉型或複製；
//! 每次 `search` 處理整批 queries，計算期間釋放 GIL。
//! Advanced C 的 `QuantHnswIndex` 與 `sq8_encode`（decision 019 §4「Binding」）只新增，既有類別與 `BUILD_INFO` 不變。

use std::path::PathBuf;

use hnsw::{
    BatchResult, HnswError, HnswParams, MemoryAccounting, PADDING_ID, PersistError, QuantIndex,
    Representation, SearchStats, sq8_train_encode,
};
use numpy::ndarray::{Array2, ShapeError};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
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

// ---- Advanced C：vector quantization（decision 019 §3、§4） ----

/// `QuantHnswIndex.search_stats` 的回傳：`(ids, distances, kernel_calls, visited_inserts)`。
type StatsOutput<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
);

/// `sq8_encode` 的回傳：`(vmin, vdiff, codes)`。
type CodecOutput<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<u8>>,
);

/// `QuantHnswIndex.codebook` 的回傳：`(vmin, vdiff)`。
type CodebookOutput<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>);

/// Python 端的表示名稱：`"fp32"`（臂 F）與 `"sq8"`（臂 S）。只接受這兩個字串，不做大小寫或別名的轉換。
fn parse_representation(name: &str) -> PyResult<Representation> {
    match name {
        "fp32" => Ok(Representation::Fp32),
        "sq8" => Ok(Representation::Sq8),
        other => Err(PyValueError::new_err(format!(
            "representation must be \"fp32\" or \"sq8\", got {other:?}"
        ))),
    }
}

fn representation_name(representation: Representation) -> &'static str {
    match representation {
        Representation::Fp32 => "fp32",
        Representation::Sq8 => "sq8",
    }
}

/// 以 Core 的 `HnswIndex::new` + `add` 建圖（019 §3：F、S 與 Core 是同一張 graph）。
fn build_core(
    data: &[f32],
    dim: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> Result<hnsw::HnswIndex, HnswError> {
    let params = HnswParams {
        m,
        ef_construction,
        seed,
    };
    let mut index = hnsw::HnswIndex::new(dim, params)?;
    index.add(data)?;
    Ok(index)
}

/// `(n, d)` 的 row-major codes 轉成 uint8 陣列（複製一份，之後修改它不影響 index）。
fn codes_array<'py>(
    py: Python<'py>,
    codes: &[u8],
    dim: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = (codes.len() / dim, dim);
    Array2::from_shape_vec(shape, codes.to_vec())
        .map(|codes| codes.into_pyarray(py))
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// `sq8_train_encode` 在這些輸入上會 panic（PyO3 會轉成 `PanicException`），所以先在這裡檢查並回傳 `ValueError`：
/// 含 NaN / Inf 的 row（訊息與 Core 的 `NonFiniteValue` 相同），以及 `vmax − vmin` 溢位成 Inf 的維度。
/// `dim = 0` 由呼叫端先檢查；min / max 只用來判斷溢位，codebook 仍由 `sq8_train_encode` 計算。
fn check_codec_input(values: &[f32], dim: usize) -> Result<(), String> {
    if let Some(row) = values
        .chunks_exact(dim)
        .position(|row| row.iter().any(|x| !x.is_finite()))
    {
        return Err(HnswError::NonFiniteValue { row }.to_string());
    }
    let mut rows = values.chunks_exact(dim);
    let Some(first) = rows.next() else {
        return Ok(());
    };
    let (mut lo, mut hi) = (first.to_vec(), first.to_vec());
    for row in rows {
        for ((&x, lo), hi) in row.iter().zip(&mut lo).zip(&mut hi) {
            *lo = lo.min(x);
            *hi = hi.max(x);
        }
    }
    match lo
        .iter()
        .zip(&hi)
        .position(|(&lo, &hi)| !(hi - lo).is_finite())
    {
        Some(j) => Err(format!(
            "the range of dimension {j} overflows f32 (vmax - vmin is infinite)"
        )),
        None => Ok(()),
    }
}

/// decision 019 §3 的 SQ8 codec（per-dim min–max，對齊 FAISS 的 `QT_8bit` + `RS_minmax`、`rangestat_arg = 0`），
/// 在 shape 為 `(n, d)` 的輸入上原樣訓練並編碼：不正規化（`QuantHnswIndex` 則在 `add` 正規化之後的 unit vectors 上訓練），
/// 供 019 §4 與 FAISS 的 `compute_codes` 在同一個輸入陣列上比對。
///
/// 回傳 `(vmin, vdiff, codes)`：float32 `(d,)`、float32 `(d,)`、uint8 `(n, d)`。`n = 0` 時 vmin、vdiff 全為 0。
/// `d = 0`、含 NaN / Inf，或某一維的範圍溢位時為 `ValueError`。計算期間釋放 GIL。
#[pyfunction]
fn sq8_encode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
) -> PyResult<CodecOutput<'py>> {
    let dim = data.shape()[1];
    let values = rows(&data, dim)?;
    if dim == 0 {
        return Err(to_py_err(HnswError::InvalidDimension(dim)));
    }
    let (vmin, vdiff, codes) = py
        .detach(|| check_codec_input(values, dim).map(|()| sq8_train_encode(values, dim)))
        .map_err(PyValueError::new_err)?;
    Ok((
        vmin.into_pyarray(py),
        vdiff.into_pyarray(py),
        codes_array(py, &codes, dim)?,
    ))
}

fn memory_dict(py: Python<'_>, accounting: MemoryAccounting) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("struct_bytes", accounting.struct_bytes)?;
    out.set_item("graph_bytes", accounting.graph_bytes)?;
    out.set_item("vectors_bytes", accounting.vectors_bytes)?;
    out.set_item("codes_bytes", accounting.codes_bytes)?;
    out.set_item("codebook_bytes", accounting.codebook_bytes)?;
    out.set_item("total_bytes", accounting.total_bytes())?;
    Ok(out)
}

/// Advanced C 的 HNSW index（decision 019 §3、§4）：以 Core 建出的 graph，搭配 FP32（臂 F）或 SQ8（臂 S）的向量表示。
///
/// 只能搜尋，不能 add，也沒有 save / load。沒有建構子，只能由 `build` / `build_pair` 建立。
#[pyclass(name = "QuantHnswIndex", module = "rust_hnsw")]
struct PyQuantHnswIndex {
    inner: QuantIndex,
}

#[pymethods]
impl PyQuantHnswIndex {
    /// 以 Core 的 `HnswIndex::new` + `add` 建圖，再轉成 `representation`（`"fp32"` 或 `"sq8"`）。
    /// 所有參數都必須明確給定，沒有預設值（019 §4）。`data` 的 shape 為 `(n, dim)`。計算期間釋放 GIL。
    #[staticmethod]
    fn build(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f32>,
        m: usize,
        ef_construction: usize,
        seed: u64,
        representation: &str,
    ) -> PyResult<Self> {
        let representation = parse_representation(representation)?;
        let dim = data.shape()[1];
        let data = rows(&data, dim)?;
        py.detach(|| {
            build_core(data, dim, m, ef_construction, seed)
                .map(|index| index.into_quant(representation))
        })
        .map(|inner| Self { inner })
        .map_err(to_py_err)
    }

    /// 只給 decision 019 §7 的診斷使用：建一次圖，clone 後分別轉成 FP32 與 SQ8，回傳 `(fp32, sq8)`。
    ///
    /// 兩者都由 clone 轉換：`Vec::clone` 會把 capacity 收緊成長度，兩者的 graph capacity 因此相同，
    /// `memory_accounting` 的差恰為 `3·n·d − 8·d`。計算期間釋放 GIL。
    #[staticmethod]
    fn build_pair(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f32>,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> PyResult<(Self, Self)> {
        let dim = data.shape()[1];
        let data = rows(&data, dim)?;
        let (fp32, sq8) = py
            .detach(|| {
                build_core(data, dim, m, ef_construction, seed).map(|index| {
                    // 先轉 SQ8（轉換時同時持有 FP32 向量與 codes），再轉 FP32，峰值記憶體較低。
                    let sq8 = index.clone().into_quant(Representation::Sq8);
                    let fp32 = index.clone().into_quant(Representation::Fp32);
                    (fp32, sq8)
                })
            })
            .map_err(to_py_err)?;
        Ok((Self { inner: fp32 }, Self { inner: sq8 }))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// `"fp32"` 或 `"sq8"`。
    #[getter]
    fn representation(&self) -> &'static str {
        representation_name(self.inner.representation())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// 一次批次搜尋，layer 0 使用 `max(ef_search, k)`；語意與輸出格式與 `HnswIndex.search` 相同。
    /// FP32 的 ids 與 distances 與 Core 逐位元相同；SQ8 的距離在輸出時夾到 `[0, 2]`（019 §3）。
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

    /// 不計時的診斷（019 §7）：結果與 `search` 相同，另外回傳每個 query 的計數。
    /// 回傳 `(ids, distances, kernel_calls, visited_inserts)`；後兩者是 uint64、shape 為 `(num_queries,)`：
    /// 呼叫距離 kernel 的次數（所有層，含 entry point；019 §2 的 D），與呼叫 visited set insert 的次數（含 entry）。
    fn search_stats<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<StatsOutput<'py>> {
        let queries = rows(&queries, self.inner.dim())?;
        let inner = &self.inner;
        let (arrays, stats) = py
            .detach(|| {
                inner
                    .search_stats(queries, k, ef_search)
                    .map(|(batch, stats)| (to_arrays(batch), stats))
            })
            .map_err(to_py_err)?;
        let (ids, distances) = into_output(py, arrays)?;
        let SearchStats {
            kernel_calls,
            visited_inserts,
        } = stats;
        Ok((
            ids,
            distances,
            kernel_calls.into_pyarray(py),
            visited_inserts.into_pyarray(py),
        ))
    }

    /// 以同一個 kernel 對所有點做 brute force（019 §7 的天花板；只作分析基準）。輸出格式與 `search` 相同。
    fn exact_search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
    ) -> PyResult<SearchOutput<'py>> {
        let queries = rows(&queries, self.inner.dim())?;
        let inner = &self.inner;
        let arrays = py
            .detach(|| inner.exact_search_batch(queries, k).map(to_arrays))
            .map_err(to_py_err)?;
        into_output(py, arrays)
    }

    /// 結構大小（019 §7 的 M_struct）：各 Vec 的 capacity × 元素大小，加上 header；與 allocator 無關。
    /// 回傳 dict：`struct_bytes`、`graph_bytes`、`vectors_bytes`、`codes_bytes`、`codebook_bytes`、`total_bytes`。
    fn memory_accounting<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        memory_dict(py, self.inner.memory_accounting())
    }

    /// SQ8 的 codes：uint8、shape 為 `(n, dim)`（複製一份）；FP32 為 `None`。
    fn codes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<u8>>>> {
        self.inner
            .codes()
            .map(|codes| codes_array(py, codes, self.inner.dim()))
            .transpose()
    }

    /// SQ8 的 codebook `(vmin, vdiff)`：float32、shape 都是 `(dim,)`（複製一份）；FP32 為 `None`。
    fn codebook<'py>(&self, py: Python<'py>) -> Option<CodebookOutput<'py>> {
        self.inner.codebook().map(|(vmin, vdiff)| {
            (
                PyArray1::from_slice(py, vmin),
                PyArray1::from_slice(py, vdiff),
            )
        })
    }

    // ---- 測試用的 graph 存取（019 §4） ----

    /// 目前的 entry point；空 index 為 `None`。
    #[getter]
    fn entry_point(&self) -> Option<u32> {
        self.inner.entry_point()
    }

    /// 節點的最高層；id 不存在為 `None`。
    fn level(&self, id: u32) -> Option<usize> {
        self.inner.level(id)
    }

    /// 節點在某一層的鄰居（依儲存順序）；id 不存在或高於該節點的 level 為 `None`。
    fn neighbors(&self, id: u32, level: usize) -> Option<Vec<u32>> {
        self.inner.neighbors(id, level).map(<[u32]>::to_vec)
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
    m.add_class::<PyQuantHnswIndex>()?;
    m.add_function(wrap_pyfunction!(sq8_encode, m)?)?;
    m.add("BUILD_INFO", build_info(m.py())?)?;
    Ok(())
}
