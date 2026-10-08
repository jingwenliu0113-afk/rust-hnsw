//! load 不得依檔案宣稱的大小預先配置記憶體（decision 005「記憶體上限」）。
//!
//! 以計數用的 global allocator 記錄 load 期間的峰值配置量。macOS 的大配置是 lazy commit，
//! 只看是否 OOM 偵測不到預先 reserve，所以要直接計數。
//! 這個檔案只能有一個 `#[test]`：同一個 test binary 內的測試會平行執行，互相干擾計數。

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::spec_codec::{self, GAMMA, SpecFile, recompute_crcs};
use hnsw::{HnswIndex, MAX_M, NodeId, PersistError};

struct PeakAlloc;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn record_growth(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_growth(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                record_growth(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

/// 執行 `f`，回傳結果與期間相對於起點的峰值配置量。
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let result = f();
    (result, PEAK.load(Ordering::SeqCst).saturating_sub(base))
}

const LIMIT: usize = 1 << 20;

#[test]
fn load_does_not_allocate_from_claimed_sizes() {
    // (a) header 宣稱的大小遠超過實際資料：
    //     極大（n = NodeId::MAX、dim = 2^20，n·dim·4 ≈ 2^54 bytes，不溢位）與中等（約 64 MiB）兩種。
    //     中等的預先 reserve 在 macOS 會 lazy 成功、不會 abort，只能靠峰值計數抓到，所以放在最前面：
    //     極大的預先 reserve 會讓配置失敗而 abort，若先執行就輪不到峰值斷言。
    for (case, len, dim, data_bytes) in [
        ("64 MiB claim, header only", 1 << 20, 16, 0),
        (
            "huge claim, header only",
            u64::from(NodeId::MAX),
            1u64 << 20,
            0usize,
        ),
        (
            "huge claim, 64 KiB of data",
            u64::from(NodeId::MAX),
            1 << 20,
            64 * 1024,
        ),
    ] {
        let mut spec = SpecFile::valid(dim, 16, 100, 5, &[], vec![]);
        spec.len = len;
        spec.rng_state = spec.seed.wrapping_add(len.wrapping_mul(GAMMA));
        spec.entry_point = 0;
        let mut bytes = spec_codec::encode_header(&spec);
        bytes.extend(std::iter::repeat_n(0u8, data_bytes));

        let (result, peak) = peak_during(|| HnswIndex::load(bytes.as_slice()));
        assert!(
            matches!(result, Err(PersistError::Truncated)),
            "{case}: {result:?}"
        );
        assert!(peak < LIMIT, "{case}: peak allocation {peak} bytes");
    }

    // (b) m 合法但很大、n 小。u32::MAX 由 `count ≤ 2m` 擋下；
    //     2m（= 2^32 − 2）通過 `count ≤ 2m`，只有 `count ≤ n − 1` 擋得下，而且必須在配置之前擋下。
    let spec = SpecFile::valid(
        2,
        MAX_M as u64,
        10,
        5,
        &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
        vec![vec![vec![1, 2]], vec![vec![0, 2]], vec![vec![0, 1]]],
    );
    let base = spec_codec::encode(&spec);
    HnswIndex::load(base.as_slice()).expect("base spec must be valid");
    let count_at = spec_codec::HEADER_LEN + 3 * 2 * 4 + 4;
    for count in [u32::MAX, u32::try_from(2 * MAX_M).unwrap()] {
        let mut bytes = base.clone();
        bytes[count_at..count_at + 4].copy_from_slice(&count.to_le_bytes());
        recompute_crcs(&mut bytes);
        let (result, peak) = peak_during(|| HnswIndex::load(bytes.as_slice()));
        assert!(
            matches!(result, Err(PersistError::Corrupt(_))),
            "count {count}: {result:?}"
        );
        assert!(peak < LIMIT, "count {count}: peak allocation {peak} bytes");
    }
}
