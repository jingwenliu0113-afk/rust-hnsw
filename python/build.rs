//! 在編譯時記錄被量測的 binary 的編譯設定（decision 008 §10），以 `rust_hnsw.BUILD_INFO` 公開。
//! runtime 讀到的 shell 環境不代表 `.so` 當初的編譯設定，所以只能在這裡記錄。

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    for key in [
        "LTO",
        "CODEGEN_UNITS",
        "OVERFLOW_CHECKS",
        "PANIC",
        "OPT_LEVEL",
        "DEBUG_ASSERTIONS",
        "CODEGEN_BACKEND",
    ] {
        println!("cargo:rerun-if-env-changed=CARGO_PROFILE_RELEASE_{key}");
    }

    for key in ["PROFILE", "OPT_LEVEL", "DEBUG"] {
        let value = env::var(key).unwrap_or_default();
        println!("cargo:rustc-env=RUST_HNSW_BUILD_{key}={value}");
    }
    // CARGO_ENCODED_RUSTFLAGS 以 0x1f 分隔各旗標。
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\x1f', " ");
    println!("cargo:rustc-env=RUST_HNSW_BUILD_RUSTFLAGS={rustflags}");

    // 以環境變數覆寫的 cargo profile 設定（例如 CARGO_PROFILE_RELEASE_LTO）。
    // cargo 設定檔中的 [profile] 無法從這裡觀察（008 §11 的已知限制）。
    let mut overrides: Vec<String> = env::vars()
        .filter(|(key, _)| key.starts_with("CARGO_PROFILE_"))
        .map(|(key, value)| format!("{key}={value}"))
        .collect();
    overrides.sort();
    println!(
        "cargo:rustc-env=RUST_HNSW_BUILD_PROFILE_ENV={}",
        overrides.join(" ")
    );

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let version = Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_default();
    println!("cargo:rustc-env=RUST_HNSW_BUILD_RUSTC={version}");
}
