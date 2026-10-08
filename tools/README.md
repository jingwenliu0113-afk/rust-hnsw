# Mac Heavy Lock

這個資料夾的 `with-mac-lock` 用來避免不同 Project 同時執行會互相污染結果的 Mac heavy workload。

## 使用方式

```bash
./tools/with-mac-lock <command> [args...]
```

## 不需要鎖的工作

- 寫 code
- unit tests
- correctness tests
- 文件
- 小型 smoke tests

## 需要鎖的工作

依 `MACHINE_POLICY.md` 定義。

不要自行刪除其他 session 建立的 lock。
