# simple.cpp 执行流程总结

## 任务图

```
       A
      / \
     B   C
      \ /
       D
```

## 核心数据：Topology->_join_counter 的变化

| 时刻 | 操作 | 变化 | 值 | 说明 |
|------|------|------|-----|------|
| 初始 | 设置 | - | 1 | 源节点数量（A） |
| A 执行 | 调度 B | +1 | 2 | B 就绪 |
| A 执行 | 调度 C | +1 | 3 | C 就绪 |
| A 完成 | _tear_down_nonasync(A) | -1 | 2 | fetch_sub 返回 3 |
| C 执行 | D 未就绪 | +0 | 2 | D._join_counter: 2→1 |
| C 完成 | _tear_down_nonasync(C) | -1 | 1 | fetch_sub 返回 2 |
| B 执行 | 调度 D | +1 | 2 | D 就绪 |
| B 完成 | _tear_down_nonasync(B) | -1 | 1 | fetch_sub 返回 2 |
| D 执行 | 无后继 | +0 | 1 | D 无后继任务 |
| D 完成 | _tear_down_nonasync(D) | -1 | 0 | **fetch_sub 返回 1，触发清理！** |

## 关键代码执行

### 任务 A 完成时

```cpp
_tear_down_nonasync(worker, A, cache) {
    if (A._parent == nullptr) {
        // 返回 3，减少后变为 2
        if (A._topology->_join_counter.fetch_sub(1, acq_rel) == 1) {
            // 3 != 1，不执行
        }
    }
}
```

### 任务 C 完成时

```cpp
_tear_down_nonasync(worker, C, cache) {
    if (C._parent == nullptr) {
        // 返回 2，减少后变为 1
        if (C._topology->_join_counter.fetch_sub(1, acq_rel) == 1) {
            // 2 != 1，不执行
        }
    }
}
```

### 任务 B 完成时

```cpp
_tear_down_nonasync(worker, B, cache) {
    if (B._parent == nullptr) {
        // 返回 2，减少后变为 1
        if (B._topology->_join_counter.fetch_sub(1, acq_rel) == 1) {
            // 2 != 1，不执行
        }
    }
}
```

### 任务 D 完成时（触发清理）

```cpp
_tear_down_nonasync(worker, D, cache) {
    if (D._parent == nullptr) {
        // 返回 1，减少后变为 0
        if (D._topology->_join_counter.fetch_sub(1, acq_rel) == 1) {
            // 1 == 1，执行！
            _tear_down_topology(worker, D._topology);
            // ↓
            // 设置 Promise
            // 唤醒 main 线程
            // 清理资源
        }
    }
}
```

## 执行顺序

1. **A** 执行（初始调度）
2. **C** 执行（从 cache，尾调用优化）
3. **B** 执行（从队列）
4. **D** 执行（从 cache，尾调用优化）

## 尾调用优化

- **A → C**：C 被缓存，通过 `goto begin_invoke` 直接执行
- **B → D**：D 被缓存，通过 `goto begin_invoke` 直接执行

**效果**：避免了 2 次 push + 2 次 pop 操作

## 为什么只有 D 触发清理？

**关键**：`fetch_sub(1)` 返回的是**减少前**的值

- A 完成：`fetch_sub` 返回 3（减少后 2）→ 不触发
- C 完成：`fetch_sub` 返回 2（减少后 1）→ 不触发
- B 完成：`fetch_sub` 返回 2（减少后 1）→ 不触发
- D 完成：`fetch_sub` 返回 1（减少后 0）→ **触发！**

只有当返回值为 1 时，说明减少后变为 0，Topology 完成。

## 并发场景

如果有多个工作线程，B 和 C 可能并发执行：

```
线程 1 执行 B：
  D._join_counter.fetch_sub(1): 2 → 1（未就绪）
  Topology->_join_counter.fetch_add(1): 1 → 2
  Topology->_join_counter.fetch_sub(1): 2 → 1（未触发）

线程 2 执行 C：
  D._join_counter.fetch_sub(1): 1 → 0（就绪！）
  Topology->_join_counter.fetch_add(1): 1 → 2
  Topology->_join_counter.fetch_sub(1): 2 → 1（未触发）

某个线程执行 D：
  Topology->_join_counter.fetch_sub(1): 1 → 0（触发！）
```

原子操作保证了即使并发执行，D 也只会被调度一次，清理也只会触发一次。

## 核心设计

1. **join_counter 机制**：
   - 每调度一个后继任务：+1
   - 每完成一个任务：-1
   - 变为 0：触发清理

2. **原子操作**：
   - `fetch_sub` 和 `fetch_add` 保证线程安全
   - 无需加锁，高效并发

3. **尾调用优化**：
   - 通过 cache 和 goto 实现任务链
   - 减少队列操作，提高性能

4. **精确的完成检测**：
   - 通过 `fetch_sub` 的返回值判断
   - 只有最后一个任务触发清理
   - 不会重复或遗漏

## 总结

`_tear_down_nonasync` 是 Taskflow 任务完成处理的核心函数：

- **每个任务都调用**：更新 join_counter
- **只有最后一个触发清理**：通过原子操作的返回值判断
- **线程安全**：原子操作保证并发正确性
- **高效**：无锁设计，尾调用优化

这个设计体现了 Taskflow 在并发编程中的精妙之处：简单、高效、安全。

