# Taskflow 任务清理函数详解

本文档详细介绍 Taskflow 中任务清理相关的函数及其调用关系。

## 函数调用关系图

```
_invoke() [任务执行主函数]
    │
    ├─> 执行任务逻辑
    │
    ├─> 调度后继任务
    │
    └─> _tear_down_nonasync() [清理非异步任务]
            │
            ├─> 情况1: 顶层任务 (parent == nullptr)
            │       │
            │       └─> Topology->_join_counter.fetch_sub(1)
            │               │
            │               └─> 如果变为 0: _tear_down_topology()
            │                       │
            │                       ├─> 情况1: 需要再次运行 (run_until)
            │                       │       └─> _set_up_topology() [重新设置]
            │                       │
            │                       └─> 情况2: 最后一次运行
            │                               ├─> 调用完成回调 (_call)
            │                               ├─> 设置 Promise (_carry_out_promise)
            │                               ├─> 处理下一个 Topology
            │                               └─> 清理资源
            │
            └─> 情况2: 子任务 (parent != nullptr)
                    │
                    └─> parent->_join_counter.fetch_sub(1)
                            │
                            └─> 如果变为 0 且 parent 被抢占:
                                    └─> _update_cache() [缓存父节点]
                                            │
                                            ├─> 如果 cache 已有任务: _schedule()
                                            └─> cache = parent
```

## 核心函数详解

### 1. `_tear_down_nonasync()`

**功能**：清理非异步任务节点，更新父节点或 Topology 的 join_counter

**调用时机**：
- 在 `_invoke()` 函数的最后阶段（阶段 5）
- 任务执行完成，后继任务已经调度完毕

**核心逻辑**：
1. 减少父节点（或 Topology）的 `join_counter`
2. 如果 `join_counter` 变为 0，说明所有子任务（或整个 Topology）完成
3. 触发相应的完成处理逻辑

**关键点**：
- **必须先检查 parent，再减少 join_counter**，避免数据竞争
- **必须在减少 join_counter 之前读取所有需要的数据**，避免悬空指针

### 2. `_update_cache()`

**功能**：更新任务缓存，实现尾调用优化（Tail Call Optimization）

**核心思想**：
- 如果只有一个后继任务就绪，不要放入队列，而是缓存起来
- 在 `_invoke()` 函数末尾通过 `goto` 跳转直接执行缓存的任务
- 避免队列操作的开销，提高缓存局部性

**调用场景**：
1. 调度后继任务时（在 `_invoke()` 的阶段 4）
2. 恢复被抢占的父节点时（在 `_tear_down_nonasync` 中）

**优化效果**：
- 避免 push + pop 的开销
- 提高 CPU 缓存命中率（任务数据仍在 L1 缓存中）
- 形成任务链，避免递归调用和栈溢出

### 3. `_tear_down_topology()`

**功能**：清理 Topology，处理完成后的逻辑

**调用时机**：
- 当 Topology 的 `join_counter` 变为 0 时（所有任务执行完毕）
- 在 `_tear_down_nonasync()` 中调用

**核心逻辑**：
1. 检查是否需要再次运行（`run_until` 场景）
2. 调用完成回调
3. 设置 Promise（通知等待的线程）
4. 处理下一个 Topology（如果有）
5. 清理资源

**两种情况**：

#### 情况 1：需要再次运行（run_until 场景）
- 判断条件：没有异常、没有被取消、谓词返回 false
- 处理：重新设置 Topology，准备下一次运行

#### 情况 2：最后一次运行（正常完成）
- 调用完成回调（如果有）
- 设置 Promise，通知等待的线程
- 如果有下一个 Topology，开始执行
- 清理资源（包括 satellite Taskflow）

## 关键概念

### join_counter 的含义

#### Topology 的 join_counter
- 记录当前正在执行或等待执行的任务数量
- 初始值：源节点（无前驱的节点）的数量
- 每调度一个后继任务：`join_counter` 加 1
- 每完成一个任务：`join_counter` 减 1
- 当 `join_counter` 变为 0：整个 Topology 执行完成

#### 父节点的 join_counter
- 记录父节点的子任务中还有多少未完成
- 初始值：子图的源节点数量
- 每调度一个子任务的后继：`join_counter` 加 1
- 每完成一个子任务：`join_counter` 减 1
- 当 `join_counter` 变为 0：所有子任务完成，父节点可以恢复执行

### 抢占（PREEMPTED）机制

**什么是抢占？**
- 父节点（Subflow、Runtime、Module）创建了子任务
- 父节点不能阻塞等待，而是被标记为 PREEMPTED 并返回
- 当所有子任务完成后，父节点需要被重新调度

**为什么需要抢占？**
- 避免工作线程阻塞等待
- 提高线程利用率
- 支持嵌套并行

### 尾调用优化（Tail Call Optimization）

**核心思想**：
- 通过 `cache` 变量缓存单个就绪的后继任务
- 在 `_invoke()` 函数末尾通过 `goto` 跳转直接执行
- 形成任务链，避免递归调用

**实现机制**：
```cpp
#define TF_INVOKE_CONTINUATION() \
    if (cache) {                 \
        node = cache;            \
        goto begin_invoke;       \
    }
```

**优化效果**：
- 避免队列操作（push + pop）
- 提高缓存局部性
- 减少上下文切换
- 避免栈溢出

## 数据竞争的避免

### 问题：为什么必须先检查 parent，再减少 join_counter？

**错误的顺序**：
1. 线程 A：`parent->_join_counter.fetch_sub(1)` 返回 1（变为 0）
2. 线程 B：parent 被删除（因为 `join_counter` 为 0）
3. 线程 A：访问 `parent->_nstate`（悬空指针！）

**正确的顺序**：
1. 线程 A：读取 `parent` 指针
2. 线程 A：读取 `parent->_nstate`（安全，因为 `join_counter` 还未减到 0）
3. 线程 A：`parent->_join_counter.fetch_sub(1)`

### 问题：为什么必须在减少 join_counter 之前读取所有需要的数据？

**原因**：`join_counter` 变为 0 后，节点可能被其他线程删除

**解决方案**：
```cpp
auto state = parent->_nstate;  // 提前读取，保存到局部变量
if (parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    if (state & NSTATE::PREEMPTED) {  // 使用局部变量，安全
        _update_cache(worker, cache, parent);
    }
}
```

## 执行示例

以 `simple.cpp` 为例，说明这些函数的执行流程：

```
任务图：
    A
   / \
  B   C
   \ /
    D
```

### 执行流程

#### 1. 初始状态
- Topology->_join_counter = 1（源节点 A）
- A._join_counter = 0（无前驱）
- B._join_counter = 1（前驱：A）
- C._join_counter = 1（前驱：A）
- D._join_counter = 2（前驱：B, C）

#### 2. 执行任务 A
```cpp
_invoke(worker, A)
    ├─> 执行 A 的工作函数
    ├─> 调度后继任务 B 和 C
    │   ├─> B._join_counter: 1 -> 0（就绪）
    │   │   ├─> Topology->_join_counter: 1 -> 2
    │   │   └─> _update_cache(worker, cache, B)  // cache = B
    │   └─> C._join_counter: 1 -> 0（就绪）
    │       ├─> Topology->_join_counter: 2 -> 3
    │       └─> _update_cache(worker, cache, C)  // 调度 B，cache = C
    └─> _tear_down_nonasync(worker, A, cache)
        └─> Topology->_join_counter: 3 -> 2（未完成）
```

#### 3. 执行任务 B（从队列取出）
```cpp
_invoke(worker, B)
    ├─> 执行 B 的工作函数
    ├─> 调度后继任务 D
    │   └─> D._join_counter: 2 -> 1（未就绪）
    └─> _tear_down_nonasync(worker, B, cache)
        └─> Topology->_join_counter: 2 -> 1（未完成）
```

#### 4. 执行任务 C（从 cache）
```cpp
_invoke(worker, C)  // 通过 TF_INVOKE_CONTINUATION() 从 cache 执行
    ├─> 执行 C 的工作函数
    ├─> 调度后继任务 D
    │   ├─> D._join_counter: 1 -> 0（就绪）
    │   ├─> Topology->_join_counter: 1 -> 2
    │   └─> _update_cache(worker, cache, D)  // cache = D
    └─> _tear_down_nonasync(worker, C, cache)
        └─> Topology->_join_counter: 2 -> 1（未完成）
```

#### 5. 执行任务 D（从 cache）
```cpp
_invoke(worker, D)  // 通过 TF_INVOKE_CONTINUATION() 从 cache 执行
    ├─> 执行 D 的工作函数
    ├─> 没有后继任务
    └─> _tear_down_nonasync(worker, D, cache)
        └─> Topology->_join_counter: 1 -> 0（完成！）
            └─> _tear_down_topology(worker, Topology)
                ├─> 调用完成回调（如果有）
                ├─> 设置 Promise（通知等待的线程）
                └─> 清理资源
```

### 关键观察

1. **join_counter 的变化**：
   - Topology 的 `join_counter` 从 1 开始，最终变为 0
   - 每调度一个后继任务，`join_counter` 加 1
   - 每完成一个任务，`join_counter` 减 1

2. **尾调用优化**：
   - 任务 C 和 D 通过 `cache` 直接执行，避免队列操作
   - 形成任务链：A -> C -> D

3. **并发执行**：
   - 任务 B 和 C 可以并发执行（都就绪）
   - 任务 D 必须等待 B 和 C 都完成

## 总结

这三个函数共同实现了 Taskflow 的任务清理和完成处理机制：

1. **`_tear_down_nonasync()`**：任务级别的清理，更新 `join_counter`
2. **`_update_cache()`**：任务调度优化，实现尾调用
3. **`_tear_down_topology()`**：Topology 级别的清理，处理完成逻辑

通过精心设计的同步机制和优化策略，Taskflow 实现了高效、安全的并发任务执行。

### 设计亮点

1. **无锁同步**：使用原子操作实现 `join_counter` 的更新
2. **尾调用优化**：通过 `cache` 和 `goto` 实现任务链式执行
3. **数据竞争避免**：精心设计的读取顺序，避免悬空指针
4. **灵活的完成处理**：支持 `run_until`、回调、Promise 等多种机制
5. **资源管理**：支持 satellite Taskflow 的自动清理

