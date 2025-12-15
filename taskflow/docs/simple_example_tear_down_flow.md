# simple.cpp 的 _tear_down_nonasync 执行流程详解

## 任务图结构

```
       +---+
 +---->| B |-----+
 |     +---+     |
+---+           +-v-+
| A |           | D |
+---+           +-^-+
 |     +---+     |
 +---->| C |-----+
       +---+
```

**依赖关系**：
- A 是源节点（无前驱）
- B 的前驱是 A
- C 的前驱是 A
- D 的前驱是 B 和 C

## 初始状态

### 各节点的 _join_counter 初始值

在 `_set_up_topology()` 中设置：

```cpp
A._join_counter = 0  // 无前驱
B._join_counter = 1  // 前驱：A
C._join_counter = 1  // 前驱：A
D._join_counter = 2  // 前驱：B, C
```

### Topology 的 _join_counter 初始值

```cpp
Topology->_join_counter = 1  // 源节点数量（只有 A）
```

### 各节点的 _parent

```cpp
A._parent = nullptr  // 顶层任务
B._parent = nullptr  // 顶层任务
C._parent = nullptr  // 顶层任务
D._parent = nullptr  // 顶层任务
```

## 详细执行流程

### 阶段 1：执行任务 A

#### 1.1 _invoke(worker, A) - 执行任务

```cpp
_invoke(worker, A) {
    // 执行 A 的工作函数
    std::cout << "TaskA\n";
    
    // 调度后继任务 B 和 C
    for (size_t i = 0; i < A._num_successors; ++i) {  // A._num_successors = 2
        auto s = A._edges[i];  // s = B, 然后 s = C
        
        // 处理后继任务 B
        if (B._join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // B._join_counter: 1 -> 0（就绪！）
            Topology->_join_counter.fetch_add(1, std::memory_order_relaxed);
            // Topology->_join_counter: 1 -> 2
            _update_cache(worker, cache, B);
            // cache = B
        }
        
        // 处理后继任务 C
        if (C._join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // C._join_counter: 1 -> 0（就绪！）
            Topology->_join_counter.fetch_add(1, std::memory_order_relaxed);
            // Topology->_join_counter: 2 -> 3
            _update_cache(worker, cache, C);
            // 先调度 B 到队列，然后 cache = C
        }
    }
}
```

**状态变化**：
- `B._join_counter`: 1 → 0（就绪）
- `C._join_counter`: 1 → 0（就绪）
- `Topology->_join_counter`: 1 → 2 → 3
- `cache`: nullptr → B → C（B 被调度到队列）

#### 1.2 _tear_down_nonasync(worker, A, cache) - 清理任务 A

```cpp
_tear_down_nonasync(worker, A, cache) {
    // 检查 A 的父节点
    if (auto parent = A._parent; parent == nullptr) {
        // ====================================================================
        // A 是顶层任务（没有父节点）
        // ====================================================================
        
        // 原子地减少 Topology 的 join_counter
        if (A._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // 如果返回 1，说明减少后变为 0，Topology 完成
            // 但这里返回 3，减少后变为 2，Topology 未完成
            _tear_down_topology(worker, A._topology);  // 不会执行
        }
    }
}
```

**状态变化**：
- `Topology->_join_counter`: 3 → 2（未完成，不调用 `_tear_down_topology`）

**当前状态**：
- 任务 A 执行完成
- 任务 B 在队列中等待执行
- 任务 C 在 cache 中，将通过 `TF_INVOKE_CONTINUATION()` 直接执行
- 任务 D 的 `_join_counter = 2`，等待 B 和 C 完成

---

### 阶段 2：执行任务 C（从 cache）

#### 2.1 TF_INVOKE_CONTINUATION() - 尾调用优化

```cpp
// 在 _invoke() 函数末尾
TF_INVOKE_CONTINUATION() {
    if (cache) {        // cache = C
        node = cache;   // node = C
        goto begin_invoke;  // 跳转到 _invoke 的开始，执行 C
    }
}
```

#### 2.2 _invoke(worker, C) - 执行任务

```cpp
_invoke(worker, C) {
    // 执行 C 的工作函数
    std::cout << "TaskC\n";
    
    // 调度后继任务 D
    for (size_t i = 0; i < C._num_successors; ++i) {  // C._num_successors = 1
        auto s = C._edges[i];  // s = D
        
        // 处理后继任务 D
        if (D._join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // D._join_counter: 2 -> 1（未就绪，因为 B 还没完成）
            // 条件不满足，不调度 D
        }
    }
}
```

**状态变化**：
- `D._join_counter`: 2 → 1（未就绪）
- `Topology->_join_counter`: 2（不变，因为没有调度新任务）

#### 2.3 _tear_down_nonasync(worker, C, cache) - 清理任务 C

```cpp
_tear_down_nonasync(worker, C, cache) {
    if (auto parent = C._parent; parent == nullptr) {
        // C 是顶层任务
        if (C._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // 返回 2，减少后变为 1，Topology 未完成
            _tear_down_topology(worker, C._topology);  // 不会执行
        }
    }
}
```

**状态变化**：
- `Topology->_join_counter`: 2 → 1（未完成）

**当前状态**：
- 任务 A 和 C 执行完成
- 任务 B 在队列中等待执行
- 任务 D 的 `_join_counter = 1`，等待 B 完成

---

### 阶段 3：执行任务 B（从队列）

#### 3.1 _invoke(worker, B) - 执行任务

```cpp
_invoke(worker, B) {
    // 执行 B 的工作函数
    std::cout << "TaskB\n";
    
    // 调度后继任务 D
    for (size_t i = 0; i < B._num_successors; ++i) {  // B._num_successors = 1
        auto s = B._edges[i];  // s = D
        
        // 处理后继任务 D
        if (D._join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // D._join_counter: 1 -> 0（就绪！）
            Topology->_join_counter.fetch_add(1, std::memory_order_relaxed);
            // Topology->_join_counter: 1 -> 2
            _update_cache(worker, cache, D);
            // cache = D
        }
    }
}
```

**状态变化**：
- `D._join_counter`: 1 → 0（就绪！）
- `Topology->_join_counter`: 1 → 2
- `cache`: nullptr → D

#### 3.2 _tear_down_nonasync(worker, B, cache) - 清理任务 B

```cpp
_tear_down_nonasync(worker, B, cache) {
    if (auto parent = B._parent; parent == nullptr) {
        // B 是顶层任务
        if (B._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // 返回 2，减少后变为 1，Topology 未完成
            _tear_down_topology(worker, B._topology);  // 不会执行
        }
    }
}
```

**状态变化**：
- `Topology->_join_counter`: 2 → 1（未完成）

**当前状态**：
- 任务 A、B、C 执行完成
- 任务 D 在 cache 中，将通过 `TF_INVOKE_CONTINUATION()` 直接执行

---

### 阶段 4：执行任务 D（从 cache）

#### 4.1 TF_INVOKE_CONTINUATION() - 尾调用优化

```cpp
TF_INVOKE_CONTINUATION() {
    if (cache) {        // cache = D
        node = cache;   // node = D
        goto begin_invoke;  // 跳转到 _invoke 的开始，执行 D
    }
}
```

#### 4.2 _invoke(worker, D) - 执行任务

```cpp
_invoke(worker, D) {
    // 执行 D 的工作函数
    std::cout << "TaskD\n";
    
    // D 没有后继任务
    // D._num_successors = 0
}
```

**状态变化**：
- 无（D 没有后继任务）

#### 4.3 _tear_down_nonasync(worker, D, cache) - 清理任务 D

```cpp
_tear_down_nonasync(worker, D, cache) {
    if (auto parent = D._parent; parent == nullptr) {
        // D 是顶层任务
        if (D._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // 返回 1，减少后变为 0，Topology 完成！
            _tear_down_topology(worker, D._topology);  // 会执行！
        }
    }
}
```

**状态变化**：
- `Topology->_join_counter`: 1 → 0（完成！）

#### 4.4 _tear_down_topology(worker, Topology) - 清理 Topology

```cpp
_tear_down_topology(worker, Topology) {
    auto& f = Topology->_taskflow;
    
    // 检查是否需要再次运行
    if (!Topology->_exception_ptr && !Topology->cancelled() && !Topology->_pred()) {
        // 这个例子中，_pred 为空（默认返回 true），所以条件不满足
    }
    else {
        // 最后一次运行
        
        // 调用完成回调（如果有）
        if (Topology->_call != nullptr) {
            Topology->_call();  // 这个例子中没有回调
        }
        
        // 检查是否有下一个 Topology
        if (std::unique_lock<std::mutex> lock(f._mutex); f._topologies.size() > 1) {
            // 这个例子中只有一个 Topology
        }
        else {
            // 最后一个 Topology
            auto fetched_tpg{std::move(f._topologies.front())};
            f._topologies.pop();
            auto satellite{f._satellite};
            
            lock.unlock();
            
            // 设置 Promise，通知等待的线程
            fetched_tpg->_carry_out_promise();
            // 这会唤醒 main 线程中的 future.wait()
            
            _decrement_topology();
            
            // 清理 satellite Taskflow（如果有）
            if (satellite) {
                std::scoped_lock<std::mutex> satellite_lock(_taskflows_mutex);
                _taskflows.erase(*satellite);
            }
        }
    }
}
```

**最终状态**：
- 所有任务执行完成
- Topology 被清理
- Promise 被设置，main 线程被唤醒
- `executor.run(taskflow).wait()` 返回

---

## 完整的状态变化表

| 阶段 | 执行任务 | A.jc | B.jc | C.jc | D.jc | Topo.jc | cache | 说明 |
|------|---------|------|------|------|------|---------|-------|------|
| 初始 | - | 0 | 1 | 1 | 2 | 1 | null | 初始状态 |
| 1.1 | A 执行 | 0 | 0 | 0 | 2 | 3 | C | B、C 就绪，B 入队列，C 缓存 |
| 1.2 | A 清理 | 0 | 0 | 0 | 2 | 2 | C | Topo.jc 减 1 |
| 2.1 | C 执行 | 0 | 0 | 0 | 1 | 2 | null | D.jc 减 1，但未就绪 |
| 2.2 | C 清理 | 0 | 0 | 0 | 1 | 1 | null | Topo.jc 减 1 |
| 3.1 | B 执行 | 0 | 0 | 0 | 0 | 2 | D | D 就绪，缓存 D |
| 3.2 | B 清理 | 0 | 0 | 0 | 0 | 1 | D | Topo.jc 减 1 |
| 4.1 | D 执行 | 0 | 0 | 0 | 0 | 1 | null | D 无后继 |
| 4.2 | D 清理 | 0 | 0 | 0 | 0 | 0 | null | Topo.jc 减 1，变为 0！ |
| 4.3 | Topo 清理 | - | - | - | - | - | - | 设置 Promise，完成 |

**注**：jc = _join_counter

---

## 关键观察

### 1. Topology->_join_counter 的变化规律

```
初始值：1（源节点数量）

执行 A：
  调度 B：+1  (1 -> 2)
  调度 C：+1  (2 -> 3)
  完成 A：-1  (3 -> 2)

执行 C：
  D 未就绪：+0  (2 -> 2)
  完成 C：-1    (2 -> 1)

执行 B：
  调度 D：+1  (1 -> 2)
  完成 B：-1  (2 -> 1)

执行 D：
  无后继：+0  (1 -> 1)
  完成 D：-1  (1 -> 0)  ← 触发 _tear_down_topology
```

**规律**：
- 每调度一个后继任务：`+1`
- 每完成一个任务：`-1`
- 最终变为 0：所有任务完成

### 2. 尾调用优化的效果

**任务执行顺序**：
1. A（从初始调度）
2. C（从 cache，尾调用）
3. B（从队列）
4. D（从 cache，尾调用）

**优化效果**：
- 任务 C 和 D 通过 cache 直接执行，避免了队列操作
- 形成任务链：A → C（尾调用）
- 形成任务链：B → D（尾调用）
- 减少了 2 次 push + 2 次 pop 操作

### 3. _tear_down_nonasync 的调用次数

每个任务执行完成后都会调用一次 `_tear_down_nonasync`：
- 任务 A 完成：调用 `_tear_down_nonasync(worker, A, cache)`
- 任务 C 完成：调用 `_tear_down_nonasync(worker, C, cache)`
- 任务 B 完成：调用 `_tear_down_nonasync(worker, B, cache)`
- 任务 D 完成：调用 `_tear_down_nonasync(worker, D, cache)`

**总共 4 次调用**，但只有最后一次（任务 D）触发了 `_tear_down_topology`。

### 4. 并发执行的可能性

虽然这个例子是单线程执行，但如果有多个工作线程：
- 任务 B 和 C 可以并发执行（都在 A 完成后就绪）
- 任务 D 必须等待 B 和 C 都完成

**并发场景下的 join_counter**：
```
线程 1 执行 B：
  D._join_counter.fetch_sub(1): 2 -> 1（未就绪）

线程 2 执行 C：
  D._join_counter.fetch_sub(1): 1 -> 0（就绪！）
  调度 D
```

原子操作保证了即使并发执行，D 也只会被调度一次。

---

## 总结

### _tear_down_nonasync 的核心作用

1. **更新 join_counter**：
   - 顶层任务：更新 `Topology->_join_counter`
   - 子任务：更新 `parent->_join_counter`

2. **检测完成**：
   - 当 `join_counter` 变为 0 时，触发完成处理
   - 顶层任务：调用 `_tear_down_topology`
   - 子任务：重新调度父节点（如果被抢占）

3. **避免数据竞争**：
   - 先读取需要的数据（如 `parent->_nstate`）
   - 再减少 `join_counter`
   - 避免悬空指针访问

### 设计精妙之处

1. **原子操作**：使用 `fetch_sub` 和 `fetch_add` 实现无锁同步
2. **尾调用优化**：通过 cache 和 goto 实现任务链式执行
3. **精确的完成检测**：通过 `join_counter` 精确跟踪任务完成状态
4. **灵活的清理机制**：支持多种完成场景（run_until、回调、Promise）

这个设计使得 Taskflow 能够高效、安全地执行复杂的任务图，同时保持代码的简洁和可维护性。

---

## 代码级别的详细追踪

### 任务 A 的 _tear_down_nonasync 调用

```cpp
// 在 _invoke(worker, A) 的末尾
_tear_down_nonasync(worker, A, cache) {
    // 步骤 1：读取 parent 指针
    auto parent = A._parent;  // parent = nullptr

    // 步骤 2：判断是否有父节点
    if (parent == nullptr) {
        // A 是顶层任务，更新 Topology 的 join_counter

        // 步骤 3：原子地减少 Topology 的 join_counter
        auto old_value = A._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel);
        // old_value = 3（减少前的值）
        // 减少后：A._topology->_join_counter = 2

        // 步骤 4：检查是否完成
        if (old_value == 1) {
            // 如果减少前是 1，说明减少后是 0，Topology 完成
            // 但这里 old_value = 3，所以不执行
            _tear_down_topology(worker, A._topology);
        }
    }
}
```

### 任务 C 的 _tear_down_nonasync 调用

```cpp
// 在 _invoke(worker, C) 的末尾
_tear_down_nonasync(worker, C, cache) {
    auto parent = C._parent;  // parent = nullptr

    if (parent == nullptr) {
        auto old_value = C._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel);
        // old_value = 2
        // 减少后：C._topology->_join_counter = 1

        if (old_value == 1) {
            // old_value = 2，不执行
            _tear_down_topology(worker, C._topology);
        }
    }
}
```

### 任务 B 的 _tear_down_nonasync 调用

```cpp
// 在 _invoke(worker, B) 的末尾
_tear_down_nonasync(worker, B, cache) {
    auto parent = B._parent;  // parent = nullptr

    if (parent == nullptr) {
        auto old_value = B._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel);
        // old_value = 2
        // 减少后：B._topology->_join_counter = 1

        if (old_value == 1) {
            // old_value = 2，不执行
            _tear_down_topology(worker, B._topology);
        }
    }
}
```

### 任务 D 的 _tear_down_nonasync 调用（触发完成）

```cpp
// 在 _invoke(worker, D) 的末尾
_tear_down_nonasync(worker, D, cache) {
    auto parent = D._parent;  // parent = nullptr

    if (parent == nullptr) {
        auto old_value = D._topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel);
        // old_value = 1（关键！）
        // 减少后：D._topology->_join_counter = 0

        if (old_value == 1) {
            // 条件满足！触发 Topology 清理
            _tear_down_topology(worker, D._topology);

            // 在 _tear_down_topology 中：
            {
                auto& f = D._topology->_taskflow;

                // 检查是否需要再次运行
                if (!D._topology->_exception_ptr &&
                    !D._topology->cancelled() &&
                    !D._topology->_pred()) {
                    // _pred() 为空，返回 true，条件不满足
                }
                else {
                    // 最后一次运行

                    // 调用完成回调（如果有）
                    if (D._topology->_call != nullptr) {
                        D._topology->_call();
                    }

                    // 检查是否有下一个 Topology
                    std::unique_lock<std::mutex> lock(f._mutex);
                    if (f._topologies.size() > 1) {
                        // 有下一个 Topology，处理队列
                    }
                    else {
                        // 最后一个 Topology
                        auto fetched_tpg{std::move(f._topologies.front())};
                        f._topologies.pop();
                        auto satellite{f._satellite};

                        lock.unlock();

                        // 设置 Promise，唤醒等待的线程
                        fetched_tpg->_carry_out_promise();
                        // 这会调用 _promise.set_value()
                        // main 线程中的 future.wait() 被唤醒

                        _decrement_topology();

                        if (satellite) {
                            std::scoped_lock<std::mutex> satellite_lock(_taskflows_mutex);
                            _taskflows.erase(*satellite);
                        }
                    }
                }
            }
        }
    }
}
```

---

## 关键时刻：Topology->_join_counter 变为 0

**为什么只有任务 D 触发了 _tear_down_topology？**

因为 `Topology->_join_counter` 的变化遵循以下规律：

```
初始值：1（源节点 A）

任务 A 执行：
  调度 B：+1  (1 → 2)
  调度 C：+1  (2 → 3)
  完成 A：-1  (3 → 2)  ← fetch_sub 返回 3，不触发

任务 C 执行：
  D 未就绪：+0
  完成 C：-1  (2 → 1)  ← fetch_sub 返回 2，不触发

任务 B 执行：
  调度 D：+1  (1 → 2)
  完成 B：-1  (2 → 1)  ← fetch_sub 返回 2，不触发

任务 D 执行：
  无后继：+0
  完成 D：-1  (1 → 0)  ← fetch_sub 返回 1，触发！
```

**关键点**：
- `fetch_sub(1)` 返回的是**减少前**的值
- 只有当返回值为 1 时，说明减少后变为 0
- 任务 D 是最后一个完成的任务，所以它触发了 Topology 的清理

---

## 内存顺序（Memory Order）的作用

### fetch_sub 使用 memory_order_acq_rel

```cpp
if (node->_topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    _tear_down_topology(worker, node->_topology);
}
```

**为什么使用 acq_rel？**

1. **Release 语义**（释放）：
   - 当前线程在 `fetch_sub` 之前的所有写操作对其他线程可见
   - 确保任务的执行结果（如修改的共享变量）对后续操作可见

2. **Acquire 语义**（获取）：
   - 当前线程在 `fetch_sub` 之后的所有读操作能看到其他线程的写操作
   - 确保在 `_tear_down_topology` 中能看到所有任务的执行结果

**示例场景**：
```
线程 1 执行任务 B：
  修改共享变量 x = 10
  fetch_sub(1, acq_rel)  ← Release：x = 10 对其他线程可见

线程 2 执行任务 D：
  fetch_sub(1, acq_rel)  ← Acquire：能看到 x = 10
  _tear_down_topology()
    读取 x  ← 保证能看到 x = 10
```

### fetch_add 使用 memory_order_relaxed

```cpp
rjc.fetch_add(1, std::memory_order_relaxed);
```

**为什么可以使用 relaxed？**

- `fetch_add` 只是增加计数器，不需要同步其他内存操作
- 最终的同步由 `fetch_sub` 的 `acq_rel` 保证
- 使用 `relaxed` 可以提高性能

---

## 总结：_tear_down_nonasync 的执行特点

### 1. 每个任务都会调用

无论任务是否触发 Topology 完成，每个任务执行完成后都会调用 `_tear_down_nonasync`。

### 2. 只有最后一个任务触发清理

通过 `fetch_sub` 的返回值判断，只有最后一个任务（使 `join_counter` 变为 0 的任务）会触发 `_tear_down_topology`。

### 3. 原子操作保证线程安全

即使多个任务并发完成，原子操作保证只有一个任务会触发清理，不会重复执行。

### 4. 精确的完成检测

通过 `join_counter` 的精确跟踪，能够准确判断 Topology 何时完成，不会过早或过晚触发清理。

这个设计体现了 Taskflow 在并发编程中的精妙之处：简单、高效、安全。


