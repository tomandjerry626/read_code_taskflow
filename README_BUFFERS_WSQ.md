# _buffers 和 _wsq 关系完全指南

## 🎯 快速回答

### 问题：任务往 _buffers 里放还是往 _wsq 里放？

**答案取决于两个因素**：

1. **谁在调度任务？**
   - 工作线程 → 优先放入 _wsq
   - 外部线程 → 直接放入 _buffers

2. **_wsq 是否已满？**
   - 未满（< 1024）→ 放入 _wsq
   - 已满（≥ 1024）→ 溢出到 _buffers

---

## 📚 核心概念

### Worker::_wsq (本地工作窃取队列)

```cpp
class Worker {
    BoundedTaskQueue<Node*> _wsq;  // 有界队列，容量 1024
};
```

**特点**：
- ✅ **有界**：固定容量 1024
- ✅ **无锁**：所有者线程 push/pop 无需加锁
- ✅ **高性能**：缓存友好，低竞争
- ✅ **私有**：每个工作线程独占

**操作**：
- 所有者线程：从底部 `push()` 和 `pop()`（LIFO）
- 其他线程：从顶部 `steal()`（FIFO）

---

### Executor::_buffers (中心化缓冲区)

```cpp
class Executor {
    Freelist<Node*> _buffers;  // 多桶无界队列
};
```

**特点**：
- ✅ **无界**：动态扩展，无容量限制
- ⚠️ **有锁**：每个桶有一个 mutex
- ✅ **共享**：所有线程可访问
- ✅ **多桶**：`floor(log2(N))` 个桶，减少竞争

**结构**：
```
_buffers
├── Bucket 0: [mutex + UnboundedTaskQueue]
├── Bucket 1: [mutex + UnboundedTaskQueue]
└── ...
```

---

## 🔀 任务分配决策树

```
产生新任务
    ↓
是否从工作线程调用？
    ↓
    ├─ 否 → 直接放入 _buffers
    │
    └─ 是 → 是否是本执行器的工作线程？
            ↓
            ├─ 否 → 放入 _buffers
            │
            └─ 是 → _wsq 是否已满？
                    ↓
                    ├─ 否 → 放入 _wsq ✓ (Fast Path)
                    │
                    └─ 是 → 溢出到 _buffers (Slow Path)
```

---

## 💻 代码实现

### 核心调度函数

```cpp
// 从工作线程调度
void Executor::_schedule(Worker& worker, Node* node) {
    if(worker._executor == this) {
        // 工作线程 → 尝试放入 _wsq
        worker._wsq.push(node, [&](){ 
            _buffers.push(node);  // 溢出回调
        });
    } else {
        // 外部线程 → 直接放入 _buffers
        _buffers.push(node);
    }
    _notifier.notify_one();
}

// 从外部线程调度
void Executor::_schedule(Node* node) {
    _buffers.push(node);  // 直接放入 _buffers
    _notifier.notify_one();
}
```

### _wsq.push() 的溢出机制

```cpp
template <typename O, typename C>
void BoundedTaskQueue::push(O&& o, C&& on_full) {
    int64_t b = _bottom.load(std::memory_order_relaxed);
    int64_t t = _top.load(std::memory_order_acquire);
    
    // 检查队列是否已满
    if TF_UNLIKELY((b - t) > BufferSize - 1) {
        on_full();  // 调用溢出回调 → _buffers.push()
        return;
    }
    
    // 队列未满，正常插入
    _buffer[b & BufferMask].store(std::forward<O>(o), ...);
    _bottom.store(b + 1, std::memory_order_release);
}
```

---

## 📊 实际场景分析

### 场景 1：外部线程提交任务

```cpp
// 主线程
tf::Executor executor(4);
tf::Taskflow taskflow;
taskflow.emplace([]{ /* Task A */ });
executor.run(taskflow);  // ← 主线程调用
```

**结果**：Task A → `_buffers`  
**原因**：主线程不是工作线程

---

### 场景 2：工作线程产生子任务

```cpp
taskflow.emplace([](tf::Subflow& sf){
    sf.emplace([]{ /* 子任务 1 */ });  // ← 工作线程调用
    sf.emplace([]{ /* 子任务 2 */ });
});
```

**结果**：子任务 1, 2 → 当前工作线程的 `_wsq`  
**原因**：调用者是工作线程，且 _wsq 未满

---

### 场景 3：_wsq 溢出

```cpp
taskflow.emplace([](tf::Subflow& sf){
    for(int i = 0; i < 2000; i++) {
        sf.emplace([i]{ /* 子任务 */ });
    }
});
```

**结果**：
- 前 1024 个子任务 → `_wsq`
- 后 976 个子任务 → `_buffers`（溢出）

---

### 场景 4：异步任务

```cpp
executor.async([]{ /* 异步任务 */ });
```

**结果**：异步任务 → `_buffers`  
**原因**：`async()` 内部调用 `_schedule(node)`，无 Worker 参数

---

## 🎨 架构图

```
┌─────────────────────────────────────────────────────┐
│                    Executor                          │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  ...     │
│  │          │  │          │  │          │          │
│  │  _wsq    │  │  _wsq    │  │  _wsq    │          │
│  │  [1024]  │  │  [1024]  │  │  [1024]  │          │
│  │  无锁    │  │  无锁    │  │  无锁    │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │ 溢出        │ 溢出        │ 溢出            │
│       ↓             ↓             ↓                 │
│  ┌──────────────────────────────────────────┐      │
│  │         _buffers (Freelist)              │      │
│  │  ┌──────────┐  ┌──────────┐             │      │
│  │  │ Bucket 0 │  │ Bucket 1 │  ...        │      │
│  │  │ 无界     │  │ 无界     │             │      │
│  │  │ + mutex  │  │ + mutex  │             │      │
│  │  └──────────┘  └──────────┘             │      │
│  └──────────────────────────────────────────┘      │
│                    ↑                                │
│                    │ 外部线程直接放入                │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 性能优势

### 为什么优先使用 _wsq？

| 方面 | _wsq | _buffers |
|------|------|----------|
| 并发控制 | 无锁 | 有锁 |
| 缓存局部性 | 高 | 低 |
| 竞争程度 | 低 | 高 |
| 适用场景 | 工作线程产生的任务 | 外部提交/溢出 |

**性能提升**：
- 无锁操作比加锁快 10-100 倍
- 缓存命中率提高 → 减少内存访问延迟
- 减少竞争 → 提高并发度

---

## 📖 相关文档

- **详细解析**: `BUFFERS_VS_WSQ_EXPLANATION.md`
- **示例代码**: `buffers_wsq_demo.cpp`
- **源代码**: 
  - `taskflow/core/executor.hpp` (_schedule 方法)
  - `taskflow/core/tsq.hpp` (BoundedTaskQueue)
  - `taskflow/core/freelist.hpp` (Freelist)

---

## 🎓 关键要点

1. **本地优先原则**：工作线程产生的任务优先放入自己的 _wsq
2. **溢出机制**：_wsq 满时自动溢出到 _buffers
3. **外部兜底**：外部线程提交的任务直接进入 _buffers
4. **两层设计**：Fast Path (_wsq) + Slow Path (_buffers)

这就是 Taskflow 高性能调度的核心秘密！🎉

