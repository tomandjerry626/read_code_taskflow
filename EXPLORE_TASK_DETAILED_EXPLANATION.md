# _explore_task() 函数详解

## 函数签名
```cpp
inline bool Executor::_explore_task(Worker& w, Node*& t)
```

## 函数作用
`_explore_task()` 是 Taskflow 工作窃取调度器的核心函数之一，负责实现**探索阶段（Explore Phase）**。当工作线程的本地队列为空时，它会尝试从其他线程的队列或中心化缓冲区中"窃取"任务。

## 返回值含义
- **`true`**: 窃取过程正常结束（可能窃取到任务，也可能没有）
- **`false`**: 工作线程收到停止信号，应该立即退出

## 完整执行流程示例

### 场景设定
假设我们有一个 4 线程的执行器：
- **工作线程**: Worker 0, Worker 1, Worker 2, Worker 3
- **中心化缓冲区**: 2 个桶（Bucket 0, Bucket 1）
- **总队列数**: 4 + 2 = 6

当前状态：
- Worker 0 的本地队列为空，需要窃取任务
- Worker 1 的队列有 5 个任务
- Worker 2 的队列为空
- Worker 3 的队列有 2 个任务
- Bucket 0 有 3 个任务
- Bucket 1 为空

### 执行流程详解

#### 第 1 步：初始化参数

```cpp
const size_t MAX_STEALS = ((num_queues() + 1) << 1);
// num_queues() = 6
// MAX_STEALS = (6 + 1) * 2 = 14
```

```cpp
std::uniform_int_distribution<size_t> udist(0, num_queues()-1);
// 创建随机数生成器，范围 [0, 5]
```

```cpp
size_t num_steals = 0;  // 窃取尝试计数器
size_t vtm = w._vtm;     // 假设 Worker 0 的 _vtm = 2
```

**当前状态**:
- `MAX_STEALS = 14`
- `num_steals = 0`
- `vtm = 2` (上次窃取的受害者索引)

---

#### 第 2 步：第一次窃取尝试（vtm = 2）

```cpp
while(true) {
    // vtm = 2 < 4 (workers.size())，所以从 Worker 2 的队列窃取
    t = _workers[2]._wsq.steal();
```

**队列索引映射**:
```
索引 0 -> Worker 0 的队列
索引 1 -> Worker 1 的队列
索引 2 -> Worker 2 的队列  ← 当前尝试窃取
索引 3 -> Worker 3 的队列
索引 4 -> Bucket 0
索引 5 -> Bucket 1
```

**结果**: Worker 2 的队列为空，`t = nullptr`

```cpp
    if(t) {  // false，窃取失败
        // 不执行
    }
```

---

#### 第 3 步：检查窃取次数

```cpp
    if (++num_steals > MAX_STEALS) {  // num_steals = 1, 1 > 14? false
        // 不执行
    }
```

---

#### 第 4 步：检查停止信号

```cpp
    if(w._done.load(std::memory_order_relaxed)) {  // false
        // 不执行
    }
```

---

#### 第 5 步：随机选择下一个受害者

```cpp
    vtm = udist(w._rdgen);  // 假设随机生成 vtm = 1
}
```

**当前状态**:
- `num_steals = 1`
- `vtm = 1` (新的受害者索引)

---

#### 第 6 步：第二次窃取尝试（vtm = 1）

```cpp
while(true) {  // 循环继续
    // vtm = 1 < 4，从 Worker 1 的队列窃取
    t = _workers[1]._wsq.steal();
```

**Worker 1 的队列状态**:
```
底部 [Task A] [Task B] [Task C] [Task D] [Task E] 顶部
                                                    ↑
                                                  steal
```

**结果**: 成功窃取到 `Task E`，`t = &Task_E`

```cpp
    if(t) {  // true，窃取成功！
        w._vtm = vtm;  // 更新 Worker 0 的 _vtm = 1
        break;         // 跳出循环
    }
}
```

---

#### 第 7 步：返回结果

```cpp
return true;  // 窃取过程正常结束
```

**最终状态**:
- `t` 指向 `Task E`
- Worker 0 的 `_vtm` 更新为 1（下次优先从 Worker 1 继续窃取）
- 函数返回 `true`

---

## 关键设计细节

### 1. 队列索引布局
```
[0, N-1]           -> 工作线程的本地队列（N = _workers.size()）
[N, num_queues()-1] -> 中心化缓冲区的桶
```

### 2. 窃取策略
- **优先延续性**: 优先从上次成功窃取的队列继续窃取（利用局部性）
- **随机性**: 窃取失败后随机选择下一个受害者（避免冲突）
- **公平性**: 所有队列（包括缓冲区）都有机会被窃取

### 3. 退出条件
- **成功窃取**: 找到任务后立即返回
- **尝试次数上限**: 超过 `150 + MAX_STEALS` 次失败后放弃
- **停止信号**: 收到 `_done` 信号时返回 `false`

### 4. CPU 优化
```cpp
if (++num_steals > MAX_STEALS) {
    std::this_thread::yield();  // 主动让出 CPU
```
避免在所有队列都为空时过度消耗 CPU 资源。

---

## 场景 2：从中心化缓冲区窃取

### 初始状态
- Worker 0 需要窃取任务
- Worker 0 的 `_vtm = 4`（指向 Bucket 0）
- 所有工作线程的队列都为空
- Bucket 0 有 3 个任务

### 执行流程

```cpp
vtm = w._vtm;  // vtm = 4
```

```cpp
// vtm = 4 >= 4 (_workers.size())，从缓冲区窃取
t = _buffers.steal(vtm - _workers.size());
// 等价于：_buffers.steal(4 - 4) = _buffers.steal(0)
// 从 Bucket 0 窃取
```

**结果**: 成功从 Bucket 0 窃取到任务

---

## 场景 3：所有队列都为空（准备休眠）

### 初始状态
- Worker 0 需要窃取任务
- 所有队列都为空
- `MAX_STEALS = 14`

### 执行流程

Worker 0 会尝试从所有队列窃取，每次都失败：

```
尝试 1:  vtm=2, 窃取失败, num_steals=1
尝试 2:  vtm=5, 窃取失败, num_steals=2
尝试 3:  vtm=0, 窃取失败, num_steals=3
...
尝试 14: vtm=3, 窃取失败, num_steals=14
尝试 15: vtm=1, 窃取失败, num_steals=15
```

当 `num_steals > 14` 时：
```cpp
if (++num_steals > MAX_STEALS) {
    std::this_thread::yield();  // 让出 CPU
    if(num_steals > 150 + MAX_STEALS) {  // 15 > 164? false
        // 继续尝试
    }
}
```

继续尝试直到 `num_steals > 164`：
```cpp
if(num_steals > 150 + MAX_STEALS) {  // 165 > 164? true
    break;  // 放弃窃取
}
```

**结果**:
- `t = nullptr`（没有窃取到任务）
- 返回 `true`
- 调用者会进入两阶段提交协议，准备休眠

---

## 场景 4：收到停止信号

### 执行流程

在窃取循环中，每次尝试后都会检查停止信号：

```cpp
while(true) {
    t = ...;  // 尝试窃取

    if(t) { break; }

    // 检查停止信号
    if(w._done.load(std::memory_order_relaxed)) {
        return false;  // 立即返回 false
    }

    vtm = udist(w._rdgen);
}
```

**结果**:
- 函数返回 `false`
- 工作线程退出调度循环
- 执行器正在关闭

---

## 与 Chase-Lev 算法的关系

`_explore_task()` 调用的 `_wsq.steal()` 实现了 **Chase-Lev 工作窃取队列**算法：

### Chase-Lev 队列特性
```
底部 [Task 1] [Task 2] [Task 3] [Task 4] 顶部
      ↑                                    ↑
    所有者 pop()                        其他线程 steal()
```

- **所有者线程**（Worker 自己）：从底部 `pop()`，LIFO 顺序
- **窃取线程**（其他 Worker）：从顶部 `steal()`，FIFO 顺序

### 无锁并发保证
- 使用原子操作和内存屏障
- 所有者和窃取者可以并发访问
- 多个窃取者之间通过 CAS 操作竞争

---

## 性能优化技巧

### 1. 受害者索引缓存
```cpp
size_t vtm = w._vtm;  // 使用上次成功的索引
```
**优势**: 如果某个队列有很多任务，连续窃取可以提高缓存命中率。

### 2. 随机化避免冲突
```cpp
vtm = udist(w._rdgen);  // 随机选择下一个受害者
```
**优势**: 多个空闲线程不会都盯着同一个队列，减少竞争。

### 3. 渐进式退避
```cpp
if (++num_steals > MAX_STEALS) {
    std::this_thread::yield();  // 第一阶段：让出 CPU
    if(num_steals > 150 + MAX_STEALS) {
        break;  // 第二阶段：彻底放弃
    }
}
```
**优势**: 避免忙等待，同时给新任务到达留出时间窗口。

---

## 调用上下文

`_explore_task()` 在 `_wait_for_task()` 中被调用：

```cpp
bool Executor::_wait_for_task(Worker& w, Node*& t) {
  explore_task:

  // 尝试窃取任务
  if(_explore_task(w, t) == false) {
    return false;  // 收到停止信号
  }

  // 窃取成功，返回执行
  if(t) {
    return true;
  }

  // 窃取失败，进入两阶段提交协议（2PC）
  _notifier.prepare_wait(w._waiter);
  // ... 检查队列是否真的为空 ...
  // ... 如果确实为空，进入休眠 ...
}
```

---

## 总结

`_explore_task()` 函数实现了高效的工作窃取机制：

1. **智能窃取策略**: 优先延续 + 随机选择
2. **公平性**: 所有队列（工作线程 + 缓冲区）都会被尝试
3. **性能优化**: 渐进式退避，避免 CPU 浪费
4. **正确性保证**: 基于 Chase-Lev 无锁队列算法
5. **响应性**: 及时响应停止信号

这个函数是 Taskflow 实现高性能并行调度的关键组件之一！


