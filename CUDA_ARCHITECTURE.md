# Taskflow CUDA 模块架构说明

## 📚 目录结构

```
taskflow/cuda/
├── cudaflow.hpp              # 主入口文件，类型定义
├── cuda_graph.hpp            # CUDA 图构建（显式 API）
├── cuda_graph_exec.hpp       # CUDA 图执行
├── cuda_capturer.hpp         # CUDA 流捕获（自动构建图）
├── cuda_optimizer.hpp        # 流捕获优化器
├── cuda_stream.hpp           # CUDA 流封装
├── cuda_device.hpp           # CUDA 设备管理
├── cuda_memory.hpp           # CUDA 内存管理
├── cuda_error.hpp            # CUDA 错误处理
├── cuda_meta.hpp             # CUDA 元编程工具
├── cuda_execution_policy.hpp # CUDA 执行策略
└── algorithm/                # CUDA 算法库
    ├── single_task.hpp       # 单任务执行
    ├── for_each.hpp          # 并行 for_each
    ├── transform.hpp         # 并行 transform
    ├── reduce.hpp            # 并行 reduce
    ├── scan.hpp              # 并行 scan
    ├── sort.hpp              # 并行 sort
    ├── merge.hpp             # 并行 merge
    ├── find.hpp              # 并行 find
    ├── matmul.hpp            # 矩阵乘法
    └── transpose.hpp         # 矩阵转置
```

## 🎯 核心概念

### 1. 两种图构建方式

#### 方式 1：显式图构建（cudaGraph）

```cpp
// 用户显式调用 API 构建图
tf::cudaGraph cg;
auto h2d = cg.copy(d_ptr, h_ptr, N);      // 添加内存拷贝节点
auto kernel = cg.kernel(...);              // 添加内核节点
auto d2h = cg.copy(h_ptr, d_ptr, N);      // 添加内存拷贝节点

h2d.precede(kernel);                       // 设置依赖关系
kernel.precede(d2h);

tf::cudaGraphExec exec(cg);                // 实例化图
tf::cudaStream stream;
stream.run(exec).synchronize();            // 执行图
```

**使用的 CUDA API**：
- `cudaGraphCreate()` - 创建空图
- `cudaGraphAddKernelNode()` - 添加内核节点
- `cudaGraphAddMemcpyNode()` - 添加内存拷贝节点
- `cudaGraphAddDependencies()` - 添加依赖边
- `cudaGraphInstantiate()` - 实例化图
- `cudaGraphLaunch()` - 执行图

#### 方式 2：流捕获（cudaFlowCapturer）

```cpp
// 系统自动捕获异步 CUDA 操作
taskflow.emplace([](tf::cudaFlowCapturer& capturer){
  auto h2d = capturer.on([&](cudaStream_t stream){
    cudaMemcpyAsync(d_ptr, h_ptr, N, cudaMemcpyHostToDevice, stream);
  });
  
  auto kernel = capturer.on([&](cudaStream_t stream){
    my_kernel<<<grid, block, 0, stream>>>(...);
  });
  
  auto d2h = capturer.on([&](cudaStream_t stream){
    cudaMemcpyAsync(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost, stream);
  });
  
  h2d.precede(kernel);
  kernel.precede(d2h);
});
```

**使用的 CUDA API**：
- `cudaStreamBeginCapture()` - 开始捕获
- `cudaStreamEndCapture()` - 结束捕获，生成图
- 任意异步 CUDA 操作（cudaMemcpyAsync, kernel<<<>>>等）

### 2. 核心类型

#### cudaGraph（CUDA 图）

```cpp
using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;
```

- 管理 `cudaGraph_t` 的智能指针
- 提供高层 API 构建 GPU 任务图
- 自动管理资源生命周期

**主要方法**：
- `noop()` - 创建空节点
- `host(callable)` - 创建 CPU 回调节点
- `kernel(grid, block, shm, func, args...)` - 创建内核节点
- `copy(dst, src, count)` - 创建内存拷贝节点
- `memcpy(dst, src, bytes)` - 创建非类型化拷贝节点
- `memset(dst, value, count)` - 创建内存设置节点
- `fill(dst, value, count)` - 创建类型化填充节点
- `zero(dst, count)` - 创建清零节点

#### cudaGraphExec（可执行图）

```cpp
using cudaGraphExec = cudaGraphExecBase<cudaGraphExecCreator, cudaGraphExecDeleter>;
```

- 管理 `cudaGraphExec_t` 的智能指针
- 从 `cudaGraph` 实例化而来
- 可以高效地多次执行

**使用方式**：
```cpp
tf::cudaGraphExec exec(cg);  // 从 cudaGraph 实例化
tf::cudaStream stream;
stream.run(exec);            // 执行图
stream.synchronize();        // 等待完成
```

#### cudaTask（CUDA 任务）

- 表示 CUDA 图中的一个节点
- 封装 `cudaGraphNode_t`
- 提供依赖关系设置接口

**主要方法**：
- `precede(tasks...)` - 设置后继任务
- `succeed(tasks...)` - 设置前驱任务
- `type()` - 查询节点类型
- `num_successors()` - 查询后继数量
- `num_predecessors()` - 查询前驱数量

**节点类型**：
- `cudaGraphNodeTypeKernel` - 内核执行
- `cudaGraphNodeTypeMemcpy` - 内存拷贝
- `cudaGraphNodeTypeMemset` - 内存设置
- `cudaGraphNodeTypeHost` - CPU 回调
- `cudaGraphNodeTypeEmpty` - 空操作
- `cudaGraphNodeTypeWaitEvent` - 等待事件
- `cudaGraphNodeTypeEventRecord` - 记录事件

#### cudaFlowCapturer（流捕获器）

- 通过流捕获自动构建 CUDA 图
- 支持任意异步 CUDA 操作
- 使用优化器转换为高效的 CUDA 图

**主要方法**：
- `on(callable)` - 捕获异步 CUDA 操作
- `noop()` - 创建空节点
- `make_optimizer<OPT>()` - 设置优化器
- `capture()` - 捕获并生成图
- `run(stream)` - 执行图

**优化器类型**：
- `cudaFlowSequentialOptimizer` - 顺序优化器（单流）
- `cudaFlowRoundRobinOptimizer` - 轮询优化器（多流，默认）
- `cudaFlowLinearOptimizer` - 线性优化器

## 🔄 图构建流程

### 显式图构建流程（cudaGraph）

```
1. 创建 cudaGraph 对象
   ↓ [调用 cudaGraphCreate()]

2. 添加各种节点
   ↓ [调用 cudaGraphAddKernelNode(), cudaGraphAddMemcpyNode() 等]

3. 设置依赖关系
   ↓ [调用 cudaGraphAddDependencies()]

4. 实例化为 cudaGraphExec
   ↓ [调用 cudaGraphInstantiate()]

5. 通过 cudaStream 执行
   ↓ [调用 cudaGraphLaunch()]

6. 等待完成
   ↓ [调用 cudaStreamSynchronize()]
```

### 流捕获构建流程（cudaFlowCapturer）

```
1. 创建 cudaFlowCapturer 对象
   ↓

2. 用户调用 capturer.on() 添加任务
   ↓ [记录到内部图 cudaFlowGraph]

3. 用户设置依赖关系
   ↓ [构建 DAG]

4. 调用 capture() 生成 CUDA 图
   ↓ [优化器分析图结构]
   ↓ [调用 cudaStreamBeginCapture()]
   ↓ [按优化顺序执行所有操作]
   ↓ [调用 cudaStreamEndCapture()]
   ↓ [生成 cudaGraph_t]

5. 实例化并执行
   ↓ [调用 cudaGraphInstantiate()]
   ↓ [调用 cudaGraphLaunch()]
```

## 🆚 与 CPU Taskflow 的区别

| 特性 | CPU Taskflow | CUDA Graph |
|------|-------------|------------|
| **执行位置** | CPU 线程池 | GPU |
| **图表示** | Node/Graph/Topology | cudaGraph_t/cudaGraphNode_t |
| **调度方式** | 动态调度（运行时） | 静态图（预先构建） |
| **调度器** | 工作窃取（Work-Stealing） | GPU 硬件调度 |
| **动态性** | 支持 Subflow/Runtime | 不支持动态修改 |
| **任务类型** | Static, Subflow, Runtime 等 | Kernel, Memcpy, Memset 等 |
| **依赖管理** | Join counter | CUDA 图依赖边 |
| **优化目标** | 负载均衡、缓存局部性 | 内核融合、启动开销 |
| **执行开销** | 每个任务有调度开销 | 整个图只有一次启动开销 |

## 🚀 性能优势

### CUDA Graph 的优势

1. **极低的启动开销**
   - 传统方式：每次内核启动 ~10μs
   - CUDA Graph：整个图只有一次启动
   - 适合大量小内核

2. **更好的并发性**
   - CUDA 运行时分析整个图
   - 自动并发执行独立内核
   - 无需手动管理流和事件

3. **内核融合机会**
   - CUDA 驱动可以优化图
   - 可能融合相邻小内核
   - 减少内存访问

4. **可重复执行**
   - 图实例化后可多次执行
   - 每次执行开销极低
   - 适合迭代算法

### Taskflow CUDA 的额外优势

1. **高层抽象**
   - 隐藏 CUDA Graph API 复杂性
   - 类似 CPU Taskflow 的接口
   - 易于学习和使用

2. **自动优化**
   - cudaFlowCapturer 自动转换图
   - 支持多种优化策略
   - 最大化并发性

3. **与 CPU 任务集成**
   - GPU 任务无缝嵌入 CPU Taskflow
   - 统一的任务图模型
   - 自动管理 CPU-GPU 同步

## 📋 CUDA API 使用总结

### 图构建 API（cudaGraph 使用）

| API | 功能 | 使用位置 |
|-----|------|---------|
| `cudaGraphCreate()` | 创建空图 | cudaGraphCreator |
| `cudaGraphAddKernelNode()` | 添加内核节点 | cudaGraph::kernel() |
| `cudaGraphAddMemcpyNode()` | 添加内存拷贝节点 | cudaGraph::copy() |
| `cudaGraphAddMemsetNode()` | 添加内存设置节点 | cudaGraph::memset() |
| `cudaGraphAddHostNode()` | 添加主机回调节点 | cudaGraph::host() |
| `cudaGraphAddEmptyNode()` | 添加空节点 | cudaGraph::noop() |
| `cudaGraphAddDependencies()` | 添加依赖边 | cudaTask::precede() |
| `cudaGraphDestroy()` | 销毁图 | cudaGraphDeleter |

### 图实例化 API

| API | 功能 | 使用位置 |
|-----|------|---------|
| `cudaGraphInstantiate()` | 实例化图 | cudaGraphExecCreator |
| `cudaGraphExecDestroy()` | 销毁实例化图 | cudaGraphExecDeleter |

### 图执行 API

| API | 功能 | 使用位置 |
|-----|------|---------|
| `cudaGraphLaunch()` | 在流上启动图 | cudaStream::run() |
| `cudaStreamSynchronize()` | 等待流完成 | cudaStream::synchronize() |

### 流捕获 API（cudaFlowCapturer 使用）

| API | 功能 | 使用位置 |
|-----|------|---------|
| `cudaStreamBeginCapture()` | 开始捕获 | cudaFlowOptimizer::_optimize() |
| `cudaStreamEndCapture()` | 结束捕获 | cudaFlowOptimizer::_optimize() |
| `cudaStreamIsCapturing()` | 检查是否在捕获中 | 调试用 |

### 图查询 API

| API | 功能 | 使用位置 |
|-----|------|---------|
| `cudaGraphGetNodes()` | 获取所有节点 | cuda_graph_get_nodes() |
| `cudaGraphGetRootNodes()` | 获取根节点 | cuda_graph_get_root_nodes() |
| `cudaGraphGetEdges()` | 获取所有边 | cuda_graph_get_edges() |
| `cudaGraphNodeGetType()` | 获取节点类型 | cuda_get_graph_node_type() |
| `cudaGraphNodeGetDependencies()` | 获取前驱节点 | cuda_graph_node_get_dependencies() |
| `cudaGraphNodeGetDependentNodes()` | 获取后继节点 | cuda_graph_node_get_dependent_nodes() |

### 图导出 API

| API | 功能 | 使用位置 |
|-----|------|---------|
| `cudaGraphDebugDotPrint()` | 导出为 DOT 格式 | cudaGraph::dump() |

## 💡 使用建议

### 何时使用 cudaGraph（显式构建）

✅ **适合场景**：
- 图结构简单明确
- 需要精确控制节点类型
- 需要多次修改图结构
- 需要查询图的详细信息

❌ **不适合场景**：
- 需要捕获复杂的 CUDA 库调用
- 图结构非常复杂
- 需要捕获第三方库的操作

### 何时使用 cudaFlowCapturer（流捕获）

✅ **适合场景**：
- 需要捕获任意异步 CUDA 操作
- 使用 cuBLAS、cuDNN 等库
- 图结构复杂，手动构建困难
- 需要自动优化并发性

❌ **不适合场景**：
- 需要精确控制每个节点
- 需要频繁修改图结构
- 捕获开销不可接受

### 优化器选择

1. **cudaFlowSequentialOptimizer（顺序优化器）**
   - 使用单个流捕获
   - 所有任务顺序执行
   - 适合依赖关系复杂的图
   - 并发性最低，但最可靠

2. **cudaFlowRoundRobinOptimizer（轮询优化器，默认）**
   - 使用多个流（默认 4 个）
   - 将任务轮询分配到不同流
   - 适合大多数场景
   - 并发性好，开销适中

3. **cudaFlowLinearOptimizer（线性优化器）**
   - 使用层级化（levelize）算法
   - 同一层的任务并发执行
   - 适合层次分明的图
   - 并发性最高，但开销较大

## 📝 完整示例

### 示例 1：矩阵乘法（显式构建）

```cpp
#include <taskflow/cuda/cudaflow.hpp>

int main() {
  const int M = 1024, N = 1024, K = 1024;

  // 分配内存
  float *ha, *hb, *hc;
  float *da, *db, *dc;

  ha = new float[M*K];
  hb = new float[K*N];
  hc = new float[M*N];

  cudaMalloc(&da, M*K*sizeof(float));
  cudaMalloc(&db, K*N*sizeof(float));
  cudaMalloc(&dc, M*N*sizeof(float));

  // 初始化数据
  // ...

  // 创建 CUDA 图
  tf::cudaGraph cg;

  // 添加任务
  auto h2d_a = cg.copy(da, ha, M*K);
  auto h2d_b = cg.copy(db, hb, K*N);
  auto matmul = cg.kernel(
    dim3((N+15)/16, (M+15)/16), dim3(16, 16), 0,
    matmul_kernel, da, db, dc, M, N, K
  );
  auto d2h_c = cg.copy(hc, dc, M*N);

  // 设置依赖关系
  matmul.succeed(h2d_a, h2d_b);
  matmul.precede(d2h_c);

  // 实例化并执行
  tf::cudaGraphExec exec(cg);
  tf::cudaStream stream;
  stream.run(exec).synchronize();

  // 清理
  delete[] ha; delete[] hb; delete[] hc;
  cudaFree(da); cudaFree(db); cudaFree(dc);

  return 0;
}
```

### 示例 2：使用流捕获

```cpp
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

int main() {
  tf::Executor executor;
  tf::Taskflow taskflow;

  // 创建 cudaFlowCapturer 任务
  taskflow.emplace([](tf::cudaFlowCapturer& capturer){

    // 捕获内存拷贝
    auto h2d = capturer.on([&](cudaStream_t stream){
      cudaMemcpyAsync(d_data, h_data, N*sizeof(float),
                      cudaMemcpyHostToDevice, stream);
    });

    // 捕获内核启动
    auto kernel = capturer.on([&](cudaStream_t stream){
      my_kernel<<<grid, block, 0, stream>>>(d_data, N);
    });

    // 捕获内存拷贝
    auto d2h = capturer.on([&](cudaStream_t stream){
      cudaMemcpyAsync(h_result, d_result, N*sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
    });

    // 设置依赖关系
    h2d.precede(kernel);
    kernel.precede(d2h);
  });

  executor.run(taskflow).wait();

  return 0;
}
```

### 示例 3：嵌入 CPU Taskflow

```cpp
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

int main() {
  tf::Executor executor;
  tf::Taskflow taskflow;

  // CPU 任务
  auto cpu_task_1 = taskflow.emplace([](){
    std::cout << "CPU Task 1\n";
  });

  // GPU 任务
  auto gpu_task = taskflow.emplace([&](){
    tf::cudaGraph cg;

    // 构建 CUDA 图
    auto kernel = cg.kernel(grid, block, 0, my_kernel, args...);

    // 执行
    tf::cudaGraphExec exec(cg);
    tf::cudaStream stream;
    stream.run(exec).synchronize();
  });

  // CPU 任务
  auto cpu_task_2 = taskflow.emplace([](){
    std::cout << "CPU Task 2\n";
  });

  // 设置依赖关系
  cpu_task_1.precede(gpu_task);
  gpu_task.precede(cpu_task_2);

  executor.run(taskflow).wait();

  return 0;
}
```

## 🔍 调试技巧

### 1. 导出图结构

```cpp
tf::cudaGraph cg;
// ... 构建图 ...

// 导出为 DOT 格式
cg.dump(std::cout);

// 或导出原生图
cg.dump_native_graph(std::cout);
```

### 2. 查询节点信息

```cpp
auto task = cg.kernel(...);

// 查询节点类型
auto type = task.type();
std::cout << "Type: " << tf::to_string(type) << "\n";

// 查询后继数量
std::cout << "Successors: " << task.num_successors() << "\n";

// 查询前驱数量
std::cout << "Predecessors: " << task.num_predecessors() << "\n";
```

### 3. 错误检查

所有 CUDA API 调用都通过 `TF_CHECK_CUDA` 宏进行错误检查：

```cpp
TF_CHECK_CUDA(
  cudaGraphCreate(&g, 0),
  "failed to create CUDA graph"
);
```

如果出错，会抛出异常并打印详细信息。

## 📚 相关文件说明

- **cudaflow.hpp**: 主入口，类型定义和架构说明
- **cuda_graph.hpp**: 核心图构建实现，包含所有节点创建函数
- **cuda_capturer.hpp**: 流捕获实现
- **cuda_optimizer.hpp**: 三种优化器实现
- **cuda_stream.hpp**: CUDA 流封装
- **cuda_device.hpp**: CUDA 设备管理
- **cuda_memory.hpp**: CUDA 内存分配和释放
- **cuda_error.hpp**: CUDA 错误处理宏
- **algorithm/**: 各种并行算法实现


