#pragma once

#include "../taskflow.hpp"
#include "cuda_graph.hpp"
#include "cuda_graph_exec.hpp"
#include "algorithm/single_task.hpp"

/**
@file taskflow/cuda/cudaflow.hpp
@brief cudaFlow include file

============================================================================
Taskflow CUDA 模块架构说明
============================================================================

【核心概念】：

Taskflow 的 CUDA 模块提供了两种方式来构建和执行 GPU 任务图：

1. **cudaGraph（显式图构建）**
   - 用户显式调用 API 构建 CUDA 图
   - 类似于 CPU 端的 Taskflow，但操作的是 GPU 任务
   - 使用 CUDA Graph API（cudaGraphCreate, cudaGraphAddNode 等）

2. **cudaFlowCapturer（流捕获）**
   - 通过 CUDA Stream Capture 自动捕获 GPU 操作
   - 用户编写异步 CUDA 代码，系统自动转换为 CUDA 图
   - 使用 CUDA Stream Capture API（cudaStreamBeginCapture 等）

============================================================================
与 CPU Taskflow 的区别
============================================================================

【相同点】：
  ✅ 都使用任务图（DAG）模型
  ✅ 都支持任务依赖关系（precede/succeed）
  ✅ 都可以嵌入到 CPU Taskflow 中作为一个任务

【不同点】：

1. **执行位置**
   - CPU Taskflow：任务在 CPU 线程池中执行
   - CUDA Graph：任务在 GPU 上执行

2. **图的构建方式**
   - CPU Taskflow：在运行时动态调度，支持 Subflow/Runtime
   - CUDA Graph：必须先构建完整的图，然后实例化执行

3. **图的表示**
   - CPU Taskflow：使用 Node/Graph/Topology 表示
   - CUDA Graph：使用 CUDA 原生的 cudaGraph_t/cudaGraphNode_t

4. **调度器**
   - CPU Taskflow：使用工作窃取调度器（Work-Stealing）
   - CUDA Graph：由 CUDA 运行时调度，GPU 硬件执行

5. **优化策略**
   - CPU Taskflow：负载均衡、缓存局部性
   - CUDA Graph：内核融合、启动开销优化、并发执行

============================================================================
核心类型说明
============================================================================

【1. cudaGraph（CUDA 图）】

类型定义：
  using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;

作用：
  - 管理 CUDA 原生图（cudaGraph_t）的智能指针
  - 提供构建 GPU 任务图的 API
  - 自动管理资源生命周期

使用方式：
  tf::cudaGraph cg;                              // 创建 CUDA 图
  tf::cudaTask h2d = cg.copy(d_ptr, h_ptr, N);  // 添加内存拷贝任务
  tf::cudaTask kernel = cg.kernel(...);          // 添加内核任务
  h2d.precede(kernel);                           // 设置依赖关系

【2. cudaGraphExec（可执行 CUDA 图）】

类型定义：
  using cudaGraphExec = cudaGraphExecBase<cudaGraphExecCreator, cudaGraphExecDeleter>;

作用：
  - 管理实例化后的 CUDA 图（cudaGraphExec_t）
  - 可以通过 cudaStream 多次执行
  - 执行开销极低（单次内核启动）

使用方式：
  tf::cudaGraphExec exec(cg);  // 从 cudaGraph 实例化
  tf::cudaStream stream;       // 创建 CUDA 流
  stream.run(exec);            // 在流上执行图
  stream.synchronize();        // 等待完成

【3. cudaTask（CUDA 任务）】

作用：
  - 表示 CUDA 图中的一个节点
  - 封装 CUDA 原生节点（cudaGraphNode_t）
  - 提供依赖关系设置接口

任务类型：
  - Kernel：GPU 内核执行
  - Memcpy：内存拷贝（H2D, D2H, D2D）
  - Memset：内存设置
  - Host：CPU 回调函数
  - Empty：空操作（用于同步）

【4. cudaFlowCapturer（流捕获器）】

作用：
  - 通过流捕获自动构建 CUDA 图
  - 支持任意异步 CUDA 操作
  - 使用优化器转换为高效的 CUDA 图

使用方式：
  taskflow.emplace([](tf::cudaFlowCapturer& capturer){
    auto task = capturer.on([](cudaStream_t stream){
      my_kernel<<<grid, block, 0, stream>>>(...);  // 捕获内核启动
    });
  });

============================================================================
CUDA API 使用说明
============================================================================

【图构建 API】（cudaGraph 使用）：
  - cudaGraphCreate()              - 创建空图
  - cudaGraphAddKernelNode()       - 添加内核节点
  - cudaGraphAddMemcpyNode()       - 添加内存拷贝节点
  - cudaGraphAddMemsetNode()       - 添加内存设置节点
  - cudaGraphAddHostNode()         - 添加主机回调节点
  - cudaGraphAddEmptyNode()        - 添加空节点
  - cudaGraphAddDependencies()     - 添加依赖边
  - cudaGraphDestroy()             - 销毁图

【图实例化 API】：
  - cudaGraphInstantiate()         - 实例化图
  - cudaGraphExecDestroy()         - 销毁实例化图

【图执行 API】：
  - cudaGraphLaunch()              - 在流上启动图

【流捕获 API】（cudaFlowCapturer 使用）：
  - cudaStreamBeginCapture()       - 开始捕获
  - cudaStreamEndCapture()         - 结束捕获，生成图
  - cudaStreamIsCapturing()        - 检查是否在捕获中

【图查询 API】：
  - cudaGraphGetNodes()            - 获取所有节点
  - cudaGraphGetRootNodes()        - 获取根节点
  - cudaGraphGetEdges()            - 获取所有边
  - cudaGraphNodeGetType()         - 获取节点类型

【图导出 API】：
  - cudaGraphDebugDotPrint()       - 导出为 DOT 格式

============================================================================
典型使用流程
============================================================================

【方式 1：显式图构建（cudaGraph）】

步骤：
  1. 创建 cudaGraph 对象
  2. 添加各种 cudaTask（kernel, copy, memset 等）
  3. 设置任务依赖关系（precede/succeed）
  4. 实例化为 cudaGraphExec
  5. 通过 cudaStream 执行
  6. 可以多次执行，开销极低

示例：
  tf::cudaGraph cg;
  auto h2d = cg.copy(d_data, h_data, N);
  auto kernel = cg.kernel(grid, block, 0, my_kernel, d_data, N);
  auto d2h = cg.copy(h_result, d_result, N);

  h2d.precede(kernel);
  kernel.precede(d2h);

  tf::cudaGraphExec exec(cg);
  tf::cudaStream stream;
  stream.run(exec).synchronize();

【方式 2：流捕获（cudaFlowCapturer）】

步骤：
  1. 在 Taskflow 中创建 cudaFlowCapturer 任务
  2. 使用 capturer.on() 捕获异步 CUDA 操作
  3. 设置任务依赖关系
  4. 系统自动优化并生成 CUDA 图
  5. 自动执行

示例：
  taskflow.emplace([](tf::cudaFlowCapturer& capturer){
    auto h2d = capturer.on([&](cudaStream_t s){
      cudaMemcpyAsync(d_data, h_data, N, cudaMemcpyHostToDevice, s);
    });

    auto kernel = capturer.on([&](cudaStream_t s){
      my_kernel<<<grid, block, 0, s>>>(d_data, N);
    });

    auto d2h = capturer.on([&](cudaStream_t s){
      cudaMemcpyAsync(h_result, d_result, N, cudaMemcpyDeviceToHost, s);
    });

    h2d.precede(kernel);
    kernel.precede(d2h);
  });

【方式 3：嵌入 CPU Taskflow】

CUDA 任务可以作为 CPU Taskflow 的一个节点：

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto cpu_task_1 = taskflow.emplace([](){
    // CPU 任务
  });

  auto gpu_task = taskflow.emplace([&](){
    tf::cudaGraph cg;
    // ... 构建 CUDA 图 ...
    tf::cudaGraphExec exec(cg);
    tf::cudaStream stream;
    stream.run(exec).synchronize();
  });

  auto cpu_task_2 = taskflow.emplace([](){
    // CPU 任务
  });

  cpu_task_1.precede(gpu_task);
  gpu_task.precede(cpu_task_2);

  executor.run(taskflow).wait();

============================================================================
性能优势
============================================================================

【CUDA Graph 的优势】：

1. **极低的启动开销**
   - 传统方式：每次内核启动都有 ~10μs 开销
   - CUDA Graph：整个图只有一次启动开销
   - 适合大量小内核的场景

2. **更好的并发性**
   - CUDA 运行时可以分析整个图的依赖关系
   - 自动并发执行独立的内核
   - 无需手动管理流和事件

3. **内核融合机会**
   - CUDA 驱动可以优化图的执行
   - 可能融合相邻的小内核
   - 减少内存访问次数

4. **可重复执行**
   - 图实例化后可以多次执行
   - 每次执行开销极低
   - 适合迭代算法

【Taskflow CUDA 的额外优势】：

1. **高层抽象**
   - 隐藏 CUDA Graph API 的复杂性
   - 提供类似 CPU Taskflow 的接口
   - 易于学习和使用

2. **自动优化**
   - cudaFlowCapturer 使用优化器自动转换图
   - 支持多种优化策略（Sequential, RoundRobin, Linear）
   - 最大化并发性

3. **与 CPU 任务集成**
   - GPU 任务可以无缝嵌入 CPU Taskflow
   - 统一的任务图模型
   - 自动管理 CPU-GPU 同步

============================================================================
*/

namespace tf {

/**
@brief default smart pointer type to manage a `cudaGraph_t` object with unique ownership

【类型说明】：
  - 这是一个智能指针类型，管理 CUDA 原生图（cudaGraph_t）
  - 使用 cudaGraphCreator 创建图
  - 使用 cudaGraphDeleter 销毁图
  - 自动管理资源生命周期，防止内存泄漏

【CUDA API】：
  - 创建：cudaGraphCreate()
  - 销毁：cudaGraphDestroy()
*/
using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;

/**
@brief default smart pointer type to manage a `cudaGraphExec_t` object with unique ownership

【类型说明】：
  - 这是一个智能指针类型，管理实例化后的 CUDA 图（cudaGraphExec_t）
  - 使用 cudaGraphExecCreator 实例化图
  - 使用 cudaGraphExecDeleter 销毁实例化图
  - 实例化后的图可以高效地多次执行

【CUDA API】：
  - 实例化：cudaGraphInstantiate()
  - 执行：cudaGraphLaunch()
  - 销毁：cudaGraphExecDestroy()
*/
using cudaGraphExec = cudaGraphExecBase<cudaGraphExecCreator, cudaGraphExecDeleter>;

}  // end of namespace tf -----------------------------------------------------


