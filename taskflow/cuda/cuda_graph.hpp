#pragma once

#include <filesystem>

#include "cuda_memory.hpp"
#include "cuda_stream.hpp"
#include "cuda_meta.hpp"

#include "../utility/traits.hpp"

/**
@file cuda_graph.hpp
@brief CUDA Graph 核心实现文件

============================================================================
文件说明
============================================================================

本文件实现了 Taskflow 的 CUDA Graph 功能，包括：

1. **辅助函数**：构建 CUDA 图节点参数（memcpy, memset, kernel 等）
2. **cudaTask 类**：表示 CUDA 图中的一个任务节点
3. **cudaGraphBase 类**：管理 CUDA 图的智能指针类
4. **图构建 API**：提供高层接口来构建 GPU 任务图

【核心概念】：

CUDA Graph 是 NVIDIA 提供的一种优化 GPU 执行的机制：
  - 将多个 GPU 操作组织成一个图（DAG）
  - 一次性提交整个图，而不是逐个提交操作
  - 大幅降低 CPU-GPU 通信开销
  - 允许 GPU 运行时优化执行顺序

【与 CPU Taskflow 的对比】：

CPU Taskflow                    | CUDA Graph
-------------------------------|--------------------------------
Node/Graph/Topology            | cudaGraphNode_t/cudaGraph_t
动态调度（运行时）              | 静态图（预先构建）
工作窃取调度器                  | GPU 硬件调度
支持 Subflow/Runtime           | 不支持动态修改
任务在 CPU 线程池执行           | 任务在 GPU 执行

============================================================================
*/

namespace tf {

// ----------------------------------------------------------------------------
// cudaGraph_t routines - CUDA 图辅助函数
// ----------------------------------------------------------------------------

/**
@brief 获取类型化内存拷贝任务的参数

【功能】：
  - 为 cudaGraphAddMemcpyNode() 准备参数
  - 支持类型化的内存拷贝（知道元素类型和数量）
  - 自动计算字节数和内存布局

【参数】：
  - tgt: 目标指针（可以是 host 或 device）
  - src: 源指针（可以是 host 或 device）
  - num: 元素数量（不是字节数）

【返回】：
  - cudaMemcpy3DParms 结构体，用于 cudaGraphAddMemcpyNode()

【CUDA API】：
  - 使用 make_cudaPos() 创建位置
  - 使用 make_cudaPitchedPtr() 创建指针描述
  - 使用 make_cudaExtent() 创建范围
  - cudaMemcpyDefault 自动检测拷贝方向（H2D, D2H, D2D）

【注意】：
  - 使用 cudaMemcpy3DParms 是为了统一接口
  - 实际上是 1D 拷贝（height=1, depth=1）
  - pitch（行间距）等于总字节数
*/
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
>
cudaMemcpy3DParms cuda_get_copy_parms(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  cudaMemcpy3DParms p;

  // 源内存配置
  p.srcArray = nullptr;  // 不使用 CUDA 数组
  p.srcPos = ::make_cudaPos(0, 0, 0);  // 从位置 (0,0,0) 开始
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
  // srcPtr 参数：指针, pitch（字节）, 宽度（元素）, 高度

  // 目标内存配置
  p.dstArray = nullptr;  // 不使用 CUDA 数组
  p.dstPos = ::make_cudaPos(0, 0, 0);  // 从位置 (0,0,0) 开始
  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);

  // 拷贝范围：num*sizeof(U) 字节，1 行，1 层
  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);

  // 自动检测拷贝方向（H2D, D2H, D2D, H2H）
  p.kind = cudaMemcpyDefault;

  return p;
}

/**
@brief 获取非类型化内存拷贝任务的参数

【功能】：
  - 为 cudaGraphAddMemcpyNode() 准备参数
  - 支持非类型化的内存拷贝（只知道字节数）
  - 用于 void* 指针的拷贝

【参数】：
  - tgt: 目标指针（void*）
  - src: 源指针（const void*）
  - bytes: 字节数

【返回】：
  - cudaMemcpy3DParms 结构体

【CUDA API】：
  - 调用 cudaGraphAddMemcpyNode() 时使用

【与 cuda_get_copy_parms 的区别】：
  - cuda_get_copy_parms: 类型化，参数是元素数量
  - cuda_get_memcpy_parms: 非类型化，参数是字节数
*/
inline cudaMemcpy3DParms cuda_get_memcpy_parms(
  void* tgt, const void* src, size_t bytes
)  {

  // cudaPitchedPtr 参数说明：
  // d   - 指向分配内存的指针
  // p   - 内存的 pitch（行间距），单位字节
  // xsz - 逻辑宽度（元素数量）
  // ysz - 逻辑高度（行数）
  cudaMemcpy3DParms p;
  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
  p.extent = ::make_cudaExtent(bytes, 1, 1);
  p.kind = cudaMemcpyDefault;

  return p;
}

/**
@brief 获取内存设置任务的参数（非类型化）

【功能】：
  - 为 cudaGraphAddMemsetNode() 准备参数
  - 类似于 memset()，将内存区域设置为指定值
  - 按字节设置

【参数】：
  - dst: 目标指针（device 内存）
  - ch: 要设置的值（0-255）
  - count: 字节数

【返回】：
  - cudaMemsetParams 结构体

【CUDA API】：
  - 调用 cudaGraphAddMemsetNode() 时使用
  - 等价于 cudaMemset(dst, ch, count)

【注意】：
  - elementSize = 1 表示按字节设置
  - width = count 表示设置 count 个字节
  - height = 1 表示只有一行
*/
inline cudaMemsetParams cuda_get_memset_parms(void* dst, int ch, size_t count) {

  cudaMemsetParams p;
  p.dst = dst;
  p.value = ch;  // 要设置的值（0-255）
  p.pitch = 0;   // 0 表示连续内存

  // 注释掉的代码是优化版本，根据对齐自动选择 elementSize
  // 但为了简单和可靠，统一使用 elementSize = 1
  //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
  //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;

  p.elementSize = 1;  // 按字节设置（可以是 1, 2, 或 4）
  p.width = count;    // 设置 count 个元素
  p.height = 1;       // 只有一行

  return p;
}

/**
@brief 获取类型化填充任务的参数

【功能】：
  - 为 cudaGraphAddMemsetNode() 准备参数
  - 支持类型化的填充（知道元素类型）
  - 可以填充任意 POD 类型的值

【模板参数】：
  - T: 元素类型，必须是 POD 且大小为 1, 2, 或 4 字节

【参数】：
  - dst: 目标指针（device 内存）
  - value: 要填充的值（类型 T）
  - count: 元素数量（不是字节数）

【返回】：
  - cudaMemsetParams 结构体

【CUDA API】：
  - 调用 cudaGraphAddMemsetNode() 时使用

【限制】：
  - 只支持 1, 2, 4 字节的 POD 类型
  - 这是 CUDA memset 的硬件限制

【示例】：
  int* d_data;
  cudaMalloc(&d_data, 100 * sizeof(int));
  auto params = cuda_get_fill_parms(d_data, 42, 100);  // 填充 100 个 int，值为 42
*/
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
>
cudaMemsetParams cuda_get_fill_parms(T* dst, T value, size_t count) {

  cudaMemsetParams p;
  p.dst = dst;

  // 执行按位拷贝（bit-wise copy）
  p.value = 0;  // 关键：先清零
  static_assert(sizeof(T) <= sizeof(p.value), "internal error");
  std::memcpy(&p.value, &value, sizeof(T));  // 将 value 的位模式拷贝到 p.value

  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief gets the memset node parameter of a zero task (typed)
*/
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
>
cudaMemsetParams cuda_get_zero_parms(T* dst, size_t count) {

  cudaMemsetParams p;
  p.dst = dst;
  p.value = 0;
  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief queries the number of root nodes in a native CUDA graph
*/
inline size_t cuda_graph_get_num_root_nodes(cudaGraph_t graph) {
  size_t num_nodes;
  TF_CHECK_CUDA(
    cudaGraphGetRootNodes(graph, nullptr, &num_nodes),
    "failed to get native graph root nodes"
  );
  return num_nodes;
}

/**
@brief queries the number of nodes in a native CUDA graph
*/
inline size_t cuda_graph_get_num_nodes(cudaGraph_t graph) {
  size_t num_nodes;
  TF_CHECK_CUDA(
    cudaGraphGetNodes(graph, nullptr, &num_nodes),
    "failed to get native graph nodes"
  );
  return num_nodes;
}

/**
@brief Handles compatibility with CUDA <= 12.x and CUDA == 13.x
 */
inline size_t cuda_graph_get_num_edges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to) {
  size_t num_edges;
  TF_CHECK_CUDA(
      TF_CUDA_PRE13(cudaGraphGetEdges(graph, from, to, &num_edges))
      TF_CUDA_POST13(cudaGraphGetEdges(graph, from, to, nullptr, &num_edges)),
      "failed to get native graph edges"
  );
  return num_edges;
}

/**
@brief Handles compatibility with CUDA <= 12.x and CUDA 13
* @param node
* @param dependencies
* @return
 */
inline size_t cuda_graph_node_get_dependencies(cudaGraphNode_t node, cudaGraphNode_t* dependencies) {
  size_t num_predecessors;
  TF_CHECK_CUDA(
      TF_CUDA_PRE13(cudaGraphNodeGetDependencies(node, dependencies, &num_predecessors))
      TF_CUDA_POST13(cudaGraphNodeGetDependencies(node, dependencies, nullptr, &num_predecessors)),
  "Failed to get number of dependencies");
  return num_predecessors;
}

/**
@brief Handles compatibility with CUDA <= 12.x and CUDA 13
@param node
@param dependent_nodes
@return
 */
inline size_t cuda_graph_node_get_dependent_nodes(cudaGraphNode_t node, cudaGraphNode_t *dependent_nodes) {
  size_t num_successors;
  TF_CHECK_CUDA(
      TF_CUDA_PRE13(cudaGraphNodeGetDependentNodes(node, dependent_nodes, &num_successors))
      TF_CUDA_POST13(cudaGraphNodeGetDependentNodes(node, dependent_nodes, nullptr, &num_successors)),
      "Failed to get CUDA dependent nodes");
  return num_successors;
}

/**
@brief Handles compatibility with CUDA <= 12.x and CUDA 13
@param graph
@param from
@param to
@param numDependencies
 */
inline void cuda_graph_add_dependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
  TF_CHECK_CUDA(
      TF_CUDA_PRE13(cudaGraphAddDependencies(graph, from, to, numDependencies))
      TF_CUDA_POST13(cudaGraphAddDependencies(graph, from, to, nullptr, numDependencies)),
      "Failed to add CUDA graph node dependencies"
      );
}

/**
@brief queries the number of edges in a native CUDA graph
*/
inline size_t cuda_graph_get_num_edges(cudaGraph_t graph) {
  return cuda_graph_get_num_edges(graph, nullptr, nullptr);
}



/**
@brief acquires the nodes in a native CUDA graph
*/
inline std::vector<cudaGraphNode_t> cuda_graph_get_nodes(cudaGraph_t graph) {
  size_t num_nodes = cuda_graph_get_num_nodes(graph);
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  TF_CHECK_CUDA(
    cudaGraphGetNodes(graph, nodes.data(), &num_nodes),
    "failed to get native graph nodes"
  );
  return nodes;
}

/**
@brief acquires the root nodes in a native CUDA graph
*/
inline std::vector<cudaGraphNode_t> cuda_graph_get_root_nodes(cudaGraph_t graph) {
  size_t num_nodes = cuda_graph_get_num_root_nodes(graph);
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  TF_CHECK_CUDA(
    cudaGraphGetRootNodes(graph, nodes.data(), &num_nodes),
    "failed to get native graph nodes"
  );
  return nodes;
}

/**
@brief acquires the edges in a native CUDA graph
*/
inline std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>>
cuda_graph_get_edges(cudaGraph_t graph) {
  size_t num_edges = cuda_graph_get_num_edges(graph);
  std::vector<cudaGraphNode_t> froms(num_edges), tos(num_edges);
  num_edges = cuda_graph_get_num_edges(graph, froms.data(), tos.data());
  std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>> edges(num_edges);
  for(size_t i=0; i<num_edges; i++) {
    edges[i] = std::make_pair(froms[i], tos[i]);
  }
  return edges;
}

/**
@brief queries the type of a native CUDA graph node

valid type values are:
  + cudaGraphNodeTypeKernel      = 0x00
  + cudaGraphNodeTypeMemcpy      = 0x01
  + cudaGraphNodeTypeMemset      = 0x02
  + cudaGraphNodeTypeHost        = 0x03
  + cudaGraphNodeTypeGraph       = 0x04
  + cudaGraphNodeTypeEmpty       = 0x05
  + cudaGraphNodeTypeWaitEvent   = 0x06
  + cudaGraphNodeTypeEventRecord = 0x07
*/
inline cudaGraphNodeType cuda_get_graph_node_type(cudaGraphNode_t node) {
  cudaGraphNodeType type;
  TF_CHECK_CUDA(
    cudaGraphNodeGetType(node, &type), "failed to get native graph node type"
  );
  return type;
}

// ----------------------------------------------------------------------------
// cudaTask Types - CUDA 任务类型
// ----------------------------------------------------------------------------

/**
@brief 将 CUDA 任务类型转换为可读字符串

【功能】：
  - 用于调试和日志输出
  - 将 cudaGraphNodeType 枚举转换为字符串

【CUDA 节点类型说明】：
  - Kernel: GPU 内核执行节点
  - Memcpy: 内存拷贝节点（H2D, D2H, D2D）
  - Memset: 内存设置节点
  - Host: CPU 回调函数节点
  - Graph: 子图节点（嵌套图）
  - Empty: 空节点（用于同步）
  - WaitEvent: 等待事件节点
  - EventRecord: 记录事件节点
  - ExtSemaphoreSignal: 外部信号量信号节点
  - ExtSemaphoreWait: 外部信号量等待节点
  - MemAlloc: 内存分配节点
  - MemFree: 内存释放节点
  - Conditional: 条件节点

【使用示例】：
  auto task = cg.kernel(...);
  auto type = task.type();
  std::cout << "Task type: " << tf::to_string(type) << "\n";
*/
constexpr const char* to_string(cudaGraphNodeType type) {
  switch (type) {
    case cudaGraphNodeTypeKernel:             return "Kernel";
    case cudaGraphNodeTypeMemcpy:             return "Memcpy";
    case cudaGraphNodeTypeMemset:             return "Memset";
    case cudaGraphNodeTypeHost:               return "Host";
    case cudaGraphNodeTypeGraph:              return "Graph";
    case cudaGraphNodeTypeEmpty:              return "Empty";
    case cudaGraphNodeTypeWaitEvent:          return "WaitEvent";
    case cudaGraphNodeTypeEventRecord:        return "EventRecord";
    case cudaGraphNodeTypeExtSemaphoreSignal: return "ExtSemaphoreSignal";
    case cudaGraphNodeTypeExtSemaphoreWait:   return "ExtSemaphoreWait";
    case cudaGraphNodeTypeMemAlloc:           return "MemAlloc";
    case cudaGraphNodeTypeMemFree:            return "MemFree";
    case cudaGraphNodeTypeConditional:        return "Conditional";
    default:                                  return "undefined";
  }
}

// ----------------------------------------------------------------------------
// cudaTask - CUDA 任务句柄
// ----------------------------------------------------------------------------

/**
@class cudaTask

@brief 表示 CUDA 图中一个节点的句柄

【核心概念】：

cudaTask 是 Taskflow 对 CUDA 图节点（cudaGraphNode_t）的封装，提供：
  1. 依赖关系设置（precede/succeed）
  2. 节点信息查询（type, num_successors 等）
  3. 与 CPU Taskflow 类似的接口

【与 CPU Task 的区别】：

CPU Task (tf::Task)              | CUDA Task (tf::cudaTask)
--------------------------------|--------------------------------
表示 CPU 任务图的节点            | 表示 CUDA 图的节点
封装 Node*                      | 封装 cudaGraphNode_t
在 CPU 线程池执行               | 在 GPU 执行
支持动态修改（Subflow/Runtime） | 静态图，不支持动态修改
有多种任务类型（Static等）       | 有多种节点类型（Kernel等）

【生命周期】：

1. 创建：通过 cudaGraph 的方法创建
   auto task = cg.kernel(...);

2. 设置依赖：使用 precede/succeed
   task1.precede(task2);

3. 执行：图实例化后执行
   tf::cudaGraphExec exec(cg);
   stream.run(exec);

【注意】：
  - cudaTask 只是一个句柄，不拥有节点
  - 节点的生命周期由 cudaGraph 管理
  - 图实例化后不能修改拓扑结构

【CUDA API】：
  - 依赖关系：cudaGraphAddDependencies()
  - 查询信息：cudaGraphNodeGetType(), cudaGraphNodeGetDependencies() 等
*/
class cudaTask {

  template <typename Creator, typename Deleter>
  friend class cudaGraphBase;

  template <typename Creator, typename Deleter>
  friend class cudaGraphExecBase;

  friend class cudaFlow;
  friend class cudaFlowCapturer;
  friend class cudaFlowCapturerBase;

  friend std::ostream& operator << (std::ostream&, const cudaTask&);

  public:

    /**
    @brief 构造空的 cudaTask

    【说明】：
      - 默认构造的 cudaTask 不指向任何节点
      - _native_graph 和 _native_node 都为 nullptr
      - 必须通过 cudaGraph 的方法创建有效的 cudaTask
    */
    cudaTask() = default;

    /**
    @brief 拷贝构造 cudaTask

    【说明】：
      - cudaTask 是轻量级句柄，拷贝开销很小
      - 拷贝后的 cudaTask 指向同一个节点
      - 不会复制节点本身
    */
    cudaTask(const cudaTask&) = default;

    /**
    @brief 拷贝赋值 cudaTask

    【说明】：
      - 赋值后指向同一个节点
      - 不影响原节点
    */
    cudaTask& operator = (const cudaTask&) = default;

    /**
    @brief 添加从当前任务到其他任务的前驱链接

    【功能】：
      - 设置当前任务为其他任务的前驱
      - 等价于：this → tasks...
      - 可以一次设置多个后继任务

    【模板参数】：
      - Ts: 参数包，可以是多个 cudaTask

    【参数】：
      - tasks: 一个或多个后继任务

    【返回】：
      - *this，支持链式调用

    【CUDA API】：
      - 调用 cudaGraphAddDependencies() 添加依赖边

    【使用示例】：
      auto h2d = cg.copy(...);
      auto kernel = cg.kernel(...);
      auto d2h = cg.copy(...);

      h2d.precede(kernel);        // h2d → kernel
      kernel.precede(d2h);        // kernel → d2h

      // 或链式调用
      h2d.precede(kernel).precede(d2h);  // 错误！precede 返回 h2d

      // 正确的链式调用
      h2d.precede(kernel);
      kernel.precede(d2h);

      // 或一次设置多个后继
      h2d.precede(kernel1, kernel2, kernel3);

    【注意】：
      - 不能创建环（会导致死锁）
      - 图实例化后不能修改依赖关系
    */
    template <typename... Ts>
    cudaTask& precede(Ts&&... tasks);

    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& succeed(Ts&&... tasks);

    /**
    @brief queries the number of successors
    */
    size_t num_successors() const;

    /**
    @brief queries the number of dependents
    */
    size_t num_predecessors() const;

    /**
    @brief queries the type of this task
    */
    auto type() const;

    /**
    @brief dumps the task through an output stream

    @param os an output stream target
    */
    void dump(std::ostream& os) const;

  private:

    cudaTask(cudaGraph_t, cudaGraphNode_t);
    
    cudaGraph_t _native_graph {nullptr};
    cudaGraphNode_t _native_node {nullptr};
};

// Constructor
inline cudaTask::cudaTask(cudaGraph_t native_graph, cudaGraphNode_t native_node) : 
  _native_graph {native_graph}, _native_node  {native_node} {
}
  
// Function: precede
template <typename... Ts>
cudaTask& cudaTask::precede(Ts&&... tasks) {
  (
    cuda_graph_add_dependencies(
      _native_graph, &_native_node, &(tasks._native_node), 1
    ), ...
  );
  return *this;
}

// Function: succeed
template <typename... Ts>
cudaTask& cudaTask::succeed(Ts&&... tasks) {
  (tasks.precede(*this), ...);
  return *this;
}

// Function: num_predecessors
inline size_t cudaTask::num_predecessors() const {
  return cuda_graph_node_get_dependencies(_native_node, nullptr);
}

// Function: num_successors
inline size_t cudaTask::num_successors() const {
  return cuda_graph_node_get_dependent_nodes(_native_node, nullptr);
}

// Function: type
inline auto cudaTask::type() const {
  cudaGraphNodeType type;
  cudaGraphNodeGetType(_native_node, &type);
  return type;
}

// Function: dump
inline void cudaTask::dump(std::ostream& os) const {
  os << "cudaTask [type=" << to_string(type()) << ']';
}

/**
@brief overload of ostream inserter operator for cudaTask
*/
inline std::ostream& operator << (std::ostream& os, const cudaTask& ct) {
  ct.dump(os);
  return os;
}

// ----------------------------------------------------------------------------
// cudaGraph
// ----------------------------------------------------------------------------

/**
 @class cudaGraphCreator

 @brief class to create functors that construct CUDA graphs
 
 This class define functors to new CUDA graphs using `cudaGraphCreate`. 
 
*/
class cudaGraphCreator {

  public:

  /**
   * @brief creates a new CUDA graph
   *
   * Calls `cudaGraphCreate` to generate a CUDA native graph and returns it.
   * If the graph creation fails, an error is reported.
   *
   * @return A newly created `cudaGraph_t` instance.
   * @throws If CUDA graph creation fails, an error is logged.
   */
  cudaGraph_t operator () () const {
    cudaGraph_t g;
    TF_CHECK_CUDA(cudaGraphCreate(&g, 0), "failed to create a CUDA native graph");
    return g;
  }
  
  /**
  @brief return the given CUDA graph
  */
  cudaGraph_t operator () (cudaGraph_t graph) const {
    return graph;
  }

};

/**
 @class cudaGraphDeleter

 @brief class to create a functor that deletes a CUDA graph
 
 This structure provides an overloaded function call operator to safely
 destroy a CUDA graph using `cudaGraphDestroy`.
 
*/
class cudaGraphDeleter {

  public:
 
  /**
   * @brief deletes a CUDA graph
   *
   * Calls `cudaGraphDestroy` to release the CUDA graph resource if it is valid.
   *
   * @param g the CUDA graph to be destroyed
   */
  void operator () (cudaGraph_t g) const {
    cudaGraphDestroy(g);
  }
};
  

/**
@class cudaGraphBase

@brief 管理 CUDA 图的智能指针类（唯一所有权）

【核心概念】：

cudaGraphBase 是一个模板类，继承自 std::unique_ptr，用于管理 CUDA 原生图（cudaGraph_t）：
  - 提供 RAII 风格的资源管理
  - 自动创建和销毁 CUDA 图
  - 提供高层 API 构建 GPU 任务图
  - 确保唯一所有权（不可拷贝，只能移动）

【模板参数】：
  - Creator: 创建图的函数对象（在构造函数中使用）
  - Deleter: 销毁图的函数对象（在析构函数中使用）

【典型实例化】：
  using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;

  其中：
  - cudaGraphCreator: 调用 cudaGraphCreate()
  - cudaGraphDeleter: 调用 cudaGraphDestroy()

【与 CPU Graph 的区别】：

CPU Graph (tf::Graph)           | CUDA Graph (tf::cudaGraph)
-------------------------------|--------------------------------
管理 Node 对象的容器            | 管理 cudaGraph_t 的智能指针
动态添加/删除节点               | 静态图，构建后不可修改
支持 Subflow/Runtime           | 不支持动态修改
使用 std::vector 存储节点       | 使用 CUDA 原生图结构

【生命周期】：

1. 创建：
   tf::cudaGraph cg;  // 调用 cudaGraphCreate()

2. 构建：
   auto task1 = cg.kernel(...);
   auto task2 = cg.copy(...);
   task1.precede(task2);

3. 实例化：
   tf::cudaGraphExec exec(cg);  // 调用 cudaGraphInstantiate()

4. 执行：
   tf::cudaStream stream;
   stream.run(exec);  // 调用 cudaGraphLaunch()

5. 销毁：
   // 自动调用 cudaGraphDestroy()

【CUDA API】：
  - 创建：cudaGraphCreate()
  - 添加节点：cudaGraphAddKernelNode(), cudaGraphAddMemcpyNode() 等
  - 添加依赖：cudaGraphAddDependencies()
  - 查询：cudaGraphGetNodes(), cudaGraphGetEdges() 等
  - 导出：cudaGraphDebugDotPrint()
  - 销毁：cudaGraphDestroy()

【注意】：
  - cudaGraphBase 不可拷贝，只能移动
  - 图构建完成后必须实例化才能执行
  - 实例化后不能修改图结构
  - 一个图可以实例化多次
*/
template <typename Creator, typename Deleter>
class cudaGraphBase : public std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, cudaGraphDeleter> {

  static_assert(std::is_pointer_v<cudaGraph_t>, "cudaGraph_t is not a pointer type");

  public:

  /**
  @brief 基类 std::unique_ptr 类型

  【说明】：
    - cudaGraphBase 继承自 std::unique_ptr
    - 自动管理 cudaGraph_t 的生命周期
    - 提供 get(), release(), reset() 等方法
  */
  using base_type = std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, Deleter>;

  /**
  @brief 构造 cudaGraph 对象

  【功能】：
    - 调用 Creator 函数对象创建 CUDA 图
    - 将创建的图交给 std::unique_ptr 管理
    - 自动设置 Deleter

  【模板参数】：
    - ArgsT: 传递给 Creator 的参数类型

  【参数】：
    - args: 传递给 Creator 的参数

  【CUDA API】：
    - Creator 通常调用 cudaGraphCreate()

  【使用示例】：
    tf::cudaGraph cg;  // 默认构造，创建空图
  */
  template <typename... ArgsT>
  explicit cudaGraphBase(ArgsT&& ... args) : base_type(
    Creator{}(std::forward<ArgsT>(args)...), Deleter()
  ) {
  }

  /**
  @brief 移动构造 cudaGraph

  【说明】：
    - cudaGraph 不可拷贝，只能移动
    - 移动后原对象不再拥有图
  */
  cudaGraphBase(cudaGraphBase&&) = default;

  /**
  @brief 移动赋值 cudaGraph

  【说明】：
    - 移动后原对象不再拥有图
    - 如果 *this 原本拥有图，会先销毁
  */
  cudaGraphBase& operator = (cudaGraphBase&&) = default;

  /**
  @brief 查询图中的节点数量

  【返回】：
    - 节点数量

  【CUDA API】：
    - 调用 cudaGraphGetNodes()
  */
  size_t num_nodes() const;

  /**
  @brief 查询图中的边数量

  【返回】：
    - 边数量（依赖关系数量）

  【CUDA API】：
    - 调用 cudaGraphGetEdges()
  */
  size_t num_edges() const;

  /**
  @brief 查询图是否为空

  【返回】：
    - true: 图为空（没有节点）
    - false: 图不为空

  【说明】：
    - 空图可以执行，但不做任何事情
  */
  bool empty() const;

  /**
  @brief 将 CUDA 图导出为 DOT 格式

  【功能】：
    - 将图结构导出为 Graphviz DOT 格式
    - 可以使用 graphviz 工具可视化

  【参数】：
    - os: 输出流

  【CUDA API】：
    - 调用 cudaGraphDebugDotPrint()

  【使用示例】：
    tf::cudaGraph cg;
    // ... 构建图 ...
    cg.dump(std::cout);  // 输出到控制台

    std::ofstream ofs("graph.dot");
    cg.dump(ofs);  // 输出到文件

    // 使用 graphviz 可视化
    // dot -Tpng graph.dot -o graph.png
  */
  void dump(std::ostream& os);

  // ------------------------------------------------------------------------
  // Graph building routines - 图构建方法
  // ------------------------------------------------------------------------

  /**
  @brief 创建空操作任务

  【功能】：
    - 创建一个不执行任何操作的节点
    - 用于传递依赖关系（同步点）

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddEmptyNode()

  【使用场景】：
    - 作为多个任务的汇聚点
    - 作为多个任务的分发点
    - 减少依赖边的数量

  【示例】：
    // 场景：2 组各 n 个任务，中间有屏障

    // 方法 1：使用空节点（推荐）
    std::vector<cudaTask> group1(n);
    for(int i = 0; i < n; i++) {
      group1[i] = cg.kernel(...);
    }

    auto barrier = cg.noop();  // 空节点作为屏障

    std::vector<cudaTask> group2(n);
    for(int i = 0; i < n; i++) {
      group2[i] = cg.kernel(...);
      group1[i].precede(barrier);  // group1 → barrier
      barrier.precede(group2[i]);  // barrier → group2
    }
    // 总共 2*n 条边

    // 方法 2：不使用空节点（不推荐）
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        group1[i].precede(group2[j]);  // 每个 group1 → 每个 group2
      }
    }
    // 总共 n^2 条边！
  */
  cudaTask noop();

  /**
  @brief 创建在 CPU 上执行的主机任务

  【功能】：
    - 在 GPU 任务图中插入 CPU 回调函数
    - 用于在 GPU 任务之间执行 CPU 代码
    - 可以访问 CPU 内存和执行 CPU 逻辑

  【模板参数】：
    - C: 可调用对象类型

  【参数】：
    - callable: 可调用对象，无参数无返回值（std::function<void()>）
    - user_data: 用户数据指针（可选）

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddHostNode()

  【限制】：
    - 主机任务只能执行 CPU 函数
    - 不能调用任何 CUDA API（如 cudaMalloc, cudaMemcpy 等）
    - 不能启动内核
    - 可以访问 CPU 内存（包括 pinned memory）

  【使用场景】：
    - 在 GPU 任务之间进行 CPU 计算
    - 检查中间结果
    - 记录日志
    - 更新 CPU 数据结构

  【示例】：
    tf::cudaGraph cg;

    auto h2d = cg.copy(d_data, h_data, N);

    auto host_task = cg.host([&](){
      std::cout << "Kernel is about to run\n";
      // 可以访问 CPU 变量
      // 不能调用 CUDA API
    }, nullptr);

    auto kernel = cg.kernel(...);

    h2d.precede(host_task);
    host_task.precede(kernel);

  【注意】：
    - 主机任务会阻塞 GPU 执行
    - 应该尽量简短
    - 不要在主机任务中做耗时操作
  */
  template <typename C>
  cudaTask host(C&& callable, void* user_data);

  /**
  @brief 创建 GPU 内核任务

  【功能】：
    - 在图中添加 GPU 内核执行节点
    - 这是 CUDA 图最核心的功能

  【模板参数】：
    - F: 内核函数类型
    - ArgsT: 内核参数类型（参数包）

  【参数】：
    - g: 网格维度（dim3）
    - b: 线程块维度（dim3）
    - s: 共享内存大小（字节）
    - f: 内核函数指针
    - args: 内核参数（按值传递）

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddKernelNode()

  【使用示例】：
    // 定义内核
    __global__ void my_kernel(int* data, int N) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if(idx < N) data[idx] *= 2;
    }

    // 添加到图
    tf::cudaGraph cg;
    dim3 grid((N+255)/256);
    dim3 block(256);
    auto task = cg.kernel(grid, block, 0, my_kernel, d_data, N);

  【注意】：
    - 参数按值传递（会拷贝）
    - 指针参数必须指向 device 内存
    - 共享内存大小 s 是动态共享内存
    - 内核函数必须是 __global__ 函数
  */
  template <typename F, typename... ArgsT>
  cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT... args);

  /**
  @brief 创建内存设置任务（按字节填充）

  【功能】：
    - 将 device 内存区域设置为指定字节值
    - 类似于 memset()

  【参数】：
    - dst: 目标指针（device 内存）
    - v: 要设置的字节值（0-255）
    - count: 字节数

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddMemsetNode()

  【使用示例】：
    int* d_data;
    cudaMalloc(&d_data, 1000 * sizeof(int));

    tf::cudaGraph cg;
    auto task = cg.memset(d_data, 0, 1000 * sizeof(int));  // 清零

  【注意】：
    - 按字节设置，不是按元素
    - 如果要设置类型化数据，使用 fill() 或 zero()
  */
  cudaTask memset(void* dst, int v, size_t count);

  /**
  @brief 创建内存拷贝任务（非类型化）

  【功能】：
    - 在图中添加内存拷贝节点
    - 支持任意方向的拷贝（H2D, D2H, D2D, H2H）

  【参数】：
    - tgt: 目标指针
    - src: 源指针
    - bytes: 字节数

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddMemcpyNode()

  【拷贝方向】：
    - H2D: Host to Device（CPU → GPU）
    - D2H: Device to Host（GPU → CPU）
    - D2D: Device to Device（GPU → GPU）
    - H2H: Host to Host（CPU → CPU，很少用）

  【使用示例】：
    float *h_data, *d_data;
    h_data = new float[N];
    cudaMalloc(&d_data, N * sizeof(float));

    tf::cudaGraph cg;
    auto h2d = cg.memcpy(d_data, h_data, N * sizeof(float));  // H2D
    auto kernel = cg.kernel(...);
    auto d2h = cg.memcpy(h_data, d_data, N * sizeof(float));  // D2H

    h2d.precede(kernel);
    kernel.precede(d2h);

  【注意】：
    - 参数是字节数，不是元素数量
    - 如果要类型化拷贝，使用 copy()
  */
  cudaTask memcpy(void* tgt, const void* src, size_t bytes);

  /**
  @brief 创建清零任务（类型化）

  【功能】：
    - 将类型化的 device 内存区域清零
    - 比 memset() 更安全（知道元素类型）

  【模板参数】：
    - T: 元素类型（大小必须是 1, 2, 或 4 字节）

  【参数】：
    - dst: 目标指针（device 内存）
    - count: 元素数量（不是字节数）

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddMemsetNode()

  【使用示例】：
    int* d_data;
    cudaMalloc(&d_data, 1000 * sizeof(int));

    tf::cudaGraph cg;
    auto task = cg.zero(d_data, 1000);  // 清零 1000 个 int

  【限制】：
    - T 必须是 POD 类型
    - sizeof(T) 必须是 1, 2, 或 4
    - 这是 CUDA memset 的硬件限制
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  cudaTask zero(T* dst, size_t count);

  /**
  @brief 创建填充任务（类型化）

  【功能】：
    - 将类型化的 device 内存区域填充为指定值
    - 支持任意 POD 类型（1, 2, 4 字节）

  【模板参数】：
    - T: 元素类型（大小必须是 1, 2, 或 4 字节）

  【参数】：
    - dst: 目标指针（device 内存）
    - value: 要填充的值（类型 T）
    - count: 元素数量（不是字节数）

  【返回】：
    - cudaTask 句柄

  【CUDA API】：
    - 调用 cudaGraphAddMemsetNode()

  【使用示例】：
    int* d_data;
    cudaMalloc(&d_data, 1000 * sizeof(int));

    tf::cudaGraph cg;
    auto task = cg.fill(d_data, 42, 1000);  // 填充 1000 个 int，值为 42

    float* d_floats;
    cudaMalloc(&d_floats, 500 * sizeof(float));
    auto task2 = cg.fill(d_floats, 3.14f, 500);  // 填充 500 个 float

  【与 memset 的区别】：
    - memset: 按字节填充，value 是 0-255
    - fill: 按元素填充，value 是类型 T 的值

  【限制】：
    - T 必须是 POD 类型
    - sizeof(T) 必须是 1, 2, 或 4
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  cudaTask fill(T* dst, T value, size_t count);

  /**
  @brief creates a memcopy task that copies typed data

  @tparam T element type (non-void)

  @param tgt pointer to the target memory block
  @param src pointer to the source memory block
  @param num number of elements to copy

  @return a tf::cudaTask handle

  A copy task transfers <tt>num*sizeof(T)</tt> bytes of data from a source location
  to a target location. Direction can be arbitrary among CPUs and GPUs.
  */
  template <typename T,
    std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
  >
  cudaTask copy(T* tgt, const T* src, size_t num);
  
  // ------------------------------------------------------------------------
  // generic algorithms
  // ------------------------------------------------------------------------

  /**
  @brief runs a callable with only a single kernel thread

  @tparam C callable type

  @param c callable to run by a single kernel thread

  @return a tf::cudaTask handle
  */
  template <typename C>
  cudaTask single_task(C c);
  
  /**
  @brief applies a callable to each dereferenced element of the data array

  @tparam I iterator type
  @tparam C callable type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  @param callable a callable object to apply to the dereferenced iterator

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  for(auto itr = first; itr != last; itr++) {
    callable(*itr);
  }
  @endcode
  */
  template <typename I, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask for_each(I first, I last, C callable);
  
  /**
  @brief applies a callable to each index in the range with the step size

  @tparam I index type
  @tparam C callable type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first beginning index
  @param last last index
  @param step step size
  @param callable the callable to apply to each element in the data array

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  // step is positive [first, last)
  for(auto i=first; i<last; i+=step) {
    callable(i);
  }

  // step is negative [first, last)
  for(auto i=first; i>last; i+=step) {
    callable(i);
  }
  @endcode
  */
  template <typename I, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask for_each_index(I first, I last, I step, C callable);
  
  /**
  @brief applies a callable to a source range and stores the result in a target range

  @tparam I input iterator type
  @tparam O output iterator type
  @tparam C unary operator type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first iterator to the beginning of the input range
  @param last iterator to the end of the input range
  @param output iterator to the beginning of the output range
  @param op the operator to apply to transform each element in the range

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  while (first != last) {
    *output++ = callable(*first++);
  }
  @endcode
  */
  template <typename I, typename O, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask transform(I first, I last, O output, C op);
  
  /**
  @brief creates a task to perform parallel transforms over two ranges of items

  @tparam I1 first input iterator type
  @tparam I2 second input iterator type
  @tparam O output iterator type
  @tparam C unary operator type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first1 iterator to the beginning of the input range
  @param last1 iterator to the end of the input range
  @param first2 iterato
  @param output iterator to the beginning of the output range
  @param op binary operator to apply to transform each pair of items in the
            two input ranges

  @return cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  while (first1 != last1) {
    *output++ = op(*first1++, *first2++);
  }
  @endcode
  */
  template <typename I1, typename I2, typename O, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);

  private:

  cudaGraphBase(const cudaGraphBase&) = delete;
  cudaGraphBase& operator = (const cudaGraphBase&) = delete;
};

// query the number of nodes
template <typename Creator, typename Deleter>
size_t cudaGraphBase<Creator, Deleter>::num_nodes() const {
  size_t n;
  TF_CHECK_CUDA(
    cudaGraphGetNodes(this->get(), nullptr, &n),
    "failed to get native graph nodes"
  );
  return n;
}

// query the emptiness
template <typename Creator, typename Deleter>
bool cudaGraphBase<Creator, Deleter>::empty() const {
  return num_nodes() == 0;
}

// query the number of edges
template <typename Creator, typename Deleter>
size_t cudaGraphBase<Creator, Deleter>::num_edges() const {
  return cuda_graph_get_num_edges(this->get());
}

//// dump the graph
//inline void cudaGraph::dump(std::ostream& os) {
//  
//  // acquire the native handle
//  auto g = this->get();
//
//  os << "digraph cudaGraph {\n";
//
//  std::stack<std::tuple<cudaGraph_t, cudaGraphNode_t, int>> stack;
//  stack.push(std::make_tuple(g, nullptr, 1));
//
//  int pl = 0;
//
//  while(stack.empty() == false) {
//
//    auto [graph, parent, l] = stack.top();
//    stack.pop();
//
//    for(int i=0; i<pl-l+1; i++) {
//      os << "}\n";
//    }
//
//    os << "subgraph cluster_p" << graph << " {\n"
//       << "label=\"cudaGraph-L" << l << "\";\n"
//       << "color=\"purple\";\n";
//
//    auto nodes = cuda_graph_get_nodes(graph);
//    auto edges = cuda_graph_get_edges(graph);
//
//    for(auto& [from, to] : edges) {
//      os << 'p' << from << " -> " << 'p' << to << ";\n";
//    }
//
//    for(auto& node : nodes) {
//      auto type = cuda_get_graph_node_type(node);
//      if(type == cudaGraphNodeTypeGraph) {
//
//        cudaGraph_t child_graph;
//        TF_CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(node, &child_graph), "");
//        stack.push(std::make_tuple(child_graph, node, l+1));
//
//        os << 'p' << node << "["
//           << "shape=folder, style=filled, fontcolor=white, fillcolor=purple, "
//           << "label=\"cudaGraph-L" << l+1
//           << "\"];\n";
//      }
//      else {
//        os << 'p' << node << "[label=\""
//           << to_string(type)
//           << "\"];\n";
//      }
//    }
//
//    // precede to parent
//    if(parent != nullptr) {
//      std::unordered_set<cudaGraphNode_t> successors;
//      for(const auto& p : edges) {
//        successors.insert(p.first);
//      }
//      for(auto node : nodes) {
//        if(successors.find(node) == successors.end()) {
//          os << 'p' << node << " -> " << 'p' << parent << ";\n";
//        }
//      }
//    }
//
//    // set the previous level
//    pl = l;
//  }
//
//  for(int i=0; i<=pl; i++) {
//    os << "}\n";
//  }
//}

// dump the graph
template <typename Creator, typename Deleter>
void cudaGraphBase<Creator, Deleter>::dump(std::ostream& os) {

  // Generate a unique temporary filename in the system's temp directory using filesystem
  auto temp_path = std::filesystem::temp_directory_path() / "graph_";
  std::random_device rd;
  std::uniform_int_distribution<int> dist(100000, 999999); // Generates a random number
  temp_path += std::to_string(dist(rd)) + ".dot";

  // Call the original function with the temporary file
  TF_CHECK_CUDA(cudaGraphDebugDotPrint(this->get(), temp_path.string().c_str(), 0), "");

  // Read the file and write to the output stream
  std::ifstream file(temp_path);
  if (file) {
    os << file.rdbuf();  // Copy file contents to the stream
    file.close();
    std::filesystem::remove(temp_path);  // Clean up the temporary file
  } else {
    TF_THROW("failed to open ", temp_path, " for dumping the CUDA graph");
  }
}

// Function: noop
template <typename Creator, typename Deleter>
cudaTask cudaGraphBase<Creator, Deleter>::noop() {

  cudaGraphNode_t node;

  TF_CHECK_CUDA(
    cudaGraphAddEmptyNode(&node, this->get(), nullptr, 0),
    "failed to create a no-operation (empty) node"
  );

  return cudaTask(this->get(), node);
}

// Function: host
template <typename Creator, typename Deleter>
template <typename C>
cudaTask cudaGraphBase<Creator, Deleter>::host(C&& callable, void* user_data) {

  cudaGraphNode_t node;
  cudaHostNodeParams p {callable, user_data};

  TF_CHECK_CUDA(
    cudaGraphAddHostNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a host node"
  );

  return cudaTask(this->get(), node);
}

// Function: kernel
template <typename Creator, typename Deleter>
template <typename F, typename... ArgsT>
cudaTask cudaGraphBase<Creator, Deleter>::kernel(
  dim3 g, dim3 b, size_t s, F f, ArgsT... args
) {

  cudaGraphNode_t node;
  cudaKernelNodeParams p;

  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };

  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    cudaGraphAddKernelNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a kernel task"
  );

  return cudaTask(this->get(), node);
}

// Function: zero
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaGraphBase<Creator, Deleter>::zero(T* dst, size_t count) {

  cudaGraphNode_t node;
  auto p = cuda_get_zero_parms(dst, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memset (zero) task"
  );

  return cudaTask(this->get(), node);
}

// Function: fill
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaGraphBase<Creator, Deleter>::fill(T* dst, T value, size_t count) {

  cudaGraphNode_t node;
  auto p = cuda_get_fill_parms(dst, value, count);
  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memset (fill) task"
  );

  return cudaTask(this->get(), node);
}

// Function: copy
template <typename Creator, typename Deleter>
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cudaGraphBase<Creator, Deleter>::copy(T* tgt, const T* src, size_t num) {

  cudaGraphNode_t node;
  auto p = cuda_get_copy_parms(tgt, src, num);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memcpy (copy) task"
  );

  return cudaTask(this->get(), node);
}

// Function: memset
template <typename Creator, typename Deleter>
cudaTask cudaGraphBase<Creator, Deleter>::memset(void* dst, int ch, size_t count) {

  cudaGraphNode_t node;
  auto p = cuda_get_memset_parms(dst, ch, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memset task"
  );

  return cudaTask(this->get(), node);
}

// Function: memcpy
template <typename Creator, typename Deleter>
cudaTask cudaGraphBase<Creator, Deleter>::memcpy(void* tgt, const void* src, size_t bytes) {

  cudaGraphNode_t node;
  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memcpy task"
  );

  return cudaTask(this->get(), node);
}





}  // end of namespace tf -----------------------------------------------------




