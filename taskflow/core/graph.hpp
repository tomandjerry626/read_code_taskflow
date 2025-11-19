#pragma once

#include "../utility/macros.hpp"
#include "../utility/traits.hpp"
#include "../utility/iterator.hpp"

#ifdef TF_ENABLE_TASK_POOL
#include "../utility/object_pool.hpp"
#endif

#include "../utility/os.hpp"
#include "../utility/math.hpp"
#include "../utility/small_vector.hpp"
#include "../utility/serializer.hpp"
#include "../utility/lazy_string.hpp"
#include "error.hpp"
#include "declarations.hpp"
#include "semaphore.hpp"
#include "environment.hpp"
#include "topology.hpp"
#include "tsq.hpp"


/**
@file graph.hpp
@brief graph include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Class: Graph
// ----------------------------------------------------------------------------

/**
@class Graph

@brief class to create a graph object

A graph is the ultimate storage for a task dependency graph and is the main
gateway to interact with an executor.
This class is mainly used for creating an opaque graph object in a custom
class to interact with the executor through taskflow composition.

A graph object is move-only.

Graph 类是任务依赖图的最终存储容器
继承自 std::vector<std::unique_ptr<Node>>，存储所有任务节点
每个节点使用 unique_ptr 管理，保证节点的唯一所有权
*/
class Graph : public std::vector<std::unique_ptr<Node>> {

  friend class Node;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Taskflow;
  friend class Executor;

  public:

  /**
  @brief constructs a graph object

  构造一个空的任务图
  */
  Graph() = default;

  /**
  @brief disabled copy constructor

  禁用拷贝构造函数，Graph 是 move-only 类型
  因为节点之间有指针引用，拷贝会导致指针失效
  */
  Graph(const Graph&) = delete;

  /**
  @brief constructs a graph using move semantics

  移动构造函数，转移图的所有权
  */
  Graph(Graph&&) = default;

  /**
  @brief disabled copy assignment operator

  禁用拷贝赋值运算符，Graph 是 move-only 类型
  */
  Graph& operator = (const Graph&) = delete;

  /**
  @brief assigns a graph using move semantics

  移动赋值运算符，转移图的所有权
  */
  Graph& operator = (Graph&&) = default;

  private:

  // 从图中删除指定节点
  // 会在 vector 中查找并移除该节点的 unique_ptr
  void _erase(Node*);

  /**
  @private

  在图的末尾添加一个新节点
  使用完美转发将参数传递给 Node 的构造函数
  返回新创建节点的原始指针
  */
  template <typename ...ArgsT>
  Node* _emplace_back(ArgsT&&...);
};

// ----------------------------------------------------------------------------
// TaskParams
// ----------------------------------------------------------------------------

/**
@class TaskParams

@brief class to create a task parameter object 
*/
class TaskParams {

  public:

  /**
  @brief name of the task
  */
  std::string name;

  /**
  @brief C-styled pointer to user data
  */
  void* data {nullptr};
};

/**
@class DefaultTaskParams

@brief class to create an empty task parameter for compile-time optimization
*/
class DefaultTaskParams {};

/**
@brief determines if the given type is a task parameter type

Task parameters can be specified in one of the following types:
  + tf::TaskParams
  + tf::DefaultTaskParams
  + std::string
*/
template <typename P>
constexpr bool is_task_params_v =
  std::is_same_v<std::decay_t<P>, TaskParams> ||
  std::is_same_v<std::decay_t<P>, DefaultTaskParams> ||
  std::is_constructible_v<std::string, P>;

// ----------------------------------------------------------------------------
// Node
// ----------------------------------------------------------------------------

/**
@private

Node 类是 Taskflow 中任务的内部表示
每个 Task 对象实际上是一个指向 Node 的句柄
Node 存储了任务的所有信息：
  - 任务类型和工作函数（_handle）
  - 依赖关系（_edges）
  - 执行状态（_nstate, _estate, _join_counter）
  - 所属的 Topology 和父节点
  - 信号量和异常处理

Node 是 Taskflow 调度器的核心数据结构
*/
class Node {

  friend class Graph;
  friend class Task;
  friend class AsyncTask;
  friend class TaskView;
  friend class Taskflow;
  friend class Executor;
  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class AnchorGuard;

#ifdef TF_ENABLE_TASK_POOL
  TF_ENABLE_POOLABLE_ON_THIS;
#endif

  // 占位符类型，表示空任务（没有实际工作）
  using Placeholder = std::monostate;

  // 静态任务句柄
  // 最基本的任务类型，执行一个无参数的函数
  struct Static {

    template <typename C>
    Static(C&&);

    // 工作函数：void()
    // 不接受任何参数，不返回任何值
    std::function<void()> work;
  };

  // 运行时任务句柄
  // 可以在运行时动态创建新任务
  struct Runtime {

    template <typename C>
    Runtime(C&&);

    // 工作函数：void(Runtime&)
    // 接受 Runtime 对象，可以通过它动态调度新任务
    std::function<void(tf::Runtime&)> work;
  };

  // 非抢占式运行时任务句柄
  // 类似 Runtime，但不能被抢占（必须执行完成）
  struct NonpreemptiveRuntime {

    template <typename C>
    NonpreemptiveRuntime(C&&);

    // 工作函数：void(NonpreemptiveRuntime&)
    // 接受 NonpreemptiveRuntime 对象，可以动态调度新任务
    // 但不支持抢占式调度
    std::function<void(tf::NonpreemptiveRuntime&)> work;
  };

  // 子流任务句柄
  // 可以创建子任务图（嵌套并行）
  struct Subflow {

    template <typename C>
    Subflow(C&&);

    // 工作函数：void(Subflow&)
    // 接受 Subflow 对象，可以通过它创建子任务
    std::function<void(tf::Subflow&)> work;

    // 子图，存储子任务的任务图
    // 子任务会被添加到这个图中
    Graph subgraph;
  };

  // 条件任务句柄
  // 根据返回值决定执行哪个后继任务
  struct Condition {

    template <typename C>
    Condition(C&&);

    // 工作函数：int()
    // 返回值是后继任务的索引（0, 1, 2, ...）
    // 返回 -1 表示不执行任何后继任务
    std::function<int()> work;
  };

  // 多条件任务句柄
  // 可以同时激活多个后继任务
  struct MultiCondition {

    template <typename C>
    MultiCondition(C&&);

    // 工作函数：SmallVector<int>()
    // 返回一个索引列表，表示要执行的后继任务
    // 例如：返回 {0, 2, 3} 表示执行第 0、2、3 个后继任务
    std::function<SmallVector<int>()> work;
  };

  // 模块任务句柄
  // 用于组合其他 Taskflow（任务图复用）
  struct Module {

    template <typename T>
    Module(T&);

    // 引用另一个 Taskflow 的图
    // 执行时会执行这个图中的所有任务
    Graph& graph;
  };

  // 异步任务句柄
  // 用于 executor.async() 创建的异步任务
  struct Async {

    template <typename T>
    Async(T&&);

    // 工作函数，支持三种类型：
    //   1. void()                      : 简单异步任务
    //   2. void(Runtime&)              : 静默异步任务（可以创建子任务）
    //   3. void(Runtime&, bool)        : 异步任务（可以创建子任务，bool 表示是否取消）
    std::variant<
      std::function<void()>,
      std::function<void(tf::Runtime&)>,       // silent async
      std::function<void(tf::Runtime&, bool)>  // async
    > work;
  };

  // 依赖异步任务句柄
  // 用于 executor.dependent_async() 创建的有依赖关系的异步任务
  struct DependentAsync {

    template <typename C>
    DependentAsync(C&&);

    // 工作函数，支持三种类型（同 Async）
    std::variant<
      std::function<void()>,
      std::function<void(tf::Runtime&)>,       // silent async
      std::function<void(tf::Runtime&, bool)>  // async
    > work;

    // 引用计数，用于管理异步任务的生命周期
    // 当引用计数减到 0 时，任务可以被销毁
    std::atomic<size_t> use_count {1};

    // 异步任务状态
    // 可能的值：
    //   - ASTATE::UNFINISHED (0) : 未完成
    //   - ASTATE::LOCKED     (1) : 被锁定（正在处理）
    //   - ASTATE::FINISHED   (2) : 已完成
    std::atomic<ASTATE::underlying_type> state {ASTATE::UNFINISHED};
  };

  using handle_t = std::variant<
    Placeholder,          // placeholder
    Static,               // static tasking
    Runtime,              // runtime tasking
    NonpreemptiveRuntime, // runtime (non-preemptive) tasking
    Subflow,              // subflow tasking
    Condition,            // conditional tasking
    MultiCondition,       // multi-conditional tasking
    Module,               // composable tasking
    Async,                // async tasking
    DependentAsync        // dependent async tasking
  >;

  // 信号量集合
  // 用于任务间的同步和资源管理
  struct Semaphores {
    // 任务执行前需要获取的信号量列表
    // 如果无法获取所有信号量，任务会等待
    SmallVector<Semaphore*> to_acquire;

    // 任务执行后需要释放的信号量列表
    // 释放信号量可能会唤醒其他等待的任务
    SmallVector<Semaphore*> to_release;
  };

  public:

  // variant 索引常量
  // 用于判断 _handle 当前存储的是哪种类型的任务
  // 通过 _handle.index() 获取当前索引，然后与这些常量比较

  // 占位符任务索引（空任务）
  constexpr static auto PLACEHOLDER           = get_index_v<Placeholder, handle_t>;
  // 静态任务索引 void()
  constexpr static auto STATIC                = get_index_v<Static, handle_t>;
  // 运行时任务索引 void(Runtime&)
  constexpr static auto RUNTIME               = get_index_v<Runtime, handle_t>;
  // 非抢占式运行时任务索引 void(NonpreemptiveRuntime&)
  constexpr static auto NONPREEMPTIVE_RUNTIME = get_index_v<NonpreemptiveRuntime, handle_t>;
  // 子流任务索引 void(Subflow&)
  constexpr static auto SUBFLOW               = get_index_v<Subflow, handle_t>;
  // 条件任务索引 int()
  constexpr static auto CONDITION             = get_index_v<Condition, handle_t>;
  // 多条件任务索引 SmallVector<int>()
  constexpr static auto MULTI_CONDITION       = get_index_v<MultiCondition, handle_t>;
  // 模块任务索引（组合其他 Taskflow）
  constexpr static auto MODULE                = get_index_v<Module, handle_t>;
  // 异步任务索引
  constexpr static auto ASYNC                 = get_index_v<Async, handle_t>;
  // 依赖异步任务索引
  constexpr static auto DEPENDENT_ASYNC       = get_index_v<DependentAsync, handle_t>;

  // 默认构造函数
  Node() = default;

  // 构造函数（带任务参数）
  // 参数：
  //   nstate       - 节点状态
  //   estate       - 异常状态
  //   params       - 任务参数（包含名称和用户数据）
  //   topology     - 所属的 Topology
  //   parent       - 父节点（用于子流）
  //   join_counter - 初始 join 计数器值
  //   args         - 传递给 handle_t 的参数（用于构造具体的任务类型）
  template <typename... Args>
  Node(nstate_t, estate_t, const TaskParams&, Topology*, Node*, size_t, Args&&...);

  // 构造函数（不带任务参数，用于编译时优化）
  template <typename... Args>
  Node(nstate_t, estate_t, const DefaultTaskParams&, Topology*, Node*, size_t, Args&&...);

  // 获取后继节点的数量
  // 返回该节点指向的节点数量（出边数量）
  size_t num_successors() const;

  // 获取前驱节点的数量
  // 返回指向该节点的节点数量（入边数量）
  size_t num_predecessors() const;

  // 获取强依赖（非条件依赖）的数量
  // 强依赖必须完成才能执行该节点
  size_t num_strong_dependencies() const;

  // 获取弱依赖（条件依赖）的数量
  // 弱依赖来自条件任务，不一定会执行
  size_t num_weak_dependencies() const;

  // 获取任务名称
  const std::string& name() const;

  private:

  // 节点状态标志位（Node State）
  // 可能的值（可以通过位运算组合）：
  //   - NSTATE::NONE           (0x00000000) : 无状态
  //   - NSTATE::CONDITIONED    (0x10000000) : 节点有条件依赖（弱依赖）
  //   - NSTATE::PREEMPTED      (0x20000000) : 节点被抢占（用于运行时任务）
  //   - NSTATE::RETAIN_SUBFLOW (0x40000000) : 保留子流（不自动销毁）
  //   - NSTATE::JOINED_SUBFLOW (0x80000000) : 子流已经 join
  // 低 28 位用于存储弱依赖（条件依赖）的数量
  nstate_t _nstate              {NSTATE::NONE};

  // 异常状态标志位（Exception State），原子操作保证线程安全
  // 可能的值（可以通过位运算组合）：
  //   - ESTATE::NONE      (0x00000000) : 正常状态
  //   - ESTATE::EXCEPTION (0x10000000) : 发生了异常
  //   - ESTATE::CANCELLED (0x20000000) : 被取消
  //   - ESTATE::ANCHORED  (0x40000000) : 被锚定（用于异常传播路径标记）
  std::atomic<estate_t> _estate {ESTATE::NONE};

  // 任务名称，用于调试和可视化
  std::string _name;

  // 用户自定义数据指针
  // 可以通过 TaskParams 设置，用于在任务间传递数据
  void* _data {nullptr};

  // 指向所属 Topology 的指针
  // Topology 是 Taskflow 的运行时实例，管理整个任务图的执行
  // 如果为 nullptr，表示这是一个异步任务（没有 Topology）
  Topology* _topology {nullptr};

  // 指向父节点的指针
  // 用于子流（Subflow）任务，指向创建它的父任务
  // 如果为 nullptr，表示这是顶层任务
  Node* _parent {nullptr};

  // 后继节点的数量
  // _edges 向量的布局：[后继节点...][前驱节点...]
  // _num_successors 标记了后继节点和前驱节点的分界点
  // 例如：_edges = [s1, s2, s3, p1, p2]，_num_successors = 3
  //       前 3 个是后继节点，后 2 个是前驱节点
  size_t _num_successors {0};

  // 边的向量，存储所有相邻节点（后继节点 + 前驱节点）
  // 布局：[0, _num_successors) 是后继节点
  //       [_num_successors, size()) 是前驱节点
  // 使用 SmallVector 优化小规模情况（4 个元素以内不需要堆分配）
  SmallVector<Node*, 4> _edges;

  // Join 计数器，用于跟踪还有多少个强依赖（前驱节点）未完成
  // 当计数器减到 0 时，表示所有前驱节点都已完成，该节点可以执行
  // 使用原子操作保证线程安全（多个前驱节点可能并发完成）
  std::atomic<size_t> _join_counter {0};

  // 任务句柄，存储任务的实际工作内容
  // 使用 std::variant 支持多种任务类型：
  //   - Placeholder          : 占位符（空任务）
  //   - Static               : 静态任务 void()
  //   - Runtime              : 运行时任务 void(Runtime&)
  //   - NonpreemptiveRuntime : 非抢占式运行时任务 void(NonpreemptiveRuntime&)
  //   - Subflow              : 子流任务 void(Subflow&)
  //   - Condition            : 条件任务 int()
  //   - MultiCondition       : 多条件任务 SmallVector<int>()
  //   - Module               : 模块任务（组合其他 Taskflow）
  //   - Async                : 异步任务
  //   - DependentAsync       : 依赖异步任务
  handle_t _handle;

  // 信号量指针，用于任务间的同步
  // 包含两个列表：
  //   - to_acquire : 任务执行前需要获取的信号量
  //   - to_release : 任务执行后需要释放的信号量
  // 使用 unique_ptr 是因为大多数任务不需要信号量，节省内存
  std::unique_ptr<Semaphores> _semaphores;

  // 异常指针，用于存储任务执行过程中捕获的异常
  // 如果任务抛出异常，异常会被存储在这里
  // 并在适当的时候重新抛出或传递给 Topology
  std::exception_ptr _exception_ptr {nullptr};

  // 判断节点是否被取消
  // 检查节点所属的 Topology 或父节点是否设置了 CANCELLED 标志
  bool _is_cancelled() const;

  // 判断节点是否是条件任务
  // 条件任务包括 CONDITION 和 MULTI_CONDITION 两种类型
  bool _is_conditioner() const;

  // 判断节点是否被抢占
  // 检查 _nstate 是否设置了 PREEMPTED 标志
  bool _is_preempted() const;

  // 尝试获取所有需要的信号量
  // 参数：nodes - 用于存储因信号量而被唤醒的节点
  // 返回：true 表示成功获取所有信号量，false 表示失败
  // 如果失败，会释放已经获取的信号量
  bool _acquire_all(SmallVector<Node*>&);

  // 释放所有需要释放的信号量
  // 参数：nodes - 用于存储因信号量而被唤醒的节点
  // 释放信号量可能会唤醒其他等待的任务
  void _release_all(SmallVector<Node*>&);

  // 添加一条从当前节点到目标节点的边（依赖关系）
  // 参数：v - 目标节点（后继节点）
  // 当前节点完成后，目标节点的 join_counter 会减 1
  void _precede(Node*);

  // 设置 join 计数器的初始值
  // 根据前驱节点的数量和类型（强依赖/弱依赖）计算初始值
  void _set_up_join_counter();

  // 重新抛出存储的异常
  // 如果 _exception_ptr 不为空，重新抛出该异常
  void _rethrow_exception();

  // 从当前节点的后继列表中移除指定节点
  // 参数：node - 要移除的后继节点
  void _remove_successors(Node*);

  // 从当前节点的前驱列表中移除指定节点
  // 参数：node - 要移除的前驱节点
  void _remove_predecessors(Node*);

  // 获取根节点的 join 计数器引用
  // 如果有父节点，返回父节点的 join_counter
  // 否则返回 Topology 的 join_counter
  // 用于子流任务向上传播完成信号
  std::atomic<size_t>& _root_join_counter();
};

// ----------------------------------------------------------------------------
// Node Object Pool
// ----------------------------------------------------------------------------

/**
@private
*/
#ifdef TF_ENABLE_TASK_POOL
inline ObjectPool<Node> _task_pool;
#endif

/**
@private
*/
template <typename... ArgsT>
TF_FORCE_INLINE Node* animate(ArgsT&&... args) {
#ifdef TF_ENABLE_TASK_POOL
  return _task_pool.animate(std::forward<ArgsT>(args)...);
#else
  return new Node(std::forward<ArgsT>(args)...);
#endif
}

/**
@private
*/
TF_FORCE_INLINE void recycle(Node* ptr) {
#ifdef TF_ENABLE_TASK_POOL
  _task_pool.recycle(ptr);
#else
  delete ptr;
#endif
}

// ----------------------------------------------------------------------------
// Definition for Node::Static
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Static::Static(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Runtime
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Runtime::Runtime(C&& c) : work {std::forward<C>(c)} {
}

// Constructor
template <typename C>
Node::NonpreemptiveRuntime::NonpreemptiveRuntime(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Subflow
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Subflow::Subflow(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Condition
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Condition::Condition(C&& c) : work {std::forward<C>(c)} {
}                                        

// ----------------------------------------------------------------------------
// Definition for Node::MultiCondition
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::MultiCondition::MultiCondition(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::Module
// ----------------------------------------------------------------------------

// Constructor
template <typename T>
inline Node::Module::Module(T& obj) : graph{ obj.graph() } {
}

// ----------------------------------------------------------------------------
// Definition for Node::Async
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Async::Async(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node::DependentAsync
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::DependentAsync::DependentAsync(C&& c) : work {std::forward<C>(c)} {
}

// ----------------------------------------------------------------------------
// Definition for Node
// ----------------------------------------------------------------------------

// Constructor
template <typename... Args>
Node::Node(
  nstate_t nstate,
  estate_t estate,
  const TaskParams& params,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _nstate       {nstate},
  _estate       {estate},
  _name         {params.name},
  _data         {params.data},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Constructor
template <typename... Args>
Node::Node(
  nstate_t nstate,
  estate_t estate,
  const DefaultTaskParams&,
  Topology* topology, 
  Node* parent, 
  size_t join_counter,
  Args&&... args
) :
  _nstate       {nstate},
  _estate       {estate},
  _topology     {topology},
  _parent       {parent},
  _join_counter {join_counter},
  _handle       {std::forward<Args>(args)...} {
}

// Procedure: _precede
/*
u successor   layout: s1, s2, s3, p1, p2 (num_successors = 3)
v predecessor layout: s1, p1, p2

add a new successor: u->v
u successor   layout: 
  s1, s2, s3, p1, p2, v (push_back v)
  s1, s2, s3, v, p2, p1 (swap adj[num_successors] with adj[n-1])
v predecessor layout: 
  s1, p1, p2, u         (push_back u)
*/ 
inline void Node::_precede(Node* v) {
  _edges.push_back(v);
  std::swap(_edges[_num_successors++], _edges[_edges.size() - 1]);
  v->_edges.push_back(this);
}

// Function: _remove_successors
inline void Node::_remove_successors(Node* node) {
  auto sit = std::remove(_edges.begin(), _edges.begin() + _num_successors, node);
  size_t new_num_successors = std::distance(_edges.begin(), sit);
  std::move(_edges.begin() + _num_successors, _edges.end(), sit);
  _edges.resize(_edges.size() - (_num_successors - new_num_successors));
  _num_successors = new_num_successors;
}

// Function: _remove_predecessors
inline void Node::_remove_predecessors(Node* node) {
  _edges.erase( 
    std::remove(_edges.begin() + _num_successors, _edges.end(), node), _edges.end()
  );
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _num_successors;
}

// Function: predecessors
inline size_t Node::num_predecessors() const {
  return _edges.size() - _num_successors;
}

// Function: num_weak_dependencies
inline size_t Node::num_weak_dependencies() const {
  size_t n = 0;
  for(size_t i=_num_successors; i<_edges.size(); i++) {
    n += _edges[i]->_is_conditioner();
  }
  return n;
}

// Function: _root_join_counter
// not supposed to be called by async task
TF_FORCE_INLINE std::atomic<size_t>& Node::_root_join_counter() {
  return (_parent) ? _parent->_join_counter : _topology->_join_counter; 
}

// Function: num_strong_dependencies
inline size_t Node::num_strong_dependencies() const {
  size_t n = 0;
  for(size_t i=_num_successors; i<_edges.size(); i++) {
    n += !_edges[i]->_is_conditioner();
  }
  return n;
}

// Function: name
inline const std::string& Node::name() const {
  return _name;
}

// Function: _is_conditioner
inline bool Node::_is_conditioner() const {
  return _handle.index() == Node::CONDITION ||
         _handle.index() == Node::MULTI_CONDITION;
}

// Function: _is_preempted
inline bool Node::_is_preempted() const {
  return _nstate & NSTATE::PREEMPTED;
}

// Function: _is_cancelled
// we currently only support cancellation of taskflow (no async task)
inline bool Node::_is_cancelled() const {
  return (_topology && (_topology->_estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED)) 
         ||
         (_parent && (_parent->_estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED));
}

// Procedure: _set_up_join_counter
inline void Node::_set_up_join_counter() {
  size_t c = 0;
  //for(auto p : _predecessors) {
  for(size_t i=_num_successors; i<_edges.size(); i++) {
    bool is_cond = _edges[i]->_is_conditioner();
    _nstate = (_nstate + is_cond) | (is_cond * NSTATE::CONDITIONED);  // weak dependency
    c += !is_cond;  // strong dependency
  }
  _join_counter.store(c, std::memory_order_relaxed);
}

// Procedure: _rethrow_exception
inline void Node::_rethrow_exception() {
  if(_exception_ptr) {
    auto e = _exception_ptr;
    _exception_ptr = nullptr;
    std::rethrow_exception(e);
  }
}

// Function: _acquire_all
inline bool Node::_acquire_all(SmallVector<Node*>& nodes) {
  // assert(_semaphores != nullptr);
  auto& to_acquire = _semaphores->to_acquire;
  for(size_t i = 0; i < to_acquire.size(); ++i) {
    if(!to_acquire[i]->_try_acquire_or_wait(this)) {
      for(size_t j = 1; j <= i; ++j) {
        to_acquire[i-j]->_release(nodes);
      }
      return false;
    }
  }
  return true;
}

// Function: _release_all
inline void Node::_release_all(SmallVector<Node*>& nodes) {
  // assert(_semaphores != nullptr);
  auto& to_release = _semaphores->to_release;
  for(const auto& sem : to_release) {
    sem->_release(nodes);
  }
}



// ----------------------------------------------------------------------------
// AnchorGuard
// ----------------------------------------------------------------------------

/**
@private

AnchorGuard 是一个 RAII 风格的守卫类
用于在异常传播过程中标记节点为"锚定"状态
当异常发生时，从抛出异常的节点向上遍历到根节点
所有路径上的节点都会被标记为 ANCHORED
这样可以防止这些节点被其他线程修改或销毁
*/
class AnchorGuard {

  public:

  // 构造函数：将节点标记为锚定状态
  // anchor is at estate as it may be accessed by multiple threads (e.g., corun's
  // parent with tear_down_async's parent).
  // 锚定标志存储在 estate 中，因为它可能被多个线程访问
  // 例如：corun 的父节点和 tear_down_async 的父节点
  AnchorGuard(Node* node) : _node{node} {
    // 使用原子操作设置 ANCHORED 标志
    _node->_estate.fetch_or(ESTATE::ANCHORED, std::memory_order_relaxed);
  }

  // 析构函数：清除节点的锚定状态
  ~AnchorGuard() {
    // 使用原子操作清除 ANCHORED 标志
    _node->_estate.fetch_and(~ESTATE::ANCHORED, std::memory_order_relaxed);
  }

  private:

  // 被守卫的节点指针
  Node* _node;
};


// ----------------------------------------------------------------------------
// Graph definition
// ----------------------------------------------------------------------------

// Function: erase
inline void Graph::_erase(Node* node) {
  erase(
    std::remove_if(begin(), end(), [&](auto& p){ return p.get() == node; }),
    end()
  );
}

/**
@private
*/
template <typename ...ArgsT>
Node* Graph::_emplace_back(ArgsT&&... args) {
  push_back(std::make_unique<Node>(std::forward<ArgsT>(args)...));
  return back().get();
}

// ----------------------------------------------------------------------------
// Graph checker
// ----------------------------------------------------------------------------

/**
@private
 */
template <typename T, typename = void>
struct has_graph : std::false_type {};

/**
@private
 */
template <typename T>
struct has_graph<T, std::void_t<decltype(std::declval<T>().graph())>>
    : std::is_same<decltype(std::declval<T>().graph()), Graph&> {};

/**
 * @brief determines if the given type has a member function `Graph& graph()`
 *
 * This trait determines if the provided type `T` contains a member function
 * with the exact signature `tf::Graph& graph()`. It uses SFINAE and `std::void_t`
 * to detect the presence of the member function and its return type.
 *
 * @tparam T The type to inspect.
 * @retval true If the type `T` has a member function `tf::Graph& graph()`.
 * @retval false Otherwise.
 *
 * Example usage:
 * @code
 *
 * struct A {
 *   tf::Graph& graph() { return my_graph; };
 *   tf::Graph my_graph;
 *
 *   // other custom members to alter my_graph
 * };
 *
 * struct C {}; // No graph function
 *
 * static_assert(has_graph_v<A>, "A has graph()");
 * static_assert(!has_graph_v<C>, "C does not have graph()");
 * @endcode
 */
template <typename T>
constexpr bool has_graph_v = has_graph<T>::value;

// ----------------------------------------------------------------------------
// detailed helper functions
// ----------------------------------------------------------------------------

namespace detail {

/**
@private
*/
template <typename T>
TF_FORCE_INLINE Node* get_node_ptr(T& node) {
  using U = std::decay_t<T>;
  if constexpr (std::is_same_v<U, Node*>) {
    return node;
  } 
  else if constexpr (std::is_same_v<U, std::unique_ptr<Node>>) {
    return node.get();
  } 
  else {
    static_assert(dependent_false_v<T>, "Unsupported type for get_node_ptr");
  }
} 

}  // end of namespace tf::detail ---------------------------------------------


}  // end of namespace tf. ----------------------------------------------------



