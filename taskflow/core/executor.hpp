#pragma once

#include "observer.hpp"
#include "taskflow.hpp"
#include "async_task.hpp"
#include "freelist.hpp"

/**
@file executor.hpp
@brief executor include file

============================================================================
Taskflow 任务类型完整说明
============================================================================

Taskflow 支持 9 种主要任务类型，每种类型适用于不同的场景：

1. Static Task（静态任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：最基本的任务类型，执行一个简单的函数                  │
   │ 特点：                                                      │
   │   - 无参数，无返回值（或返回 void）                         │
   │   - 任务图在编译时确定（静态）                              │
   │   - 不能动态创建新任务                                      │
   │   - 性能最高，开销最小                                      │
   │ 使用场景：                                                  │
   │   - 简单的计算任务                                          │
   │   - 数据处理                                                │
   │   - I/O 操作                                                │
   │   - 大部分常规任务（最常用）                                │
   │ 示例：                                                      │
   │   auto task = taskflow.emplace([](){                        │
   │     std::cout << "Static Task\n";                           │
   │   });                                                       │
   └─────────────────────────────────────────────────────────────┘

2. Runtime Task（运行时任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：接受 tf::Runtime& 参数，可动态调度其他任务           │
   │ 特点：                                                      │
   │   - 接受 tf::Runtime& 参数                                  │
   │   - 可调用 rt.schedule(task) 动态调度任务                   │
   │   - 可抢占（Preemptive）：任务可被暂停，等待子任务完成      │
   │   - 可强制调度原本不会执行的任务                            │
   │ 使用场景：                                                  │
   │   - 需要根据运行时条件动态调度任务                          │
   │   - 打破静态依赖关系                                        │
   │   - 强制执行某些任务                                        │
   │   - 动态工作流                                              │
   │ 示例：                                                      │
   │   auto task = taskflow.emplace([&](tf::Runtime& rt){        │
   │     rt.schedule(other_task);  // 动态调度                   │
   │   });                                                       │
   └─────────────────────────────────────────────────────────────┘

3. Subflow Task（子流任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：接受 tf::Subflow& 参数，可动态创建子任务图           │
   │ 特点：                                                      │
   │   - 接受 tf::Subflow& 参数                                  │
   │   - 可调用 subflow.emplace() 创建子任务                     │
   │   - 子任务形成独立的任务图                                  │
   │   - 父任务等待所有子任务完成后才结束（join）                │
   │   - 子任务默认自动清理（可通过 retain() 保留）              │
   │ 使用场景：                                                  │
   │   - 递归算法（快速排序、归并排序、分治算法）                │
   │   - 动态并行（任务数量在运行时确定）                        │
   │   - 嵌套并行（任务内部还有并行）                            │
   │   - 树形结构的并行处理                                      │
   │ 示例：                                                      │
   │   auto task = taskflow.emplace([](tf::Subflow& sf){         │
   │     auto A = sf.emplace([](){...});                         │
   │     auto B = sf.emplace([](){...});                         │
   │     A.precede(B);                                           │
   │   });                                                       │
   └─────────────────────────────────────────────────────────────┘

4. Condition Task（条件任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：返回整数的任务，根据返回值选择后继任务                │
   │ 特点：                                                      │
   │   - 返回类型是 int                                          │
   │   - 返回值决定执行哪个后继任务（0=第一个，1=第二个，...）   │
   │   - 可实现循环、分支、条件跳转                              │
   │   - 支持多个后继任务                                        │
   │ 使用场景：                                                  │
   │   - 循环控制（while、for 循环）                             │
   │   - 条件分支（if-else）                                     │
   │   - 状态机                                                  │
   │   - 迭代算法                                                │
   │ 示例：                                                      │
   │   auto cond = taskflow.emplace([&](){                       │
   │     if(counter < 10) return 0;  // 继续循环                 │
   │     return 1;  // 退出循环                                  │
   │   });                                                       │
   │   cond.precede(loop_body, exit_task);                       │
   └─────────────────────────────────────────────────────────────┘

5. MultiCondition Task（多条件任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：返回整数容器的任务，可同时选择多个后继任务            │
   │ 特点：                                                      │
   │   - 返回类型是 tf::SmallVector<int>                         │
   │   - 可同时选择多个后继任务执行                              │
   │   - 比 Condition Task 更灵活                                │
   │ 使用场景：                                                  │
   │   - 需要同时触发多个分支                                    │
   │   - 复杂的控制流                                            │
   │   - 多路分发                                                │
   │ 示例：                                                      │
   │   auto multi = taskflow.emplace([]() -> tf::SmallVector<int>│
   │     return {0, 2};  // 同时选择第0和第2个后继               │
   │   });                                                       │
   │   multi.precede(task_A, task_B, task_C);                    │
   └─────────────────────────────────────────────────────────────┘

6. Module Task（模块任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：将一个 Taskflow 作为另一个 Taskflow 的子模块         │
   │ 特点：                                                      │
   │   - 使用 taskflow.composed_of(other_taskflow) 创建          │
   │   - 实现任务图的组合和复用                                  │
   │   - 模块可以嵌套（模块中包含模块）                          │
   │   - 模块任务可以有前驱和后继                                │
   │ 使用场景：                                                  │
   │   - 代码复用（将常用的任务图封装成模块）                    │
   │   - 模块化设计（大型任务图分解成小模块）                    │
   │   - 层次化结构（构建复杂的任务层次）                        │
   │   - 库函数封装（将算法封装成可复用的模块）                  │
   │ 示例：                                                      │
   │   tf::Taskflow module_tf;                                   │
   │   // ... 创建 module_tf 的任务 ...                          │
   │   auto module = main_tf.composed_of(module_tf);             │
   └─────────────────────────────────────────────────────────────┘

7. Async Task（异步任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：独立于 Taskflow 的异步任务，立即开始执行             │
   │ 特点：                                                      │
   │   - 使用 executor.async() 创建                              │
   │   - 返回 std::future，可获取结果                            │
   │   - 立即开始执行，不需要 run()                              │
   │   - 不属于任何 Taskflow                                     │
   │   - 可在 Subflow/Runtime 中创建                             │
   │ 使用场景：                                                  │
   │   - 独立的后台任务                                          │
   │   - 不需要依赖关系的任务                                    │
   │   - 将 Executor 当作线程池使用                              │
   │   - 快速启动异步操作                                        │
   │ 示例：                                                      │
   │   std::future<int> fu = executor.async([](){                │
   │     return 42;                                              │
   │   });                                                       │
   │   int result = fu.get();                                    │
   └─────────────────────────────────────────────────────────────┘

8. DependentAsync Task（依赖异步任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：有依赖关系的异步任务                                  │
   │ 特点：                                                      │
   │   - 使用 executor.dependent_async() 创建                    │
   │   - 可指定依赖关系（前驱任务）                              │
   │   - 返回 AsyncTask 句柄和 future                            │
   │   - 前驱任务完成后才开始执行                                │
   │   - 不需要创建 Taskflow                                     │
   │ 使用场景：                                                  │
   │   - 需要依赖关系的异步任务                                  │
   │   - 动态构建任务图（不使用 Taskflow）                       │
   │   - 轻量级的任务依赖                                        │
   │   - 快速原型开发                                            │
   │ 示例：                                                      │
   │   auto [A, fuA] = executor.dependent_async([](){...});      │
   │   auto [B, fuB] = executor.dependent_async([](){...}, A);   │
   │   fuB.get();  // 等待 B 完成                                │
   └─────────────────────────────────────────────────────────────┘

9. NonpreemptiveRuntime Task（非抢占式运行时任务）
   ┌─────────────────────────────────────────────────────────────┐
   │ 定义：类似 Runtime Task，但不可抢占                         │
   │ 特点：                                                      │
   │   - 接受 tf::Runtime& 参数                                  │
   │   - 可动态调度任务                                          │
   │   - 不可抢占：任务不会被暂停                                │
   │   - 性能比 Runtime Task 稍高（无抢占开销）                  │
   │ 使用场景：                                                  │
   │   - 需要动态调度，但不需要抢占                              │
   │   - 性能敏感的场景                                          │
   │   - 简单的动态工作流                                        │
   │ 注意：                                                      │
   │   - 这是内部任务类型，用户通常不需要直接使用                │
   │   - Runtime Task 会根据需要自动选择抢占或非抢占模式         │
   └─────────────────────────────────────────────────────────────┘

============================================================================
任务类型选择指南
============================================================================

【决策树】：

1. 需要动态创建子任务？
   ├─ 是 → 使用 Subflow Task（递归算法、动态并行）
   └─ 否 → 继续

2. 需要循环或条件分支？
   ├─ 是 → 使用 Condition Task 或 MultiCondition Task
   └─ 否 → 继续

3. 需要动态调度任务（打破静态依赖）？
   ├─ 是 → 使用 Runtime Task
   └─ 否 → 继续

4. 需要复用任务图？
   ├─ 是 → 使用 Module Task
   └─ 否 → 继续

5. 需要独立的异步任务？
   ├─ 是 → 使用 Async Task 或 DependentAsync Task
   └─ 否 → 使用 Static Task（默认选择）

【性能排序】（从快到慢）：
  Static Task > Condition Task > Runtime Task > Subflow Task > Module Task

【使用频率】（从高到低）：
  Static Task (90%) > Subflow Task (5%) > Condition Task (3%) > 其他 (2%)

============================================================================
工作窃取模式与线程优先级
============================================================================

【核心问题】：在 Taskflow 的工作窃取模式下，设置线程优先级有效吗？

【简短回答】：效果有限，不建议依赖线程优先级来控制任务执行顺序

【详细分析】：

1. 工作窃取模式的特点
   ┌─────────────────────────────────────────┐
   │  共享队列（_buffers）                   │
   │  ├─ 初始任务                            │
   │  └─ 溢出任务                            │
   └─────────────────────────────────────────┘
            ↓         ↓         ↓         ↓
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │Worker 0│  │Worker 1│  │Worker 2│  │Worker 3│
   │High Pri│  │High Pri│  │Low Pri │  │Low Pri │
   │ _wsq 0 │  │ _wsq 1 │  │ _wsq 2 │  │ _wsq 3 │
   └────────┘  └────────┘  └────────┘  └────────┘
       ↑           ↑           ↑           ↑
       └───────────┴───────────┴───────────┘
             任何 Worker 都可以窃取任何队列的任务

   关键特性：
   - 任务可以被任何 Worker 窃取
   - 窃取目标是随机选择的
   - 无法保证任务只在特定优先级的线程上执行

2. 为什么线程优先级效果有限？

   原因 1：任务迁移
   ┌─────────────────────────────────────────────────────────┐
   │ 1. 高优先级 Worker 0 从 _buffers 获取任务 A            │
   │ 2. 任务 A 创建 100 个子任务，放入 Worker 0 的 _wsq     │
   │ 3. 低优先级 Worker 2, 3 窃取这些子任务                 │
   │ 4. 结果：子任务在低优先级线程上执行！                   │
   └─────────────────────────────────────────────────────────┘

   原因 2：共享队列竞争
   - 所有 Worker 竞争从 _buffers 获取任务
   - 线程优先级不影响队列的竞争结果
   - 低优先级线程也可能先获取到任务

   原因 3：窃取的随机性
   - 窃取目标是随机选择的（通过 _rdgen 随机数生成器）
   - 无法控制哪个线程窃取哪个任务

3. 什么情况下线程优先级有效？

   ✅ 有效场景：

   场景 1：CPU 密集型任务，且任务数量 ≤ 线程数量
   - 每个线程执行一个长时间运行的任务
   - 没有工作窃取发生
   - 线程优先级决定 CPU 时间分配

   场景 2：所有任务都从 main 线程提交，且没有 Subflow/Runtime
   - 任务均匀分布在所有线程的队列中
   - 高优先级线程更快地完成任务

   场景 3：使用 CPU affinity 绑定线程到特定核心
   - 结合线程优先级和 CPU 绑定
   - 可以实现更好的隔离

   ❌ 无效场景：

   - 大量短任务（工作窃取频繁）
   - 使用 Subflow 动态创建子任务
   - 使用 Runtime 动态调度任务
   - 递归算法（任务集中在一个队列）

4. 推荐方案：使用多个独立的 Executor

   ❌ 不推荐：在同一个 Executor 内设置不同线程优先级

   ✅ 推荐：创建多个独立的 Executor，每个 Executor 不同优先级

   示例：

   // 高优先级 Executor：2 个线程，处理关键任务
   class HighPriorityWorker : public tf::WorkerInterface {
     void scheduler_prologue(tf::Worker& w) override {
       // 设置高优先级（Linux: nice = -5）
       setpriority(PRIO_PROCESS, 0, -5);
     }
     void scheduler_epilogue(tf::Worker&, std::exception_ptr) override {}
   };

   tf::Executor high_priority_executor(
     2,
     tf::make_worker_interface<HighPriorityWorker>()
   );

   // 低优先级 Executor：4 个线程，处理后台任务
   class LowPriorityWorker : public tf::WorkerInterface {
     void scheduler_prologue(tf::Worker& w) override {
       // 设置低优先级（Linux: nice = 10）
       setpriority(PRIO_PROCESS, 0, 10);
     }
     void scheduler_epilogue(tf::Worker&, std::exception_ptr) override {}
   };

   tf::Executor low_priority_executor(
     4,
     tf::make_worker_interface<LowPriorityWorker>()
   );

   // 关键任务在高优先级 Executor 上执行
   tf::Taskflow critical_taskflow;
   // ... 创建关键任务 ...
   high_priority_executor.run(critical_taskflow);

   // 后台任务在低优先级 Executor 上执行
   tf::Taskflow background_taskflow;
   // ... 创建后台任务 ...
   low_priority_executor.run(background_taskflow);

   优势：
   - 两个 Executor 完全独立，没有工作窃取
   - 可以保证关键任务的执行时间
   - 线程优先级真正有效

5. 性能对比：工作窃取 vs 优先级线程池

   指标                  | 优先级线程池 | Taskflow 工作窃取
   ---------------------|-------------|------------------
   任务优先级保证        | ⭐⭐⭐⭐⭐    | ⭐
   线程优先级效果        | ⭐⭐⭐⭐⭐    | ⭐⭐
   负载均衡             | ⭐⭐         | ⭐⭐⭐⭐⭐
   缓存局部性           | ⭐⭐         | ⭐⭐⭐⭐⭐
   递归算法性能         | ⭐⭐         | ⭐⭐⭐⭐⭐
   实时性保证           | ⭐⭐⭐⭐⭐    | ⭐
   吞吐量               | ⭐⭐⭐       | ⭐⭐⭐⭐⭐
   可预测性             | ⭐⭐⭐⭐⭐    | ⭐⭐

6. 总结建议

   如果需要严格的优先级控制：
   → 使用传统优先级线程池
   → 或者使用多个独立的 Executor（每个 Executor 不同优先级）

   如果需要高吞吐量和负载均衡：
   → 使用 Taskflow 工作窃取模式
   → 线程优先级作为辅助手段，不要依赖它

   如果需要实时性保证：
   → 使用实时调度策略（SCHED_FIFO）
   → 结合 CPU affinity 绑定核心
   → 避免使用工作窃取
   → 考虑使用专门的实时任务调度器

============================================================================
*/

namespace tf {

// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------

/** 
@class Executor

@brief class to create an executor 

An tf::Executor manages a set of worker threads to run tasks 
using an efficient work-stealing scheduling algorithm.

@code{.cpp}
// Declare an executor and a taskflow
tf::Executor executor;
tf::Taskflow taskflow;

// Add three tasks into the taskflow
tf::Task A = taskflow.emplace([] () { std::cout << "This is TaskA\n"; });
tf::Task B = taskflow.emplace([] () { std::cout << "This is TaskB\n"; });
tf::Task C = taskflow.emplace([] () { std::cout << "This is TaskC\n"; });

// Build precedence between tasks
A.precede(B, C);

tf::Future<void> fu = executor.run(taskflow);
fu.wait();  // block until the execution completes

executor.run(taskflow, [](){ std::cout << "end of 1 run"; }).wait();
executor.run_n(taskflow, 4);
executor.wait_for_all();  // block until all associated executions finish
executor.run_n(taskflow, 4, [](){ std::cout << "end of 4 runs"; }).wait();
executor.run_until(taskflow, [cnt=0] () mutable { return ++cnt == 10; });
@endcode

Most executor methods are @em thread-safe. 
For example, you can submit multiple taskflows to an executor concurrently 
from different threads, while other threads simultaneously create asynchronous tasks.

@code{.cpp}
std::thread t1([&](){ executor.run(taskflow); };
std::thread t2([&](){ executor.async([](){ std::cout << "async task from t2\n"; }); });
executor.async([&](){ std::cout << "async task from the main thread\n"; });
@endcode

@note
To know more about tf::Executor, please refer to @ref ExecuteTaskflow.
*/
class Executor {

  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class Algorithm;

  public:

  /**
  @brief constructs the executor with @c N worker threads

  @param N number of workers (default std::thread::hardware_concurrency)
  @param wix interface class instance to configure workers' behaviors

  The constructor spawns @c N worker threads to run tasks in a
  work-stealing loop. The number of workers must be greater than zero
  or an exception will be thrown.
  By default, the number of worker threads is equal to the maximum
  hardware concurrency returned by std::thread::hardware_concurrency.

  Users can alter the worker behavior, such as changing thread affinity,
  via deriving an instance from tf::WorkerInterface.

  @attention
  An exception will be thrown if executor construction fails.
  */
  explicit Executor(
    size_t N = std::thread::hardware_concurrency(),
    std::shared_ptr<WorkerInterface> wix = nullptr
  );

  /**
  @brief destructs the executor

  The destructor calls Executor::wait_for_all to wait for all submitted
  taskflows to complete and then notifies all worker threads to stop
  and join these threads.
  */
  ~Executor();

  /**
  @brief runs a taskflow once

  @param taskflow a tf::Taskflow object

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow once and returns a tf::Future
  object that eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run(taskflow);
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run(Taskflow& taskflow);

  /**
  @brief runs a moved taskflow once

  @param taskflow a moved tf::Taskflow object

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow once and returns a tf::Future
  object that eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run(std::move(taskflow));
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run(Taskflow&& taskflow);

  /**
  @brief runs a taskflow once and invoke a callback upon completion

  @param taskflow a tf::Taskflow object
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow once and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run(taskflow, [](){ std::cout << "done"; });
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.

  */
  template<typename C>
  tf::Future<void> run(Taskflow& taskflow, C&& callable);

  /**
  @brief runs a moved taskflow once and invoke a callback upon completion

  @param taskflow a moved tf::Taskflow object
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow once and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run(
    std::move(taskflow), [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename C>
  tf::Future<void> run(Taskflow&& taskflow, C&& callable);

  /**
  @brief runs a taskflow for @c N times

  @param taskflow a tf::Taskflow object
  @param N number of runs

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow @c N times and returns a tf::Future
  object that eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_n(taskflow, 2);  // run taskflow 2 times
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run_n(Taskflow& taskflow, size_t N);

  /**
  @brief runs a moved taskflow for @c N times

  @param taskflow a moved tf::Taskflow object
  @param N number of runs

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow @c N times and returns a tf::Future
  object that eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run_n(
    std::move(taskflow), 2    // run the moved taskflow 2 times
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run_n(Taskflow&& taskflow, size_t N);

  /**
  @brief runs a taskflow for @c N times and then invokes a callback

  @param taskflow a tf::Taskflow
  @param N number of runs
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow @c N times and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run(
    taskflow, 2, [](){ std::cout << "done"; }  // runs taskflow 2 times and invoke
                                               // the lambda to print "done"
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename C>
  tf::Future<void> run_n(Taskflow& taskflow, size_t N, C&& callable);

  /**
  @brief runs a moved taskflow for @c N times and then invokes a callback

  @param taskflow a moved tf::Taskflow
  @param N number of runs
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow @c N times and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_n(
    // run the moved taskflow 2 times and invoke the lambda to print "done"
    std::move(taskflow), 2, [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename C>
  tf::Future<void> run_n(Taskflow&& taskflow, size_t N, C&& callable);

  /**
  @brief runs a taskflow multiple times until the predicate becomes true

  @param taskflow a tf::Taskflow
  @param pred a boolean predicate to return @c true for stop

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow multiple times until
  the predicate returns @c true.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    taskflow, [](){ return rand()%10 == 0 }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P>
  tf::Future<void> run_until(Taskflow& taskflow, P&& pred);

  /**
  @brief runs a moved taskflow and keeps running it
         until the predicate becomes true

  @param taskflow a moved tf::Taskflow object
  @param pred a boolean predicate to return @c true for stop

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow multiple times until
  the predicate returns @c true.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    std::move(taskflow), [](){ return rand()%10 == 0 }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P>
  tf::Future<void> run_until(Taskflow&& taskflow, P&& pred);

  /**
  @brief runs a taskflow multiple times until the predicate becomes true and
         then invokes the callback

  @param taskflow a tf::Taskflow
  @param pred a boolean predicate to return @c true for stop
  @param callable a callable object to be invoked after this run completes

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow multiple times until
  the predicate returns @c true and then invokes the given callable when
  the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    taskflow, [](){ return rand()%10 == 0 }, [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P, typename C>
  tf::Future<void> run_until(Taskflow& taskflow, P&& pred, C&& callable);

  /**
  @brief runs a moved taskflow and keeps running
         it until the predicate becomes true and then invokes the callback

  @param taskflow a moved tf::Taskflow
  @param pred a boolean predicate to return @c true for stop
  @param callable a callable object to be invoked after this run completes

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow multiple times until
  the predicate returns @c true and then invokes the given callable when
  the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    std::move(taskflow),
    [](){ return rand()%10 == 0 }, [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P, typename C>
  tf::Future<void> run_until(Taskflow&& taskflow, P&& pred, C&& callable);

  /**
  @brief runs a target graph and waits until it completes using 
         an internal worker of this executor
  
  @tparam T target type which has `tf::Graph& T::graph()` defined
  @param target the target task graph object

  The method coruns a target graph cooperatively with other workers in the same executor
  and block until the execution completes.
  Under cooperative execution, a worker is not preempted. Instead, it continues 
  participating in the work-stealing loop, executing available tasks alongside 
  other workers.  
  
  @code{.cpp}
  tf::Executor executor(2);
  tf::Taskflow taskflow;
  std::array<tf::Taskflow, 1000> others;
  
  std::atomic<size_t> counter{0};
  
  for(size_t n=0; n<1000; n++) {
    for(size_t i=0; i<1000; i++) {
      others[n].emplace([&](){ counter++; });
    }
    taskflow.emplace([&executor, &tf=others[n]](){
      executor.corun(tf);
      //executor.run(tf).wait();  <- blocking the worker without doing anything
      //                             will introduce deadlock
    });
  }
  executor.run(taskflow).wait();
  @endcode 

  The method is thread-safe as long as the target is not concurrently
  ran by two or more threads.

  @attention
  You must call tf::Executor::corun from a worker of the calling executor
  or an exception will be thrown.
  */
  template <typename T>
  void corun(T& target);

  /**
  @brief keeps running the work-stealing loop until the predicate returns `true`
  
  @tparam P predicate type
  @param predicate a boolean predicate to indicate when to stop the loop

  The method keeps the caller worker running in the work-stealing loop
  until the stop predicate becomes true.

  The method keeps the calling worker running available tasks cooperatively 
  with other workers in the same executor and block until the predicate return `true`.
  Under cooperative execution, a worker is not preempted. Instead, it continues 
  participating in the work-stealing loop, executing available tasks alongside 
  other workers.  


  @code{.cpp}
  taskflow.emplace([&](){
    std::future<void> fu = std::async([](){ std::sleep(100s); });
    executor.corun_until([](){
      return fu.wait_for(std::chrono::seconds(0)) == future_status::ready;
    });
  });
  @endcode

  @attention
  You must call tf::Executor::corun_until from a worker of the calling executor
  or an exception will be thrown.
  */
  template <typename P>
  void corun_until(P&& predicate);

  /**
  @brief waits for all tasks to complete

  This member function waits until all submitted tasks
  (e.g., taskflows, asynchronous tasks) to finish.

  @code{.cpp}
  executor.run(taskflow1);
  executor.run_n(taskflow2, 10);
  executor.run_n(taskflow3, 100);
  executor.wait_for_all();  // wait until the above submitted taskflows finish
  @endcode
  */
  void wait_for_all();

  /**
  @brief queries the number of worker threads

  Each worker represents one unique thread spawned by an executor
  upon its construction time.

  @code{.cpp}
  tf::Executor executor(4);
  std::cout << executor.num_workers();    // 4
  @endcode
  */
  size_t num_workers() const noexcept;
  
  /**
  @brief queries the number of workers that are currently not making any stealing attempts
  */
  size_t num_waiters() const noexcept;
  
  /**
  @brief queries the number of queues used in the work-stealing loop
  */
  size_t num_queues() const noexcept;

  /**
  @brief queries the number of running topologies at the time of this call

  When a taskflow is submitted to an executor, a topology is created to store
  runtime metadata of the running taskflow.
  When the execution of the submitted taskflow finishes,
  its corresponding topology will be removed from the executor.

  @code{.cpp}
  executor.run(taskflow);
  std::cout << executor.num_topologies();  // 0 or 1 (taskflow still running)
  @endcode
  */
  size_t num_topologies() const;

  /**
  @brief queries the number of running taskflows with moved ownership

  @code{.cpp}
  executor.run(std::move(taskflow));
  std::cout << executor.num_taskflows();  // 0 or 1 (taskflow still running)
  @endcode
  */
  size_t num_taskflows() const;

  /**
  @brief queries pointer to the calling worker if it belongs to this executor, otherwise returns `nullptr`

  Returns a pointer to the per-worker storage associated with this executor. 
  If the calling thread is not a worker of this executor, the function returns `nullptr`.

  @code{.cpp}
  auto w = executor.this_worker();
  tf::Taskflow taskflow;
  tf::Executor executor;
  executor.async([&](){
    assert(executor.this_worker() != nullptr);
    assert(executor.this_worker()->executor() == &executor);
  });
  @endcode
  */
  Worker* this_worker();
  
  /**
  @brief queries the id of the caller thread within this executor

  Each worker has an unique id in the range of @c 0 to @c N-1 associated with
  its parent executor.
  If the caller thread does not belong to the executor, @c -1 is returned.

  @code{.cpp}
  tf::Executor executor(4);   // 4 workers in the executor
  executor.this_worker_id();  // -1 (main thread is not a worker)

  taskflow.emplace([&](){
    std::cout << executor.this_worker_id();  // 0, 1, 2, or 3
  });
  executor.run(taskflow);
  @endcode
  */
  int this_worker_id() const;
 
  // --------------------------------------------------------------------------
  // Observer methods
  // --------------------------------------------------------------------------

  /**
  @brief constructs an observer to inspect the activities of worker threads

  @tparam Observer observer type derived from tf::ObserverInterface
  @tparam ArgsT argument parameter pack

  @param args arguments to forward to the constructor of the observer

  @return a shared pointer to the created observer

  Each executor manages a list of observers with shared ownership with callers.
  For each of these observers, the two member functions,
  tf::ObserverInterface::on_entry and tf::ObserverInterface::on_exit
  will be called before and after the execution of a task.

  This member function is not thread-safe.
  */
  template <typename Observer, typename... ArgsT>
  std::shared_ptr<Observer> make_observer(ArgsT&&... args);

  /**
  @brief removes an observer from the executor

  This member function is not thread-safe.
  */
  template <typename Observer>
  void remove_observer(std::shared_ptr<Observer> observer);

  /**
  @brief queries the number of observers
  */
  size_t num_observers() const noexcept;

  // --------------------------------------------------------------------------
  // Async Task Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief creates a parameterized asynchronous task to run the given function

  @tparam P task parameter type
  @tparam F callable type

  @param params task parameters
  @param func callable object

  @return a @std_future that will hold the result of the execution
  
  The method creates a parameterized asynchronous task 
  to run the given function and return a @std_future object 
  that eventually will hold the result of the execution.

  @code{.cpp}
  std::future<int> future = executor.async("name", [](){
    std::cout << "create an asynchronous task with a name and returns 1\n";
    return 1;
  });
  future.get();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F>
  auto async(P&& params, F&& func);

  /**
  @brief runs a given function asynchronously

  @tparam F callable type

  @param func callable object

  @return a @std_future that will hold the result of the execution

  The method creates an asynchronous task to run the given function
  and return a @std_future object that eventually will hold the result
  of the return value.

  @code{.cpp}
  std::future<int> future = executor.async([](){
    std::cout << "create an asynchronous task and returns 1\n";
    return 1;
  });
  future.get();
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  auto async(F&& func);

  /**
  @brief similar to tf::Executor::async but does not return a future object

  @tparam F callable type

  @param params task parameters
  @param func callable object

  The method creates a parameterized asynchronous task 
  to run the given function without returning any @std_future object.
  This member function is more efficient than tf::Executor::async 
  and is encouraged to use when applications do not need a @std_future to acquire
  the result or synchronize the execution.

  @code{.cpp}
  executor.silent_async("name", [](){
    std::cout << "create an asynchronous task with a name and no return\n";
  });
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F>
  void silent_async(P&& params, F&& func);
  
  /**
  @brief similar to tf::Executor::async but does not return a future object
  
  @tparam F callable type
  
  @param func callable object

  The method creates an asynchronous task 
  to run the given function without returning any @std_future object.
  This member function is more efficient than tf::Executor::async 
  and is encouraged to use when applications do not need a @std_future to acquire
  the result or synchronize the execution.

  @code{.cpp}
  executor.silent_async([](){
    std::cout << "create an asynchronous task with no return\n";
  });
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  void silent_async(F&& func);

  // --------------------------------------------------------------------------
  // Silent Dependent Async Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish

  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async([](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async([](){ printf("B\n"); });
  executor.silent_dependent_async([](){ printf("C runs after A and B\n"); }, A, B);
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F, typename... Tasks,
    std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish
  
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async("A", [](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async("B", [](){ printf("B\n"); });
  executor.silent_dependent_async(
    "C", [](){ printf("C runs after A and B\n"); }, A, B
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F, typename... Tasks,
    std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async([](){ printf("A\n"); }),
    executor.silent_dependent_async([](){ printf("B\n"); })
  };
  executor.silent_dependent_async(
    [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F, typename I, 
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param params tasks parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)

  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async("A", [](){ printf("A\n"); }),
    executor.silent_dependent_async("B", [](){ printf("B\n"); })
  };
  executor.silent_dependent_async(
    "C", [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F, typename I, 
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, I first, I last);
  
  // --------------------------------------------------------------------------
  // Dependent Async Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish
  
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async([](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async([](){ printf("B\n"); });
  auto [C, fuC] = executor.dependent_async(
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    A, B
  );
  fuC.get();  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename F, typename... Tasks,
    std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  auto dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously
         when the given predecessors finish
  
  @tparam P task parameters type
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask
  
  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async("A", [](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async("B", [](){ printf("B\n"); });
  auto [C, fuC] = executor.dependent_async(
    "C",
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    A, B
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename P, typename F, typename... Tasks,
    std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  auto dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async([](){ printf("A\n"); }),
    executor.silent_dependent_async([](){ printf("B\n"); })
  };
  auto [C, fuC] = executor.dependent_async(
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    array.begin(), array.end()
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename F, typename I,
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam P task parameters type
  @tparam F callable type
  @tparam I iterator type 
  
  @param params task parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async("A", [](){ printf("A\n"); }),
    executor.silent_dependent_async("B", [](){ printf("B\n"); })
  };
  auto [C, fuC] = executor.dependent_async(
    "C",
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    array.begin(), array.end()
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename P, typename F, typename I,
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto dependent_async(P&& params, F&& func, I first, I last);

  private:

  // 保护 _taskflows 列表的互斥锁
  // 用于在多线程环境下安全地添加或移除由执行器管理的 taskflow 对象
  std::mutex _taskflows_mutex;

  // 工作线程池，存储所有工作线程对象
  // 每个工作线程在执行器构造时创建，并在析构时销毁
  // 工作线程数量在构造时确定，运行期间保持不变
  std::vector<Worker> _workers;

  // 通知器对象，用于实现高效的线程唤醒机制
  // 当有新任务到达时，通过通知器唤醒休眠的工作线程
  // 支持 notify_one、notify_all 和 notify_n 等操作
  // 根据编译选项和 C++ 版本选择不同的实现（AtomicNotifier 或 NonblockingNotifier）
  DefaultNotifier _notifier;

#if __cplusplus >= TF_CPP20
  // 当前正在运行的拓扑（topology）数量（C++20 版本使用原子变量）
  // 拓扑表示一个 taskflow 的运行时实例，包含执行状态和元数据
  // 用于 wait_for_all() 等待所有提交的任务完成
  std::atomic<size_t> _num_topologies {0};
#else
  // 条件变量，用于在拓扑计数变化时通知等待的线程（C++17 版本）
  std::condition_variable _topology_cv;

  // 保护 _num_topologies 的互斥锁（C++17 版本）
  std::mutex _topology_mutex;

  // 当前正在运行的拓扑（topology）数量（C++17 版本使用普通变量 + 锁）
  size_t _num_topologies {0};
#endif

  // 由执行器管理的 taskflow 对象列表
  // 当使用 run(std::move(taskflow)) 提交 taskflow 时，执行器会接管其生命周期
  // 这些 taskflow 在执行完成后会被自动清理
  std::list<Taskflow> _taskflows;

  // 中心化任务缓冲区（Freelist），用于存储待执行的任务节点
  // 当工作线程的本地队列满时，任务会溢出到这个中心化缓冲区
  // 当外部线程（非工作线程）提交任务时，任务也会放入这个缓冲区
  // 使用多个桶（bucket）来减少竞争，桶的数量为 floor(log2(N))
  Freelist<Node*> _buffers;

  // 用户自定义的工作线程接口，用于配置工作线程的行为
  // 可以在工作线程进入/退出调度循环时执行自定义操作（如设置线程亲和性）
  std::shared_ptr<WorkerInterface> _worker_interface;

  // 观察者集合，用于监控任务执行过程
  // 观察者可以在任务执行前后收到通知，用于性能分析、日志记录等
  std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

  // 线程 ID 到工作线程对象的映射表
  // 用于快速查找当前线程对应的工作线程对象
  // 在 this_worker() 和 this_worker_id() 等方法中使用
  std::unordered_map<std::thread::id, Worker*> _t2w;

  // 关闭执行器，等待所有任务完成并停止所有工作线程
  void _shutdown();

  // 在任务执行前调用所有观察者的 on_entry 回调
  void _observer_prologue(Worker&, Node*);

  // 在任务执行后调用所有观察者的 on_exit 回调
  void _observer_epilogue(Worker&, Node*);

  // 创建指定数量的工作线程并启动它们
  void _spawn(size_t);

  // 利用阶段（Exploit）：工作线程持续从自己的本地队列中取出并执行任务
  // 这是工作窃取算法的第一阶段，优先执行本地任务以提高缓存局部性
  void _exploit_task(Worker&, Node*&);

  // 探索阶段（Explore）：当本地队列为空时，尝试从其他工作线程的队列中窃取任务
  // 这是工作窃取算法的第二阶段，实现负载均衡
  // 返回 false 表示工作线程应该退出（收到停止信号）
  bool _explore_task(Worker&, Node*&);

  // 调度单个任务节点（从工作线程调用）
  // 如果调用者是本执行器的工作线程，任务会被放入其本地队列；否则放入中心化缓冲区
  void _schedule(Worker&, Node*);

  // 调度单个任务节点（从外部线程调用）
  // 任务会被直接放入中心化缓冲区
  void _schedule(Node*);

  // 设置拓扑（topology）以准备执行
  // 初始化任务图中所有节点的运行时状态，并调度源节点（无前驱的节点）
  void _set_up_topology(Worker*, Topology*);

  // 拆除拓扑（topology）
  // 当拓扑中的所有任务执行完成时调用，清理资源并触发回调
  void _tear_down_topology(Worker&, Topology*);

  // 拆除异步任务节点
  void _tear_down_async(Worker&, Node*, Node*&);

  // 拆除依赖异步任务节点
  void _tear_down_dependent_async(Worker&, Node*, Node*&);

  // 拆除非异步任务节点（普通任务）
  // 递减父节点或拓扑的 join counter，当计数归零时触发后续操作
  void _tear_down_nonasync(Worker&, Node*, Node*&);

  // 拆除调用节点的通用逻辑
  void _tear_down_invoke(Worker&, Node*, Node*&);

  // 原子地增加正在运行的拓扑计数
  void _increment_topology();

  // 原子地减少正在运行的拓扑计数，并在计数归零时通知等待的线程
  void _decrement_topology();

  // 调用（执行）任务节点
  // 这是任务执行的核心入口，根据任务类型分发到不同的 _invoke_* 方法
  void _invoke(Worker&, Node*);

  // 调用静态任务（Static Task）
  void _invoke_static_task(Worker&, Node*);

  // 调用非抢占式运行时任务
  void _invoke_nonpreemptive_runtime_task(Worker&, Node*);

  // 调用条件任务（Condition Task），返回单个整数表示后继分支
  void _invoke_condition_task(Worker&, Node*, SmallVector<int>&);

  // 调用多条件任务（Multi-Condition Task），返回多个整数表示多个后继分支
  void _invoke_multi_condition_task(Worker&, Node*, SmallVector<int>&);

  // 处理依赖异步任务的依赖关系
  void _process_dependent_async(Node*, tf::AsyncTask&, size_t&);

  // 处理任务执行过程中抛出的异常
  void _process_exception(Worker&, Node*);

  // 调度异步任务
  void _schedule_async_task(Node*);

  // 更新任务缓存，用于优化任务调度
  // 将当前缓存的任务调度出去，并将新任务设置为缓存
  void _update_cache(Worker&, Node*&, Node*);

  // 等待任务到达
  // 当工作线程无任务可执行时，进入等待状态（使用两阶段提交协议）
  // 返回 false 表示工作线程应该退出
  bool _wait_for_task(Worker&, Node*&);

  // 调用子流任务（Subflow Task）
  // 返回 true 表示任务被抢占，需要稍后继续执行
  bool _invoke_subflow_task(Worker&, Node*);

  // 调用模块任务（Module Task）
  // 返回 true 表示任务被抢占
  bool _invoke_module_task(Worker&, Node*);

  // 调用模块任务的实现
  bool _invoke_module_task_impl(Worker&, Node*, Graph&);

  // 调用异步任务
  bool _invoke_async_task(Worker&, Node*);

  // 调用依赖异步任务
  bool _invoke_dependent_async_task(Worker&, Node*);

  // 调用运行时任务（Runtime Task）
  bool _invoke_runtime_task(Worker&, Node*);

  // 调用运行时任务的实现（单参数版本）
  bool _invoke_runtime_task_impl(Worker&, Node*, std::function<void(Runtime&)>&);

  // 调用运行时任务的实现（双参数版本，第二个参数表示是否被抢占）
  bool _invoke_runtime_task_impl(Worker&, Node*, std::function<void(Runtime&, bool)>&);

  // 设置任务图（Graph）以准备执行
  // 遍历图中的节点，初始化运行时状态，并返回可以立即执行的节点迭代器
  // 参数：[begin, end) 节点范围，拓扑对象，父节点
  template <typename I>
  I _set_up_graph(I, I, Topology*, Node*);

  // 协作运行直到满足停止条件
  // 工作线程继续参与工作窃取循环，直到谓词返回 true
  // 用于实现 corun_until 功能，避免阻塞工作线程
  template <typename P>
  void _corun_until(Worker&, P&&);

  // 协作运行任务图
  // 工作线程协作执行指定范围内的任务节点
  template <typename I>
  void _corun_graph(Worker&, Node*, I, I);

  // 调度一批任务节点（从工作线程调用）
  // 将 [begin, end) 范围内的任务节点加入调度队列
  template <typename I>
  void _schedule(Worker&, I, I);

  // 调度一批任务节点（从外部线程调用）
  // 将 [begin, end) 范围内的任务节点加入中心化缓冲区
  template <typename I>
  void _schedule(I, I);

  // 调度带有父节点的任务图
  // 用于子流（Subflow）场景，将子任务关联到父任务
  template <typename I>
  void _schedule_graph_with_parent(Worker&, I, I, Node*);

  // 创建异步任务的内部实现
  // 返回 std::future 对象用于获取任务执行结果
  template <typename P, typename F>
  auto _async(P&&, F&&, Topology*, Node*);

  // 创建静默异步任务的内部实现
  // 不返回 std::future，性能更高
  template <typename P, typename F>
  void _silent_async(P&&, F&&, Topology*, Node*);

  // 创建依赖异步任务的内部实现
  // 任务在指定的前驱任务完成后才会执行
  // 返回 AsyncTask 句柄和 std::future 对象
  template <typename P, typename F, typename I,
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto _dependent_async(P&&, F&&, I, I, Topology*, Node*);

  // 创建静默依赖异步任务的内部实现
  // 任务在指定的前驱任务完成后才会执行，但不返回 std::future
  template <typename P, typename F, typename I,
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto _silent_dependent_async(P&&, F&&, I, I, Topology*, Node*);
};

#ifndef DOXYGEN_GENERATING_OUTPUT

// Constructor
inline Executor::Executor(size_t N, std::shared_ptr<WorkerInterface> wix) :
  _workers  (N),
  _notifier (N),
  _buffers  (N),
  _worker_interface(std::move(wix)) {

  if(N == 0) {
    TF_THROW("executor must define at least one worker");
  }
  
  // If spawning N threads fails, shut down any created threads before 
  // rethrowing the exception.
#ifndef TF_DISABLE_EXCEPTION_HANDLING
  try {
#endif
    _spawn(N);
#ifndef TF_DISABLE_EXCEPTION_HANDLING
  }
  catch(...) {
    _shutdown();
    std::rethrow_exception(std::current_exception());
  }
#endif

  // initialize the default observer if requested
  if(has_env(TF_ENABLE_PROFILER)) {
    TFProfManager::get()._manage(make_observer<TFProfObserver>());
  }
}

// Destructor
inline Executor::~Executor() {
  _shutdown();
}

// Function: _shutdown
inline void Executor::_shutdown() {

  // wait for all topologies to complete
  wait_for_all();

  // shut down the scheduler
  for(size_t i=0; i<_workers.size(); ++i) {
  #if __cplusplus >= TF_CPP20
    _workers[i]._done.test_and_set(std::memory_order_relaxed);
  #else
    _workers[i]._done.store(true, std::memory_order_relaxed);
  #endif
  }
  
  _notifier.notify_all();
  
  // Only join the thread if it is joinable, as std::thread construction 
  // may fail and throw an exception.
  for(auto& w : _workers) {
    if(w._thread.joinable()) {
      w._thread.join();
    }
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const noexcept {
  return _workers.size();
}

// Function: num_waiters
inline size_t Executor::num_waiters() const noexcept {
#if __cplusplus >= TF_CPP20
  return _notifier.num_waiters();
#else
  // Unfortunately, nonblocking notifier does not have an easy way to return
  // the number of workers that are not making stealing attempts.
  return 0;
#endif
}

// Function: num_queues
inline size_t Executor::num_queues() const noexcept {
  return _workers.size() + _buffers.size();
}

// Function: num_topologies
inline size_t Executor::num_topologies() const {
#if __cplusplus >= TF_CPP20
  return _num_topologies.load(std::memory_order_relaxed);
#else
  return _num_topologies;
#endif
}

// Function: num_taskflows
inline size_t Executor::num_taskflows() const {
  return _taskflows.size();
}

// Function: this_worker
inline Worker* Executor::this_worker() {
  auto itr = _t2w.find(std::this_thread::get_id());
  return itr == _t2w.end() ? nullptr : itr->second;
}

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  auto i = _t2w.find(std::this_thread::get_id());
  return i == _t2w.end() ? -1 : static_cast<int>(i->second->_id);
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N) {

  for(size_t id=0; id<N; ++id) {
    _workers[id]._id = id;
    _workers[id]._vtm = id;
    _workers[id]._executor = this;
    _workers[id]._waiter = &_notifier._waiters[id];
    _workers[id]._thread = std::thread([&, &w=_workers[id]] () {

      // initialize the random engine and seed for work-stealing loop
      w._rdgen.seed(static_cast<std::default_random_engine::result_type>(
        std::hash<std::thread::id>()(std::this_thread::get_id()))
      );

      // before entering the work-stealing loop, call the scheduler prologue
      if(_worker_interface) {
        _worker_interface->scheduler_prologue(w);
      }

      Node* t = nullptr;
      std::exception_ptr ptr = nullptr;

      // must use 1 as condition instead of !done because
      // the previous worker may stop while the following workers
      // are still preparing for entering the scheduling loop
#ifndef TF_DISABLE_EXCEPTION_HANDLING
      try {
#endif

        // worker loop
        while(1) {

          // drain out the local queue
          _exploit_task(w, t);

          // steal and wait for tasks
          if(_wait_for_task(w, t) == false) {
            break;
          }
        }

#ifndef TF_DISABLE_EXCEPTION_HANDLING
      } 
      catch(...) {
        ptr = std::current_exception();
      }
#endif
      
      // call the user-specified epilogue function
      if(_worker_interface) {
        _worker_interface->scheduler_epilogue(w, ptr);
      }

    });

    _t2w.emplace(_workers[id]._thread.get_id(), &_workers[id]);
  }
}

// Function: _corun_until
template <typename P>
void Executor::_corun_until(Worker& w, P&& stop_predicate) {

  const size_t MAX_STEALS = ((num_queues() + 1) << 1);
    
  std::uniform_int_distribution<size_t> udist(0, num_queues()-1);
  
  exploit:

  while(!stop_predicate()) {
    
    // here we don't do while-loop to drain out the local queue as it can
    // potentially enter a very deep recursive corun, cuasing stack overflow
    if(auto t = w._wsq.pop(); t) {
      _invoke(w, t);
    }
    else {
      size_t num_steals = 0;
      size_t vtm = w._vtm;

      explore:

      t = (vtm < _workers.size()) ? _workers[vtm]._wsq.steal() : 
                                    _buffers.steal(vtm - _workers.size());

      if(t) {
        _invoke(w, t);
        w._vtm = vtm;
        goto exploit;
      }
      else if(!stop_predicate()) {
        if(++num_steals > MAX_STEALS) {
          std::this_thread::yield();
        }
        vtm = udist(w._rdgen);
        goto explore;
      }
      else {
        break;
      }
    }
  }
}

// Function: _explore_task
// 探索阶段（Explore）：工作窃取算法的核心函数
// 当工作线程的本地队列为空时，尝试从其他线程的队列或中心化缓冲区中窃取任务
// 参数：
//   w - 当前工作线程对象
//   t - 输出参数，窃取到的任务节点指针（如果窃取成功）
// 返回值：
//   true  - 窃取过程正常结束（可能窃取到任务，也可能没有）
//   false - 工作线程收到停止信号，应该退出
inline bool Executor::_explore_task(Worker& w, Node*& t) {

  //assert(!t);

  // 计算最大窃取尝试次数
  // num_queues() = _workers.size() + _buffers.size()（工作线程数 + 缓冲区桶数）
  // 公式：MAX_STEALS = (N + 1) * 2，其中 N 是总队列数
  // 这个值确保每个队列至少被尝试窃取 2 次
  const size_t MAX_STEALS = ((num_queues() + 1) << 1);

  // 创建均匀分布的随机数生成器，范围是 [0, num_queues()-1]
  // 用于随机选择下一个受害者队列
  std::uniform_int_distribution<size_t> udist(0, num_queues()-1);

  // 记录当前已经尝试窃取的次数（包括失败的尝试）
  size_t num_steals = 0;

  // 获取当前工作线程的受害者索引（victim thread index）
  // 这是上一次窃取时使用的受害者索引，优先从这个索引继续窃取
  size_t vtm = w._vtm;

  // 窃取循环：持续尝试从不同的队列中窃取任务
  // Make the worker steal immediately from the assigned victim.
  while(true) {

    // 【核心窃取逻辑】根据受害者索引决定从哪里窃取任务
    //
    // 队列索引布局：
    //   [0, _workers.size()-1]           -> 工作线程的本地队列
    //   [_workers.size(), num_queues()-1] -> 中心化缓冲区的桶
    //
    // If the worker's victim thread is within the worker pool, steal from the worker's queue.
    // Otherwise, steal from the buffer, adjusting the victim index based on the worker pool size.
    t = (vtm < _workers.size())
      ? _workers[vtm]._wsq.steal()                    // 从其他工作线程的队列顶部窃取
      : _buffers.steal(vtm - _workers.size());        // 从中心化缓冲区的某个桶中窃取

    // 窃取成功！
    if(t) {
      // 更新工作线程的受害者索引，下次优先从这个索引继续窃取
      // （因为这个队列可能还有更多任务）
      w._vtm = vtm;
      break;  // 跳出循环，返回窃取到的任务
    }

    // 窃取失败（队列为空），增加失败计数
    // Increment the steal count, and if it exceeds MAX_STEALS, yield the thread.
    // If the number of empty steals reaches MAX_STEALS, exit the loop.
    if (++num_steals > MAX_STEALS) {
      // 已经尝试了足够多次，主动让出 CPU 时间片
      // 避免在所有队列都为空时过度消耗 CPU
      std::this_thread::yield();

      // 如果窃取失败次数超过 150 + MAX_STEALS，彻底放弃
      // 这表示系统中很可能没有任务了，准备进入休眠状态
      if(num_steals > 150 + MAX_STEALS) {
        break;  // 跳出循环，t 仍然是 nullptr
      }
    }

    // 检查工作线程是否收到停止信号
  #if __cplusplus >= TF_CPP20
    if(w._done.test(std::memory_order_relaxed)) {
  #else
    if(w._done.load(std::memory_order_relaxed)) {
  #endif
      return false;  // 返回 false 表示线程应该退出
    }

    // 随机选择下一个受害者索引
    // 使用工作线程自己的随机数生成器，避免多线程竞争
    // Randomely generate a next victim.
    vtm = udist(w._rdgen); //w._rdvtm();
  }

  // 返回 true 表示窃取过程正常结束
  // 注意：此时 t 可能是 nullptr（没有窃取到任务）或有效指针（窃取成功）
  return true;
}

// Procedure: _exploit_task
//
// 利用本地队列执行任务的核心函数 - 工作线程的"快速路径"
//
// 函数作用：
//   当工作线程有任务可执行时（从本地队列或窃取得到），持续执行本地队列中的任务
//   这是工作窃取调度器的"Exploit"阶段
//
// 执行流程：
//   1. 执行当前任务 t（通过 _invoke）
//   2. 从本地队列底部 pop 下一个任务
//   3. 如果有任务，继续循环；如果没有，退出
//
// 为什么是 while 循环？
//   - 优先执行本地队列的任务（缓存局部性好，无竞争）
//   - 避免频繁进入 Explore 阶段（窃取开销大）
//   - 实现深度优先执行（LIFO），适合递归任务
//
// 参数：
//   w - 当前工作线程
//   t - 要执行的任务（输入/输出参数）
//       输入：窃取到的任务或本地队列的任务
//       输出：循环结束后为 nullptr
inline void Executor::_exploit_task(Worker& w, Node*& t) {
  while(t) {
    // 执行当前任务（详见 _invoke 函数的注释）
    _invoke(w, t);

    // 从本地队列底部 pop 下一个任务
    // 注意：这是 LIFO 顺序（后进先出），所有者从底部 pop
    // 窃取者从顶部 steal（FIFO），这是 Chase-Lev 队列的特性
    t = w._wsq.pop();
  }
  // 循环结束：本地队列为空，返回到调用者（通常是 _wait_for_task）
  // 调用者会进入 Explore 阶段尝试窃取任务
}

// Function: _wait_for_task
//
// 等待任务到达的核心函数 - 使用两阶段提交协议（2PC）实现高效且安全的线程休眠
//
// 函数作用：
//   当工作线程无任务可执行时，避免忙等待浪费 CPU，通过 2PC 协议安全地进入休眠状态
//
// 执行流程：
//   1. 调用 _explore_task() 尝试窃取任务
//   2. 如果窃取成功（t != nullptr），返回 true 继续执行
//   3. 如果窃取失败（t == nullptr），进入 2PC 协议：
//      a) prepare_wait() - 第一阶段：声明"我准备休眠"
//      b) 检查三个关键条件，确保所有队列真的为空
//      c) 如果发现任务或停止信号，cancel_wait() 并采取相应行动
//      d) 如果确认无任务，commit_wait() 进入休眠
//   4. 被唤醒后，跳转到 explore_task 重新开始窃取循环
//
// 为什么需要两阶段提交协议（2PC）？
//   问题：如果线程在准备休眠时，其他线程刚好提交了新任务，可能导致：
//     - 所有线程都休眠
//     - 新任务无人执行
//     - 系统死锁
//
//   解决方案：在 prepare_wait() 和 commit_wait() 之间再次检查所有队列
//     - 如果有新任务到达，至少有一个线程会发现并取消休眠
//     - 提交任务的线程会看到 waiter 状态并尝试唤醒休眠线程
//     - 避免所有线程同时休眠导致的死锁
//
// 返回值：
//   - true  : 继续工作循环（窃取到任务或被唤醒）
//   - false : 工作线程应该退出（收到停止信号 w._done == true）
inline bool Executor::_wait_for_task(Worker& w, Node*& t) {

  explore_task:

  // 1. 进入窃取模式：线程 A 首先将自己的角色转变为“窃取者”（thief），并调用 explore_task 尝试从其他线程的队列以及共享队列中窃取任务
  if(_explore_task(w, t) == false) { // 返回false是线程收到停止信号，跳出循环的标志.不是窃取失败的标志，窃取成功或者失败都会返回true.
    return false;
  }
  
  // Go exploit the task if we successfully steal one.
  /*
  窃取成功（简单情况）：如果 explore_task 成功找到了一个任务，线程 A 就重新变回“活跃的”（active）工作线程，并开始执行这个任务。
  段落中还补充了一个关键细节：如果线程 A 之前是系统里最后一个窃取者，那么它在转为活跃状态后，必须负责唤醒另一个休眠的线程来接替它“窃取者”的角色。这确保了只要系统里有活跃的线程，就至少会有一个线程在不断地寻找新任务，从而保持系统的响应性。
  */
  if(t) {
    // 窃取成功，返回true到'_exploit_task'中执行.
    return true;
  }

  // 窃取失败，当前线程要进入2PC等待模式.
  // Entering the 2PC guard as all queues should be empty after many stealing attempts.
  _notifier.prepare_wait(w._waiter);
  
  // ============================================================================
  // 检查条件 #1: 中心化缓冲区（_buffers）必须为空
  // ============================================================================
  // 为什么要检查 _buffers？
  //   1. 外部线程提交的任务会放入 _buffers（主线程调用 run/async）
  //   2. 工作线程队列溢出的任务也会放入 _buffers（_wsq 容量为 1024）
  //
  // 检查时机：在 prepare_wait() 之后
  //   - 如果在 prepare_wait() 之前有任务到达，我们会在这里发现
  //   - 如果在检查过程中有任务到达，提交任务的线程会看到 waiter 状态并唤醒我们
  for(size_t vtm=0; vtm<_buffers.size(); ++vtm) {
    if(!_buffers._buckets[vtm].queue.empty()) {
      // 发现任务！取消休眠，重新开始窃取
      _notifier.cancel_wait(w._waiter);
      w._vtm = vtm + _workers.size();
      goto explore_task;
    }
  }
  
  // ============================================================================
  // 检查条件 #2: 所有工作线程的队列（_wsq）必须为空
  // ============================================================================
  // 为什么要检查其他工作线程的队列？
  //   1. 其他工作线程可能刚刚产生了新任务（执行 Subflow/Runtime 任务时）
  //   2. 这些任务可以被窃取（Chase-Lev 队列支持并发访问）
  //
  // 为什么跳过自己的队列？
  //   - 如果自己的队列不为空，在 _exploit_task() 阶段就已经处理了
  //   - 到这里时，自己的队列一定是空的
  //
  // Note: 使用基于索引的循环避免与 _spawn() 的数据竞争
  // _spawn() 可能在同时初始化其他工作线程的数据结构

  // 检查 ID 小于当前线程的工作线程
  for(size_t vtm=0; vtm<w._id; ++vtm) {
    if(!_workers[vtm]._wsq.empty()) {
      // 发现任务！取消休眠，重新开始窃取
      _notifier.cancel_wait(w._waiter);
      w._vtm = vtm;
      goto explore_task;
    }
  }

  // 跳过自己的队列（w._id）
  // due to the property of the work-stealing queue, we don't need to check
  // the queue of this worker

  // 检查 ID 大于当前线程的工作线程
  for(size_t vtm=w._id+1; vtm<_workers.size(); vtm++) {
    if(!_workers[vtm]._wsq.empty()) {
      // 发现任务！取消休眠，重新开始窃取
      _notifier.cancel_wait(w._waiter);
      w._vtm = vtm;
      goto explore_task;
    }
  }
  
  // ============================================================================
  // 检查条件 #3: 工作线程必须存活（未收到停止信号）
  // ============================================================================
  // 为什么要检查 _done？
  //   1. 执行器正在关闭（用户调用 wait_for_all() 或析构 Executor）
  //   2. 避免线程在关闭过程中进入休眠，导致无法正常退出
  //   3. 检查时机：在所有队列检查之后，确保不会错过任何任务
#if __cplusplus >= TF_CPP20
  if(w._done.test(std::memory_order_relaxed)) {
#else
  if(w._done.load(std::memory_order_relaxed)) {
#endif
    // 收到停止信号，取消休眠并退出
    _notifier.cancel_wait(w._waiter);
    return false;
  }

  // ============================================================================
  // 阶段 4: 提交休眠（commit_wait）
  // ============================================================================
  // 所有检查都通过：
  //   ✓ _buffers 为空
  //   ✓ 所有工作线程的 _wsq 为空
  //   ✓ 线程未收到停止信号
  //
  // 现在可以安全地进入休眠状态
  // commit_wait() 会阻塞当前线程，直到被其他线程唤醒
  //
  // 唤醒条件：
  //   - 其他线程提交了新任务，调用 notify_one() 或 notify_all()
  //   - 执行器关闭，调用 notify_all()
  //
  // 被唤醒后，跳转到 explore_task 重新开始窃取循环
  _notifier.commit_wait(w._waiter);
  goto explore_task;
}

// Function: make_observer
template<typename Observer, typename... ArgsT>
std::shared_ptr<Observer> Executor::make_observer(ArgsT&&... args) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  // use a local variable to mimic the constructor
  auto ptr = std::make_shared<Observer>(std::forward<ArgsT>(args)...);

  ptr->set_up(_workers.size());

  _observers.emplace(std::static_pointer_cast<ObserverInterface>(ptr));

  return ptr;
}

// Procedure: remove_observer
template <typename Observer>
void Executor::remove_observer(std::shared_ptr<Observer> ptr) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  _observers.erase(std::static_pointer_cast<ObserverInterface>(ptr));
}

// Function: num_observers
inline size_t Executor::num_observers() const noexcept {
  return _observers.size();
}

// Procedure: _schedule
//
// ============================================================================
// 调度单个任务节点 - 两层队列架构的核心调度函数
// ============================================================================
//
// 函数作用：
//   将一个任务节点放入队列，等待工作线程执行
//   这是任务调度的核心函数，决定任务放入哪个队列（_buffers 或 _wsq）
//
// ============================================================================
// 何时放入 _buffers（共享队列）？
// ============================================================================
//   1. 外部线程（如 main）调用 executor.run(taskflow)
//      - main 线程不是工作线程，worker._executor != this
//      - 所有初始任务放入 _buffers
//
//   2. 外部线程调用 executor.async(callable)
//      - 异步任务从外部提交
//
//   3. 工作线程的本地队列满了（容量 1024）
//      - 溢出回调自动将任务放入 _buffers
//
//   4. 其他执行器的工作线程提交任务
//      - worker._executor != this
//
// ============================================================================
// 何时放入 _wsq（本地队列）？
// ============================================================================
//   1. Subflow 中创建的子任务
//      - 父任务在工作线程中执行
//      - sf.emplace() 创建的子任务放入当前工作线程的 _wsq
//
//   2. Runtime 中动态调度的任务
//      - rt.schedule() 调度的任务放入当前工作线程的 _wsq
//
//   3. 任务执行后产生的后继任务
//      - 任务 A 完成后，调度后继任务 B
//      - B 放入当前工作线程的 _wsq（或通过 cache 直接执行）
//
//   4. 工作线程自己产生的任务
//      - 任何在工作线程中调用 _schedule 的情况
//
// ============================================================================
// 为什么需要两层队列？设计优势是什么？
// ============================================================================
//
//   【问题】：如果只有一个全局队列（_buffers）会怎样？
//   ┌─────────────────────────────────────────────────────────────┐
//   │ 单队列架构的问题：                                          │
//   │   1. 高竞争：所有线程竞争同一个队列，锁竞争严重             │
//   │   2. 缓存失效：任务在不同线程间跳转，缓存命中率低           │
//   │   3. 扩展性差：线程数增加，性能下降                         │
//   │   4. 不适合递归：FIFO 顺序不适合分治算法                    │
//   └─────────────────────────────────────────────────────────────┘
//
//   【解决方案】：两层队列架构（Work-Stealing）
//   ┌─────────────────────────────────────────────────────────────┐
//   │ 优势 1：减少竞争                                            │
//   │   - 99% 的任务在本地队列，无需竞争全局锁                    │
//   │   - 只有初始任务和溢出任务进入 _buffers                     │
//   │   - 大幅减少锁竞争，提高并发性能                            │
//   │                                                             │
//   │ 优势 2：提高性能                                            │
//   │   - 本地队列无锁操作（所有者访问底部）                      │
//   │   - 缓存友好（任务数据可能还在 L1 缓存中）                  │
//   │   - 减少上下文切换（任务在同一线程连续执行）                │
//   │                                                             │
//   │ 优势 3：负载均衡                                            │
//   │   - 其他线程可以从本地队列窃取任务（从顶部 steal）          │
//   │   - 忙碌的线程分担任务给空闲的线程                          │
//   │   - 自动实现负载均衡，无需手动调度                          │
//   │                                                             │
//   │ 优势 4：适合递归任务                                        │
//   │   - 本地队列 LIFO 顺序（后进先出）                          │
//   │   - 深度优先执行，适合分治算法（快速排序、归并排序等）      │
//   │   - 减少内存占用（深度优先比广度优先占用更少内存）          │
//   │                                                             │
//   │ 优势 5：灵活性                                              │
//   │   - 支持外部提交（_buffers）                                │
//   │   - 支持动态任务生成（_wsq）                                │
//   │   - 支持多种任务模式（Subflow、Runtime、Module 等）         │
//   └─────────────────────────────────────────────────────────────┘
//
// ============================================================================
// 实际执行示例
// ============================================================================
//
//   示例 1：从 main 线程提交任务（你当前的情况）
//   ┌─────────────────────────────────────────────────────────────┐
//   │ int main() {                                                │
//   │   tf::Executor executor(4);  // 4 个工作线程                │
//   │   tf::Taskflow taskflow;                                    │
//   │                                                             │
//   │   for(int i = 0; i < 10; i++) {                            │
//   │     taskflow.emplace([i](){ ... });                        │
//   │   }                                                         │
//   │                                                             │
//   │   executor.run(taskflow).wait();  // ← main 线程调用       │
//   │ }                                                           │
//   │                                                             │
//   │ 执行流程：                                                  │
//   │   1. main 线程不是工作线程（worker._executor != this）      │
//   │   2. 所有 10 个任务放入 _buffers（共享队列）                │
//   │   3. 4 个工作线程从 _buffers 窃取任务执行                   │
//   │   4. 如果任务没有后继，执行完毕                             │
//   │   5. 如果任务有后继，后继任务放入工作线程的 _wsq            │
//   └─────────────────────────────────────────────────────────────┘
//
//   示例 2：Subflow 动态创建子任务（本地队列的主要场景）
//   ┌─────────────────────────────────────────────────────────────┐
//   │ taskflow.emplace([](tf::Subflow& sf){                      │
//   │   // 这段代码在工作线程中执行                               │
//   │   for(int i = 0; i < 100; i++) {                           │
//   │     sf.emplace([i](){ ... });  // ← 工作线程创建子任务      │
//   │   }                                                         │
//   │ });                                                         │
//   │                                                             │
//   │ 执行流程：                                                  │
//   │   1. 父任务从 _buffers 被窃取（main 提交）                  │
//   │   2. Worker 0 执行父任务                                    │
//   │   3. 创建 100 个子任务，放入 Worker 0 的 _wsq               │
//   │   4. Worker 0 优先执行自己的 _wsq（无锁，快速）             │
//   │   5. 其他 Worker 可以窃取 Worker 0 的任务（负载均衡）       │
//   │                                                             │
//   │ 性能优势：                                                  │
//   │   - 100 个子任务大部分在 Worker 0 的 _wsq 中                │
//   │   - 无需竞争 _buffers 的锁                                  │
//   │   - 缓存局部性好（任务数据在 L1 缓存中）                    │
//   │   - 如果 Worker 0 忙不过来，其他 Worker 会帮忙窃取          │
//   └─────────────────────────────────────────────────────────────┘
//
//   示例 3：任务链的执行（cache 优化 + 本地队列）
//   ┌─────────────────────────────────────────────────────────────┐
//   │ A.precede(B);  // A -> B -> C -> D -> E                    │
//   │ B.precede(C);                                              │
//   │ C.precede(D);                                              │
//   │ D.precede(E);                                              │
//   │                                                             │
//   │ 执行流程：                                                  │
//   │   1. 任务 A 从 _buffers 被窃取（main 提交）                 │
//   │   2. Worker 0 执行任务 A                                    │
//   │   3. A 完成，调度后继任务 B                                 │
//   │   4. B 通过 cache 直接执行（尾调用优化）                    │
//   │      或放入 Worker 0 的 _wsq                                │
//   │   5. Worker 0 继续执行 B（缓存热，性能好）                  │
//   │   6. B -> C -> D -> E 可能在同一个线程连续执行              │
//   │   7. 避免了任务在线程间跳转，缓存命中率高                   │
//   │                                                             │
//   │ 性能优势：                                                  │
//   │   - 任务链在同一个线程执行，缓存局部性极好                  │
//   │   - 无需竞争队列，无锁操作                                  │
//   │   - 减少上下文切换                                          │
//   └─────────────────────────────────────────────────────────────┘
//
// 参数：
//   worker - 调用者的工作线程对象
//   node   - 要调度的任务节点
//
inline void Executor::_schedule(Worker& worker, Node* node) {

  // ============================================================================
  // 【关键判断】检查调用者是否是本执行器的工作线程
  // ============================================================================
  // 这个判断决定了任务放入哪个队列：
  //   - 如果是工作线程（worker._executor == this）：
  //     放入本地队列（_wsq）- 快速路径（Fast Path）
  //   - 如果不是工作线程（worker._executor != this）：
  //     放入共享队列（_buffers）- 慢速路径（Slow Path）
  if(worker._executor == this) {
    // ========================================================================
    // 情况 1：调用者是本执行器的工作线程 - 快速路径（Fast Path）
    // ========================================================================
    // 优先放入工作线程的本地队列（_wsq）
    //
    // 为什么优先使用本地队列？
    //   1. 无锁操作：所有者访问底部，无需加锁
    //   2. 缓存友好：任务数据可能还在 L1 缓存中
    //   3. 无竞争：只有所有者 push/pop 底部
    //   4. LIFO 顺序：深度优先，适合递归任务
    //
    // worker._wsq.push() 的第二个参数是溢出回调（on_full callback）
    // 当本地队列满时（容量 1024），会自动调用这个 lambda，将任务放入 _buffers
    worker._wsq.push(node, [&](){ _buffers.push(node); });
    //                     ↑
    //                     溢出回调：_wsq 满时自动调用
    //
    // 溢出场景：
    //   - 工作线程快速产生大量任务（例如：深度递归）
    //   - 本地队列容量 1024 不够用
    //   - 自动溢出到 _buffers，避免任务丢失

    // 唤醒一个休眠的工作线程来执行任务
    // 即使任务放入本地队列，也需要唤醒其他线程（负载均衡）
    _notifier.notify_one();
    return;
  }

  // ============================================================================
  // 情况 2：调用者不是本执行器的工作线程 - 慢速路径（Slow Path）
  // ============================================================================
  // 直接放入中心化缓冲区（_buffers）
  //
  // 适用场景：
  //   1. 外部线程（如 main）调用 executor.run()
  //   2. 外部线程调用 executor.async()
  //   3. 其他执行器的工作线程提交任务
  //
  // 为什么使用 _buffers？
  //   - 外部线程没有本地队列（_wsq）
  //   - 必须使用共享队列
  //   - 工作线程会从 _buffers 窃取任务
  _buffers.push(node);
  _notifier.notify_one();
}

// Procedure: _schedule
// 调度单个任务节点（从外部线程调用）
// 这个重载版本没有 Worker 参数，表示调用者一定不是工作线程
// 参数：
//   node - 要调度的任务节点
inline void Executor::_schedule(Node* node) {
  // 直接放入中心化缓冲区
  // 使用场景：
  //   1. 用户主线程调用 executor.run(taskflow)
  //   2. 外部线程调用 executor.async()
  //   3. 其他非工作线程提交任务
  _buffers.push(node);
  _notifier.notify_one();
}

// Procedure: _schedule
template <typename I>
void Executor::_schedule(Worker& worker, I first, I last) {

  size_t num_nodes = last - first;
  
  if(num_nodes == 0) {
    return;
  }
  
  // NOTE: We cannot use first/last in the for-loop (e.g., for(; first != last; ++first)).
  // This is because when a node v is inserted into the queue, v can run and finish 
  // immediately. If v is the last node in the graph, it will tear down the parent task vector
  // which cause the last ++first to fail. This problem is specific to MSVC which has a stricter
  // iterator implementation in std::vector than GCC/Clang.
  if(worker._executor == this) {
    for(size_t i=0; i<num_nodes; i++) {
      auto node = detail::get_node_ptr(first[i]);
      worker._wsq.push(node, [&](){ _buffers.push(node); });
      _notifier.notify_one();
    }
    return;
  }
  
  // caller is not a worker of this executor - go through the centralized queue
  for(size_t i=0; i<num_nodes; i++) {
    _buffers.push(detail::get_node_ptr(first[i]));
  }
  _notifier.notify_n(num_nodes);
}

// Procedure: _schedule
template <typename I>
inline void Executor::_schedule(I first, I last) {
  
  size_t num_nodes = last - first;

  if(num_nodes == 0) {
    return;
  }

  // NOTE: We cannot use first/last in the for-loop (e.g., for(; first != last; ++first)).
  // This is because when a node v is inserted into the queue, v can run and finish 
  // immediately. If v is the last node in the graph, it will tear down the parent task vector
  // which cause the last ++first to fail. This problem is specific to MSVC which has a stricter
  // iterator implementation in std::vector than GCC/Clang.
  for(size_t i=0; i<num_nodes; i++) {
    _buffers.push(detail::get_node_ptr(first[i]));
  }
  _notifier.notify_n(num_nodes);
}
  
template <typename I>
void Executor::_schedule_graph_with_parent(Worker& worker, I beg, I end, Node* parent) {
  auto send = _set_up_graph(beg, end, parent->_topology, parent);
  parent->_join_counter.fetch_add(send - beg, std::memory_order_relaxed);
  _schedule(worker, beg, send);
}

TF_FORCE_INLINE void Executor::_update_cache(Worker& worker, Node*& cache, Node* node) {
  if(cache) {
    _schedule(worker, cache);
  }
  cache = node;
}
  
// Procedure: _invoke
//
// 执行单个任务的核心函数 - Taskflow 调度器的心脏
//
// 函数作用：
//   执行一个任务节点，并处理其后续任务的调度
//   这是整个 Taskflow 执行引擎最核心的函数
//
// 执行流程（5 个阶段）：
//   【阶段 1】前置检查：抢占、取消、信号量
//   【阶段 2】执行任务：根据任务类型调用相应的执行函数
//   【阶段 3】释放信号量：如果任务持有信号量，释放并调度等待者
//   【阶段 4】调度后继任务：减少后继任务的 join_counter，调度就绪的后继任务
//   【阶段 5】清理工作：更新 topology 的 join_counter，处理任务完成
//
// 关键机制：
//   1. **cache 变量**：用于优化后继任务的调度
//      - 如果只有一个后继任务就绪，将其缓存在 cache 中
//      - 通过 goto begin_invoke 直接执行，避免放入队列的开销
//      - 实现了"尾调用优化"，减少队列操作
//
//   2. **TF_INVOKE_CONTINUATION 宏**：实现任务链式执行
//      - 如果 cache 不为空，直接跳转到 begin_invoke 执行缓存的任务
//      - 避免将单个后继任务放入队列再取出的开销
//
//   3. **goto 标签**：
//      - begin_invoke：函数入口点，用于链式执行
//      - invoke_task：跳过前置检查，直接执行任务（用于抢占恢复）
//
// 参数：
//   worker - 执行任务的工作线程
//   node   - 要执行的任务节点
inline void Executor::_invoke(Worker& worker, Node* node) {

  // ============================================================================
  // TF_INVOKE_CONTINUATION 宏：实现任务链式执行的关键机制
  // ============================================================================
  // 作用：如果 cache 中有后继任务，直接跳转执行，避免队列操作
  //
  // 使用场景：
  //   - 当前任务执行完毕后，如果只有一个后继任务就绪
  //   - 将该后继任务缓存在 cache 中
  //   - 调用此宏直接跳转到 begin_invoke，形成任务链
  //
  // 优化效果：
  //   - 避免 _schedule() 将任务放入队列
  //   - 避免 _wsq.pop() 从队列取出任务
  //   - 减少队列操作的原子操作开销
  //   - 提高缓存局部性（任务数据仍在 L1 缓存中）
  #define TF_INVOKE_CONTINUATION()  \
  if (cache) {                      \
    node = cache;                   \
    goto begin_invoke;              \
  }

  // ============================================================================
  // begin_invoke 标签：函数的真正入口点
  // ============================================================================
  // 为什么需要这个标签？
  //   - 支持 TF_INVOKE_CONTINUATION 宏的跳转
  //   - 实现任务链式执行（尾调用优化）
  //   - 避免递归调用导致的栈溢出
  begin_invoke:

  // cache 变量：用于缓存单个就绪的后继任务
  // 初始化为 nullptr，在调度后继任务时可能被设置
  Node* cache {nullptr};

  // ============================================================================
  // 【阶段 1.1】检查任务是否被抢占
  // ============================================================================
  // 什么是抢占（PREEMPTED）？
  //   - 某些任务（如 Subflow、Runtime）可能需要等待子任务完成
  //   - 工作线程不能阻塞等待，而是将任务标记为 PREEMPTED 并返回
  //   - 当子任务完成后，父任务会被重新调度
  //
  // 为什么跳过前置检查？
  //   - 抢占的任务已经完成了前置检查（取消、信号量）
  //   - 直接跳转到 invoke_task 继续执行任务逻辑
  if(node->_nstate & NSTATE::PREEMPTED) {
    goto invoke_task;
  }

  // ============================================================================
  // 【阶段 1.2】检查任务是否被取消
  // ============================================================================
  // 什么情况下任务会被取消？
  //   - 用户显式取消任务
  //   - 父任务抛出异常，导致子任务被取消
  //   - Topology 被取消
  //
  // 为什么需要 tear_down_invoke？
  //   - 异步任务（Async、DependentAsync）需要回收节点
  //   - 更新 topology 的 join_counter
  //   - 可能触发 topology 的完成回调
  if(node->_is_cancelled()) {
    _tear_down_invoke(worker, node, cache);
    TF_INVOKE_CONTINUATION();  // 如果 cache 有后继任务，继续执行
    return;
  }

  // ============================================================================
  // 【阶段 1.3】获取信号量
  // ============================================================================
  // 什么是信号量（Semaphore）？
  //   - 用于限制并发执行的任务数量
  //   - 例如：限制同时访问数据库的任务数
  //
  // 获取失败怎么办？
  //   - _acquire_all() 返回 false，说明信号量不足
  //   - 当前任务会被放入等待队列（waiters）
  //   - 调度等待队列中的其他任务（这些任务在等待信号量释放）
  //   - 当前任务返回，等待信号量可用时被重新调度
  if(node->_semaphores && !node->_semaphores->to_acquire.empty()) {
    SmallVector<Node*> waiters;
    if(!node->_acquire_all(waiters)) {
      // 获取信号量失败，调度等待队列中的任务
      _schedule(worker, waiters.begin(), waiters.end());
      return;  // 当前任务返回，等待信号量可用
    }
  }

  // ============================================================================
  // invoke_task 标签：执行任务的入口点
  // ============================================================================
  // 为什么需要这个标签？
  //   - 抢占的任务跳过前置检查，直接从这里开始执行
  invoke_task:

  // ============================================================================
  // 【阶段 2】执行任务：根据任务类型调用相应的执行函数
  // ============================================================================
  // conds 变量：用于存储条件任务的返回值
  // 条件任务可以返回多个分支索引，决定执行哪些后继任务
  SmallVector<int> conds;

  // 使用 switch 而不是 if-else：编译器会生成跳转表，性能更好
  // node->_handle 是 std::variant，存储不同类型的任务句柄
  switch(node->_handle.index()) {

    // ------------------------------------------------------------------------
    // Static Task：静态任务（最常见的任务类型）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 简单的可调用对象（函数、lambda）
    //   - 没有子任务，没有条件分支
    //   - 执行完毕后直接调度所有后继任务
    //
    // 执行流程：
    //   1. 调用 observer 的 on_entry（如果有）
    //   2. 执行用户的可调用对象
    //   3. 调用 observer 的 on_exit（如果有）
    case Node::STATIC:{
      _invoke_static_task(worker, node);
    }
    break;

    // ------------------------------------------------------------------------
    // Runtime Task：运行时任务（可以动态调度新任务）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 可以在执行过程中动态创建和调度新任务
    //   - 可以被抢占（如果有子任务需要等待）
    //   - 返回 true 表示任务被抢占，需要等待子任务完成
    //
    // 执行流程：
    //   1. 执行用户的可调用对象（传入 Runtime 对象）
    //   2. 如果有子任务，标记为 PREEMPTED 并返回 true
    //   3. 当子任务完成后，任务会被重新调度
    case Node::RUNTIME:{
      if(_invoke_runtime_task(worker, node)) {
        return;  // 任务被抢占，等待子任务完成
      }
    }
    break;

    // ------------------------------------------------------------------------
    // Non-preemptive Runtime Task：非抢占式运行时任务
    // ------------------------------------------------------------------------
    // 特点：
    //   - 类似 Runtime Task，但不会被抢占
    //   - 必须等待所有子任务完成后才能返回
    //   - 适合需要同步等待的场景
    case Node::NONPREEMPTIVE_RUNTIME:{
      _invoke_nonpreemptive_runtime_task(worker, node);
    }
    break;

    // ------------------------------------------------------------------------
    // Subflow Task：子流任务（可以创建子任务图）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 可以创建一个子任务图（DAG）
    //   - 可以被抢占（如果子任务图未完成）
    //   - 返回 true 表示任务被抢占，需要等待子任务图完成
    //
    // 执行流程：
    //   1. 执行用户的可调用对象（传入 Subflow 对象）
    //   2. 用户在 Subflow 中创建子任务
    //   3. 如果子任务未完成，标记为 PREEMPTED 并返回 true
    //   4. 当子任务完成后，任务会被重新调度
    case Node::SUBFLOW: {
      if(_invoke_subflow_task(worker, node)) {
        return;  // 任务被抢占，等待子任务图完成
      }
    }
    break;

    // ------------------------------------------------------------------------
    // Condition Task：条件任务（单分支条件）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 返回一个整数，表示选择哪个后继任务
    //   - 只能选择一个分支
    //   - 返回值是后继任务的索引（0, 1, 2, ...）
    //
    // 执行流程：
    //   1. 执行用户的可调用对象，获取返回值
    //   2. 将返回值存储在 conds 中
    //   3. 在阶段 4 中，只调度对应索引的后继任务
    case Node::CONDITION: {
      _invoke_condition_task(worker, node, conds);
    }
    break;

    // ------------------------------------------------------------------------
    // Multi-Condition Task：多条件任务（多分支条件）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 返回一个整数集合，表示选择哪些后继任务
    //   - 可以选择多个分支
    //   - 返回值是后继任务的索引集合
    //
    // 执行流程：
    //   1. 执行用户的可调用对象，获取返回值集合
    //   2. 将返回值存储在 conds 中
    //   3. 在阶段 4 中，调度所有对应索引的后继任务
    case Node::MULTI_CONDITION: {
      _invoke_multi_condition_task(worker, node, conds);
    }
    break;

    // ------------------------------------------------------------------------
    // Module Task：模块任务（嵌入另一个 Taskflow）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 将另一个 Taskflow 作为子图嵌入
    //   - 可以被抢占（如果子图未完成）
    //   - 返回 true 表示任务被抢占，需要等待子图完成
    //
    // 执行流程：
    //   1. 设置子图的 topology
    //   2. 调度子图的所有入口任务
    //   3. 如果子图未完成，标记为 PREEMPTED 并返回 true
    //   4. 当子图完成后，任务会被重新调度
    case Node::MODULE: {
      if(_invoke_module_task(worker, node)) {
        return;  // 任务被抢占，等待子图完成
      }
    }
    break;

    // ------------------------------------------------------------------------
    // Async Task：异步任务（独立的异步任务）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 没有前驱任务，可以随时调度
    //   - 执行完毕后需要特殊的清理逻辑
    //   - 返回 true 表示任务被抢占
    //
    // 执行流程：
    //   1. 执行用户的可调用对象
    //   2. 调用 _tear_down_async 清理任务
    //   3. 如果有缓存的后继任务，继续执行
    case Node::ASYNC: {
      if(_invoke_async_task(worker, node)) {
        return;  // 任务被抢占
      }
      _tear_down_async(worker, node, cache);
      TF_INVOKE_CONTINUATION();  // 如果 cache 有后继任务，继续执行
      return;
    }
    break;

    // ------------------------------------------------------------------------
    // Dependent Async Task：依赖异步任务（有前驱的异步任务）
    // ------------------------------------------------------------------------
    // 特点：
    //   - 有前驱任务，需要等待前驱完成
    //   - 执行完毕后需要特殊的清理逻辑
    //   - 返回 true 表示任务被抢占
    //
    // 执行流程：
    //   1. 执行用户的可调用对象
    //   2. 调用 _tear_down_dependent_async 清理任务
    //   3. 如果有缓存的后继任务，继续执行
    case Node::DEPENDENT_ASYNC: {
      if(_invoke_dependent_async_task(worker, node)) {
        return;  // 任务被抢占
      }
      _tear_down_dependent_async(worker, node, cache);
      TF_INVOKE_CONTINUATION();  // 如果 cache 有后继任务，继续执行
      return;
    }
    break;

    // ------------------------------------------------------------------------
    // Monostate (Placeholder)：占位符
    // ------------------------------------------------------------------------
    // 特点：
    //   - 空任务，不执行任何操作
    //   - 用于占位或特殊用途
    default:
    break;
  }

  // ============================================================================
  // 【阶段 3】释放信号量
  // ============================================================================
  // 为什么在这里释放信号量？
  //   - 任务已经执行完毕，不再需要持有信号量
  //   - 释放信号量后，等待的任务可以获取信号量并执行
  //
  // waiters 是什么？
  //   - 等待当前任务释放信号量的任务列表
  //   - 这些任务之前因为信号量不足而被阻塞
  //   - 现在信号量可用，可以调度这些任务
  if(node->_semaphores && !node->_semaphores->to_release.empty()) {
    SmallVector<Node*> waiters;
    node->_release_all(waiters);  // 释放信号量，获取等待者列表
    _schedule(worker, waiters.begin(), waiters.end());  // 调度等待者
  }

  // ============================================================================
  // 重置 join_counter 以支持循环图
  // ============================================================================
  // 为什么需要重置 join_counter？
  //   - 支持循环依赖图（例如：Pipeline）
  //   - 任务可能被多次执行，每次执行前需要重置 join_counter
  //
  // 为什么必须在调度后继任务之前重置？
  //   - 避免与后继任务访问 _predecessors 的竞争条件
  //   - 如果后继任务先执行，可能会修改 _predecessors
  //
  // 为什么使用 fetch_add 而不是直接赋值？
  //   - 用户可能在任务执行过程中显式调度这个任务（例如：Pipeline）
  //   - 直接赋值会覆盖用户的调度，导致 join_counter 不正确
  //   - fetch_add 是原子操作，可以正确累加
  //
  // 计算公式：
  //   重置值 = num_predecessors() - (node->_nstate & ~NSTATE::MASK)
  //   - num_predecessors()：前驱任务的数量
  //   - (node->_nstate & ~NSTATE::MASK)：强依赖的数量（已经在 join_counter 中）
  node->_join_counter.fetch_add(
    node->num_predecessors() - (node->_nstate & ~NSTATE::MASK), std::memory_order_relaxed
  );

  // ============================================================================
  // 【阶段 4】调度后继任务
  // ============================================================================
  // 这是任务执行的关键阶段：决定哪些后继任务可以执行
  //
  // 核心机制：join_counter（连接计数器）
  //   - 每个任务都有一个 join_counter，初始值等于前驱任务的数量
  //   - 当前驱任务完成时，join_counter 减 1
  //   - 当 join_counter 变为 0 时，任务就绪，可以被调度
  //
  // rjc（root_join_counter）：
  //   - Topology 的根连接计数器，记录未完成的任务数量
  //   - 每调度一个后继任务，rjc 加 1
  //   - 当 rjc 变为 0 时，整个 Taskflow 执行完毕
  //
  // 两种调度策略：
  //   1. 条件任务：只调度选中的分支
  //   2. 非条件任务：尝试调度所有后继任务
  switch(auto& rjc = node->_root_join_counter(); node->_handle.index()) {

    // ------------------------------------------------------------------------
    // 条件任务和多条件任务：只调度选中的分支
    // ------------------------------------------------------------------------
    case Node::CONDITION:
    case Node::MULTI_CONDITION: {
      // 遍历条件任务返回的所有分支索引
      for(auto cond : conds) {
        // 检查分支索引是否有效
        if(cond >= 0 && static_cast<size_t>(cond) < node->_num_successors) {
          auto s = node->_edges[cond];  // 获取对应的后继任务

          // 将后继任务的 join_counter 清零
          // 为什么清零？
          //   - 条件任务的后继可能有多个前驱
          //   - 但条件任务只选择一个分支，其他前驱不会执行
          //   - 清零确保后继任务可以立即执行，不等待其他前驱
          s->_join_counter.store(0, std::memory_order_relaxed);

          // 增加 topology 的未完成任务计数
          rjc.fetch_add(1, std::memory_order_relaxed);

          // 调度后继任务（可能缓存在 cache 中）
          _update_cache(worker, cache, s);
        }
      }
    }
    break;

    // ------------------------------------------------------------------------
    // 非条件任务：尝试调度所有后继任务
    // ------------------------------------------------------------------------
    default: {
      // 遍历所有后继任务
      for(size_t i=0; i<node->_num_successors; ++i) {
        auto s = node->_edges[i];

        // 原子地减少后继任务的 join_counter
        // fetch_sub 返回减少前的值
        // 如果返回值为 1，说明减少后变为 0，任务就绪
        if(s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          // 后继任务就绪，可以调度

          // 增加 topology 的未完成任务计数
          rjc.fetch_add(1, std::memory_order_relaxed);

          // 调度后继任务
          // _update_cache 的逻辑：
          //   - 如果 cache 为空，将后继任务缓存在 cache 中
          //   - 如果 cache 不为空，将 cache 中的任务调度到队列，然后缓存新任务
          //
          // 为什么使用 cache？
          //   - 如果只有一个后继任务就绪，可以直接执行（尾调用优化）
          //   - 避免将任务放入队列再取出的开销
          //   - 提高缓存局部性
          _update_cache(worker, cache, s);
        }
      }
    }
    break;
  }


  // ============================================================================
  // 【阶段 5】清理工作
  // ============================================================================
  // 清理当前任务，更新 topology 的状态
  // 如果 topology 完成，触发完成回调
  _tear_down_nonasync(worker, node, cache);

  // 如果 cache 中有后继任务，跳转到 begin_invoke 继续执行
  // 这是尾调用优化的关键：避免递归调用，避免栈溢出
  TF_INVOKE_CONTINUATION();
}

// Procedure: _tear_down_nonasync
inline void Executor::_tear_down_nonasync(Worker& worker, Node* node, Node*& cache) {
  // we must check parent first before subtracting the join counter,
  // or it can introduce data race
  if(auto parent = node->_parent; parent == nullptr) {
    if(node->_topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      _tear_down_topology(worker, node->_topology);
    }
  }
  else {  
    // needs to fetch every data before join counter becomes zero at which
    // the node may be deleted
    auto state = parent->_nstate;
    if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if(state & NSTATE::PREEMPTED) {
        _update_cache(worker, cache, parent);
      }
    }
  }
}

// Procedure: _tear_down_invoke
inline void Executor::_tear_down_invoke(Worker& worker, Node* node, Node*& cache) {
  switch(node->_handle.index()) {
    case Node::ASYNC:
      _tear_down_async(worker, node, cache);
    break;

    case Node::DEPENDENT_ASYNC:
      _tear_down_dependent_async(worker, node, cache);
    break;

    default:
      _tear_down_nonasync(worker, node, cache);
    break;
  }
}

// Procedure: _observer_prologue
inline void Executor::_observer_prologue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_entry(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _observer_epilogue
inline void Executor::_observer_epilogue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_exit(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _process_exception
inline void Executor::_process_exception(Worker&, Node* node) {

  constexpr static auto flag = ESTATE::EXCEPTION | ESTATE::CANCELLED;

  // find the anchor and mark the entire path with exception so recursive
  // or nested tasks can be cancelled properly
  // since exception can come from asynchronous task (with runtime), the node
  // itself can be anchored
  auto anchor = node;
  while(anchor && (anchor->_estate.load(std::memory_order_relaxed) & ESTATE::ANCHORED) == 0) {
    anchor->_estate.fetch_or(flag, std::memory_order_relaxed);
    anchor = anchor->_parent;
  }

  // the exception occurs under a blocking call (e.g., corun, join)
  if(anchor) {
    // multiple tasks may throw, and we only take the first thrown exception
    if((anchor->_estate.fetch_or(flag, std::memory_order_relaxed) & ESTATE::EXCEPTION) == 0) {
      anchor->_exception_ptr = std::current_exception();
      return;
    }
  }
  // otherwise, we simply store the exception in the topology and cancel it
  else if(auto tpg = node->_topology; tpg) {
    // multiple tasks may throw, and we only take the first thrown exception
    if((tpg->_estate.fetch_or(flag, std::memory_order_relaxed) & ESTATE::EXCEPTION) == 0) {
      tpg->_exception_ptr = std::current_exception();
      return;
    }
  }
  
  // for now, we simply store the exception in this node; this can happen in an 
  // execution that does not have any external control to capture the exception,
  // such as silent async task
  node->_exception_ptr = std::current_exception();
}

// Procedure: _invoke_static_task
inline void Executor::_invoke_static_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    std::get_if<Node::Static>(&node->_handle)->work();
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_subflow_task
inline bool Executor::_invoke_subflow_task(Worker& worker, Node* node) {
    
  auto& h = *std::get_if<Node::Subflow>(&node->_handle);
  auto& g = h.subgraph;

  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    
    // set up the subflow
    Subflow sf(*this, worker, node, g);

    // invoke the subflow callable
    _observer_prologue(worker, node);
    TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
      h.work(sf);
    });
    _observer_epilogue(worker, node);
    
    // spawn the subflow if it is joinable and its graph is non-empty
    // implicit join is faster than Subflow::join as it does not involve corun
    if(sf.joinable() && g.size()) {

      // signal the executor to preempt this node
      node->_nstate |= NSTATE::PREEMPTED;

      // set up and schedule the graph
      _schedule_graph_with_parent(worker, g.begin(), g.end(), node);
      return true;
    }
  }
  else {
    node->_nstate &= ~NSTATE::PREEMPTED;
  }

  // the subflow has finished or joined
  if((node->_nstate & NSTATE::RETAIN_SUBFLOW) == 0) {
    g.clear();
  }

  return false;
}

// Procedure: _invoke_condition_task
inline void Executor::_invoke_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::Condition>(&node->_handle)->work;
    conds = { work() };
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_multi_condition_task
inline void Executor::_invoke_multi_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    conds = std::get_if<Node::MultiCondition>(&node->_handle)->work();
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_module_task
inline bool Executor::_invoke_module_task(Worker& w, Node* node) {
  return _invoke_module_task_impl(w, node, std::get_if<Node::Module>(&node->_handle)->graph);  
}

// Procedure: _invoke_module_task_impl
inline bool Executor::_invoke_module_task_impl(Worker& w, Node* node, Graph& graph) {

  // No need to do anything for empty graph
  if(graph.empty()) {
    return false;
  }

  // first entry - not spawned yet
  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    // signal the executor to preempt this node
    node->_nstate |= NSTATE::PREEMPTED;
    _schedule_graph_with_parent(w, graph.begin(), graph.end(), node);
    return true;
  }

  // second entry - already spawned
  node->_nstate &= ~NSTATE::PREEMPTED;

  return false;
}


// Procedure: _invoke_async_task
inline bool Executor::_invoke_async_task(Worker& worker, Node* node) {
  auto& work = std::get_if<Node::Async>(&node->_handle)->work;
  switch(work.index()) {
    // void()
    case 0:
      _observer_prologue(worker, node);
      TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
        std::get_if<0>(&work)->operator()();
      });
      _observer_epilogue(worker, node);
    break;
    
    // void(Runtime&)
    case 1:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<1>(&work))) {
        return true;
      }
    break;
    
    // void(Runtime&, bool)
    case 2:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<2>(&work))) {
        return true;
      }
    break;
  }

  return false;
}

// Procedure: _invoke_dependent_async_task
inline bool Executor::_invoke_dependent_async_task(Worker& worker, Node* node) {
  auto& work = std::get_if<Node::DependentAsync>(&node->_handle)->work;
  switch(work.index()) {
    // void()
    case 0:
      _observer_prologue(worker, node);
      TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
        std::get_if<0>(&work)->operator()();
      });
      _observer_epilogue(worker, node);
    break;
    
    // void(Runtime&) - silent async
    case 1:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<1>(&work))) {
        return true;
      }
    break;

    // void(Runtime&, bool) - async
    case 2:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<2>(&work))) {
        return true;
      }
    break;
  }
  return false;
}

// Function: run
inline tf::Future<void> Executor::run(Taskflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
inline tf::Future<void> Executor::run(Taskflow&& f) {
  return run_n(std::move(f), 1, [](){});
}

// Function: run
template <typename C>
tf::Future<void> Executor::run(Taskflow& f, C&& c) {
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run
template <typename C>
tf::Future<void> Executor::run(Taskflow&& f, C&& c) {
  return run_n(std::move(f), 1, std::forward<C>(c));
}

// Function: run_n
inline tf::Future<void> Executor::run_n(Taskflow& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
inline tf::Future<void> Executor::run_n(Taskflow&& f, size_t repeat) {
  return run_n(std::move(f), repeat, [](){});
}

// Function: run_n
template <typename C>
tf::Future<void> Executor::run_n(Taskflow& f, size_t repeat, C&& c) {
  return run_until(
    f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_n
template <typename C>
tf::Future<void> Executor::run_n(Taskflow&& f, size_t repeat, C&& c) {
  return run_until(
    std::move(f), [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_until
template<typename P>
tf::Future<void> Executor::run_until(Taskflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: run_until
template<typename P>
tf::Future<void> Executor::run_until(Taskflow&& f, P&& pred) {
  return run_until(std::move(f), std::forward<P>(pred), [](){});
}

// Function: run_until
template <typename P, typename C>
tf::Future<void> Executor::run_until(Taskflow& f, P&& p, C&& c) {

  _increment_topology();

  //// Need to check the empty under the lock since subflow task may
  //// define detached blocks that modify the taskflow at the same time
  //bool empty;
  //{
  //  std::lock_guard<std::mutex> lock(f._mutex);
  //  empty = f.empty();
  //}

  // No need to create a real topology but returns an dummy future
  if(f.empty() || p()) {
    c();
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology();
    return tf::Future<void>(promise.get_future());
  }

  // create a topology for this run
  auto t = std::make_shared<Topology>(f, std::forward<P>(p), std::forward<C>(c));

  // need to create future before the topology got torn down quickly
  tf::Future<void> future(t->_promise.get_future(), t);

  // modifying topology needs to be protected under the lock
  {
    std::lock_guard<std::mutex> lock(f._mutex);
    f._topologies.push(t);
    if(f._topologies.size() == 1) {
      _set_up_topology(this_worker(), t.get());
    }
  }

  return future;
}

// Function: run_until
template <typename P, typename C>
tf::Future<void> Executor::run_until(Taskflow&& f, P&& pred, C&& c) {

  std::list<Taskflow>::iterator itr;

  {
    std::scoped_lock<std::mutex> lock(_taskflows_mutex);
    itr = _taskflows.emplace(_taskflows.end(), std::move(f));
    itr->_satellite = itr;
  }

  return run_until(*itr, std::forward<P>(pred), std::forward<C>(c));
}

// Function: corun
template <typename T>
void Executor::corun(T& target) {

  static_assert(has_graph_v<T>, "target must define a member function 'Graph& graph()'");
  
  Worker* w = this_worker();
  if(w == nullptr || w->_executor != this) {
    TF_THROW("corun must be called by a worker of the executor");
  }

  Node anchor;
  _corun_graph(*w, &anchor, target.graph().begin(), target.graph().end());
}

// Function: corun_until
template <typename P>
void Executor::corun_until(P&& predicate) {
  
  Worker* w = this_worker();
  if(w == nullptr || w->_executor != this) {
    TF_THROW("corun_until must be called by a worker of the executor");
  }

  _corun_until(*w, std::forward<P>(predicate));
}

// Procedure: _corun_graph
template <typename I>
void Executor::_corun_graph(Worker& w, Node* p, I first, I last) {

  // empty graph
  if(first == last) {
    return;
  }
  
  // anchor this parent as the blocking point
  {
    AnchorGuard anchor(p);
    _schedule_graph_with_parent(w, first, last, p);
    _corun_until(w, [p] () -> bool { 
      return p->_join_counter.load(std::memory_order_acquire) == 0; }
    );
  }

  // rethrow the exception to the blocker
  p->_rethrow_exception();
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
#if __cplusplus >= TF_CPP20
  _num_topologies.fetch_add(1, std::memory_order_relaxed);
#else
  std::lock_guard<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
#endif
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
#if __cplusplus >= TF_CPP20
  if(_num_topologies.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    _num_topologies.notify_all();
  }
#else
  std::lock_guard<std::mutex> lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
#endif
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
#if __cplusplus >= TF_CPP20
  size_t n = _num_topologies.load(std::memory_order_acquire);
  while(n != 0) {
    _num_topologies.wait(n, std::memory_order_acquire);
    n = _num_topologies.load(std::memory_order_acquire);
  }
#else
  std::unique_lock<std::mutex> lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
#endif
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Worker* w, Topology* tpg) {

  // ---- under taskflow lock ----
  auto& g = tpg->_taskflow._graph;
  
  auto send = _set_up_graph(g.begin(), g.end(), tpg, nullptr);
  tpg->_join_counter.store(send - g.begin(), std::memory_order_relaxed);

  w ? _schedule(*w, g.begin(), send) : _schedule(g.begin(), send);
}

// Function: _set_up_graph
template <typename I>
I Executor::_set_up_graph(I first, I last, Topology* tpg, Node* parent) {

  auto send = first;
  for(; first != last; ++first) {

    auto node = first->get();
    node->_topology = tpg;
    node->_parent = parent;
    node->_nstate = NSTATE::NONE;
    node->_estate.store(ESTATE::NONE, std::memory_order_relaxed);
    node->_set_up_join_counter();
    node->_exception_ptr = nullptr;

    // move source to the first partition
    // root, root, root, v1, v2, v3, v4, ...
    if(node->num_predecessors() == 0) {
      std::iter_swap(send++, first);
    }
  }
  return send;
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Worker& worker, Topology* tpg) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!tpg->_exception_ptr && !tpg->cancelled() && !tpg->_pred()) {
    //assert(tpg->_join_counter == 0);
    std::lock_guard<std::mutex> lock(f._mutex);
    _set_up_topology(&worker, tpg);
  }
  // case 2: the final run of this topology
  else {

    // invoke the callback after each run
    if(tpg->_call != nullptr) {
      tpg->_call();
    }

    // If there is another run (interleave between lock)
    if(std::unique_lock<std::mutex> lock(f._mutex); f._topologies.size()>1) {
      //assert(tpg->_join_counter == 0);

      // Set the promise
      tpg->_carry_out_promise();
      f._topologies.pop();
      tpg = f._topologies.front().get();

      // decrement the topology
      _decrement_topology();

      // set up topology needs to be under the lock or it can
      // introduce memory order error with pop
      _set_up_topology(&worker, tpg);
    }
    else {
      //assert(f._topologies.size() == 1);

      auto fetched_tpg {std::move(f._topologies.front())};
      f._topologies.pop();
      auto satellite {f._satellite};

      lock.unlock();
      
      // Soon after we carry out the promise, there is no longer any guarantee
      // for the lifetime of the associated taskflow.
      fetched_tpg->_carry_out_promise();

      _decrement_topology();

      // remove the taskflow if it is managed by the executor
      // TODO: in the future, we may need to synchronize on wait
      // (which means the following code should the moved before set_value)
      if(satellite) {
        std::scoped_lock<std::mutex> satellite_lock(_taskflows_mutex);
        _taskflows.erase(*satellite);
      }
    }
  }
}

// ############################################################################
// Forward Declaration: Subflow
// ############################################################################

inline void Subflow::join() {

  if(!joinable()) {
    TF_THROW("subflow already joined");
  }
    
  _executor._corun_graph(_worker, _parent, _graph.begin(), _graph.end());
  
  // join here since corun graph may throw exception
  _parent->_nstate |= NSTATE::JOINED_SUBFLOW;
}

#endif




}  // end of namespace tf -----------------------------------------------------






