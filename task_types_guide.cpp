// ============================================================================
// Taskflow 任务类型完整指南
// ============================================================================
// 本程序演示 Taskflow 中所有主要任务类型的用法和使用场景
//
// 任务类型列表：
//   1. Static Task         - 静态任务（最基本的任务类型）
//   2. Runtime Task        - 运行时任务（可抢占，可动态调度）
//   3. NonpreemptiveRuntime- 非抢占式运行时任务
//   4. Subflow Task        - 子流任务（动态创建子任务图）
//   5. Condition Task      - 条件任务（单条件分支）
//   6. MultiCondition Task - 多条件任务（多分支跳转）
//   7. Module Task         - 模块任务（组合其他 Taskflow）
//   8. Async Task          - 异步任务（独立执行，返回 future）
//   9. DependentAsync Task - 依赖异步任务（有依赖关系的异步任务）
// ============================================================================

#include <taskflow/taskflow.hpp>
#include <iostream>
#include <vector>
#include <numeric>

// ============================================================================
// 1. Static Task - 静态任务
// ============================================================================
// 【定义】：最基本的任务类型，执行一个简单的函数
// 【特点】：
//   - 无参数，无返回值（或返回 void）
//   - 任务图在编译时确定（静态）
//   - 不能动态创建新任务
//   - 性能最高，开销最小
// 【使用场景】：
//   - 简单的计算任务
//   - 数据处理
//   - I/O 操作
//   - 大部分常规任务
// ============================================================================
void demo_static_task() {
  std::cout << "\n========================================\n";
  std::cout << "1. Static Task - 静态任务\n";
  std::cout << "========================================\n";
  
  tf::Executor executor;
  tf::Taskflow taskflow("Static Task Demo");
  
  // 创建静态任务：lambda 无参数，无返回值
  auto A = taskflow.emplace([](){
    std::cout << "任务 A：执行简单计算\n";
  }).name("A");
  
  auto B = taskflow.emplace([](){
    std::cout << "任务 B：处理数据\n";
  }).name("B");
  
  auto C = taskflow.emplace([](){
    std::cout << "任务 C：输出结果\n";
  }).name("C");
  
  A.precede(B, C);  // A -> B, A -> C
  
  executor.run(taskflow).wait();
  
  std::cout << "\n【总结】：\n";
  std::cout << "- Static Task 是最常用的任务类型\n";
  std::cout << "- 适合大部分场景，性能最高\n";
  std::cout << "- 任务图在运行前确定，不能动态修改\n";
}

// ============================================================================
// 2. Runtime Task - 运行时任务
// ============================================================================
// 【定义】：接受 tf::Runtime& 参数的任务，可以在运行时动态调度其他任务
// 【特点】：
//   - 接受 tf::Runtime& 参数
//   - 可以调用 rt.schedule(task) 动态调度任务
//   - 可抢占（Preemptive）：任务可以被暂停，等待子任务完成
//   - 可以强制调度原本不会执行的任务
// 【使用场景】：
//   - 需要根据运行时条件动态调度任务
//   - 打破静态依赖关系
//   - 强制执行某些任务
//   - 动态工作流
// ============================================================================
void demo_runtime_task() {
  std::cout << "\n========================================\n";
  std::cout << "2. Runtime Task - 运行时任务\n";
  std::cout << "========================================\n";
  
  tf::Executor executor;
  tf::Taskflow taskflow("Runtime Task Demo");
  
  tf::Task A, B, C, D;
  
  // A 是条件任务，返回 0（选择第一个后继）
  A = taskflow.emplace([](){
    std::cout << "任务 A：返回 0，按静态依赖只会执行 B\n";
    return 0;
  }).name("A");
  
  // B 是 Runtime 任务，可以动态调度 C
  B = taskflow.emplace([&C](tf::Runtime& rt){
    std::cout << "任务 B：使用 Runtime 强制调度任务 C\n";
    rt.schedule(C);  // 强制调度 C，打破静态依赖
  }).name("B");
  
  // C 原本不会被执行（A 返回 0，不选择 C）
  C = taskflow.emplace([](){
    std::cout << "任务 C：被 Runtime 强制调度执行\n";
  }).name("C");
  
  D = taskflow.emplace([](){
    std::cout << "任务 D：最后执行\n";
  }).name("D");
  
  A.precede(B, C, D);  // A 的三个后继
  
  executor.run(taskflow).wait();
  
  std::cout << "\n【总结】：\n";
  std::cout << "- Runtime Task 可以动态调度任务\n";
  std::cout << "- 打破静态依赖关系，实现动态工作流\n";
  std::cout << "- 适合需要运行时决策的场景\n";
}

// ============================================================================
// 3. Subflow Task - 子流任务
// ============================================================================
// 【定义】：接受 tf::Subflow& 参数的任务，可以动态创建子任务图
// 【特点】：
//   - 接受 tf::Subflow& 参数
//   - 可以调用 subflow.emplace() 创建子任务
//   - 子任务形成独立的任务图
//   - 父任务等待所有子任务完成后才结束（join）
//   - 子任务默认自动清理（可以通过 retain() 保留）
// 【使用场景】：
//   - 递归算法（快速排序、归并排序、分治算法）
//   - 动态并行（任务数量在运行时确定）
//   - 嵌套并行（任务内部还有并行）
//   - 树形结构的并行处理
// ============================================================================
void demo_subflow_task() {
  std::cout << "\n========================================\n";
  std::cout << "3. Subflow Task - 子流任务\n";
  std::cout << "========================================\n";

  tf::Executor executor;
  tf::Taskflow taskflow("Subflow Task Demo");

  auto A = taskflow.emplace([](){
    std::cout << "任务 A：开始\n";
  }).name("A");

  // B 是 Subflow 任务，动态创建子任务
  // 注意：使用 lambda 捕获列表捕获数据，确保数据生命周期正确
  auto B = taskflow.emplace([data = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}](tf::Subflow& subflow){
    std::cout << "任务 B：创建 3 个子任务...\n";

    auto B1 = subflow.emplace([&data](){
      int sum = std::accumulate(data.begin(), data.end(), 0);
      std::cout << "  子任务 B1：计算和 = " << sum << "\n";
    }).name("B1");

    auto B2 = subflow.emplace([&data](){
      int product = std::accumulate(data.begin(), data.end(), 1, std::multiplies<int>());
      std::cout << "  子任务 B2：计算积 = " << product << "\n";
    }).name("B2");

    auto B3 = subflow.emplace([](){
      std::cout << "  子任务 B3：所有计算完成\n";
    }).name("B3");

    B1.precede(B3);  // B1 -> B3
    B2.precede(B3);  // B2 -> B3

    // 父任务 B 会等待所有子任务（B1, B2, B3）完成
  }).name("B");

  auto C = taskflow.emplace([](){
    std::cout << "任务 C：结束\n";
  }).name("C");

  A.precede(B);
  B.precede(C);

  executor.run(taskflow).wait();

  std::cout << "\n【总结】：\n";
  std::cout << "- Subflow 适合动态创建子任务图\n";
  std::cout << "- 父任务自动等待所有子任务完成（join）\n";
  std::cout << "- 适合递归算法和动态并行\n";
}

// ============================================================================
// 4. Condition Task - 条件任务
// ============================================================================
// 【定义】：返回整数的任务，根据返回值选择后继任务
// 【特点】：
//   - 返回类型是 int
//   - 返回值决定执行哪个后继任务（0 = 第一个，1 = 第二个，...）
//   - 可以实现循环、分支、条件跳转
//   - 支持多个后继任务
// 【使用场景】：
//   - 循环控制（while、for 循环）
//   - 条件分支（if-else）
//   - 状态机
//   - 迭代算法
// ============================================================================
void demo_condition_task() {
  std::cout << "\n========================================\n";
  std::cout << "4. Condition Task - 条件任务\n";
  std::cout << "========================================\n";

  tf::Executor executor;
  tf::Taskflow taskflow("Condition Task Demo");

  int counter = 0;

  auto A = taskflow.emplace([&](){
    std::cout << "任务 A：初始化 counter = 0\n";
    counter = 0;
  }).name("A");

  auto B = taskflow.emplace([&](){
    std::cout << "任务 B：counter++ = " << ++counter << "\n";
  }).name("B");

  // C 是条件任务，返回 int
  auto C = taskflow.emplace([&](){
    std::cout << "任务 C：检查 counter = " << counter;
    if(counter < 5) {
      std::cout << " -> 继续循环（返回 0，跳转到 B）\n";
      return 0;  // 返回 0：选择第一个后继（B）
    }
    std::cout << " -> 退出循环（返回 1，跳转到 D）\n";
    return 1;  // 返回 1：选择第二个后继（D）
  }).name("C");

  auto D = taskflow.emplace([&](){
    std::cout << "任务 D：循环结束，counter = " << counter << "\n";
  }).name("D");

  A.precede(B);
  B.precede(C);
  C.precede(B);  // 第一个后继：循环回 B
  C.precede(D);  // 第二个后继：跳转到 D

  executor.run(taskflow).wait();

  std::cout << "\n【总结】：\n";
  std::cout << "- Condition Task 实现循环和分支\n";
  std::cout << "- 返回值决定执行哪个后继任务\n";
  std::cout << "- 适合迭代算法和状态机\n";
}

// ============================================================================
// 5. MultiCondition Task - 多条件任务
// ============================================================================
// 【定义】：返回整数容器的任务，可以同时选择多个后继任务
// 【特点】：
//   - 返回类型是 tf::SmallVector<int>
//   - 可以同时选择多个后继任务执行
//   - 比 Condition Task 更灵活
// 【使用场景】：
//   - 需要同时触发多个分支
//   - 复杂的控制流
//   - 多路分发
// ============================================================================
void demo_multi_condition_task() {
  std::cout << "\n========================================\n";
  std::cout << "5. MultiCondition Task - 多条件任务\n";
  std::cout << "========================================\n";

  tf::Executor executor;
  tf::Taskflow taskflow("MultiCondition Task Demo");

  // A 是多条件任务，返回 SmallVector<int>
  auto A = taskflow.emplace([]() -> tf::SmallVector<int> {
    std::cout << "任务 A：同时选择后继 0 和 2（B 和 D）\n";
    return {0, 2};  // 选择第 0 个和第 2 个后继
  }).name("A");

  auto B = taskflow.emplace([](){
    std::cout << "  任务 B：被选择执行\n";
  }).name("B");

  auto C = taskflow.emplace([](){
    std::cout << "  任务 C：不会执行\n";
  }).name("C");

  auto D = taskflow.emplace([](){
    std::cout << "  任务 D：被选择执行\n";
  }).name("D");

  A.precede(B, C, D);  // A 有三个后继

  executor.run(taskflow).wait();

  std::cout << "\n【总结】：\n";
  std::cout << "- MultiCondition Task 可以同时选择多个后继\n";
  std::cout << "- 比 Condition Task 更灵活\n";
  std::cout << "- 适合复杂的控制流\n";
}

// ============================================================================
// 6. Module Task - 模块任务
// ============================================================================
// 【定义】：将一个 Taskflow 作为另一个 Taskflow 的子模块
// 【特点】：
//   - 使用 taskflow.composed_of(other_taskflow) 创建
//   - 实现任务图的组合和复用
//   - 模块可以嵌套（模块中包含模块）
//   - 模块任务可以有前驱和后继
// 【使用场景】：
//   - 代码复用（将常用的任务图封装成模块）
//   - 模块化设计（大型任务图分解成小模块）
//   - 层次化结构（构建复杂的任务层次）
//   - 库函数封装（将算法封装成可复用的模块）
// ============================================================================
void demo_module_task() {
  std::cout << "\n========================================\n";
  std::cout << "6. Module Task - 模块任务\n";
  std::cout << "========================================\n";

  tf::Executor executor;

  // 创建一个可复用的模块 Taskflow
  tf::Taskflow module_tf("Module");
  auto M1 = module_tf.emplace([](){
    std::cout << "  模块任务 M1\n";
  }).name("M1");

  auto M2 = module_tf.emplace([](){
    std::cout << "  模块任务 M2\n";
  }).name("M2");

  auto M3 = module_tf.emplace([](){
    std::cout << "  模块任务 M3\n";
  }).name("M3");

  M1.precede(M3);
  M2.precede(M3);

  // 创建主 Taskflow，使用模块
  tf::Taskflow main_tf("Main");

  auto A = main_tf.emplace([](){
    std::cout << "任务 A：开始\n";
  }).name("A");

  // 将 module_tf 作为模块任务插入
  auto module_task = main_tf.composed_of(module_tf).name("Module");

  auto B = main_tf.emplace([](){
    std::cout << "任务 B：结束\n";
  }).name("B");

  A.precede(module_task);
  module_task.precede(B);

  executor.run(main_tf).wait();

  std::cout << "\n【总结】：\n";
  std::cout << "- Module Task 实现任务图的组合和复用\n";
  std::cout << "- 适合模块化设计和代码复用\n";
  std::cout << "- 可以构建层次化的任务结构\n";
}

// ============================================================================
// 7. Async Task - 异步任务
// ============================================================================
// 【定义】：独立于 Taskflow 的异步任务，立即开始执行
// 【特点】：
//   - 使用 executor.async() 创建
//   - 返回 std::future，可以获取结果
//   - 立即开始执行，不需要 run()
//   - 不属于任何 Taskflow
//   - 可以在 Subflow/Runtime 中创建
// 【使用场景】：
//   - 独立的后台任务
//   - 不需要依赖关系的任务
//   - 将 Executor 当作线程池使用
//   - 快速启动异步操作
// ============================================================================
void demo_async_task() {
  std::cout << "\n========================================\n";
  std::cout << "7. Async Task - 异步任务\n";
  std::cout << "========================================\n";

  tf::Executor executor;

  // 创建异步任务，返回 future
  std::future<int> fu1 = executor.async([](){
    std::cout << "异步任务 1：计算 1 + 2\n";
    return 1 + 2;
  });

  std::future<int> fu2 = executor.async([](){
    std::cout << "异步任务 2：计算 3 * 4\n";
    return 3 * 4;
  });

  // silent_async 不返回 future
  executor.silent_async([](){
    std::cout << "静默异步任务：不返回结果\n";
  });

  // 获取结果
  std::cout << "异步任务 1 结果：" << fu1.get() << "\n";
  std::cout << "异步任务 2 结果：" << fu2.get() << "\n";

  executor.wait_for_all();  // 等待所有异步任务完成

  std::cout << "\n【总结】：\n";
  std::cout << "- Async Task 独立执行，不属于 Taskflow\n";
  std::cout << "- 返回 future，可以获取结果\n";
  std::cout << "- 适合独立的后台任务\n";
}

// ============================================================================
// 8. DependentAsync Task - 依赖异步任务
// ============================================================================
// 【定义】：有依赖关系的异步任务
// 【特点】：
//   - 使用 executor.dependent_async() 创建
//   - 可以指定依赖关系（前驱任务）
//   - 返回 AsyncTask 句柄和 future
//   - 前驱任务完成后才开始执行
//   - 不需要创建 Taskflow
// 【使用场景】：
//   - 需要依赖关系的异步任务
//   - 动态构建任务图（不使用 Taskflow）
//   - 轻量级的任务依赖
//   - 快速原型开发
// ============================================================================
void demo_dependent_async_task() {
  std::cout << "\n========================================\n";
  std::cout << "8. DependentAsync Task - 依赖异步任务\n";
  std::cout << "========================================\n";

  tf::Executor executor;

  // 创建依赖异步任务
  auto [A, fuA] = executor.dependent_async([](){
    std::cout << "任务 A：第一个任务\n";
  });

  // B 依赖 A
  auto [B, fuB] = executor.dependent_async([](){
    std::cout << "任务 B：依赖 A\n";
  }, A);

  // C 依赖 A
  auto [C, fuC] = executor.dependent_async([](){
    std::cout << "任务 C：依赖 A\n";
  }, A);

  // D 依赖 B 和 C
  auto [D, fuD] = executor.dependent_async([](){
    std::cout << "任务 D：依赖 B 和 C\n";
  }, B, C);

  fuD.get();  // 等待 D 完成

  std::cout << "\n【总结】：\n";
  std::cout << "- DependentAsync Task 支持依赖关系\n";
  std::cout << "- 不需要创建 Taskflow，更轻量级\n";
  std::cout << "- 适合动态构建任务图\n";
}

// ============================================================================
// 9. NonpreemptiveRuntime Task - 非抢占式运行时任务
// ============================================================================
// 【定义】：类似 Runtime Task，但不可抢占
// 【特点】：
//   - 接受 tf::Runtime& 参数
//   - 可以动态调度任务
//   - 不可抢占：任务不会被暂停
//   - 性能比 Runtime Task 稍高（无抢占开销）
// 【使用场景】：
//   - 需要动态调度，但不需要抢占
//   - 性能敏感的场景
//   - 简单的动态工作流
// 【注意】：
//   - 在 Taskflow 3.x 中，Runtime Task 默认是可抢占的
//   - 如果不需要抢占，可以使用普通的 Static Task
// ============================================================================
void demo_nonpreemptive_runtime_task() {
  std::cout << "\n========================================\n";
  std::cout << "9. NonpreemptiveRuntime Task - 非抢占式运行时任务\n";
  std::cout << "========================================\n";

  std::cout << "【说明】：\n";
  std::cout << "- NonpreemptiveRuntime 是内部任务类型\n";
  std::cout << "- 用户通常不需要直接使用\n";
  std::cout << "- Runtime Task 会根据需要自动选择抢占或非抢占模式\n";
  std::cout << "- 如果不需要动态调度，使用 Static Task 即可\n";
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
  std::cout << "╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  Taskflow 任务类型完整指南                                ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n";

  demo_static_task();
  demo_runtime_task();
  demo_subflow_task();
  demo_condition_task();
  demo_multi_condition_task();
  demo_module_task();
  demo_async_task();
  demo_dependent_async_task();
  demo_nonpreemptive_runtime_task();

  std::cout << "\n========================================\n";
  std::cout << "总结：任务类型选择指南\n";
  std::cout << "========================================\n";
  std::cout << "1. 大部分场景：使用 Static Task\n";
  std::cout << "2. 需要动态创建子任务：使用 Subflow Task\n";
  std::cout << "3. 需要循环/分支：使用 Condition Task\n";
  std::cout << "4. 需要动态调度：使用 Runtime Task\n";
  std::cout << "5. 需要代码复用：使用 Module Task\n";
  std::cout << "6. 需要独立异步任务：使用 Async Task\n";
  std::cout << "7. 需要轻量级依赖：使用 DependentAsync Task\n";
  std::cout << "8. 需要多路分发：使用 MultiCondition Task\n";

  return 0;
}

