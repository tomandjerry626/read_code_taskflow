// ============================================================================
// Taskflow 工作窃取模式 vs 优先级线程池 - 性能分析和对比
// ============================================================================
//
// 本程序演示：
//   1. Taskflow 的工作窃取模式特点
//   2. 设置线程优先级在 Taskflow 中的效果
//   3. 与传统优先级线程池的对比
//
// ============================================================================

#include <taskflow/taskflow.hpp>
#include <iostream>
#include <chrono>
#include <thread>

#if defined(__linux__)
  #include <sched.h>
  #include <pthread.h>
  #include <sys/resource.h>
#endif

// ============================================================================
// 辅助函数：设置线程优先级
// ============================================================================

#if defined(__linux__)
// 设置 Linux 线程优先级
bool set_thread_priority(std::thread& thread, int priority) {
  // Linux 支持两种调度策略：
  // 1. SCHED_OTHER（默认）：普通调度，nice 值范围 -20 到 19
  // 2. SCHED_FIFO/SCHED_RR：实时调度，优先级范围 1 到 99
  
  pthread_t native_handle = thread.native_handle();
  
  // 方法 1：使用 nice 值（SCHED_OTHER）
  // nice 值越小，优先级越高（-20 最高，19 最低）
  // 注意：需要 root 权限才能设置负 nice 值
  if(setpriority(PRIO_PROCESS, 0, priority) == 0) {
    std::cout << "成功设置线程 nice 值为 " << priority << "\n";
    return true;
  }
  
  // 方法 2：使用实时调度策略（需要 root 权限）
  // struct sched_param param;
  // param.sched_priority = priority;
  // if(pthread_setschedparam(native_handle, SCHED_FIFO, &param) == 0) {
  //   return true;
  // }
  
  return false;
}

// 获取当前线程优先级
int get_thread_priority() {
  return getpriority(PRIO_PROCESS, 0);
}
#else
bool set_thread_priority(std::thread& thread, int priority) {
  std::cout << "线程优先级设置仅在 Linux 上支持\n";
  return false;
}

int get_thread_priority() {
  return 0;
}
#endif

// ============================================================================
// 自定义 WorkerInterface：为不同 Worker 设置不同优先级
// ============================================================================

class PriorityWorkerInterface : public tf::WorkerInterface {
public:
  void scheduler_prologue(tf::Worker& w) override {
    size_t worker_id = w.id();
    
    std::cout << "Worker " << worker_id << " 启动\n";
    
    #if defined(__linux__)
    // 为前半部分 Worker 设置高优先级（nice = -5）
    // 为后半部分 Worker 设置低优先级（nice = 5）
    int nice_value = (worker_id < 2) ? -5 : 5;
    
    // 注意：设置负 nice 值需要 root 权限
    // 如果没有权限，会设置失败
    if(set_thread_priority(w.thread(), nice_value)) {
      std::cout << "  → Worker " << worker_id 
                << " 优先级设置为 " << nice_value << "\n";
    } else {
      std::cout << "  → Worker " << worker_id 
                << " 优先级设置失败（可能需要 root 权限）\n";
    }
    
    // 验证当前优先级
    int current_priority = get_thread_priority();
    std::cout << "  → Worker " << worker_id 
              << " 当前 nice 值：" << current_priority << "\n";
    #endif
  }
  
  void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {
    std::cout << "Worker " << w.id() << " 退出\n";
  }
};

// ============================================================================
// 测试场景 1：CPU 密集型任务
// ============================================================================

void test_cpu_intensive() {
  std::cout << "\n========================================\n";
  std::cout << "测试场景 1：CPU 密集型任务\n";
  std::cout << "========================================\n";
  
  // 创建带优先级的 Executor
  tf::Executor executor(4, tf::make_worker_interface<PriorityWorkerInterface>());
  tf::Taskflow taskflow;
  
  std::atomic<int> counter{0};
  
  // 创建 100 个 CPU 密集型任务
  for(int i = 0; i < 100; i++) {
    taskflow.emplace([&counter, i](){
      // 模拟 CPU 密集型计算
      volatile int sum = 0;
      for(int j = 0; j < 1000000; j++) {
        sum += j;
      }
      
      int current = counter.fetch_add(1);
      if(current % 10 == 0) {
        std::cout << "任务 " << i << " 完成（第 " << current << " 个）\n";
      }
    });
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "\n总执行时间：" << duration.count() << " 毫秒\n";
}

// ============================================================================
// 测试场景 2：演示工作窃取的效果
// ============================================================================

void test_work_stealing_effect() {
  std::cout << "\n========================================\n";
  std::cout << "测试场景 2：工作窃取效果演示\n";
  std::cout << "========================================\n";

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  std::mutex mtx;
  std::map<size_t, int> worker_task_count;  // 统计每个 worker 执行的任务数

  // 创建 100 个任务
  for(int i = 0; i < 100; i++) {
    taskflow.emplace([&mtx, &worker_task_count, i](){
      // 获取当前执行的 worker ID
      // 注意：这需要通过 TLS 或其他方式获取，这里简化处理

      // 模拟短时间计算
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

      std::lock_guard<std::mutex> lock(mtx);
      std::cout << "任务 " << i << " 执行\n";
    });
  }

  executor.run(taskflow).wait();

  std::cout << "\n【观察】：\n";
  std::cout << "- 所有任务最初都放入 _buffers（共享队列）\n";
  std::cout << "- 4 个 worker 竞争从 _buffers 获取任务\n";
  std::cout << "- 任务的后继任务会放入执行它的 worker 的本地队列\n";
  std::cout << "- 其他 worker 可以窃取这些后继任务\n";
  std::cout << "- 最终负载会自动均衡\n";
}

// ============================================================================
// 测试场景 3：Subflow 中的工作窃取
// ============================================================================

void test_subflow_work_stealing() {
  std::cout << "\n========================================\n";
  std::cout << "测试场景 3：Subflow 中的工作窃取\n";
  std::cout << "========================================\n";

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  // 创建一个 Subflow 任务，动态生成大量子任务
  taskflow.emplace([](tf::Subflow& sf){
    std::cout << "主任务：创建 100 个子任务\n";

    for(int i = 0; i < 100; i++) {
      sf.emplace([i](){
        // 模拟计算
        volatile int sum = 0;
        for(int j = 0; j < 100000; j++) {
          sum += j;
        }

        if(i % 20 == 0) {
          std::cout << "  子任务 " << i << " 完成\n";
        }
      });
    }
  });

  auto start = std::chrono::high_resolution_clock::now();
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "\n执行时间：" << duration.count() << " 毫秒\n";

  std::cout << "\n【关键点】：\n";
  std::cout << "- 所有子任务最初放入创建它们的 worker 的本地队列\n";
  std::cout << "- 其他 3 个 worker 会窃取这些子任务\n";
  std::cout << "- 即使设置了线程优先级，任务也会被其他 worker 窃取\n";
  std::cout << "- 这就是为什么线程优先级在 Taskflow 中效果有限\n";
}

int main() {
  std::cout << "╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  Taskflow 工作窃取 vs 优先级线程池 - 性能分析            ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n";

  std::cout << "\n【重要说明】：\n";
  std::cout << "- 本程序需要 root 权限才能设置负 nice 值\n";
  std::cout << "- 运行方式：sudo ./priority_analysis\n";
  std::cout << "- 如果没有 root 权限，优先级设置会失败，但程序仍会运行\n";

  test_cpu_intensive();
  test_work_stealing_effect();
  test_subflow_work_stealing();

  // 打印详细分析
  std::cout << "\n========================================\n";
  std::cout << "详细分析：Taskflow vs 优先级线程池\n";
  std::cout << "========================================\n";

  std::cout << R"(
【1. 传统优先级线程池的特点】

架构：
  ┌─────────────────────────────────────────┐
  │  高优先级队列（High Priority Queue）    │
  │  ├─ Task 1                              │
  │  ├─ Task 2                              │
  │  └─ Task 3                              │
  ├─────────────────────────────────────────┤
  │  低优先级队列（Low Priority Queue）     │
  │  ├─ Task 4                              │
  │  ├─ Task 5                              │
  │  └─ Task 6                              │
  └─────────────────────────────────────────┘
           ↓         ↓         ↓
  ┌────────┐  ┌────────┐  ┌────────┐
  │Worker 0│  │Worker 1│  │Worker 2│
  │High Pri│  │High Pri│  │Low Pri │
  └────────┘  └────────┘  └────────┘

特点：
  ✅ 任务优先级明确：高优先级任务总是先执行
  ✅ 线程优先级有效：高优先级线程获得更多 CPU 时间
  ✅ 可预测性强：适合实时系统
  ❌ 负载不均衡：高优先级线程可能空闲，低优先级线程繁忙
  ❌ 缓存局部性差：任务在不同线程间切换
  ❌ 不适合递归：递归任务会堆积在一个队列中

【2. Taskflow 工作窃取模式的特点】

架构：
  ┌─────────────────────────────────────────┐
  │  共享队列（_buffers）                   │
  │  ├─ 初始任务                            │
  │  └─ 溢出任务                            │
  └─────────────────────────────────────────┘
           ↓         ↓         ↓         ↓
  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  │Worker 0│  │Worker 1│  │Worker 2│  │Worker 3│
  │ _wsq 0 │  │ _wsq 1 │  │ _wsq 2 │  │ _wsq 3 │
  │ [T1]   │  │ [T2]   │  │ [T3]   │  │ [T4]   │
  │ [T5]   │  │ [T6]   │  │ [T7]   │  │ [T8]   │
  └────────┘  └────────┘  └────────┘  └────────┘
      ↑           ↑           ↑           ↑
      └───────────┴───────────┴───────────┘
            可以互相窃取任务

特点：
  ✅ 负载自动均衡：空闲线程会窃取繁忙线程的任务
  ✅ 缓存局部性好：优先执行本地队列的任务
  ✅ 适合递归：子任务放入本地队列，深度优先执行
  ✅ 无锁操作：本地队列的 push/pop 无锁
  ❌ 任务优先级难以保证：任务可能被任何线程执行
  ❌ 线程优先级效果有限：高优先级线程可能窃取低优先级任务
  ❌ 不适合实时系统：执行顺序不可预测

【3. 设置线程优先级在 Taskflow 中的效果】

问题：为什么线程优先级在 Taskflow 中效果有限？

原因 1：任务可以被任何线程窃取
  - 即使任务最初在高优先级线程的队列中
  - 低优先级线程也可以窃取它
  - 最终任务可能在低优先级线程上执行

原因 2：工作窃取的随机性
  - 窃取目标是随机选择的
  - 无法保证高优先级任务只在高优先级线程上执行

原因 3：共享队列的竞争
  - 所有线程竞争从 _buffers 获取任务
  - 线程优先级不影响队列的竞争结果

示例：
  1. 高优先级 Worker 0 从 _buffers 获取任务 A
  2. 任务 A 创建 100 个子任务，放入 Worker 0 的 _wsq
  3. Worker 1, 2, 3（低优先级）窃取这些子任务
  4. 结果：子任务在低优先级线程上执行！

【4. 什么情况下线程优先级有效？】

有效场景：
  ✅ CPU 密集型任务，且任务数量 ≤ 线程数量
     - 每个线程执行一个长时间运行的任务
     - 没有工作窃取发生
     - 线程优先级决定 CPU 时间分配

  ✅ 所有任务都从 main 线程提交，且没有 Subflow/Runtime
     - 任务均匀分布在所有线程的队列中
     - 高优先级线程更快地完成任务

  ✅ 使用 CPU affinity 绑定线程到特定核心
     - 结合线程优先级和 CPU 绑定
     - 可以实现更好的隔离

无效场景：
  ❌ 大量短任务（工作窃取频繁）
  ❌ 使用 Subflow 动态创建子任务
  ❌ 使用 Runtime 动态调度任务
  ❌ 递归算法（任务集中在一个队列）

【5. 性能对比总结】

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

【6. 建议】

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

混合方案：
  → 创建多个 Executor，每个 Executor 不同优先级
  → 高优先级 Executor：少量线程，处理关键任务
  → 低优先级 Executor：多个线程，处理后台任务
  → 避免在同一个 Executor 内设置不同线程优先级
)";

  return 0;
}

