// ============================================================================
// 混合方案：使用多个 Executor 实现优先级控制
// ============================================================================
//
// 本程序演示如何在 Taskflow 中实现有效的优先级控制：
//   - 创建多个独立的 Executor
//   - 每个 Executor 设置不同的线程优先级
//   - 高优先级 Executor 处理关键任务
//   - 低优先级 Executor 处理后台任务
//
// ============================================================================

#include <taskflow/taskflow.hpp>
#include <iostream>
#include <chrono>
#include <atomic>

#if defined(__linux__)
  #include <sched.h>
  #include <pthread.h>
  #include <sys/resource.h>
#endif

// ============================================================================
// 高优先级 Worker Interface
// ============================================================================

class HighPriorityWorker : public tf::WorkerInterface {
public:
  void scheduler_prologue(tf::Worker& w) override {
    std::cout << "[高优先级] Worker " << w.id() << " 启动\n";
    
    #if defined(__linux__)
    // 设置高优先级（nice = -5，需要 root 权限）
    if(setpriority(PRIO_PROCESS, 0, -5) == 0) {
      std::cout << "  → 成功设置为高优先级（nice = -5）\n";
    } else {
      std::cout << "  → 优先级设置失败（需要 root 权限）\n";
    }
    #endif
  }
  
  void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {
    std::cout << "[高优先级] Worker " << w.id() << " 退出\n";
  }
};

// ============================================================================
// 低优先级 Worker Interface
// ============================================================================

class LowPriorityWorker : public tf::WorkerInterface {
public:
  void scheduler_prologue(tf::Worker& w) override {
    std::cout << "[低优先级] Worker " << w.id() << " 启动\n";
    
    #if defined(__linux__)
    // 设置低优先级（nice = 10）
    if(setpriority(PRIO_PROCESS, 0, 10) == 0) {
      std::cout << "  → 成功设置为低优先级（nice = 10）\n";
    }
    #endif
  }
  
  void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {
    std::cout << "[低优先级] Worker " << w.id() << " 退出\n";
  }
};

// ============================================================================
// 测试场景：高优先级任务 vs 低优先级任务
// ============================================================================

void test_multi_executor_priority() {
  std::cout << "\n========================================\n";
  std::cout << "测试：多 Executor 优先级控制\n";
  std::cout << "========================================\n";
  
  // 创建两个独立的 Executor
  // 高优先级 Executor：2 个线程
  tf::Executor high_priority_executor(
    2, 
    tf::make_worker_interface<HighPriorityWorker>()
  );
  
  // 低优先级 Executor：4 个线程
  tf::Executor low_priority_executor(
    4, 
    tf::make_worker_interface<LowPriorityWorker>()
  );
  
  // 创建高优先级任务流
  tf::Taskflow high_priority_taskflow("HighPriority");
  std::atomic<int> high_counter{0};
  
  for(int i = 0; i < 50; i++) {
    high_priority_taskflow.emplace([&high_counter, i](){
      // 模拟关键任务
      volatile int sum = 0;
      for(int j = 0; j < 1000000; j++) {
        sum += j;
      }
      
      int current = high_counter.fetch_add(1);
      if(current % 10 == 0) {
        std::cout << "[高优先级] 任务 " << i << " 完成\n";
      }
    });
  }
  
  // 创建低优先级任务流
  tf::Taskflow low_priority_taskflow("LowPriority");
  std::atomic<int> low_counter{0};
  
  for(int i = 0; i < 100; i++) {
    low_priority_taskflow.emplace([&low_counter, i](){
      // 模拟后台任务
      volatile int sum = 0;
      for(int j = 0; j < 1000000; j++) {
        sum += j;
      }
      
      int current = low_counter.fetch_add(1);
      if(current % 20 == 0) {
        std::cout << "[低优先级] 任务 " << i << " 完成\n";
      }
    });
  }
  
  // 同时启动两个任务流
  std::cout << "\n开始执行任务...\n\n";
  
  auto high_start = std::chrono::high_resolution_clock::now();
  auto low_start = std::chrono::high_resolution_clock::now();
  
  auto high_future = high_priority_executor.run(high_priority_taskflow);
  auto low_future = low_priority_executor.run(low_priority_taskflow);
  
  // 等待高优先级任务完成
  high_future.wait();
  auto high_end = std::chrono::high_resolution_clock::now();
  
  // 等待低优先级任务完成
  low_future.wait();
  auto low_end = std::chrono::high_resolution_clock::now();
  
  auto high_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    high_end - high_start
  );
  auto low_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    low_end - low_start
  );
  
  std::cout << "\n========================================\n";
  std::cout << "执行结果\n";
  std::cout << "========================================\n";
  std::cout << "高优先级任务（50 个）：" << high_duration.count() << " 毫秒\n";
  std::cout << "低优先级任务（100 个）：" << low_duration.count() << " 毫秒\n";
  
  std::cout << "\n【分析】：\n";
  std::cout << "- 高优先级 Executor 有 2 个线程，处理 50 个任务\n";
  std::cout << "- 低优先级 Executor 有 4 个线程，处理 100 个任务\n";
  std::cout << "- 如果有 root 权限，高优先级任务会更快完成\n";
  std::cout << "- 两个 Executor 完全独立，没有工作窃取\n";
  std::cout << "- 这样可以保证高优先级任务的执行时间\n";
}

int main() {
  std::cout << "╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  混合方案：多 Executor 优先级控制                         ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n";
  
  std::cout << "\n【方案说明】：\n";
  std::cout << "- 创建多个独立的 Executor\n";
  std::cout << "- 每个 Executor 设置不同的线程优先级\n";
  std::cout << "- 高优先级 Executor：少量线程，处理关键任务\n";
  std::cout << "- 低优先级 Executor：多个线程，处理后台任务\n";
  std::cout << "- 避免在同一个 Executor 内设置不同线程优先级\n";
  
  test_multi_executor_priority();
  
  return 0;
}

