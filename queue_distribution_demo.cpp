// 演示任务如何分配到 _buffers 和 worker 本地队列的示例
// 展示两层队列架构的性能优势

#include <taskflow/taskflow.hpp>
#include <iostream>
#include <chrono>
#include <atomic>

// 全局计数器，用于统计任务分配情况
std::atomic<int> tasks_from_main{0};
std::atomic<int> tasks_from_worker{0};

// 场景 1：所有任务从 main 线程提交（你当前的理解）
void scenario1_all_from_main() {
  std::cout << "\n========================================\n";
  std::cout << "场景 1：所有任务从 main 线程提交\n";
  std::cout << "========================================\n";
  
  tf::Executor executor(4);
  tf::Taskflow taskflow;
  
  // 创建 10 个简单任务
  for(int i = 0; i < 10; i++) {
    taskflow.emplace([i](){
      std::cout << "Task " << i << " 执行\n";
    });
  }
  
  std::cout << "\n【分析】：\n";
  std::cout << "- 所有任务都是从 main 线程调用 executor.run() 提交的\n";
  std::cout << "- main 线程不是 executor 的工作线程\n";
  std::cout << "- 因此所有任务都会放入 _buffers（共享队列）\n";
  std::cout << "- 工作线程从 _buffers 中窃取任务执行\n\n";
  
  executor.run(taskflow).wait();
}

// 场景 2：任务动态产生子任务（Subflow）
void scenario2_subflow() {
  std::cout << "\n========================================\n";
  std::cout << "场景 2：任务动态产生子任务（Subflow）\n";
  std::cout << "========================================\n";
  
  tf::Executor executor(4);
  tf::Taskflow taskflow;
  
  // 父任务：从 main 提交，放入 _buffers
  auto parent = taskflow.emplace([](tf::Subflow& sf){
    std::cout << "父任务执行中，创建 5 个子任务...\n";
    
    // 这些子任务是在工作线程中创建的！
    for(int i = 0; i < 5; i++) {
      sf.emplace([i](){
        std::cout << "  子任务 " << i << " 执行\n";
      });
    }
    
    std::cout << "\n【关键点】：\n";
    std::cout << "- 父任务在工作线程中执行\n";
    std::cout << "- 子任务是在工作线程中动态创建的\n";
    std::cout << "- 子任务会放入当前工作线程的本地队列（_wsq）\n";
    std::cout << "- 这是本地队列的主要使用场景！\n\n";
  });
  
  executor.run(taskflow).wait();
}

// 场景 3：后继任务的调度
void scenario3_successors() {
  std::cout << "\n========================================\n";
  std::cout << "场景 3：后继任务的调度\n";
  std::cout << "========================================\n";

  tf::Executor executor(4);
  tf::Taskflow taskflow;

  // 创建一个任务链：A -> B -> C -> D -> E
  auto A = taskflow.emplace([](){
    std::cout << "任务 A 执行\n";
  }).name("A");

  auto B = taskflow.emplace([](){
    std::cout << "任务 B 执行\n";
  }).name("B");

  auto C = taskflow.emplace([](){
    std::cout << "任务 C 执行\n";
  }).name("C");

  auto D = taskflow.emplace([](){
    std::cout << "任务 D 执行\n";
  }).name("D");

  auto E = taskflow.emplace([](){
    std::cout << "任务 E 执行\n";
  }).name("E");

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);

  std::cout << "\n【关键点】：\n";
  std::cout << "- 任务 A 从 _buffers 被窃取（main 线程提交）\n";
  std::cout << "- 任务 A 完成后，调度后继任务 B\n";
  std::cout << "- 任务 B 会放入当前工作线程的本地队列（_wsq）\n";
  std::cout << "- 通过 cache 变量优化，B 可能直接执行（尾调用优化）\n";
  std::cout << "- 这样 A->B->C->D->E 可能在同一个线程连续执行\n\n";

  executor.run(taskflow).wait();
}

// 场景 4：对比性能 - 展示本地队列的优势
void scenario4_performance() {
  std::cout << "\n========================================\n";
  std::cout << "场景 4：性能对比 - 递归任务\n";
  std::cout << "========================================\n";
  
  tf::Executor executor(4);
  
  // 模拟递归任务（例如：快速排序、分治算法）
  std::function<void(tf::Subflow&, int, int)> recursive_task;
  recursive_task = [&](tf::Subflow& sf, int depth, int id) {
    if(depth == 0) return;
    
    // 每个任务产生 2 个子任务（二叉树）
    for(int i = 0; i < 2; i++) {
      sf.emplace([&, depth, id, i](tf::Subflow& sf){
        recursive_task(sf, depth - 1, id * 2 + i);
      });
    }
  };
  
  tf::Taskflow taskflow;
  taskflow.emplace([&](tf::Subflow& sf){
    std::cout << "创建递归任务树（深度 5，共 31 个任务）...\n";
    recursive_task(sf, 5, 0);
  });
  
  auto start = std::chrono::high_resolution_clock::now();
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "\n【性能分析】：\n";
  std::cout << "- 执行时间：" << duration.count() << " 微秒\n";
  std::cout << "- 大部分子任务放入本地队列（_wsq）\n";
  std::cout << "- 本地队列优势：\n";
  std::cout << "  1. 无锁操作（所有者访问底部）\n";
  std::cout << "  2. 缓存局部性好（L1 缓存命中率高）\n";
  std::cout << "  3. 无竞争（只有所有者 push/pop）\n";
  std::cout << "  4. LIFO 顺序（深度优先，适合递归）\n\n";
}

int main() {
  std::cout << "╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  Taskflow 两层队列架构演示                                ║\n";
  std::cout << "║  _buffers (共享队列) vs _wsq (本地队列)                   ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n";
  
  scenario1_all_from_main();
  scenario2_subflow();
  scenario3_successors();
  scenario4_performance();
  
  std::cout << "\n========================================\n";
  std::cout << "总结\n";
  std::cout << "========================================\n";
  std::cout << "【何时放入 _buffers】：\n";
  std::cout << "  1. 外部线程（如 main）调用 executor.run()\n";
  std::cout << "  2. 外部线程调用 executor.async()\n";
  std::cout << "  3. 工作线程的本地队列满了（溢出）\n\n";
  
  std::cout << "【何时放入 _wsq（本地队列）】：\n";
  std::cout << "  1. Subflow 中创建的子任务\n";
  std::cout << "  2. Runtime 中动态调度的任务\n";
  std::cout << "  3. 工作线程执行任务后产生的后继任务\n\n";
  
  std::cout << "【设计优势】：\n";
  std::cout << "  1. 减少竞争：大部分任务在本地队列，无需竞争全局锁\n";
  std::cout << "  2. 提高性能：本地队列无锁，缓存友好\n";
  std::cout << "  3. 负载均衡：其他线程可以从本地队列窃取任务\n";
  std::cout << "  4. 灵活性：支持外部提交和动态任务生成\n\n";
  
  return 0;
}

