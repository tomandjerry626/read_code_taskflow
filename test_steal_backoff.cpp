// 测试 TaskFlow 的窃取退避策略
// 编译: g++ -std=c++17 -O2 -I. test_steal_backoff.cpp -pthread -o test_steal_backoff
// 运行: ./test_steal_backoff

#include <taskflow/taskflow.hpp>
#include <iostream>
#include <chrono>
#include <atomic>

// 全局计数器，用于统计窃取行为
std::atomic<size_t> total_steals{0};
std::atomic<size_t> yield_count{0};

int main() {
    // 创建 4 个线程的执行器
    tf::Executor executor(4);
    
    std::cout << "=== 测试 1: 任务密集场景 ===" << std::endl;
    {
        tf::Taskflow taskflow;
        
        // 创建 100 个独立任务
        for (int i = 0; i < 100; ++i) {
            taskflow.emplace([i]() {
                // 模拟短时间计算
                volatile int sum = 0;
                for (int j = 0; j < 1000; ++j) {
                    sum += j;
                }
            });
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        executor.run(taskflow).wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "执行时间: " << duration.count() << " 微秒" << std::endl;
    }
    
    std::cout << "\n=== 测试 2: 任务稀疏场景 ===" << std::endl;
    {
        tf::Taskflow taskflow;
        
        // 创建 4 个任务，每个任务执行时间较长
        for (int i = 0; i < 4; ++i) {
            taskflow.emplace([i]() {
                std::cout << "任务 " << i << " 开始执行" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "任务 " << i << " 完成" << std::endl;
            });
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        executor.run(taskflow).wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "执行时间: " << duration.count() << " 毫秒" << std::endl;
    }
    
    std::cout << "\n=== 测试 3: 任务空窗期 ===" << std::endl;
    {
        tf::Taskflow taskflow1, taskflow2;
        
        // 第一个 taskflow：1 个任务
        taskflow1.emplace([]() {
            std::cout << "Taskflow 1 执行" << std::endl;
        });
        
        // 第二个 taskflow：1 个任务
        taskflow2.emplace([]() {
            std::cout << "Taskflow 2 执行" << std::endl;
        });
        
        // 执行第一个 taskflow
        executor.run(taskflow1).wait();
        std::cout << "Taskflow 1 完成，等待 50ms..." << std::endl;
        
        // 模拟任务空窗期
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // 执行第二个 taskflow
        std::cout << "提交 Taskflow 2" << std::endl;
        executor.run(taskflow2).wait();
    }
    
    std::cout << "\n=== 测试 4: 依赖链场景 ===" << std::endl;
    {
        tf::Taskflow taskflow;
        
        // 创建一个长依赖链：A -> B -> C -> D -> E
        auto A = taskflow.emplace([]() {
            std::cout << "任务 A 执行" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        
        auto B = taskflow.emplace([]() {
            std::cout << "任务 B 执行" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        
        auto C = taskflow.emplace([]() {
            std::cout << "任务 C 执行" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        
        auto D = taskflow.emplace([]() {
            std::cout << "任务 D 执行" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        
        auto E = taskflow.emplace([]() {
            std::cout << "任务 E 执行" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        
        A.precede(B);
        B.precede(C);
        C.precede(D);
        D.precede(E);
        
        auto start = std::chrono::high_resolution_clock::now();
        executor.run(taskflow).wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "执行时间: " << duration.count() << " 毫秒" << std::endl;
        std::cout << "理论最小时间: 50 毫秒（5 个任务 × 10ms）" << std::endl;
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    
    return 0;
}

