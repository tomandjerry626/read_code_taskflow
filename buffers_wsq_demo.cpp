/**
 * _buffers 和 _wsq 关系演示
 * 
 * 这个示例展示了任务如何在 _wsq 和 _buffers 之间分配
 */

#include <iostream>
#include <vector>
#include <string>
#include <taskflow/taskflow.hpp>

int main() {
    
    std::cout << "=== Taskflow 任务调度机制演示 ===\n\n";
    
    // 创建 4 个工作线程的执行器
    tf::Executor executor(4);
    
    // ========================================
    // 场景 1：外部线程提交任务
    // ========================================
    std::cout << "【场景 1】外部线程（主线程）提交任务\n";
    std::cout << "结果：所有任务都会放入 _buffers\n";
    std::cout << "原因：主线程不是执行器的工作线程\n\n";
    
    {
        tf::Taskflow taskflow("外部提交");
        
        auto A = taskflow.emplace([](){ 
            std::cout << "  Task A 执行\n"; 
        }).name("A");
        
        auto B = taskflow.emplace([](){ 
            std::cout << "  Task B 执行\n"; 
        }).name("B");
        
        auto C = taskflow.emplace([](){ 
            std::cout << "  Task C 执行\n"; 
        }).name("C");
        
        A.precede(B, C);
        
        // 主线程调用 run()
        // 内部会调用 _schedule(node)，直接放入 _buffers
        executor.run(taskflow).wait();
    }
    
    std::cout << "\n";
    
    // ========================================
    // 场景 2：工作线程产生子任务
    // ========================================
    std::cout << "【场景 2】工作线程在执行任务时产生子任务\n";
    std::cout << "结果：子任务优先放入当前工作线程的 _wsq\n";
    std::cout << "原因：调用者是本执行器的工作线程\n\n";
    
    {
        tf::Taskflow taskflow("子任务生成");
        
        auto parent = taskflow.emplace([&](tf::Subflow& subflow){
            std::cout << "  父任务执行，产生 3 个子任务\n";
            
            // 这些子任务会放入当前工作线程的 _wsq
            // 因为调用者是工作线程
            auto sub1 = subflow.emplace([](){ 
                std::cout << "    子任务 1 执行 (来自 _wsq)\n"; 
            });
            
            auto sub2 = subflow.emplace([](){ 
                std::cout << "    子任务 2 执行 (来自 _wsq)\n"; 
            });
            
            auto sub3 = subflow.emplace([](){ 
                std::cout << "    子任务 3 执行 (来自 _wsq)\n"; 
            });
            
            sub1.precede(sub2, sub3);
        }).name("父任务");
        
        executor.run(taskflow).wait();
    }
    
    std::cout << "\n";
    
    // ========================================
    // 场景 3：_wsq 溢出
    // ========================================
    std::cout << "【场景 3】工作线程产生大量任务，_wsq 溢出\n";
    std::cout << "结果：前 1024 个任务放入 _wsq，后续任务溢出到 _buffers\n";
    std::cout << "原因：_wsq 容量有限（1024）\n\n";
    
    {
        tf::Taskflow taskflow("大量任务");
        
        auto generator = taskflow.emplace([&](tf::Subflow& subflow){
            std::cout << "  生成 2000 个子任务...\n";
            
            // 产生 2000 个任务
            // 前 1024 个 → 当前工作线程的 _wsq
            // 后 976 个  → _buffers (溢出)
            for(int i = 0; i < 2000; i++) {
                subflow.emplace([i](){
                    // 任务执行
                    if(i < 5 || i >= 1995) {
                        std::cout << "    任务 " << i << " 执行\n";
                    } else if(i == 5) {
                        std::cout << "    ... (省略中间任务) ...\n";
                    }
                });
            }
            
            std::cout << "  前 1024 个任务 → _wsq (无锁)\n";
            std::cout << "  后 976 个任务  → _buffers (溢出)\n";
        }).name("任务生成器");
        
        executor.run(taskflow).wait();
    }
    
    std::cout << "\n";
    
    // ========================================
    // 场景 4：异步任务
    // ========================================
    std::cout << "【场景 4】从外部线程提交异步任务\n";
    std::cout << "结果：异步任务直接放入 _buffers\n";
    std::cout << "原因：async() 内部调用 _schedule(node)，没有 Worker 参数\n\n";
    
    {
        auto future1 = executor.async([](){
            std::cout << "  异步任务 1 执行 (来自 _buffers)\n";
            return 42;
        });
        
        auto future2 = executor.async([](){
            std::cout << "  异步任务 2 执行 (来自 _buffers)\n";
            return 100;
        });
        
        std::cout << "  异步任务 1 结果: " << future1.get() << "\n";
        std::cout << "  异步任务 2 结果: " << future2.get() << "\n";
    }
    
    std::cout << "\n";
    
    // ========================================
    // 总结
    // ========================================
    std::cout << "=== 总结 ===\n";
    std::cout << "1. 工作线程产生的任务 → 优先放入 _wsq (快速路径)\n";
    std::cout << "2. 外部线程提交的任务 → 直接放入 _buffers (慢速路径)\n";
    std::cout << "3. _wsq 满时 (>1024)  → 溢出到 _buffers\n";
    std::cout << "4. 窃取优先级: 自己的 _wsq > 其他 _wsq > _buffers\n";
    
    return 0;
}

