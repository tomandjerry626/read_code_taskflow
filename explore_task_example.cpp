/**
 * _explore_task() 函数工作原理演示
 * 
 * 这个示例代码模拟了 _explore_task() 的核心逻辑
 * 帮助理解工作窃取算法的实现细节
 */

#include <iostream>
#include <vector>
#include <random>
#include <string>

// 模拟任务节点
struct Task {
    std::string name;
    Task(const std::string& n) : name(n) {}
};

// 模拟工作窃取队列
struct WorkStealingQueue {
    std::vector<Task*> tasks;
    
    Task* steal() {
        if (tasks.empty()) return nullptr;
        Task* t = tasks.front();
        tasks.erase(tasks.begin());
        return t;
    }
    
    bool empty() const { return tasks.empty(); }
};

// 模拟缓冲区桶
struct Bucket {
    WorkStealingQueue queue;
    
    Task* steal() {
        return queue.steal();
    }
};

// 模拟工作线程
struct Worker {
    int id;
    size_t vtm;  // 受害者索引
    WorkStealingQueue wsq;
    std::default_random_engine rdgen;
    
    Worker(int i) : id(i), vtm(0), rdgen(i) {}
};

// 模拟执行器
class Executor {
public:
    std::vector<Worker> workers;
    std::vector<Bucket> buffers;
    
    Executor(size_t num_workers, size_t num_buffers) {
        for (size_t i = 0; i < num_workers; ++i) {
            workers.emplace_back(i);
        }
        buffers.resize(num_buffers);
    }
    
    size_t num_queues() const {
        return workers.size() + buffers.size();
    }
    
    // 核心函数：_explore_task() 的简化版本
    bool explore_task(Worker& w, Task*& t) {
        std::cout << "\n=== Worker " << w.id << " 开始窃取任务 ===\n";
        
        // 计算最大窃取次数
        const size_t MAX_STEALS = ((num_queues() + 1) << 1);
        std::cout << "MAX_STEALS = " << MAX_STEALS << "\n";
        
        // 创建随机数生成器
        std::uniform_int_distribution<size_t> udist(0, num_queues() - 1);
        
        size_t num_steals = 0;
        size_t vtm = w.vtm;
        
        std::cout << "初始受害者索引 vtm = " << vtm << "\n\n";
        
        // 窃取循环
        while (true) {
            std::cout << "尝试 #" << (num_steals + 1) << ": ";
            
            // 根据 vtm 决定从哪里窃取
            if (vtm < workers.size()) {
                std::cout << "从 Worker " << vtm << " 的队列窃取... ";
                t = workers[vtm].wsq.steal();
            } else {
                size_t bucket_idx = vtm - workers.size();
                std::cout << "从 Bucket " << bucket_idx << " 窃取... ";
                t = buffers[bucket_idx].steal();
            }
            
            // 检查是否窃取成功
            if (t) {
                std::cout << "✓ 成功！窃取到任务: " << t->name << "\n";
                w.vtm = vtm;  // 更新受害者索引
                std::cout << "更新 Worker " << w.id << " 的 vtm = " << vtm << "\n";
                break;
            } else {
                std::cout << "✗ 失败（队列为空）\n";
            }
            
            // 增加失败计数
            if (++num_steals > MAX_STEALS) {
                std::cout << "已尝试 " << num_steals << " 次，超过 MAX_STEALS\n";
                if (num_steals > 150 + MAX_STEALS) {
                    std::cout << "达到最大尝试次数，放弃窃取\n";
                    break;
                }
            }
            
            // 随机选择下一个受害者
            vtm = udist(w.rdgen);
            std::cout << "随机选择下一个受害者 vtm = " << vtm << "\n\n";
        }
        
        std::cout << "=== 窃取过程结束 ===\n";
        return true;
    }
    
    // 打印当前状态
    void print_state() {
        std::cout << "\n【当前队列状态】\n";
        for (size_t i = 0; i < workers.size(); ++i) {
            std::cout << "Worker " << i << " 队列: ";
            if (workers[i].wsq.empty()) {
                std::cout << "空";
            } else {
                std::cout << workers[i].wsq.tasks.size() << " 个任务";
            }
            std::cout << "\n";
        }
        for (size_t i = 0; i < buffers.size(); ++i) {
            std::cout << "Bucket " << i << " 队列: ";
            if (buffers[i].queue.empty()) {
                std::cout << "空";
            } else {
                std::cout << buffers[i].queue.tasks.size() << " 个任务";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

int main() {
    // 创建执行器：4 个工作线程，2 个缓冲区桶
    Executor executor(4, 2);
    
    // 设置初始状态
    executor.workers[1].wsq.tasks.push_back(new Task("Task-A"));
    executor.workers[1].wsq.tasks.push_back(new Task("Task-B"));
    executor.workers[3].wsq.tasks.push_back(new Task("Task-C"));
    executor.buffers[0].queue.tasks.push_back(new Task("Task-D"));
    
    // Worker 0 的初始受害者索引设为 2
    executor.workers[0].vtm = 2;
    
    executor.print_state();
    
    // Worker 0 尝试窃取任务
    Task* stolen_task = nullptr;
    executor.explore_task(executor.workers[0], stolen_task);
    
    if (stolen_task) {
        std::cout << "\n✓ Worker 0 成功窃取到任务: " << stolen_task->name << "\n";
    } else {
        std::cout << "\n✗ Worker 0 没有窃取到任务\n";
    }
    
    return 0;
}

