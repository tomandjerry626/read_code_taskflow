// 测试随机窃取的概率分布
// 编译: g++ -std=c++17 -O2 test_random_steal_probability.cpp -o test_random_steal_probability
// 运行: ./test_random_steal_probability

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    const size_t NUM_QUEUES = 6;        // 6 个队列
    const size_t MAX_STEALS = 14;       // 14 次尝试
    const size_t NUM_SIMULATIONS = 100000;  // 模拟 10 万次
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, NUM_QUEUES - 1);
    
    // 统计数据
    size_t all_queues_visited = 0;      // 所有队列都被访问的次数
    size_t some_queue_missed = 0;       // 至少有一个队列没被访问的次数
    std::vector<size_t> visit_counts(NUM_QUEUES, 0);  // 每个队列的平均访问次数
    
    std::cout << "=== 随机窃取概率分析 ===" << std::endl;
    std::cout << "队列数量: " << NUM_QUEUES << std::endl;
    std::cout << "窃取次数: " << MAX_STEALS << std::endl;
    std::cout << "模拟次数: " << NUM_SIMULATIONS << std::endl;
    std::cout << std::endl;
    
    // 进行模拟
    for (size_t sim = 0; sim < NUM_SIMULATIONS; ++sim) {
        std::vector<bool> visited(NUM_QUEUES, false);
        std::vector<size_t> counts(NUM_QUEUES, 0);
        
        // 模拟 MAX_STEALS 次随机选择
        for (size_t i = 0; i < MAX_STEALS; ++i) {
            size_t vtm = dist(gen);
            visited[vtm] = true;
            counts[vtm]++;
        }
        
        // 统计结果
        bool all_visited = true;
        for (size_t i = 0; i < NUM_QUEUES; ++i) {
            if (!visited[i]) {
                all_visited = false;
            }
            visit_counts[i] += counts[i];
        }
        
        if (all_visited) {
            all_queues_visited++;
        } else {
            some_queue_missed++;
        }
    }
    
    // 输出结果
    std::cout << "=== 模拟结果 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    double all_visited_percent = (double)all_queues_visited / NUM_SIMULATIONS * 100;
    double some_missed_percent = (double)some_queue_missed / NUM_SIMULATIONS * 100;
    
    std::cout << "所有队列都被访问: " << all_queues_visited 
              << " 次 (" << all_visited_percent << "%)" << std::endl;
    std::cout << "至少一个队列未访问: " << some_queue_missed 
              << " 次 (" << some_missed_percent << "%)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== 每个队列的平均访问次数 ===" << std::endl;
    for (size_t i = 0; i < NUM_QUEUES; ++i) {
        double avg = (double)visit_counts[i] / NUM_SIMULATIONS;
        std::cout << "队列 " << i << ": " << avg << " 次" << std::endl;
    }
    std::cout << std::endl;
    
    // 理论计算
    std::cout << "=== 理论计算 ===" << std::endl;
    
    // 某个队列一次都没被访问的概率
    double prob_not_visited = std::pow((double)(NUM_QUEUES - 1) / NUM_QUEUES, MAX_STEALS);
    std::cout << "某个特定队列一次都没被访问的概率: " 
              << prob_not_visited * 100 << "%" << std::endl;
    
    // 某个队列至少被访问一次的概率
    double prob_visited_once = 1.0 - prob_not_visited;
    std::cout << "某个特定队列至少被访问一次的概率: " 
              << prob_visited_once * 100 << "%" << std::endl;
    
    // 所有队列都至少被访问一次的概率（近似）
    // 注意：这是一个近似值，实际计算需要用容斥原理
    double prob_all_visited_approx = std::pow(prob_visited_once, NUM_QUEUES);
    std::cout << "所有队列都至少被访问一次的概率（近似）: " 
              << prob_all_visited_approx * 100 << "%" << std::endl;
    
    // 每个队列的期望访问次数
    double expected_visits = (double)MAX_STEALS / NUM_QUEUES;
    std::cout << "每个队列的期望访问次数: " << expected_visits << " 次" << std::endl;
    std::cout << std::endl;
    
    // 结论
    std::cout << "=== 结论 ===" << std::endl;
    std::cout << "1. 随机选择 " << MAX_STEALS << " 次，有约 " 
              << some_missed_percent << "% 的概率会遗漏某些队列" << std::endl;
    std::cout << "2. 这意味着 MAX_STEALS = (N+1)*2 并不能保证每个队列都被访问 2 次" << std::endl;
    std::cout << "3. 这个公式只是一个经验值，用于在概率上提供足够的覆盖" << std::endl;
    std::cout << "4. 实际上，某些队列可能被访问 0 次，某些可能被访问 5+ 次" << std::endl;
    
    return 0;
}

