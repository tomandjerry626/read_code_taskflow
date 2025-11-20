/**
 * Taskflow CUDA 模块使用示例
 * 
 * 本文件展示了如何使用 Taskflow 的 CUDA 模块构建和执行 GPU 任务图
 * 
 * 编译命令（需要安装 CUDA Toolkit）：
 *   nvcc -std=c++17 -I. cuda_example_guide.cpp -o cuda_example
 * 
 * 或使用 g++：
 *   g++ -std=c++17 -I. cuda_example_guide.cpp -o cuda_example -lcudart
 */

#include <iostream>
#include <vector>

// 注意：以下代码需要 CUDA 环境才能编译运行
// 这里仅作为示例说明，展示 API 使用方式

// #include <taskflow/cuda/cudaflow.hpp>

/**
 * ============================================================================
 * 示例 1：显式图构建（cudaGraph）
 * ============================================================================
 * 
 * 使用场景：
 *   - 图结构简单明确
 *   - 需要精确控制节点类型
 *   - 需要查询图的详细信息
 */

void example1_explicit_graph() {
  std::cout << "\n=== 示例 1：显式图构建 ===\n";
  
  /*
  const int N = 1024;
  
  // 1. 分配内存
  float *h_data, *d_data;
  h_data = new float[N];
  cudaMalloc(&d_data, N * sizeof(float));
  
  // 初始化数据
  for(int i = 0; i < N; i++) {
    h_data[i] = i;
  }
  
  // 2. 创建 CUDA 图
  tf::cudaGraph cg;
  
  // 3. 添加任务节点
  
  // 3.1 添加 H2D 内存拷贝任务
  auto h2d = cg.copy(d_data, h_data, N);
  
  // 3.2 添加内核任务
  dim3 grid((N + 255) / 256);
  dim3 block(256);
  auto kernel = cg.kernel(grid, block, 0, my_kernel, d_data, N);
  
  // 3.3 添加 D2H 内存拷贝任务
  auto d2h = cg.copy(h_data, d_data, N);
  
  // 4. 设置依赖关系
  h2d.precede(kernel);   // h2d → kernel
  kernel.precede(d2h);   // kernel → d2h
  
  // 5. 查询图信息
  std::cout << "图中节点数量: " << cg.num_nodes() << "\n";
  std::cout << "图中边数量: " << cg.num_edges() << "\n";
  
  // 6. 导出图结构（DOT 格式）
  cg.dump(std::cout);
  
  // 7. 实例化图
  tf::cudaGraphExec exec(cg);
  
  // 8. 创建流并执行图
  tf::cudaStream stream;
  stream.run(exec).synchronize();
  
  // 9. 验证结果
  for(int i = 0; i < 10; i++) {
    std::cout << "h_data[" << i << "] = " << h_data[i] << "\n";
  }
  
  // 10. 清理
  delete[] h_data;
  cudaFree(d_data);
  */
  
  std::cout << "【说明】：\n";
  std::cout << "  1. 使用 cudaGraph 显式构建图\n";
  std::cout << "  2. 添加 copy, kernel 等节点\n";
  std::cout << "  3. 使用 precede() 设置依赖关系\n";
  std::cout << "  4. 实例化后可以多次执行\n";
  std::cout << "  5. 整个图只有一次启动开销\n";
}

/**
 * ============================================================================
 * 示例 2：流捕获（cudaFlowCapturer）
 * ============================================================================
 * 
 * 使用场景：
 *   - 需要捕获任意异步 CUDA 操作
 *   - 使用 cuBLAS、cuDNN 等库
 *   - 图结构复杂，手动构建困难
 */

void example2_stream_capture() {
  std::cout << "\n=== 示例 2：流捕获 ===\n";
  
  /*
  tf::Executor executor;
  tf::Taskflow taskflow;
  
  // 创建 cudaFlowCapturer 任务
  taskflow.emplace([](tf::cudaFlowCapturer& capturer){
    
    // 捕获 H2D 内存拷贝
    auto h2d = capturer.on([&](cudaStream_t stream){
      cudaMemcpyAsync(d_data, h_data, N * sizeof(float),
                      cudaMemcpyHostToDevice, stream);
    });
    
    // 捕获内核启动
    auto kernel = capturer.on([&](cudaStream_t stream){
      my_kernel<<<grid, block, 0, stream>>>(d_data, N);
    });
    
    // 捕获 D2H 内存拷贝
    auto d2h = capturer.on([&](cudaStream_t stream){
      cudaMemcpyAsync(h_data, d_data, N * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
    });
    
    // 设置依赖关系
    h2d.precede(kernel);
    kernel.precede(d2h);
  });
  
  // 执行
  executor.run(taskflow).wait();
  */
  
  std::cout << "【说明】：\n";
  std::cout << "  1. 使用 cudaFlowCapturer 自动捕获\n";
  std::cout << "  2. 可以捕获任意异步 CUDA 操作\n";
  std::cout << "  3. 支持 cuBLAS、cuDNN 等库\n";
  std::cout << "  4. 自动优化并发性\n";
  std::cout << "  5. 适合复杂的 GPU 工作流\n";
}

int main() {
  std::cout << "Taskflow CUDA 模块使用示例\n";
  std::cout << "============================\n";
  
  example1_explicit_graph();
  example2_stream_capture();
  
  std::cout << "\n【总结】：\n";
  std::cout << "  - 显式构建：适合简单明确的图结构\n";
  std::cout << "  - 流捕获：适合复杂的 GPU 操作\n";
  std::cout << "  - 两种方式都可以嵌入 CPU Taskflow\n";
  std::cout << "  - CUDA Graph 大幅降低启动开销\n";
  
  return 0;
}

