#pragma once

namespace tf {

// ----------------------------------------------------------------------------

// class: Topology
// Topology 类表示一个 Taskflow 的运行时实例
// 每次调用 executor.run(taskflow) 都会创建一个新的 Topology 对象
// Topology 管理任务图的执行状态、同步和异常处理
class Topology {

  friend class Executor;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class Node;

  template <typename T>
  friend class Future;

  public:

    template <typename P, typename C>
    Topology(Taskflow&, P&&, C&&);

    bool cancelled() const;

  private:

    // 指向所属的 Taskflow 对象的引用
    // 用于访问任务图的定义和元数据
    Taskflow& _taskflow;

    // Promise 对象，用于与 Future 通信
    // 当 Topology 执行完成时，通过 _promise.set_value() 通知等待的线程
    // 如果发生异常，通过 _promise.set_exception() 传递异常
    std::promise<void> _promise;

    // 谓词函数，用于判断是否应该继续执行
    // 在 executor.run_until() 中使用，返回 false 时停止执行
    std::function<bool()> _pred;

    // 回调函数，在 Topology 完成后调用
    // 用于执行清理工作或后续操作
    std::function<void()> _call;

    // Join 计数器，用于跟踪还有多少个任务未完成
    // 当计数器减到 0 时，表示整个 Topology 执行完成
    // 使用原子操作保证线程安全
    std::atomic<size_t> _join_counter {0};

    // 异常状态标志位（Exception State）
    // 可能的值：
    //   - ESTATE::NONE      : 正常状态
    //   - ESTATE::EXCEPTION : 发生了异常
    //   - ESTATE::CANCELLED : 被取消
    //   - ESTATE::ANCHORED  : 被锚定（用于异常传播）
    std::atomic<ESTATE::underlying_type> _estate {ESTATE::NONE};

    // 异常指针，用于存储执行过程中捕获的异常
    // 如果任何任务抛出异常，异常会被存储在这里
    // 在 Topology 完成时通过 _promise.set_exception() 传递给调用者
    std::exception_ptr _exception_ptr {nullptr};

    // 完成 Promise，设置返回值或异常
    // 在 Topology 执行完成时由 Executor 调用
    void _carry_out_promise();
};

// Constructor
template <typename P, typename C>
Topology::Topology(Taskflow& tf, P&& p, C&& c):
  _taskflow(tf),
  _pred {std::forward<P>(p)},
  _call {std::forward<C>(c)} {
}

// Procedure
inline void Topology::_carry_out_promise() {
  if(_exception_ptr) {
    auto e = _exception_ptr;
    _exception_ptr = nullptr;
    _promise.set_exception(e);
  }
  else {
    _promise.set_value();
  }
}

// Function: cancelled
inline bool Topology::cancelled() const {
  return _estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED;
}

}  // end of namespace tf. ----------------------------------------------------
