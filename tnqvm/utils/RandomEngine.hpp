#pragma once
#include <random>
namespace tnqvm {
struct randomEngine {
  randomEngine(const randomEngine &) = delete;
  randomEngine &operator=(const randomEngine &) = delete;
  double randProb() {
    const auto val = std::uniform_real_distribution<double>(0.0, 1.0)(m_engine);
    return val;
  }

  thread_local static randomEngine &get_instance() {
    thread_local static randomEngine instance;
    thread_local static size_t seed = 0;
    if (seed != globalSeed) {
      instance.m_engine.seed(globalSeed);
      seed = globalSeed;
    }
    return instance;
  }
  std::mt19937_64 m_engine;
  static inline size_t globalSeed = []() {
    std::random_device rd;
    return rd();
  }();
  static void setSeed(size_t seed) { globalSeed = seed; }

private:
  randomEngine() = default;
};
} // namespace tnqvm