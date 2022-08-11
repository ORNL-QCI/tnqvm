#pragma once
#include <algorithm>
#include <random>
#include <mutex>
#include <utility>
#include <vector>
namespace tnqvm {
struct randomEngine {
  randomEngine(const randomEngine &) = delete;
  randomEngine &operator=(const randomEngine &) = delete;
  double randProb() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return std::uniform_real_distribution<double>(0.0, 1.0)(m_engine);
  }

  static randomEngine &get_instance() {
    static randomEngine instance;
    return instance;
  }

  void setSeed(size_t seed) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_engine.seed(seed);
  }

  std::vector<double> sortedRandProbs(uint64_t num_samples) {
    std::vector<double> rs;
    rs.reserve(num_samples + 1);
    std::lock_guard<std::mutex> lock(m_mutex);
    for (uint64_t i = 0; i < num_samples; ++i) {
      rs.emplace_back(
          std::uniform_real_distribution<double>(0.0, 1.0)(m_engine));
    }
    std::sort(rs.begin(), rs.end());
    return rs;
  }

private:
  randomEngine() {
    std::random_device rd;
    setSeed(rd());
  }
  std::mt19937_64 m_engine;
  std::mutex m_mutex;
};
} // namespace tnqvm