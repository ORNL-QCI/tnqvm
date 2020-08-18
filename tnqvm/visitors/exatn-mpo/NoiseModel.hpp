#pragma once

#include <vector>
#include "Gate.hpp"

namespace xacc {
namespace quantum {
class Gate;
}
} // namespace xacc

namespace tnqvm {
// Kraus amplitude (per-qubit)
struct KrausAmpl {
  KrausAmpl() : KrausAmpl(0.0, 0.0) {}
  KrausAmpl(double in_probAD, double in_probDP)
      : probAD(in_probAD), probDP(in_probDP) {}
  bool isZero() const {
    return std::abs(probAD) < 1e-12 && std::abs(probDP) < 1e-12;
  }
  // Probability of Amplitude Damping on any single qubit
  double probAD;
  // Probability of Depolarizing noise on any single qubit
  double probDP;
};

class INoiseModel {
public:
  virtual std::vector<double>
  calculateAmplitudeDamping(xacc::quantum::Gate &in_gate) const = 0;
  virtual std::vector<double>
  calculateDepolarizing(xacc::quantum::Gate &in_gate, const std::vector<double>& in_amplitudeDamping = {}) const = 0;
  virtual std::pair<double, double> getRoErrorProbs(size_t in_bitIdx) const = 0;
  bool applyRoError(size_t in_bitIdx, bool in_exactMeasure) const;
};

// Helper class to parse and construct noise model
// from IBM's backend JSON.
class IBMNoiseModel : public INoiseModel {
public:
  IBMNoiseModel(const std::string &in_jsonString);
  virtual std::vector<double>
  calculateAmplitudeDamping(xacc::quantum::Gate &in_gate) const override;
  virtual std::vector<double>
  calculateDepolarizing(xacc::quantum::Gate &in_gate, const std::vector<double>& in_amplitudeDamping) const override;
  virtual std::pair<double, double>
  getRoErrorProbs(size_t in_bitIdx) const override {
    return m_roErrors[in_bitIdx];
  }

private:
  // Gets the name of the equivalent universal gate.
  // e.g. an Rz <=> u1; H <==> u2, etc.
  std::string getUniversalGateEquiv(xacc::quantum::Gate &in_gate) const;

private:
  // Parsed parameters needed for noise model construction.
  size_t m_nbQubits;
  std::vector<double> m_qubitT1;
  std::vector<double> m_qubitT2;
  std::unordered_map<std::string, double> m_gateErrors;
  std::unordered_map<std::string, double> m_gateDurations;
  std::vector<std::pair<double, double>> m_roErrors;
};
} // namespace tnqvm
