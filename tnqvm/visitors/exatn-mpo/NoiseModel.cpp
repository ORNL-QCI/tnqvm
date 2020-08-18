#include "NoiseModel.hpp"
#include "json.hpp"
#include <random>

namespace {
inline double generateRandomProbability() {
  auto randFunc =
      std::bind(std::uniform_real_distribution<double>(0, 1),
                std::mt19937(std::chrono::high_resolution_clock::now()
                                 .time_since_epoch()
                                 .count()));
  return randFunc();
}
} // namespace

namespace tnqvm {
IBMNoiseModel::IBMNoiseModel(const std::string &in_jsonString) {
  auto backEndJson = nlohmann::json::parse(in_jsonString);
  // Parse qubit data:
  auto qubitsData = backEndJson["qubits"];
  size_t nbQubit = 0;
  for (auto qubitIter = qubitsData.begin(); qubitIter != qubitsData.end();
       ++qubitIter) {
    std::optional<double> meas0Prep1, meas1Prep0;
    // Each qubit contains an array of properties.
    for (auto probIter = qubitIter->begin(); probIter != qubitIter->end();
         ++probIter) {
      const auto probObj = *probIter;
      const std::string probName = probObj["name"].get<std::string>();
      const double probVal = probObj["value"].get<double>();
      const std::string unit = probObj["unit"].get<std::string>();
      // std::cout << probName << " = " << probVal << " " << unit << "\n";
      if (probName == "T1") {
        assert(unit == "µs" || unit == "ns");
        m_qubitT1.emplace_back(unit == "µs" ? 1000.0 * probVal : probVal);
      }

      if (probName == "T2") {
        assert(unit == "µs" || unit == "ns");
        m_qubitT2.emplace_back(unit == "µs" ? 1000.0 * probVal : probVal);
      }

      if (probName == "prob_meas0_prep1") {
        assert(unit.empty());
        meas0Prep1 = probVal;
      }

      if (probName == "prob_meas1_prep0") {
        assert(unit.empty());
        meas1Prep0 = probVal;
      }
    }
    assert(meas0Prep1.has_value() && meas1Prep0.has_value());
    m_roErrors.emplace_back(std::make_pair(*meas0Prep1, *meas1Prep0));

    nbQubit++;
  }
  m_nbQubits = nbQubit;
  assert(m_qubitT1.size() == m_nbQubits);
  assert(m_qubitT2.size() == m_nbQubits);
  assert(m_roErrors.size() == m_nbQubits);

  // Parse gate data:
  auto gateData = backEndJson["gates"];
  for (auto gateIter = gateData.begin(); gateIter != gateData.end();
       ++gateIter) {
    auto gateObj = *gateIter;
    const std::string gateName = gateObj["name"].get<std::string>();
    auto gateParams = gateObj["parameters"];
    for (auto it = gateParams.begin(); it != gateParams.end(); ++it) {
      auto paramObj = *it;
      const std::string paramName = paramObj["name"].get<std::string>();
      if (paramName == "gate_length") {
        const std::string unit = paramObj["unit"].get<std::string>();
        assert(unit == "µs" || unit == "ns");
        const double gateLength = unit == "µs"
                                      ? 1000.0 * paramObj["value"].get<double>()
                                      : paramObj["value"].get<double>();
        const bool insertOk =
            m_gateDurations.insert({gateName, gateLength}).second;
        // Must not contain duplicates.
        assert(insertOk);
      }

      if (paramName == "gate_error") {
        assert(paramObj["unit"].get<std::string>().empty());
        const bool insertOk =
            m_gateErrors.insert({gateName, paramObj["value"].get<double>()})
                .second;
        // Must not contain duplicates.
        assert(insertOk);
      }
    }
  }
}

bool INoiseModel::applyRoError(size_t in_bitIdx, bool in_exactMeasure) const {
  const auto [meas0Prep1, meas1Prep0] = getRoErrorProbs(in_bitIdx);
  // If exact measurement is 1 (true), use meas0Prep1 as the error probability
  // and vice versa.
  const double flipProb = in_exactMeasure ? meas0Prep1 : meas1Prep0;
  return (generateRandomProbability() < flipProb) ? !in_exactMeasure
                                                  : in_exactMeasure;
}

std::vector<double>
IBMNoiseModel::calculateAmplitudeDamping(xacc::quantum::Gate &in_gate) const {
  const std::string universalGateName = getUniversalGateEquiv(in_gate);
  const auto gateDurationIter = m_gateDurations.find(universalGateName);
  assert(gateDurationIter != m_gateDurations.end());
  const double gateDuration = gateDurationIter->second;
  std::vector<double> amplitudeDamping;
  for (const auto &qubitIdx : in_gate.bits()) {
    const double qubitT1 = m_qubitT1[qubitIdx];
    const double dampingRate = 1.0 / qubitT1;
    const double resetProb = 1.0 * std::exp(-gateDuration * dampingRate);
    amplitudeDamping.emplace_back(1.0 - resetProb);
  }
  return amplitudeDamping;
}

std::string
IBMNoiseModel::getUniversalGateEquiv(xacc::quantum::Gate &in_gate) const {

  if (in_gate.bits().size() == 1 && in_gate.name() != "Measure") {
    // Note: rotation around Z is a noiseless *u1* operation;
    // *u2* operations are those that requires a half-length rotation;
    // *u3* operations are those that requires a full-length rotation.
    static const std::unordered_map<std::string, std::string>
        SINGLE_QUBIT_GATE_MAP{{"X", "u3"},   {"Y", "u3"},  {"Z", "u1"},
                              {"H", "u2"},   {"U", "u3"},  {"T", "u1"},
                              {"Tdg", "u1"}, {"S", "u1"},  {"Sdg", "u1"},
                              {"Rz", "u1"},  {"Rx", "u3"}, {"Ry", "u3"}};
    const auto iter = SINGLE_QUBIT_GATE_MAP.find(in_gate.name());
    // If cannot find the gate, just treat that as a noiseless u1 op.
    const std::string universalGateName =
        (iter == SINGLE_QUBIT_GATE_MAP.end()) ? "u1" : iter->second;
    return universalGateName + "_" + std::to_string(in_gate.bits()[0]);
  }

  if (in_gate.bits().size() == 2) {
    return "cx" + std::to_string(in_gate.bits()[0]) + "_" +
           std::to_string(in_gate.bits()[1]);
  }

  return "id_" + std::to_string(in_gate.bits()[0]);
}

std::vector<double> IBMNoiseModel::calculateDepolarizing(
    xacc::quantum::Gate &in_gate,
    const std::vector<double> &in_amplitudeDamping) const {
  //  Compute the depolarizing channel error parameter in the
  //  presence of T1/T2 thermal relaxation.
  //  Hence we have that the depolarizing error probability
  //  for the composed depolarization channel is
  //  p = dim * (F(E_relax) - F) / (dim * F(E_relax) - 1)
  const double averageThermalError =
      in_amplitudeDamping.empty()
          ? 0.0
          : std::accumulate(in_amplitudeDamping.begin(),
                            in_amplitudeDamping.end(), 0.0) /
                in_amplitudeDamping.size();
  const std::string universalGateName = getUniversalGateEquiv(in_gate);
  // Retrieve the error rate:
  const auto gateErrorIter = m_gateErrors.find(universalGateName);
  const double gateError =
      (gateErrorIter == m_gateErrors.end()) ? 0.0 : gateErrorIter->second;
  // If the backend gate error (estimated by randomized benchmarking) is more
  // than thermal relaxation error. We need to add depolarization to simulate
  // the total gate error.
  if (gateError > averageThermalError) {
    // Model gate error entirely as depolarizing error
    const double depolError = 2 * (gateError - averageThermalError) /
                              (2 * (1 - averageThermalError) - 1);
    return {depolError};
  }
  return {0.0};
}
} // namespace tnqvm