#include "NoiseModel.hpp"
#include "json.hpp"

namespace tnqvm {
void IBMNoiseModel::loadJson(const std::string &in_jsonString) {
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
} // namespace tnqvm