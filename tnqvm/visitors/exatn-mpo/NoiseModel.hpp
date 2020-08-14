#pragma once

#include <vector>
#include "Gate.hpp"

namespace tnqvm {
// Kraus amplitude (per-qubit)
struct KrausAmpl
{
    KrausAmpl() : KrausAmpl(0.0, 0.0) {}
    KrausAmpl(double in_probAD, double in_probDP):
        probAD(in_probAD),
        probDP(in_probDP)
        {}
    bool isZero() const { return std::abs(probAD) < 1e-12 && std::abs(probDP) < 1e-12; }
    KrausAmpl operator*(double in_time) const { return KrausAmpl(probAD * in_time, probDP * in_time); }
    // Probability of Amplitude Damping on any single qubit
    double probAD;
    // Probability of Depolarizing noise on any single qubit
    double probDP;
};

// Abstract gate-time provider, 
// i.e. to provide execution time information for an arbitrary gate.
struct IGateTimeConfigProvider
{
    virtual double getGateTime(const xacc::quantum::Gate& in_gate) const = 0;
};

// A default (dummy) gate-time config provider:
// all gates have gate time of 1.0
struct DefaultGateTimeConfigProvider : public IGateTimeConfigProvider
{
    virtual double getGateTime(const xacc::quantum::Gate& in_gate) const override { return 1.0; }
};

// TODO: Remove this struct
// Configuration for noise channels
class KrausConfig
{
public:
    KrausConfig(IGateTimeConfigProvider* gateTimeProvider, const std::vector<KrausAmpl>& krausAmpls):
        m_gateTimeProvider(gateTimeProvider),
        m_krausAmp(krausAmpls)
    {}
    // Returns Kraus amplitudes corresponding to the particular input gate.
    // Note: the order of the result elements is the same as the qubit order of the gate.
    // (e.g. for two-qubit gates)
    std::vector<KrausAmpl> computeKrausAmplitudes(xacc::quantum::Gate& in_gate) const
    {
        const double gateTime = m_gateTimeProvider->getGateTime(in_gate);
        std::vector<KrausAmpl> result;
        for (const auto& bitIdx : in_gate.bits())
        {
            result.emplace_back(m_krausAmp[bitIdx] * gateTime);
        }
        return result;
    }
private:
    // Per-qubit *raw* Kraus amplitudes (i.e. normalized to gate time)
    std::vector<KrausAmpl> m_krausAmp;
    IGateTimeConfigProvider* m_gateTimeProvider;
};

// Helper class to parse and construct noise model
// from IBM's backend JSON.
class IBMNoiseModel
{
public:
    void loadJson(const std::string& in_jsonString);
private:
    // Parsed parameters needed for noise model construction.
    size_t m_nbQubits;
    std::vector<double> m_qubitT1;
    std::vector<double> m_qubitT2;
    std::vector<std::unordered_map<std::string, double>> m_gateErrors;
    std::vector<std::unordered_map<std::string, double>> m_gateDurations;
    std::vector<std::pair<double, double>> m_roErrors;
};
}
