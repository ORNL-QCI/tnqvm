#pragma once
#include "xacc.hpp"
#include "Identifiable.hpp"

namespace tnqvm {
struct GateTensor
{
    std::string uniqueName;
    std::vector<int> tensorShape;
    std::vector<std::complex<double>> tensorData;
    std::pair<std::vector<unsigned int>,  std::vector<unsigned int>> tensorIsometry; 
};

// Shared service to construct Gate Tensor
class GateTensorConstructor : public xacc::Identifiable
{
public: 
    const std::string name() const override { return "default"; }
    const std::string description() const override { return ""; }
    GateTensor getGateTensor(xacc::Instruction& in_gate);
};
}