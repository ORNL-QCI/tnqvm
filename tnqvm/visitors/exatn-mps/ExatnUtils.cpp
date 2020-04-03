#include "ExatnUtils.hpp"
#include "base/Gates.hpp"

namespace tnqvm {
GateTensor GateTensorConstructor::getGateTensor(xacc::Instruction& in_gate)
{
    GateTensor resultTensor;
    static const std::vector<int> SINGLE_QUBIT_SHAPE{2, 2};
    static const std::vector<int> TWO_QUBIT_SHAPE{2, 2};
    static const std::pair<std::vector<unsigned int>,  std::vector<unsigned int>> SINGLE_QUBIT_ISO{{0}, {1}};
    static const std::pair<std::vector<unsigned int>,  std::vector<unsigned int>> TWO_QUBIT_ISO{{0, 1}, {2, 3}};
    
    resultTensor.tensorShape = (in_gate.nRequiredBits() == 1 ? SINGLE_QUBIT_SHAPE : TWO_QUBIT_SHAPE);
    resultTensor.tensorIsometry = (in_gate.nRequiredBits() == 1 ? SINGLE_QUBIT_ISO : TWO_QUBIT_ISO);
    
    const auto generateGateName = [](xacc::Instruction& in_quantumGate)->std::string {
        if (in_quantumGate.getParameters().empty())
        {
            // Non-parametric gate
            return in_quantumGate.name();
        }

        std::vector<std::string> gateParams;
        
        for (const auto& param: in_quantumGate.getParameters())
        {
            gateParams.emplace_back(std::to_string(param.as<double>()));
        }

        return in_quantumGate.name() + 
            "(" + 
            [&]() -> std::string {
                std::string paramList;
                for (size_t i = 0; i < gateParams.size() - 1; ++i) {
                    paramList.append(gateParams[i] + ",");
                }
                paramList.append(gateParams.back());

                return paramList;
            }() + 
            ")";
    };

    resultTensor.uniqueName = generateGateName(in_gate);

    const auto gateEnum = GetGateType(in_gate.name());
    const auto getMatrix = [&](){
        switch (gateEnum)
        {
            case CommonGates::Rx: return GetGateMatrix<CommonGates::Rx>(in_gate.getParameter(0).as<double>());
            case CommonGates::Ry: return GetGateMatrix<CommonGates::Ry>(in_gate.getParameter(0).as<double>());
            case CommonGates::Rz: return GetGateMatrix<CommonGates::Rz>(in_gate.getParameter(0).as<double>());
            case CommonGates::I: return GetGateMatrix<CommonGates::I>();
            case CommonGates::H: return GetGateMatrix<CommonGates::H>();
            case CommonGates::X: return GetGateMatrix<CommonGates::X>();
            case CommonGates::Y: return GetGateMatrix<CommonGates::Y>();
            case CommonGates::Z: return GetGateMatrix<CommonGates::Z>();
            case CommonGates::CNOT: return GetGateMatrix<CommonGates::CNOT>();
            case CommonGates::Swap: return GetGateMatrix<CommonGates::Swap>();
            default: return GetGateMatrix<CommonGates::I>();
        }
    };
    
    const auto flattenGateMatrix = [](const std::vector<std::vector<std::complex<double>>>& in_gateMatrix){
        std::vector<std::complex<double>> resultVector;
        resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
        for (const auto &row : in_gateMatrix) 
        {
            for (const auto &entry : row) 
            {
                resultVector.emplace_back(entry);
            }
        }

        return resultVector;
    };

    resultTensor.tensorData = flattenGateMatrix(getMatrix());

    return resultTensor;
}
}