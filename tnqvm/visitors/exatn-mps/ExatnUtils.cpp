/***********************************************************************************
 * Copyright (c) 2017, UT-Battelle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the xacc nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Implementation - Thien Nguyen
 *
*/

#include "ExatnUtils.hpp"
#include "base/Gates.hpp"

namespace tnqvm {
GateTensor GateTensorConstructor::getGateTensor(xacc::Instruction& in_gate)
{
    GateTensor resultTensor;
    static const std::vector<int> SINGLE_QUBIT_SHAPE{2, 2};
    static const std::vector<int> TWO_QUBIT_SHAPE{2, 2, 2, 2};
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
            case CommonGates::T: return GetGateMatrix<CommonGates::T>();
            case CommonGates::Tdg:
              return GetGateMatrix<CommonGates::Tdg>();
            case CommonGates::U:
              return GetGateMatrix<CommonGates::U>(
                  in_gate.getParameter(0).as<double>(),
                  in_gate.getParameter(1).as<double>(),
                  in_gate.getParameter(2).as<double>());
            case CommonGates::CNOT:
              return GetGateMatrix<CommonGates::CNOT>();
            case CommonGates::CY:
              return GetGateMatrix<CommonGates::CY>();
            case CommonGates::CZ:
              return GetGateMatrix<CommonGates::CZ>();
            case CommonGates::CH:
              return GetGateMatrix<CommonGates::CH>();
            case CommonGates::CRZ:
              return GetGateMatrix<CommonGates::CRZ>(in_gate.getParameter(0).as<double>());
            case CommonGates::CPhase:
              return GetGateMatrix<CommonGates::CPhase>(in_gate.getParameter(0).as<double>());
            case CommonGates::Swap: return GetGateMatrix<CommonGates::Swap>();
            case CommonGates::iSwap: return GetGateMatrix<CommonGates::iSwap>();
            case CommonGates::fSim: return GetGateMatrix<CommonGates::fSim>(in_gate.getParameter(0).as<double>(), in_gate.getParameter(1).as<double>());
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
