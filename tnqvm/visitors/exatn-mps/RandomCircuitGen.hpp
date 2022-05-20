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

#pragma once

#include "xacc.hpp"
#include "xacc_service.hpp"
#include "Circuit.hpp"
#include "IRProvider.hpp"

namespace xacc {
namespace circuits {
class RCS : public xacc::quantum::Circuit {
public:
    RCS() : Circuit("rcs") {}

    bool expand(const xacc::HeterogeneousMap& runtimeOptions) override
    {
        if (runtimeOptions.keyExists<int>("nq") && runtimeOptions.keyExists<int>("nlayers"))
        {
            const int nbQubits = runtimeOptions.get<int>("nq");
            const int nbLayers = runtimeOptions.get<int>("nlayers");
            bool parametricGate = true;

            if (runtimeOptions.keyExists<bool>("parametric-gates"))
            {
                parametricGate  = runtimeOptions.get<bool>("parametric-gates");
            }

            if (nbQubits < 2 || nbLayers < 1)
            {
                return false;
            }
            // Defines the single-qubit gate set to draw from.
            // For now, just hard-coded a pre-defined gate set
            const std::vector<std::string> GATE_SET = parametricGate ?
                std::vector<std::string> { "H", "X", "Y", "Z", "T", "Rx", "Ry", "Rz" } :
                std::vector<std::string> { "H", "X", "Y", "Z", "T" };

            auto gateRegistry = xacc::getService<IRProvider>("quantum");
            const auto randomGateGen = [&](size_t qubitIdx) -> std::shared_ptr<Instruction> {
                auto randIt = GATE_SET.begin();
                std::advance(randIt, std::rand() % GATE_SET.size());

                const std::string selectedGate = *randIt;

                if (selectedGate == "Rx" || selectedGate == "Ry" || selectedGate == "Rz")
                {
                    const auto randomAngle = []() -> double {
                        // Random floating point b/w 0 and 1.0
                        const double random = ((double) rand()) / (double) RAND_MAX;
                        // Random angle in the -pi -> pi range
                        return random * 2 * M_PI - M_PI;
                    };

                    InstructionParameter angle(randomAngle());
                    auto gate = gateRegistry->createInstruction(selectedGate, std::vector<std::size_t>{ qubitIdx });
                    gate->setParameter(0, angle);
                    return gate;
                }
                else
                {
                    return gateRegistry->createInstruction(selectedGate, std::vector<std::size_t>{ qubitIdx });
                }
            };

            for (size_t i = 0; i < nbLayers; ++i)
            {
                // Single-qubit gate layer
                for (size_t j = 0; j < nbQubits; ++j)
                {
                    addInstruction(randomGateGen(j));
                }

                // CNOT layers
                for (size_t j = 0; j < nbQubits - 1; ++j)
                {
                    auto cnot = gateRegistry->createInstruction("CX", std::vector<std::size_t>{j, j + 1});
                    addInstruction(cnot);
                }
            }

            // Add measurement instructions on all qubits:
            for (size_t j = 0; j < nbQubits; ++j)
            {
                auto measGate = gateRegistry->createInstruction("Measure", std::vector<std::size_t>{ j });
                addInstruction(measGate);
            }

            return true;
        }
        return false;
    }

    const std::vector<std::string> requiredKeys() override
    {
        // Number of qubits and number of layers
        return { "nq", "nlayers"};
    }

    DEFINE_CLONE(RCS);
};
}
}
