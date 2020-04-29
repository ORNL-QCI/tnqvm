#pragma once
#include "xacc.hpp"

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

            if (nbQubits < 2 || nbLayers < 1)
            {
                return false;
            }
            
            auto gateRegistry = xacc::getService<IRProvider>("quantum");
            const auto randomGateGen = [&gateRegistry](size_t qubitIdx) -> std::shared_ptr<Instruction> {
                // Defines the single-qubit gate set to draw from.
                // For now, just hard-coded a pre-defined gate set
                const std::vector<std::string> GATE_SET { "H", "X", "Y", "Z", "T", "Rx", "Ry", "Rz" };
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