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
#include "IRTransformation.hpp"

namespace xacc {
namespace quantum {
class NearestNeighborTransform : public IRTransformation
{
public:
    NearestNeighborTransform() {}
    void apply(std::shared_ptr<CompositeInstruction> in_program,
                const std::shared_ptr<Accelerator> in_accelerator,
                const HeterogeneousMap& in_options = {}) override
    {
        // The option is qubit-distance (default is 1, adjacent qubits)
        // any 2-qubit gates that are further than this distance are converted into
        // Swap + Original Gate (at qubit distance) + Swap
        // Strategy: meet in the middle:
        // e.g. CNOT q0, q9;  qubit-distance = 2 (max distance is next neighbor, e.g. CNOT q0, q2; CNOT q7, q9 are okay)
        // then we will Swap q0->q4; q9->q6; then CNOT q4, q6; then Swap q4->q0 and Swap q6->q9

        int maxDistance = 1;
        if (in_options.keyExists<int>("max-distance"))
        {
            maxDistance = in_options.get<int>("max-distance");
        }

        auto provider = xacc::getIRProvider("quantum");
        auto flattenedProgram = provider->createComposite(in_program->name() + "_Flattened");
        InstructionIterator it(in_program);
        while (it.hasNext())
        {
            auto nextInst = it.next();
            if (nextInst->isEnabled() && !nextInst->isComposite())
            {
                flattenedProgram->addInstruction(nextInst->clone());
            }
        }

        auto transformedProgram = provider->createComposite(in_program->name() + "_Transformed");
        for (int i = 0; i < flattenedProgram->nInstructions(); ++i)
        {
            auto inst = flattenedProgram->getInstruction(i);

            const auto exceedMaxDistance = [&maxDistance](int q1, int q2)->bool {
                return std::abs(q1 - q2) > maxDistance;
            };

            if (inst->bits().size() == 2 && exceedMaxDistance(inst->bits()[0], inst->bits()[1]))
            {
                const int origLowerIdx = std::min({inst->bits()[0], inst->bits()[1]});
                const int origUpperIdx = std::max({inst->bits()[0], inst->bits()[1]});
                size_t lowerIdx = origLowerIdx;
                size_t upperIdx = origUpperIdx;
                // Insert swaps
                for (;;/*Break inside*/)
                {
                    transformedProgram->addInstruction(provider->createInstruction("Swap", {lowerIdx, lowerIdx + 1}));
                    lowerIdx++;

                    if (!exceedMaxDistance(lowerIdx, upperIdx))
                    {
                        break;
                    }

                    transformedProgram->addInstruction(provider->createInstruction("Swap", {upperIdx, upperIdx - 1}));
                    upperIdx--;

                    if (!exceedMaxDistance(lowerIdx, upperIdx))
                    {
                        break;
                    }
                }

                // Run new gate
                const bool bitCompare = inst->bits()[0] < inst->bits()[1];
                inst->setBits(bitCompare ? std::vector<size_t> {lowerIdx, upperIdx} : std::vector<size_t> {upperIdx, lowerIdx});
                transformedProgram->addInstruction(inst);

                // Insert swaps
                for (size_t i = lowerIdx; i > origLowerIdx; --i)
                {
                    transformedProgram->addInstruction(provider->createInstruction("Swap", {i, i - 1}));
                }

                for (size_t i = upperIdx; i < origUpperIdx; ++i)
                {
                    transformedProgram->addInstruction(provider->createInstruction("Swap", {i, i + 1}));
                }
            }
            else
            {
                transformedProgram->addInstruction(inst);
            }

        }
        // DEBUG:
        // std::cout << "After transform: \n" <<  transformedProgram->toString() << "\n";
        in_program->clear();
        in_program->addInstructions(transformedProgram->getInstructions());

        return;
    }

    const IRTransformationType type() const override { return IRTransformationType::Placement; }
    const std::string name() const override { return "lnn-transform"; }
    const std::string description() const override { return ""; }
};
} // namespace quantum
} // namespace xacc