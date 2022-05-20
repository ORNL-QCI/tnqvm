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
#include "Identifiable.hpp"
#include <chrono>

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
    static GateTensor getGateTensor(xacc::Instruction& in_gate);
};

// Stat utilities
namespace Stat
{
struct FunctionCallStat
{
    std::string name;
    std::chrono::duration<double>  totalDuration;
    int numberCalls;
    std::chrono::duration<double>  maxDuration;
    std::chrono::duration<double>  minDuration;

    FunctionCallStat(const std::string& in_name):
        name(in_name),
        totalDuration(std::chrono::duration<double>::zero()),
        numberCalls(0),
        maxDuration(std::chrono::duration<double>::zero()),
        minDuration(std::chrono::duration<double>::zero())
    {};

    void addSample(std::chrono::system_clock::time_point in_begin, std::chrono::system_clock::time_point in_end)
    {
        const std::chrono::duration<double> callDuration = in_end - in_begin;
        if (numberCalls == 0)
        {
            maxDuration = callDuration;
            minDuration = callDuration;
        }

        numberCalls++;
        totalDuration += callDuration;
        if (callDuration > maxDuration)
        {
            maxDuration = callDuration;
        }

        if (callDuration < minDuration)
        {
            minDuration = callDuration;
        }
    };

    std::string toString(bool in_shouldClear = false)
    {
        std::string resultStr;
        const std::string preFix = "<<" + name + ">>: ";
        if (numberCalls == 0)
        {
            resultStr = preFix + "NO DATA.";
        }
        else
        {
            const auto avg = totalDuration/numberCalls;

            resultStr = preFix + "Number of calls = " + std::to_string(numberCalls) +
                            "; Total time = " + std::to_string(totalDuration.count()) + " [secs]; " +
                            "\nAvg = " + std::to_string(avg.count()) + " [secs]; " +
                            "\nMin = " + std::to_string(minDuration.count()) + " [secs]; " +
                            "\nMax = " + std::to_string(maxDuration.count()) + " [secs]; ";
        }

        if (in_shouldClear)
        {
            totalDuration = std::chrono::duration<double>::zero();
            numberCalls = 0;
            maxDuration = std::chrono::duration<double>::zero();
            minDuration = std::chrono::duration<double>::zero();
        }
        return resultStr;
    };
};
}
}
