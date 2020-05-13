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