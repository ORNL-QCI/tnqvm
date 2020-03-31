#pragma once
#include <unordered_set>
#include "xacc.hpp"
#include <cassert>

namespace tnqvm {
// Describes aggration configurations
struct AggreratorConfigs
{
    AggreratorConfigs(int in_width): 
        maxWidth(in_width)
    {}
    int maxWidth;
};

struct AggreratedGroup
{
   std::unordered_set<int> qubitIdx;
   std::vector<xacc::Instruction*> instructions;   
};

class TensorAggrerator
{
public:
    static constexpr int DEFAULT_WIDTH = 4;
    TensorAggrerator() :
        m_configs(DEFAULT_WIDTH)
    {}

    TensorAggrerator(const AggreratorConfigs& in_configs) :
        m_configs(in_configs)
    {}

    void addGate(xacc::Instruction* in_gateInstruction)
    {
        AggreratedGroup& group = getGroup(in_gateInstruction);
        group.instructions.emplace_back(in_gateInstruction);
    }

private:
    void flush(const AggreratedGroup& in_group) 
    {
        // TODO: construct sub-tensor network associated with this group
        // DEBUG:
        std::cout << "Flushing qubit line ";
        for (const auto& id : in_group.qubitIdx)
        {
            std::cout << id << ", ";
        }
        std::cout << "\n";

        for (const auto& inst : in_group.instructions)
        {
            std::cout << inst->toString() << "\n";
        }
    }   
    
    AggreratedGroup& getGroup(xacc::Instruction* in_gateInstruction)
    {
        std::cout << "Process " << in_gateInstruction->toString() << "\n";
        if (in_gateInstruction->bits().size() == 1)
        {
            // If 1-qubit gate:
            auto existingGroupIter = m_qubitToGroup.find(in_gateInstruction->bits()[0]);
            if (existingGroupIter != m_qubitToGroup.end())
            {
                return *(existingGroupIter->second);
            }
            else
            {
                // New group
                AggreratedGroup newGroup;
                newGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);

                m_groups.emplace_back(newGroup);
                m_qubitToGroup.emplace(in_gateInstruction->bits()[0], &m_groups.back());

                return m_groups.back();
            }
        }
        else if (in_gateInstruction->bits().size() == 2)
        {
            // If 2-qubit gate:
            auto existingGroupIter1 = m_qubitToGroup.find(in_gateInstruction->bits()[0]);
            auto existingGroupIter2 = m_qubitToGroup.find(in_gateInstruction->bits()[1]);
            

            // Both qubits are in the same group (including non-existence)
            if (existingGroupIter1 == m_qubitToGroup.end() && existingGroupIter2 == m_qubitToGroup.end())
            {
                // Non-exist: by default, we will group both of qubits into the same group on first encounter.
                // This can be done otherwise, but will need multiple-passes or some forms of look-ahead.
                AggreratedGroup newGroup;
                newGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);
                newGroup.qubitIdx.emplace(in_gateInstruction->bits()[1]);

                m_groups.emplace_back(newGroup);
                m_qubitToGroup.emplace(in_gateInstruction->bits()[0], &m_groups.back());
                m_qubitToGroup.emplace(in_gateInstruction->bits()[1], &m_groups.back());

                return m_groups.back();
            }

            // First qubit doesn't belong to any groups
            if (existingGroupIter1 == m_qubitToGroup.end())
            {
                // First qubit hasn't been tracked in any group.
                assert(existingGroupIter2 != m_qubitToGroup.end());
                if (existingGroupIter2->second->qubitIdx.size() < m_configs.maxWidth)
                {
                    // Still have some room, add first qubit to this group 
                    std::cout << "Current bits:"; 
                    for (const auto& x : existingGroupIter2->second->qubitIdx)
                    {
                        std::cout << x << ", ";
                    }
                    std::cout << "\n Insert bit: " << in_gateInstruction->bits()[0] << "\n"; 
                    existingGroupIter2->second->qubitIdx.emplace(in_gateInstruction->bits()[0]);
                    return *(existingGroupIter2->second);
                }
                else
                {
                    // Max room => flush the group
                    flush(*(existingGroupIter2->second));
                    for (const auto& idx :  existingGroupIter2->second->qubitIdx)
                    {
                        // Erase the group tracking after flushing
                        // i.e. these lines are avalable for free aggregation
                        m_qubitToGroup.erase(idx);
                    }
                    // Now, both qubit lines are free, create a new aggreration group
                    AggreratedGroup newGroup;
                    newGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);
                    newGroup.qubitIdx.emplace(in_gateInstruction->bits()[1]);

                    m_groups.emplace_back(newGroup);
                    m_qubitToGroup.emplace(in_gateInstruction->bits()[0], &m_groups.back());
                    m_qubitToGroup.emplace(in_gateInstruction->bits()[1], &m_groups.back());

                    return m_groups.back();
                }
            }

            // Second qubit doesn't belong to any groups
            if(existingGroupIter2 == m_qubitToGroup.end())
            {
                if (existingGroupIter1->second->qubitIdx.size() < m_configs.maxWidth)
                {
                    // Still have some room, add second qubit to this group 
                    existingGroupIter1->second->qubitIdx.emplace(in_gateInstruction->bits()[1]);
                    return *(existingGroupIter1->second);
                }
                else
                {
                    // Max room => flush the group
                    flush(*(existingGroupIter1->second));
                    for (const auto& idx :  existingGroupIter1->second->qubitIdx)
                    {
                        // Erase the group tracking after flushing
                        // i.e. these lines are avalable for free aggregation
                        m_qubitToGroup.erase(idx);
                    }
                    // Now, both qubit lines are free, create a new aggreration group
                    AggreratedGroup newGroup;
                    newGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);
                    newGroup.qubitIdx.emplace(in_gateInstruction->bits()[1]);

                    m_groups.emplace_back(newGroup);
                    m_qubitToGroup.emplace(in_gateInstruction->bits()[0], &m_groups.back());
                    m_qubitToGroup.emplace(in_gateInstruction->bits()[1], &m_groups.back());

                    return m_groups.back();
                }
            }

            // Both qubits are having a group here.
            assert(existingGroupIter1 != m_qubitToGroup.end() && existingGroupIter2 != m_qubitToGroup.end());
            if (existingGroupIter1->second->qubitIdx == existingGroupIter2->second->qubitIdx)
            {
                // They are the same group
                return *(existingGroupIter1->second);
            }
            else
            {
                // Both qubits have been assigned to two different groups
                // Must flush the aggregated tensor networks associated with both groups.
                flush(*(existingGroupIter1->second));
                for (const auto& idx :  existingGroupIter1->second->qubitIdx)
                {
                    m_qubitToGroup.erase(idx);
                }

                flush(*(existingGroupIter2->second));
                for (const auto& idx :  existingGroupIter2->second->qubitIdx)
                {
                    m_qubitToGroup.erase(idx);
                }
                
                // Now, both qubit lines are free, create a new aggreration group
                AggreratedGroup newGroup;
                newGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);
                newGroup.qubitIdx.emplace(in_gateInstruction->bits()[1]);

                m_groups.emplace_back(newGroup);
                m_qubitToGroup.emplace(in_gateInstruction->bits()[0], &m_groups.back());
                m_qubitToGroup.emplace(in_gateInstruction->bits()[1], &m_groups.back());

                return m_groups.back();
            }
        }
        else
        {
            xacc::error("Unsupported gates encountered!");
            throw; 
        }        
    }

private:
    AggreratorConfigs m_configs;
    std::unordered_map<int, AggreratedGroup*> m_qubitToGroup;
    std::deque<AggreratedGroup> m_groups;
};
}