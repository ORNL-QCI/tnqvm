#pragma once
#include <unordered_set>
#include "xacc.hpp"
#include <cassert>

namespace tnqvm {
// Describes aggregation configurations
struct AggregatorConfigs
{
    AggregatorConfigs(int in_width): 
        maxWidth(in_width)
    {}
    int maxWidth;
};

struct AggregatedGroup
{
    AggregatedGroup():
        flushed(false)
    {}
    std::unordered_set<int> qubitIdx;
    std::vector<xacc::Instruction*> instructions;   
    bool flushed;
};

struct IAggregatorListener
{
    virtual void onFlush(const AggregatedGroup& in_group) = 0;
};

class TensorAggregator
{
public:
    static constexpr int DEFAULT_WIDTH = 4;
    TensorAggregator(IAggregatorListener* io_listener) :
        m_configs(DEFAULT_WIDTH),
        m_listener(io_listener)
    {}

    TensorAggregator(const AggregatorConfigs& in_configs, IAggregatorListener* io_listener) :
        m_configs(in_configs),
        m_listener(io_listener)

    {}

    void addGate(xacc::Instruction* in_gateInstruction)
    {
        AggregatedGroup& group = getGroup(in_gateInstruction);
        group.instructions.emplace_back(in_gateInstruction);
    }
    // Flush all remaining groups (not yet flushed while adding gates)
    void flushAll()
    {
        for(auto& group : m_groups)
        {
            if(!group.flushed)
            {
                flush(group);
            }
        }

        if (!m_pendingGroup.instructions.empty())
        {
            flush(m_pendingGroup);
        }
    }

private:
    void flush(AggregatedGroup& io_group) 
    {
        io_group.flushed = true;
        m_listener->onFlush(io_group);
    }   
    
    AggregatedGroup& getGroup(xacc::Instruction* in_gateInstruction)
    {
        // std::cout << "Process " << in_gateInstruction->toString() << "\n";
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
                if (m_pendingGroup.qubitIdx.size() < m_configs.maxWidth)
                {
                    m_pendingGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);
                    return m_pendingGroup;
                }
                else
                {
                    // Add the pending group to the master list
                    // since it is full.
                    m_groups.emplace_back(m_pendingGroup);
                    for (const auto& bit : m_pendingGroup.qubitIdx)
                    {
                        m_qubitToGroup.emplace(bit, &m_groups.back());
                    }
                    // Crete a new pending group for 1-q gates
                    AggregatedGroup newGroup;
                    newGroup.qubitIdx.emplace(in_gateInstruction->bits()[0]);
                    m_pendingGroup = newGroup;
                    return m_pendingGroup;
                }
            }
        }
        else if (in_gateInstruction->bits().size() == 2)
        {
            // Elevate the pending group if it contains a qubit line of this 2-qubit gate.
            if (m_pendingGroup.qubitIdx.find(in_gateInstruction->bits()[0]) != m_pendingGroup.qubitIdx.end() ||
                m_pendingGroup.qubitIdx.find(in_gateInstruction->bits()[1]) != m_pendingGroup.qubitIdx.end())
            {
                m_groups.emplace_back(m_pendingGroup);
                for (const auto& bit : m_pendingGroup.qubitIdx)
                {
                    m_qubitToGroup.emplace(bit, &m_groups.back());
                }
                // Crete a new pending group 
                AggregatedGroup newGroup;
                m_pendingGroup = newGroup;
            }

            // If 2-qubit gate:
            auto existingGroupIter1 = m_qubitToGroup.find(in_gateInstruction->bits()[0]);
            auto existingGroupIter2 = m_qubitToGroup.find(in_gateInstruction->bits()[1]);
            

            // Both qubits are in the same group (including non-existence)
            if (existingGroupIter1 == m_qubitToGroup.end() && existingGroupIter2 == m_qubitToGroup.end())
            {
                // Non-exist: by default, we will group both of qubits into the same group on first encounter.
                // This can be done otherwise, but will need multiple-passes or some forms of look-ahead.
                AggregatedGroup newGroup;
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
                    // std::cout << "Current bits:"; 
                    // for (const auto& x : existingGroupIter2->second->qubitIdx)
                    // {
                    //     std::cout << x << ", ";
                    // }
                    // std::cout << "\n Insert bit: " << in_gateInstruction->bits()[0] << "\n"; 
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
                    // Now, both qubit lines are free, create a new aggregation group
                    AggregatedGroup newGroup;
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
                    // Now, both qubit lines are free, create a new aggregation group
                    AggregatedGroup newGroup;
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
                
                // Now, both qubit lines are free, create a new aggregation group
                AggregatedGroup newGroup;
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
    AggregatorConfigs m_configs;
    std::unordered_map<int, AggregatedGroup*> m_qubitToGroup;
    std::deque<AggregatedGroup> m_groups;
    // Pending group of 1-qubit gate:
    // We group 1-qubit gate into 1 group (up to max-width).
    AggregatedGroup m_pendingGroup;
    IAggregatorListener* m_listener;
};
}