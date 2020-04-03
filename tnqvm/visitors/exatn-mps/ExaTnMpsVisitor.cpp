#include "ExaTnMpsVisitor.hpp"

namespace tnqvm {

ExatnMpsVisitor::ExatnMpsVisitor():
    m_aggrerator(this)
{
    // TODO
}

void ExatnMpsVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) 
{ 
    // Check if we have any specific config for the gate aggregator
    if (options.keyExists<int>("agg-width"))
    {
        const int aggregatorWidth = options.get<int>("agg-width");
        AggreratorConfigs configs(aggregatorWidth);
        TensorAggrerator newAggr(configs, this);
        m_aggrerator = newAggr;
    }
    // TODO 
}

void ExatnMpsVisitor::finalize() { 
    // TODO 
}


void ExatnMpsVisitor::visit(Identity& in_IdentityGate) 
{ 
    // TODO 
    m_aggrerator.addGate(&in_IdentityGate);
}

void ExatnMpsVisitor::visit(Hadamard& in_HadamardGate) 
{ 
    // TODO 
    m_aggrerator.addGate(&in_HadamardGate);
}

void ExatnMpsVisitor::visit(X& in_XGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_XGate); 
}

void ExatnMpsVisitor::visit(Y& in_YGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_YGate); 
}

void ExatnMpsVisitor::visit(Z& in_ZGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_ZGate); 
}

void ExatnMpsVisitor::visit(Rx& in_RxGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_RxGate); 
}

void ExatnMpsVisitor::visit(Ry& in_RyGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_RyGate); 
}

void ExatnMpsVisitor::visit(Rz& in_RzGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_RzGate); 
}

void ExatnMpsVisitor::visit(T& in_TGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_TGate); 
}

void ExatnMpsVisitor::visit(Tdg& in_TdgGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_TdgGate); 
}

// others
void ExatnMpsVisitor::visit(Measure& in_MeasureGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_MeasureGate); 
}

void ExatnMpsVisitor::visit(U& in_UGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_UGate); 
}

// two-qubit gates: 
// NOTE: these gates are IMPORTANT for gate clustering consideration
void ExatnMpsVisitor::visit(CNOT& in_CNOTGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_CNOTGate); 
}

void ExatnMpsVisitor::visit(Swap& in_SwapGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_SwapGate); 
}

void ExatnMpsVisitor::visit(CZ& in_CZGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_CZGate); 
}

void ExatnMpsVisitor::visit(CPhase& in_CPhaseGate) 
{ 
    // TODO
    m_aggrerator.addGate(&in_CPhaseGate); 
}

const double ExatnMpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
{ 
    // TODO
    return 0.0;
}

void ExatnMpsVisitor::onFlush(const AggreratedGroup& in_group)
{
    // DEBUG:
    std::cout << "Flushing qubit line ";
    for (const auto& id : in_group.qubitIdx)
    {
        std::cout << id << ", ";
    }
    std::cout << "|| Number of gates = " << in_group.instructions.size() << "\n";

    for (const auto& inst : in_group.instructions)
    {
        std::cout << inst->toString() << "\n";
    }
    std::cout << "=============================================\n";
}
}
