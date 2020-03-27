#include "ExaTnMpsVisitor.hpp"

namespace tnqvm {

ExatnMpsVisitor::ExatnMpsVisitor()
{
    // TODO
}

void ExatnMpsVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) 
{ 
    // TODO 
}

void ExatnMpsVisitor::finalize() { 
    // TODO 
}


void ExatnMpsVisitor::visit(Identity& in_IdentityGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Hadamard& in_HadamardGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(X& in_XGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Y& in_YGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Z& in_ZGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Rx& in_RxGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Ry& in_RyGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Rz& in_RzGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(T& in_TGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Tdg& in_TdgGate) 
{ 
    // TODO 
}

// others
void ExatnMpsVisitor::visit(Measure& in_MeasureGate) 
{ 
    // TODO 
}



void ExatnMpsVisitor::visit(CPhase& in_CPhaseGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(U& in_UGate) 
{ 
    // TODO 
}

// two-qubit gates: 
// NOTE: these gates are IMPORTANT for gate clustering consideration
void ExatnMpsVisitor::visit(CNOT& in_CNOTGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(Swap& in_SwapGate) 
{ 
    // TODO 
}

void ExatnMpsVisitor::visit(CZ& in_CZGate) 
{ 
    // TODO 
}


const double ExatnMpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
{ 
    // TODO
    return 0.0;
}

}
