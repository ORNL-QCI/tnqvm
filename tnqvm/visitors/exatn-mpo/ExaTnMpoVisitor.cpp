#include "ExaTnMpoVisitor.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include <map>
#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

#ifdef TNQVM_MPI_ENABLED
#include "mpi.h"
#endif

namespace tnqvm {
ExatnMpoVisitor::ExatnMpoVisitor()
{
    // TODO
}

void ExatnMpoVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) 
{ 
    // TODO
}
void ExatnMpoVisitor::finalize() 
{ 
    // TODO
}

void ExatnMpoVisitor::visit(Identity& in_IdentityGate) 
{ 
    
}

void ExatnMpoVisitor::visit(Hadamard& in_HadamardGate) 
{ 
   
}

void ExatnMpoVisitor::visit(X& in_XGate) 
{ 
   
}

void ExatnMpoVisitor::visit(Y& in_YGate) 
{ 
    
}

void ExatnMpoVisitor::visit(Z& in_ZGate) 
{ 
    
}

void ExatnMpoVisitor::visit(Rx& in_RxGate) 
{ 
    
}

void ExatnMpoVisitor::visit(Ry& in_RyGate) 
{ 
    
}

void ExatnMpoVisitor::visit(Rz& in_RzGate) 
{ 
    
}

void ExatnMpoVisitor::visit(T& in_TGate) 
{ 
   
}

void ExatnMpoVisitor::visit(Tdg& in_TdgGate) 
{ 
   
}

// others
void ExatnMpoVisitor::visit(Measure& in_MeasureGate) 
{ 
}

void ExatnMpoVisitor::visit(U& in_UGate) 
{ 
    
}

// two-qubit gates: 
void ExatnMpoVisitor::visit(CNOT& in_CNOTGate) 
{ 
    
}

void ExatnMpoVisitor::visit(Swap& in_SwapGate) 
{ 
    
}

void ExatnMpoVisitor::visit(CZ& in_CZGate) 
{ 
   
}

void ExatnMpoVisitor::visit(CPhase& in_CPhaseGate) 
{ 
   
}

void ExatnMpoVisitor::visit(iSwap& in_iSwapGate) 
{
    
}

void ExatnMpoVisitor::visit(fSim& in_fsimGate) 
{
   
}

const double ExatnMpoVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
{ 
    return 0.0;
}
}
