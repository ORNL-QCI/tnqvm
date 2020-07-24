#include "ExaTnPmpsVisitor.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include <map>
#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

namespace tnqvm {
ExaTnPmpsVisitor::ExaTnPmpsVisitor()
{
    // TODO
}

void ExaTnPmpsVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) 
{ 
    // Initialize ExaTN (if not already initialized)
    if (!exatn::isInitialized()) 
    {
#ifdef TNQVM_EXATN_USES_MKL_BLAS
        // Fix for TNQVM bug #30
        void *core_handle =
            dlopen("@EXATN_MKL_PATH@/libmkl_core@CMAKE_SHARED_LIBRARY_SUFFIX@",
                    RTLD_LAZY | RTLD_GLOBAL);
        if (core_handle == nullptr) {
            std::string err = std::string(dlerror());
            xacc::error("Could not load mkl_core - " + err);
        }

        void *thread_handle = dlopen(
            "@EXATN_MKL_PATH@/libmkl_gnu_thread@CMAKE_SHARED_LIBRARY_SUFFIX@",
            RTLD_LAZY | RTLD_GLOBAL);
        if (thread_handle == nullptr) {
            std::string err = std::string(dlerror());
            xacc::error("Could not load mkl_gnu_thread - " + err);
        }
#endif
        // If exaTN has not been initialized, do it now.
        exatn::initialize();
        // ExaTN and XACC logging levels are always in-synced.
        exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
        xacc::subscribeLoggingLevel([](int level) {
            exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
        });
    }

    m_buffer = buffer;
    m_pmpsTensorNetwork = buildInitialNetwork(buffer->size());
}

exatn::TensorNetwork ExaTnPmpsVisitor::buildInitialNetwork(size_t in_nbQubits) const
{
    exatn::TensorNetwork purifiedMps("PMPS_Network");
    // TODO: Construct the network, e.g. using exaTN network builder.
    return purifiedMps;
}

void ExaTnPmpsVisitor::finalize() 
{ 
    // TODO
}

void ExaTnPmpsVisitor::visit(Identity& in_IdentityGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Hadamard& in_HadamardGate) 
{ 
   
}

void ExaTnPmpsVisitor::visit(X& in_XGate) 
{ 
   
}

void ExaTnPmpsVisitor::visit(Y& in_YGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Z& in_ZGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Rx& in_RxGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Ry& in_RyGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Rz& in_RzGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(T& in_TGate) 
{ 
   
}

void ExaTnPmpsVisitor::visit(Tdg& in_TdgGate) 
{ 
   
}

// others
void ExaTnPmpsVisitor::visit(Measure& in_MeasureGate) 
{ 
}

void ExaTnPmpsVisitor::visit(U& in_UGate) 
{ 
    
}

// two-qubit gates: 
void ExaTnPmpsVisitor::visit(CNOT& in_CNOTGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Swap& in_SwapGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(CZ& in_CZGate) 
{ 
   
}

void ExaTnPmpsVisitor::visit(CPhase& in_CPhaseGate) 
{ 
   
}

void ExaTnPmpsVisitor::visit(iSwap& in_iSwapGate) 
{
    
}

void ExaTnPmpsVisitor::visit(fSim& in_fsimGate) 
{
   
}

const double ExaTnPmpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
{ 
    return 0.0;
}
}
