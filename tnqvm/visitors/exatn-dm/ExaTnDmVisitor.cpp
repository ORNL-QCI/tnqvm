#include "ExaTnDmVisitor.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include "base/Gates.hpp"
#include "NoiseModel.hpp"
#include "xacc_service.hpp"
#include "xacc_plugin.hpp"
#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

namespace tnqvm {
ExaTnDmVisitor::ExaTnDmVisitor() {
  // TODO
}

void ExaTnDmVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer,
                                int nbShots) {
  // Initialize ExaTN (if not already initialized)
  if (!exatn::isInitialized()) {
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
    exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel()
                                                  : 0);
    xacc::subscribeLoggingLevel([](int level) {
      exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
    });
  }

  // TODO
}

exatn::TensorNetwork
ExaTnDmVisitor::buildInitialNetwork(size_t in_nbQubits,
                                    bool in_createQubitTensors) const {
  // TODO
}

void ExaTnDmVisitor::finalize() {
  // TODO
}

void ExaTnDmVisitor::applySingleQubitGate(
    xacc::quantum::Gate &in_gateInstruction) {
  // TODO
}

void ExaTnDmVisitor::applyTwoQubitGate(
    xacc::quantum::Gate &in_gateInstruction) {
  // TODO
}
void ExaTnDmVisitor::applyKrausOp(const KrausOp &in_op) {
  // TODO
}

void ExaTnDmVisitor::visit(Identity &in_IdentityGate) {
  applySingleQubitGate(in_IdentityGate);
}

void ExaTnDmVisitor::visit(Hadamard &in_HadamardGate) {
  applySingleQubitGate(in_HadamardGate);
  // DEBUG:
  // std::cout << "Apply: " << in_HadamardGate.toString() << "\n";
  // printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

void ExaTnDmVisitor::visit(X &in_XGate) {
  applySingleQubitGate(in_XGate);
  // DEBUG:
  // std::cout << "Apply: " << in_XGate.toString() << "\n";
  // printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

void ExaTnDmVisitor::visit(Y &in_YGate) { applySingleQubitGate(in_YGate); }

void ExaTnDmVisitor::visit(Z &in_ZGate) { applySingleQubitGate(in_ZGate); }

void ExaTnDmVisitor::visit(Rx &in_RxGate) { applySingleQubitGate(in_RxGate); }

void ExaTnDmVisitor::visit(Ry &in_RyGate) { applySingleQubitGate(in_RyGate); }

void ExaTnDmVisitor::visit(Rz &in_RzGate) { applySingleQubitGate(in_RzGate); }

void ExaTnDmVisitor::visit(T &in_TGate) { applySingleQubitGate(in_TGate); }

void ExaTnDmVisitor::visit(Tdg &in_TdgGate) {
  applySingleQubitGate(in_TdgGate);
}

// others
void ExaTnDmVisitor::visit(Measure &in_MeasureGate) {
  m_measuredBits.emplace_back(in_MeasureGate.bits()[0]);
}

void ExaTnDmVisitor::visit(U &in_UGate) { applySingleQubitGate(in_UGate); }

// two-qubit gates:
void ExaTnDmVisitor::visit(CNOT &in_CNOTGate) {
  applyTwoQubitGate(in_CNOTGate);
  // DEBUG:
  // std::cout << "Apply: " << in_CNOTGate.toString() << "\n";
  // printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

void ExaTnDmVisitor::visit(Swap &in_SwapGate) {
  applySingleQubitGate(in_SwapGate);
}

void ExaTnDmVisitor::visit(CZ &in_CZGate) { applySingleQubitGate(in_CZGate); }

void ExaTnDmVisitor::visit(CPhase &in_CPhaseGate) {
  applySingleQubitGate(in_CPhaseGate);
}

void ExaTnDmVisitor::visit(iSwap &in_iSwapGate) {
  applySingleQubitGate(in_iSwapGate);
}

void ExaTnDmVisitor::visit(fSim &in_fsimGate) {
  applySingleQubitGate(in_fsimGate);
}

const double ExaTnDmVisitor::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> in_function) {
  return 0.0;
}
} // namespace tnqvm

// Register with CppMicroservices
REGISTER_PLUGIN(tnqvm::ExaTnDmVisitor, tnqvm::TNQVMVisitor)
