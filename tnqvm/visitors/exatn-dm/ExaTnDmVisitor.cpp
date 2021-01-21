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

#define INITIAL_BOND_DIM 1
#define INITIAL_KRAUS_DIM 1
#define QUBIT_DIM 2

namespace {
std::vector<std::complex<double>> Q_ZERO_TENSOR_BODY(size_t in_volume) {
  std::vector<std::complex<double>> body(in_volume, {0.0, 0.0});
  body[0] = std::complex<double>(1.0, 0.0);
  return body;
}

const std::string ROOT_TENSOR_NAME = "Root";

void printDensityMatrix(const exatn::TensorNetwork &in_tensorNet,
                        size_t in_nbQubit) {
  exatn::TensorNetwork tempNetwork(in_tensorNet);
  tempNetwork.rename("__TEMP__" + in_tensorNet.getName());
  const bool evaledOk = exatn::evaluateSync(tempNetwork);
  assert(evaledOk);
  auto talsh_tensor =
      exatn::getLocalTensor(tempNetwork.getTensor(0)->getName());
  const auto expectedDensityMatrixVolume =
      (1ULL << in_nbQubit) * (1ULL << in_nbQubit);
  const auto nbRows = 1ULL << in_nbQubit;
  if (talsh_tensor) {
    const std::complex<double> *body_ptr;
    const bool access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr);
    if (!access_granted) {
      std::cout << "Failed to retrieve tensor data!!!\n";
    } else {
      assert(expectedDensityMatrixVolume == talsh_tensor->getVolume());
      std::complex<double> traceVal{0.0, 0.0};
      int rowId = -1;
      for (int i = 0; i < talsh_tensor->getVolume(); ++i) {
        if (i % nbRows == 0) {
          std::cout << "\n";
          rowId++;
        }
        if (i == (rowId + rowId * nbRows)) {
          traceVal += body_ptr[i];
        }
        const auto &elem = body_ptr[i];
        std::cout << elem << " ";
      }
      std::cout << "\n";
      // Verify trace(density matrix) == 1.0
      assert(std::abs(traceVal.real() - 1.0) < 1e-12);
      assert(std::abs(traceVal.imag()) < 1e-12);
    }
  } else {
    std::cout << "Failed to retrieve tensor data!!!\n";
  }
}
std::vector<std::complex<double>>
getGateMatrix(const xacc::Instruction &in_gate) {
  using namespace tnqvm;
  const auto gateEnum = GetGateType(in_gate.name());
  const auto getMatrix = [&]() {
    switch (gateEnum) {
    case CommonGates::Rx:
      return GetGateMatrix<CommonGates::Rx>(
          in_gate.getParameter(0).as<double>());
    case CommonGates::Ry:
      return GetGateMatrix<CommonGates::Ry>(
          in_gate.getParameter(0).as<double>());
    case CommonGates::Rz:
      return GetGateMatrix<CommonGates::Rz>(
          in_gate.getParameter(0).as<double>());
    case CommonGates::I:
      return GetGateMatrix<CommonGates::I>();
    case CommonGates::H:
      return GetGateMatrix<CommonGates::H>();
    case CommonGates::X:
      return GetGateMatrix<CommonGates::X>();
    case CommonGates::Y:
      return GetGateMatrix<CommonGates::Y>();
    case CommonGates::Z:
      return GetGateMatrix<CommonGates::Z>();
    case CommonGates::T:
      return GetGateMatrix<CommonGates::T>();
    case CommonGates::Tdg:
      return GetGateMatrix<CommonGates::Tdg>();
    case CommonGates::CNOT:
      return GetGateMatrix<CommonGates::CNOT>();
    case CommonGates::Swap:
      return GetGateMatrix<CommonGates::Swap>();
    case CommonGates::iSwap:
      return GetGateMatrix<CommonGates::iSwap>();
    case CommonGates::fSim:
      return GetGateMatrix<CommonGates::fSim>(
          in_gate.getParameter(0).as<double>(),
          in_gate.getParameter(1).as<double>());
    default:
      return GetGateMatrix<CommonGates::I>();
    }
  };

  const auto flattenGateMatrix =
      [](const std::vector<std::vector<std::complex<double>>> &in_gateMatrix) {
        std::vector<std::complex<double>> resultVector;
        resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
        for (const auto &row : in_gateMatrix) {
          for (const auto &entry : row) {
            resultVector.emplace_back(entry);
          }
        }

        return resultVector;
      };

  return flattenGateMatrix(getMatrix());
}
} // namespace

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
  m_buffer = buffer;
  m_tensorNetwork = buildInitialNetwork(buffer->size());
  m_tensorIdCounter = m_tensorNetwork.getMaxTensorId();
}

exatn::TensorNetwork
ExaTnDmVisitor::buildInitialNetwork(size_t in_nbQubits) const {
  for (int i = 0; i < in_nbQubits; ++i) {
    const std::string tensorName = "Q" + std::to_string(i);
    auto tensor = [&]() {
      if (in_nbQubits == 1) {
        assert(tensorName == "Q0");
        return std::make_shared<exatn::Tensor>(
            tensorName, exatn::TensorShape{QUBIT_DIM, INITIAL_BOND_DIM});
      }
      if ((i == 0) || (i == (in_nbQubits - 1))) {
        return std::make_shared<exatn::Tensor>(
            tensorName,
            exatn::TensorShape{QUBIT_DIM, INITIAL_BOND_DIM, INITIAL_KRAUS_DIM});
      }

      return std::make_shared<exatn::Tensor>(
          tensorName, exatn::TensorShape{QUBIT_DIM, INITIAL_BOND_DIM,
                                         INITIAL_KRAUS_DIM, INITIAL_BOND_DIM});
    }();

    const bool created =
        exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
    assert(created);
    const bool initialized = exatn::initTensorDataSync(
        tensorName, Q_ZERO_TENSOR_BODY(tensor->getVolume()));
    assert(initialized);
  }

  const auto buildTensorMap = [](size_t in_nbQubits) {
    const std::vector<int> qubitTensorDim(in_nbQubits, QUBIT_DIM);
    const std::vector<int> ancTensorDim(in_nbQubits, INITIAL_KRAUS_DIM);
    // Root tensor dimension: 2 .. 2 (upper legs/system dimensions) 1 ... 1
    // (lower legs/anc dimension)
    std::vector<int> rootTensorDim;
    rootTensorDim.insert(rootTensorDim.end(), qubitTensorDim.begin(),
                         qubitTensorDim.end());
    rootTensorDim.insert(rootTensorDim.end(), ancTensorDim.begin(),
                         ancTensorDim.end());
    auto rootTensor =
        std::make_shared<exatn::Tensor>(ROOT_TENSOR_NAME, rootTensorDim);
    std::map<std::string, std::shared_ptr<exatn::Tensor>> tensorMap;
    tensorMap.emplace(ROOT_TENSOR_NAME, rootTensor);
    for (int i = 0; i < in_nbQubits; ++i) {
      const std::string qTensorName = "Q" + std::to_string(i);
      tensorMap.emplace(qTensorName, exatn::getTensor(qTensorName));
    }
    return tensorMap;
  };

  const std::string rootVarNameList = [](size_t in_nbQubits) {
    std::string result = "(";
    // Primary qubit legs
    for (int i = 0; i < in_nbQubits; ++i) {
      result += ("i" + std::to_string(i) + ",");
    }

    // Anc legs:
    for (int i = 0; i < in_nbQubits; ++i) {
      result += ("k" + std::to_string(i) + ",");
    }
    assert(result.back() == ',');
    result.back() = ')';
    return result;
  }(in_nbQubits);

  const auto qubitTensorVarNameList = [](int in_qIdx,
                                         int in_nbQubits) -> std::string {
    if (in_nbQubits == 1) {
      assert(in_qIdx == 0);
      return "(i0,k0)";
    }
    if (in_qIdx == 0) {
      return "(i0,j0,k0)";
    }
    if (in_qIdx == in_nbQubits - 1) {
      return "(i" + std::to_string(in_nbQubits - 1) + ",j" +
             std::to_string(in_nbQubits - 2) + ",k" +
             std::to_string(in_nbQubits - 1) + ")";
    }

    return "(i" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx - 1) +
           ",k" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx) +
           ")";
  };

  const std::string tensorNetworkString = [&]() {
    std::string result = ROOT_TENSOR_NAME + rootVarNameList + "=";
    for (int i = 0; i < in_nbQubits - 1; ++i) {
      result += ("Q" + std::to_string(i) +
                 qubitTensorVarNameList(i, in_nbQubits) + "*");
    }
    result += ("Q" + std::to_string(m_buffer->size() - 1) +
               qubitTensorVarNameList(in_nbQubits - 1, in_nbQubits));
    return result;
  }();

  exatn::TensorNetwork qubitTensorNet("Tensor_Network", tensorNetworkString,
                                      buildTensorMap(in_nbQubits));
  // qubitTensorNet.printIt();

  // Conjugate of the network:
  exatn::TensorNetwork conjugate(qubitTensorNet);
  conjugate.rename("Conjugate");
  conjugate.conjugate();
  // conjugate.printIt();
  // Pair all ancilla legs
  std::vector<std::pair<unsigned int, unsigned int>> pairings;
  for (size_t i = 0; i < in_nbQubits; ++i) {
    pairings.emplace_back(std::make_pair(i + in_nbQubits, i + in_nbQubits));
  }

  qubitTensorNet.appendTensorNetwork(std::move(conjugate), pairings);
  std::cout << "Qubit Tensor network:\n";
  qubitTensorNet.printIt();
  return qubitTensorNet;
}

void ExaTnDmVisitor::finalize() {
  std::unordered_set<std::string> tensorList;
  for (auto iter = m_tensorNetwork.cbegin(); iter != m_tensorNetwork.cend();
       ++iter) {
    const auto &tensorName = iter->second.getTensor()->getName();
    // Not a root tensor
    if (!tensorName.empty() && tensorName[0] != '_') {
      tensorList.emplace(iter->second.getTensor()->getName());
    }
  }

  for (const auto &tensorName : tensorList) {
    const bool destroyed = exatn::destroyTensor(tensorName);
    assert(destroyed);
  }

  m_buffer.reset();
}

void ExaTnDmVisitor::applySingleQubitGate(
    xacc::quantum::Gate &in_gateInstruction) {
  m_tensorIdCounter++;
  assert(in_gateInstruction.bits().size() == 1);
  const auto gateMatrix = getGateMatrix(in_gateInstruction);
  assert(gateMatrix.size() == 4);
  const std::string uniqueGateName =
      in_gateInstruction.name() + std::to_string(m_tensorIdCounter);
  // Create the tensor
  const bool created = exatn::createTensorSync(
      uniqueGateName, exatn::TensorElementType::COMPLEX64,
      exatn::TensorShape{2, 2});
  assert(created);
  // Init tensor body data
  const bool initialized =
      exatn::initTensorDataSync(uniqueGateName, gateMatrix);
  assert(initialized);

  const std::vector<unsigned int> gatePairing{
      static_cast<unsigned int>(in_gateInstruction.bits()[0])};
  // Append the tensor for this gate to the network
  const bool appended = m_tensorNetwork.appendTensorGate(
      m_tensorIdCounter,
      // Get the gate tensor data which must have been initialized.
      exatn::getTensor(uniqueGateName),
      // which qubits that the gate is acting on
      gatePairing);
  assert(appended);

  m_tensorIdCounter++;
  const std::vector<unsigned int> gatePairingConj{static_cast<unsigned int>(
      m_buffer->size() + in_gateInstruction.bits()[0])};
  const bool conjAppended = m_tensorNetwork.appendTensorGate(
      m_tensorIdCounter,
      // Get the gate tensor data which must have been initialized.
      exatn::getTensor(uniqueGateName),
      // which qubits that the gate is acting on
      gatePairingConj,
      // conjugate
      true);
  assert(conjAppended);
}

void ExaTnDmVisitor::applyTwoQubitGate(
    xacc::quantum::Gate &in_gateInstruction) {
  m_tensorIdCounter++;
  assert(in_gateInstruction.bits().size() == 2);
  const auto gateMatrix = getGateMatrix(in_gateInstruction);
  assert(gateMatrix.size() == 16);
  const std::string uniqueGateName =
      in_gateInstruction.name() + std::to_string(m_tensorIdCounter);
  // Create the tensor
  const bool created = exatn::createTensorSync(
      uniqueGateName, exatn::TensorElementType::COMPLEX64,
      exatn::TensorShape{2, 2, 2, 2});
  assert(created);
  // Init tensor body data
  const bool initialized =
      exatn::initTensorDataSync(uniqueGateName, gateMatrix);
  assert(initialized);

  const std::vector<unsigned int> gatePairing{
      static_cast<unsigned int>(in_gateInstruction.bits()[1]),
      static_cast<unsigned int>(in_gateInstruction.bits()[0])};
  // Append the tensor for this gate to the network
  const bool appended = m_tensorNetwork.appendTensorGate(
      m_tensorIdCounter,
      // Get the gate tensor data which must have been initialized.
      exatn::getTensor(uniqueGateName),
      // which qubits that the gate is acting on
      gatePairing);
  assert(appended);

  m_tensorIdCounter++;
  const std::vector<unsigned int> gatePairingConj{
      static_cast<unsigned int>(m_buffer->size() +
                                in_gateInstruction.bits()[1]),
      static_cast<unsigned int>(m_buffer->size() +
                                in_gateInstruction.bits()[0])};
  const bool conjAppended = m_tensorNetwork.appendTensorGate(
      m_tensorIdCounter,
      // Get the gate tensor data which must have been initialized.
      exatn::getTensor(uniqueGateName),
      // which qubits that the gate is acting on
      gatePairingConj,
      // conjugate
      true);
  assert(conjAppended);
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
  std::cout << "Apply: " << in_XGate.toString() << "\n";
  m_tensorNetwork.printIt();
  printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

void ExaTnDmVisitor::visit(Y &in_YGate) { applySingleQubitGate(in_YGate); }

void ExaTnDmVisitor::visit(Z &in_ZGate) { applySingleQubitGate(in_ZGate); }

void ExaTnDmVisitor::visit(Rx &in_RxGate) { applySingleQubitGate(in_RxGate); }

void ExaTnDmVisitor::visit(Ry &in_RyGate) { applySingleQubitGate(in_RyGate); }

void ExaTnDmVisitor::visit(Rz &in_RzGate) { applySingleQubitGate(in_RzGate); }

void ExaTnDmVisitor::visit(T &in_TGate) { 
  applySingleQubitGate(in_TGate); 
  // DEBUG:
  std::cout << "Apply: " << in_TGate.toString() << "\n";
  m_tensorNetwork.printIt();
  printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

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
  m_tensorNetwork.printIt();
  printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

void ExaTnDmVisitor::visit(Swap &in_SwapGate) {
  applyTwoQubitGate(in_SwapGate);
}

void ExaTnDmVisitor::visit(CZ &in_CZGate) { applyTwoQubitGate(in_CZGate); }

void ExaTnDmVisitor::visit(CPhase &in_CPhaseGate) {
  applyTwoQubitGate(in_CPhaseGate);
}

void ExaTnDmVisitor::visit(iSwap &in_iSwapGate) {
  applyTwoQubitGate(in_iSwapGate);
}

void ExaTnDmVisitor::visit(fSim &in_fsimGate) {
  applyTwoQubitGate(in_fsimGate);
}

const double ExaTnDmVisitor::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> in_function) {
  return 0.0;
}
} // namespace tnqvm

// Register with CppMicroservices
REGISTER_PLUGIN(tnqvm::ExaTnDmVisitor, tnqvm::TNQVMVisitor)
