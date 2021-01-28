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
#include <bits/stdc++.h> 

#define QUBIT_DIM 2

namespace {
std::vector<std::complex<double>> Q_ZERO_TENSOR_BODY(size_t in_volume) {
  std::vector<std::complex<double>> body(in_volume, {0.0, 0.0});
  body[0] = std::complex<double>(1.0, 0.0);
  return body;
}

const std::string ROOT_TENSOR_NAME = "Root";

void printDensityMatrix(const exatn::TensorNetwork &in_tensorNet,
                        size_t in_nbQubit, bool in_checkTrace = true) {
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
      if (in_checkTrace) {
        // Verify trace(density matrix) == 1.0
        assert(std::abs(traceVal.real() - 1.0) < 1e-12);
        assert(std::abs(traceVal.imag()) < 1e-12);
      }
    }
  } else {
    std::cout << "Failed to retrieve tensor data!!!\n";
  }
}

bool checkDmTensorNetwork(const exatn::TensorNetwork &in_tensorNet,
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
          rowId++;
        }
        if (i == (rowId + rowId * nbRows)) {
          traceVal += body_ptr[i];
        }
        const auto &elem = body_ptr[i];
      }
      return (std::abs(traceVal.real() - 1.0) < 1e-12) &&
             (std::abs(traceVal.imag()) < 1e-12);
    }
  } else {
    std::cout << "Failed to retrieve tensor data!!!\n";
  }
  return false;
}

std::vector<std::complex<double>> flattenGateMatrix(
    const std::vector<std::vector<std::complex<double>>> &in_gateMatrix) {
  std::vector<std::complex<double>> resultVector;
  resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
  for (const auto &row : in_gateMatrix) {
    for (const auto &entry : row) {
      resultVector.emplace_back(entry);
    }
  }

  return resultVector;
}

std::vector<std::vector<std::complex<double>>> conjugateMatrix(
    const std::vector<std::vector<std::complex<double>>> &in_mat) {
  auto result = in_mat;
  const auto dim = in_mat.size();
  for (size_t rowId = 0; rowId < dim; ++rowId) {
    for (size_t colId = 0; colId < dim; ++colId) {
      result[colId][rowId] = std::conj(in_mat[colId][rowId]);
    }
  }
  return result;
}

std::vector<std::complex<double>>
getGateMatrix(const xacc::Instruction &in_gate, bool in_dagger = false) {
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

  const auto gateMatrix = getMatrix();
  return in_dagger ? flattenGateMatrix(conjugateMatrix(gateMatrix))
                   : flattenGateMatrix(gateMatrix);
}

void recursiveFindAllCombinations(std::vector<std::vector<unsigned int>>& io_result,
const std::vector<unsigned int> &arr,
                std::vector<unsigned int> &data, int start, int end, int index,
                int r) {
  // Current combination is ready
  if (index == r) {
    std::vector<unsigned int> combination;
    for (int j = 0; j < r; j++) {
      combination.emplace_back(data[j]);
    }
    io_result.emplace_back(combination);
    return;
  }
  for (int i = start; i <= end && end - i + 1 >= r - index; i++) {
    data[index] = arr[i];
    recursiveFindAllCombinations(io_result, arr, data, i + 1, end, index + 1, r);
  }
}

std::vector<std::vector<unsigned int>>
findAllPermutations(const std::vector<unsigned int> &in_elements) {
  auto a = in_elements;
  std::sort(a.begin(), a.end());
  std::vector<std::vector<unsigned int>> result;
  do {
    result.emplace_back(a);

  } while (std::next_permutation(a.begin(), a.end()));
  return result;
}

std::vector<unsigned int>
filterList(const std::vector<unsigned int> &in_totalList,
           const std::vector<unsigned int> &in_filter) {
  std::vector<unsigned int> result;
  for (const auto &el : in_totalList) {
    if (!xacc::container::contains(in_filter, el)) {
      result.emplace_back(el);
    }
  }
  return result;
}

// Helper to find all input->output leg connection permutation.
// This is *only* used for development purposes to figure out the
// correct connection configs to preseve the order of the root density matrix
// tensor.
std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
findAllGateTensorPairings(int in_tensorRank) {
  assert(in_tensorRank % 2 == 0);
  std::vector<unsigned int> legIds;
  std::vector<unsigned int> tempVec(in_tensorRank / 2);
  for (unsigned int i = 0; i < in_tensorRank; ++i) {
    legIds.emplace_back(i);
  }
  // Find all possible input leg combination:
  std::vector<std::vector<unsigned int>> inputCombinations;
  recursiveFindAllCombinations(inputCombinations, legIds, tempVec, 0,
                               in_tensorRank - 1, 0, in_tensorRank / 2);

  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> result;
  for (const auto &inputLegs : inputCombinations) {
    const auto inputLegPerm = findAllPermutations(inputLegs);
    const auto outputLegPerm =
        findAllPermutations(filterList(legIds, inputLegs));
    for (const auto &inputPerm : inputLegPerm) {
      for (const auto &outputPerm : outputLegPerm) {
        assert(inputPerm.size() == outputPerm.size());
        std::vector<std::pair<unsigned int, unsigned int>> combo;
        for (size_t i = 0; i < inputPerm.size(); ++i) {
          combo.emplace_back(std::make_pair(inputPerm[i], outputPerm[i]));
        }
        result.emplace_back(combo);
      }
    }
  }

  return result;
}

void checkKrausTensorConfig(const exatn::TensorNetwork &in_tensorNet,
                            unsigned int in_q0, unsigned int in_q1,
                            unsigned int in_numQubits,
                            const std::string &in_krausTensorName) {
  auto noiseTensor = exatn::getTensor(in_krausTensorName);
  std::cout << "Checking Kraus tensor connections:\n";
  auto test = findAllGateTensorPairings(noiseTensor->getDimExtents().size());
  for (const auto &config : test) {
    const std::vector<
        std::pair<unsigned int, std::pair<unsigned int, unsigned int>>>
        noisePairing{{in_q0, config[0]},
                     {in_q0 + in_numQubits, config[1]},
                     {in_q1, config[2]},
                     {in_q1 + in_numQubits, config[3]}};

    // Append the tensor for this gate to the network
    auto temp = in_tensorNet;
    const auto tensorIdCounter = temp.getMaxTensorId() + 1;
    const bool appended = temp.appendTensorGateGeneral(
        tensorIdCounter, noiseTensor, noisePairing);
    assert(appended);
    if (checkDmTensorNetwork(temp, in_numQubits)) {
      std::cout << "============================== \n Valid: ";
      for (const auto &[iLeg, oLeg] : config) {
        std::cout << "(" << iLeg << "-->" << oLeg << ")";
      }
      printDensityMatrix(temp, in_numQubits);
      std::cout << "==============================\n";
    }
  }
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
  if (options.pointerLikeExists<xacc::NoiseModel>("noise-model")) {
    m_noiseConfig = xacc::as_shared_ptr(
        options.getPointerLike<xacc::NoiseModel>("noise-model"));
  }
}

exatn::TensorNetwork
ExaTnDmVisitor::buildInitialNetwork(size_t in_nbQubits) const {
  for (int i = 0; i < in_nbQubits; ++i) {
    const std::string tensorName = "Q" + std::to_string(i);
    auto tensor = std::make_shared<exatn::Tensor>(
            tensorName, exatn::TensorShape{QUBIT_DIM, QUBIT_DIM});
    const bool created =
        exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
    assert(created);
    const bool initialized = exatn::initTensorDataSync(
        tensorName, Q_ZERO_TENSOR_BODY(tensor->getVolume()));
    assert(initialized);
  }
  exatn::TensorNetwork qubitTensorNet("Tensor_Network");
  // Append the qubit tensors to the tensor network
  size_t tensorIdCounter = 0;
  for (int i = 0; i < in_nbQubits; ++i) {
    tensorIdCounter++;
    const std::string tensorName = "Q" + std::to_string(i);
    qubitTensorNet.appendTensor(
        tensorIdCounter, exatn::getTensor(tensorName),
        std::vector<std::pair<unsigned int, unsigned int>>{});
  }
  std::cout << "Qubit Tensor network:\n";
  qubitTensorNet.printIt();
  return qubitTensorNet;
}

void ExaTnDmVisitor::finalize() {
  executionInfo.clear();
  // Max number of qubits that we allow for a full density matrix retrieval.
  // For more qubits, only expectation contraction is supported.
  constexpr size_t MAX_SIZE_TO_COLLAPSE_DM = 10; 
  
  if (m_buffer->size() <= MAX_SIZE_TO_COLLAPSE_DM) {
    xacc::ExecutionInfo::DensityMatrixType densityMatrix;
    const auto nbRows = 1ULL << m_buffer->size();
    densityMatrix.reserve(1 << m_buffer->size());
    exatn::TensorNetwork tempNetwork(m_tensorNetwork);
    tempNetwork.rename("__TEMP__" + m_tensorNetwork.getName());
    const bool evaledOk = exatn::evaluateSync(tempNetwork);
    assert(evaledOk);
    auto talsh_tensor =
        exatn::getLocalTensor(tempNetwork.getTensor(0)->getName());
    const auto expectedDensityMatrixVolume = nbRows * nbRows;
    if (talsh_tensor) {
      const std::complex<double> *body_ptr;
      const bool access_granted =
          talsh_tensor->getDataAccessHostConst(&body_ptr);
      if (!access_granted) {
        std::cout << "Failed to retrieve tensor data!!!\n";
      } else {
        assert(expectedDensityMatrixVolume == talsh_tensor->getVolume());
        xacc::ExecutionInfo::WaveFuncType dmRow;
        dmRow.reserve(nbRows);
        for (int i = 0; i < talsh_tensor->getVolume(); ++i) {
          if ((i != 0) && (i % nbRows == 0)) {
            assert(dmRow.size() == nbRows);
            densityMatrix.emplace_back(std::move(dmRow));
            dmRow.clear();
            dmRow.reserve(nbRows);
          }
          const auto &elem = body_ptr[i];
          dmRow.emplace_back(elem);
        }
        assert(dmRow.size() == nbRows);
        densityMatrix.emplace_back(std::move(dmRow));
        assert(densityMatrix.size() == nbRows);
      }
    }
    
    executionInfo.insert(ExecutionInfo::DmKey, std::make_shared<ExecutionInfo::DensityMatrixType>(std::move(densityMatrix)));
  }

  // Expectation value calculation:
  // Exp = Trace(rho * Op)
  if (!m_measuredBits.empty()) {
    auto tensorIdCounter = m_tensorIdCounter;
    auto expValTensorNet = m_tensorNetwork;
    const std::string measZTensorName = "MEAS_Z";
    {
      xacc::quantum::Z zGate(0);
      const auto gateMatrix = getGateMatrix(zGate);
      assert(gateMatrix.size() == 4);
      // Create the tensor
      const bool created = exatn::createTensorSync(
          measZTensorName, exatn::TensorElementType::COMPLEX64,
          exatn::TensorShape{2, 2});
      assert(created);
      // Init tensor body data
      const bool initialized =
          exatn::initTensorDataSync(measZTensorName, gateMatrix);
      assert(initialized);
    }
    // Add Z tensors for measurement
    for (const auto &measBit : m_measuredBits) {
      tensorIdCounter++;
      const std::vector<unsigned int> gatePairing{
          static_cast<unsigned int>(measBit)};
      // Append the tensor for this gate to the network
      const bool appended = expValTensorNet.appendTensorGate(
          tensorIdCounter,
          // Get the gate tensor data which must have been initialized.
          exatn::getTensor(measZTensorName),
          // which qubits that the gate is acting on
          gatePairing);
      assert(appended);
    }
    // DEBUG:
    std::cout << "TENSOR NETWORK TO COMPUTE THE TRACE:\n";
    printDensityMatrix(expValTensorNet, m_buffer->size(), false);
    // Compute the trace, closing the tensor network:
    const std::string idTensor = "ID_TRACE";
    {
      xacc::quantum::Identity idGate(0);
      const auto idGateMatrix = getGateMatrix(idGate);
      // Create the tensor
      const bool created =
          exatn::createTensorSync(idTensor, exatn::TensorElementType::COMPLEX64,
                                  exatn::TensorShape{2, 2});
      assert(created);
      // Init tensor body data
      const bool initialized =
          exatn::initTensorDataSync(idTensor, idGateMatrix);
      assert(initialized);
    }

    for (size_t qId = 0; qId < m_buffer->size(); ++qId) {
      tensorIdCounter++;
      const std::vector<std::pair<unsigned int, unsigned int>> tracePairing{
          {static_cast<unsigned int>(0), 0},
          {static_cast<unsigned int>(m_buffer->size() - qId), 1}};
      const bool appended = expValTensorNet.appendTensor(
          tensorIdCounter, exatn::getTensor(idTensor), tracePairing);
      assert(appended);
    }

    // Evaluate the trace by contraction:
    const bool evaledOk = exatn::evaluateSync(expValTensorNet);
    assert(evaledOk);
    auto talsh_tensor =
        exatn::getLocalTensor(expValTensorNet.getTensor(0)->getName());
    if (talsh_tensor) {
      const std::complex<double> *body_ptr;
      const bool access_granted =
          talsh_tensor->getDataAccessHostConst(&body_ptr);
      if (!access_granted) {
        xacc::error("Failed to retrieve tensor data!");
      } else {
        // Should only have 1 element:
        assert(talsh_tensor->getVolume() == 1);
        const std::complex<double> traceVal = *body_ptr;
        assert(traceVal.imag() < 1e-3);
        const double expValZ = traceVal.real();
        std::cout << "Exp-val Z = " << expValZ << "\n";
        m_buffer->addExtraInfo("exp-val-z", expValZ);
      }
    }
    bool destroyed = exatn::destroyTensor(measZTensorName);
    assert(destroyed);
    destroyed = exatn::destroyTensor(idTensor);
    assert(destroyed);
  }

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
  m_noiseConfig.reset();
}

void ExaTnDmVisitor::applySingleQubitGate(
    xacc::quantum::Gate &in_gateInstruction) {
  {
    m_tensorIdCounter++;
    assert(in_gateInstruction.bits().size() == 1);
    const auto gateMatrix = getGateMatrix(in_gateInstruction);
    // std::cout << "Gate mattrix:\n";
    // for (const auto &el : gateMatrix) {
    //   std::cout << el << "\n";
    // }
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
  }

  {
    // Append the dagger gate
    m_tensorIdCounter++;
    const auto gateMatrix = getGateMatrix(in_gateInstruction, true);
    // std::cout << "Gate mattrix dagger:\n";
    // for (const auto &el : gateMatrix) {
    //   std::cout << el << "\n";
    // }
    assert(gateMatrix.size() == 4);
    const std::string uniqueGateName =
        in_gateInstruction.name() + "_CONJ_" + std::to_string(m_tensorIdCounter);
    // Create the tensor
    const bool created = exatn::createTensorSync(
        uniqueGateName, exatn::TensorElementType::COMPLEX64,
        exatn::TensorShape{2, 2});
    assert(created);
    // Init tensor body data
    const bool initialized =
        exatn::initTensorDataSync(uniqueGateName, gateMatrix);
    assert(initialized);
    const std::vector<unsigned int> gatePairingConj{static_cast<unsigned int>(
        m_buffer->size() + in_gateInstruction.bits()[0])};
    const bool conjAppended = m_tensorNetwork.appendTensorGate(
        m_tensorIdCounter,
        // Get the gate tensor data which must have been initialized.
        exatn::getTensor(uniqueGateName),
        // which qubits that the gate is acting on
        gatePairingConj);
    assert(conjAppended);
  }

  // DEBUG
  std::cout << "Before noise:\n";
  m_tensorNetwork.printIt();

  // Adding noise tensors.
  applyNoise(in_gateInstruction);
}

void ExaTnDmVisitor::applyTwoQubitGate(
    xacc::quantum::Gate &in_gateInstruction) {
  {
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
  }
  {
    m_tensorIdCounter++;
    const auto gateMatrix = getGateMatrix(in_gateInstruction, true);
    assert(gateMatrix.size() == 16);
    const std::string uniqueGateName =
        in_gateInstruction.name() + "_CONJ_" + std::to_string(m_tensorIdCounter);
    // Create the tensor
    const bool created = exatn::createTensorSync(
        uniqueGateName, exatn::TensorElementType::COMPLEX64,
        exatn::TensorShape{2, 2, 2, 2});
    assert(created);
    // Init tensor body data
    const bool initialized =
        exatn::initTensorDataSync(uniqueGateName, gateMatrix);
    assert(initialized);
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
        gatePairingConj);
    assert(conjAppended);
  }

  // Adding noise tensors.
  applyNoise(in_gateInstruction);
}

void ExaTnDmVisitor::applyNoise(xacc::quantum::Gate &in_gateInstruction) {
  if (!m_noiseConfig) {
    return;
  }
  std::cout << "Before noise:\n";
  printDensityMatrix(m_tensorNetwork, m_buffer->size());

  const auto noiseChannels =
      m_noiseConfig->getNoiseChannels(in_gateInstruction);
  auto noiseUtils = xacc::getService<NoiseModelUtils>("default");

  for (auto &channel : noiseChannels) {
    auto noiseMat = noiseUtils->krausToChoi(channel.mats);

    // for (const auto &row : noiseMat) {
    //   for (const auto &el : row) {
    //     std::cout << el << " ";
    //   }
    //   std::cout << "\n";
    // }

    m_tensorIdCounter++;
    const std::string noiseTensorName = in_gateInstruction.name() + "_Noise_" +
                                        std::to_string(m_tensorIdCounter);
    if (channel.noise_qubits.size() == 1) {
      // Create the tensor
      const bool created = exatn::createTensorSync(
          noiseTensorName, exatn::TensorElementType::COMPLEX64,
          exatn::TensorShape{2, 2, 2, 2});
      assert(created);
      // Init tensor body data
      const bool initialized = exatn::initTensorDataSync(
          noiseTensorName, flattenGateMatrix(noiseMat));
      assert(initialized);

      const std::vector<
          std::pair<unsigned int, std::pair<unsigned int, unsigned int>>>
          noisePairing{
              {static_cast<unsigned int>(channel.noise_qubits[0]), {1, 0}},
              {static_cast<unsigned int>(channel.noise_qubits[0] +
                                         m_buffer->size()),
               {3, 2}}};
      // Append the tensor for this gate to the network
      const bool appended = m_tensorNetwork.appendTensorGateGeneral(
          m_tensorIdCounter, exatn::getTensor(noiseTensorName),
          // which qubits that the gate is acting on
          noisePairing);
      assert(appended);
    } else if (channel.noise_qubits.size() == 2) {
      // Create the tensor
      const bool created = exatn::createTensorSync(
          noiseTensorName, exatn::TensorElementType::COMPLEX64,
          exatn::TensorShape{2, 2, 2, 2, 2, 2, 2, 2});
      assert(created);
      // Init tensor body data
      const bool initialized = exatn::initTensorDataSync(
          noiseTensorName, flattenGateMatrix(noiseMat));
      assert(initialized);

      // DEBUG: Using this to check all possible permutations.
      // checkKrausTensorConfig(m_tensorNetwork, channel.noise_qubits[0],
      //                        channel.noise_qubits[1], m_buffer->size(),
      //                        noiseTensorName);

      const std::vector<
          std::pair<unsigned int, std::pair<unsigned int, unsigned int>>>
          noisePairingMsb{
              {static_cast<unsigned int>(channel.noise_qubits[0]), {5, 6}},
              {static_cast<unsigned int>(channel.noise_qubits[0] +
                                         m_buffer->size()),
               {1, 2}},
              {static_cast<unsigned int>(channel.noise_qubits[1]), {4, 7}},
              {static_cast<unsigned int>(channel.noise_qubits[1] +
                                         m_buffer->size()),
               {0, 3}}};

      const std::vector<
          std::pair<unsigned int, std::pair<unsigned int, unsigned int>>>
          noisePairingLsb{
              {static_cast<unsigned int>(channel.noise_qubits[1]), {5, 6}},
              {static_cast<unsigned int>(channel.noise_qubits[1] +
                                         m_buffer->size()),
               {1, 2}},
              {static_cast<unsigned int>(channel.noise_qubits[0]), {4, 7}},
              {static_cast<unsigned int>(channel.noise_qubits[0] +
                                         m_buffer->size()),
               {0, 3}}};
      const auto noisePairing = (channel.bit_order == KrausMatBitOrder::MSB)
                                    ? noisePairingMsb
                                    : noisePairingLsb;
      // Append the tensor for this gate to the network
      const bool appended = m_tensorNetwork.appendTensorGateGeneral(
          m_tensorIdCounter, exatn::getTensor(noiseTensorName),
          // which qubits that the gate is acting on
          noisePairing);
      assert(appended);
    } else {
      xacc::error("Unsupported noise data.");
    }
  }

  // DEBUG:
  std::cout << "Apply Noise: " << in_gateInstruction.toString() << "\n";
  m_tensorNetwork.printIt();
  printDensityMatrix(m_tensorNetwork, m_buffer->size());
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

void ExaTnDmVisitor::visit(Rx &in_RxGate) { 
  applySingleQubitGate(in_RxGate); 
  std::cout << "Apply: " << in_RxGate.toString() << "\n";
  m_tensorNetwork.printIt();
  printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

void ExaTnDmVisitor::visit(Ry &in_RyGate) { 
  applySingleQubitGate(in_RyGate); 
  std::cout << "Apply: " << in_RyGate.toString() << "\n";
  m_tensorNetwork.printIt();
  printDensityMatrix(m_tensorNetwork, m_buffer->size());
}

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
