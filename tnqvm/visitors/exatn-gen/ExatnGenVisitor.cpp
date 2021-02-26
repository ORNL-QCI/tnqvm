/***********************************************************************************
 * Copyright (c) 2017, UT-Battelle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the xacc nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Initial sketch - Mengsu Chen 2017/07/17;
 *   Implementation - Dmitry Lyakh 2017/10/05 - active;
 *
 **********************************************************************************/
#ifdef TNQVM_HAS_EXATN
#define _DEBUG_DIL

#include "ExatnGenVisitor.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "Instruction.hpp"
#include "talshxx.hpp"
#include <numeric>
#include <random>
#include <chrono>
#include <functional>
#include <unordered_set>
#include "xacc_plugin.hpp"

#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

namespace {
// Helper to construct qubit tensor name:
std::string generateQubitTensorName(int qubitIndex) {
  return "Q" + std::to_string(qubitIndex);
};
// The max number of qubits that we allow full state vector contraction.
// Above this limit, only tensor-based calculation is allowed.
// e.g. simulating bit-string measurement by tensor contraction.
const int MAX_NUMBER_QUBITS_FOR_STATE_VEC = 50;

// Max memory size: 8GB
const int64_t MAX_TALSH_MEMORY_BUFFER_SIZE_BYTES = 8 * (1ULL << 30);

template <typename TNQVM_COMPLEX_TYPE>
std::vector<TNQVM_COMPLEX_TYPE> flattenGateMatrix(
    const std::vector<std::vector<TNQVM_COMPLEX_TYPE>> &in_gateMatrix) {

  std::vector<TNQVM_COMPLEX_TYPE> resultVector;
  resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
  for (const auto &row : in_gateMatrix) {
    for (const auto &entry : row) {
      resultVector.emplace_back(entry);
    }
  }

  return resultVector;
}

template <typename TNQVM_COMPLEX_TYPE>
bool checkStateVectorNorm(const std::vector<TNQVM_COMPLEX_TYPE> &in_stateVec) {

  const double norm =
      std::accumulate(in_stateVec.begin(), in_stateVec.end(), 0.0,
                      [](double runningNorm, TNQVM_COMPLEX_TYPE vecComponent) {
                        return runningNorm + std::norm(vecComponent);
                      });

  return (std::abs(norm - 1.0) < 1e-12);
}

template <typename TNQVM_COMPLEX_TYPE>
double calcExpValueZ(const std::vector<int> &in_bits,
                     const std::vector<TNQVM_COMPLEX_TYPE> &in_stateVec) {
  TNQVM_TELEMETRY_ZONE("calcExpValueZ", __FILE__, __LINE__);
  const auto hasEvenParity =
      [](uint64_t x, const std::vector<int> &in_qubitIndices) -> bool {
    size_t count = 0;
    for (const auto &bitIdx : in_qubitIndices) {
      if (x & (1ULL << bitIdx)) {
        count++;
      }
    }
    return (count % 2) == 0;
  };

  double result = 0.0;
  for (uint64_t i = 0; i < in_stateVec.size(); ++i) {
    result +=
        (hasEvenParity(i, in_bits) ? 1.0 : -1.0) * std::norm(in_stateVec[i]);
  }

  return result;
}
} // namespace

namespace tnqvm {
GateInstanceIdentifier::GateInstanceIdentifier(const std::string &in_gateName)
    : m_gateName(in_gateName) {}

template <typename... GateParams>
GateInstanceIdentifier::GateInstanceIdentifier(const std::string &in_gateName,
                                               const GateParams &... in_params)
    : GateInstanceIdentifier(in_gateName) {
  addParam(in_params...);
}

std::string formatGateParamStr(const std::string &in_paramStr) {
  std::string result = in_paramStr;
  // Remove special characters:
  // ExaTN doesn't allow special characters (only underscore is allowed).
  // Hence, update format the param string accordingly.
  std::replace(result.begin(), result.end(), '-', '_');
  std::replace(result.begin(), result.end(), '.', '_');
  result.erase(std::remove(result.begin(), result.end(), '+'), result.end());

  return result;
}

template <typename GateParam>
void GateInstanceIdentifier::addParam(const GateParam &in_param) {
  m_gateParams.emplace_back(formatGateParamStr(std::to_string(in_param)));
}

template <typename GateParam, typename... MoreParams>
void GateInstanceIdentifier::addParam(const GateParam &in_param,
                                      const MoreParams &... in_moreParams) {
  m_gateParams.emplace_back(formatGateParamStr(std::to_string(in_param)));
  addParam(in_moreParams...);
}

std::string GateInstanceIdentifier::toNameString() const {
  if (m_gateParams.empty()) {
    return m_gateName;
  } else {
    return m_gateName + "__" + [&]() -> std::string {
      std::string paramList;
      for (size_t i = 0; i < m_gateParams.size() - 1; ++i) {
        paramList.append(m_gateParams[i] + "__");
      }
      paramList.append(m_gateParams.back());

      return paramList;
    }() + "__";
  }
}

template <typename TNQVM_COMPLEX_TYPE>
ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::ExatnGenVisitor() {}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::initialize(
    std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) {
  int64_t talshHostBufferSizeInBytes = MAX_TALSH_MEMORY_BUFFER_SIZE_BYTES;
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
    exatn::ParamConf exatnParams;
    if (options.keyExists<int>("exatn-buffer-size-gb")) {
      int bufferSizeGb = options.get<int>("exatn-buffer-size-gb");
      if (bufferSizeGb < 1) {
        std::cout << "Minimum buffer size is 1 GB.\n";
        bufferSizeGb = 1;
      }
      // Set the memory buffer size:
      const int64_t memorySizeBytes = bufferSizeGb * (1ULL << 30);
      std::cout << "Set ExaTN host memory buffer to " << memorySizeBytes
                << " bytes.\n";
      const bool success =
          exatnParams.setParameter("host_memory_buffer_size", memorySizeBytes);
      assert(success);
      // Update buffer size.
      talshHostBufferSizeInBytes = memorySizeBytes;
    } else {
      // Use default buffer size
      const bool success = exatnParams.setParameter(
          "host_memory_buffer_size", MAX_TALSH_MEMORY_BUFFER_SIZE_BYTES);
      assert(success);
    }
// This is a flag from ExaTN indicating that ExaTN was compiled
// w/ MPI enabled.
#ifdef MPI_ENABLED
    {
      if (options.keyExists<void *>("mpi-communicator")) {
        xacc::info("Setting ExaTN MPI_COMMUNICATOR...");
        auto communicator = options.get<void *>("mpi-communicator");
        exatn::MPICommProxy commProxy(communicator);
        exatn::initialize(commProxy, exatnParams);
      } else {
        // No specific communicator is specified,
        // exaTN will automatically use MPI_COMM_WORLD.
        exatn::initialize(exatnParams);
      }
      exatn::activateContrSeqCaching();
    }
#else
    {
      exatn::initialize(exatnParams);
      exatn::activateContrSeqCaching();
    }
#endif

    if (exatn::getDefaultProcessGroup().getSize() > 1) {
      // Multiple MPI processes:
      // if verbose is set, we must redirect log to files
      // with custom names for each process.
      if (xacc::verbose) {
        // Get the rank of this process.
        const int processRank = exatn::getProcessRank();
        const std::string fileNamePrefix =
            "process" + std::to_string(processRank);
        // Redirect log to files (one for each MPI process)
        xacc::logToFile(true, fileNamePrefix);
      }
    }

    if (options.stringExists("exatn-contract-seq-optimizer")) {
      const std::string optimizerName =
          options.getString("exatn-contract-seq-optimizer");
      std::cout << "Using '" << optimizerName << "' optimizer.\n";
      exatn::resetContrSeqOptimizer(optimizerName);
    }

    // ExaTN and XACC logging levels are always in-synced.
    // Note: If xacc::verbose is not set, we always set ExaTN logging level to
    // 0.
    exatn::resetClientLoggingLevel(xacc::verbose ? 1 : 0);
    exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel()
                                                  : 0);
    xacc::subscribeLoggingLevel([](int level) {
      exatn::resetClientLoggingLevel(xacc::verbose ? 1 : 0);
      exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
    });
  }
  m_buffer = buffer;
  // Create the qubit register tensor
  for (int i = 0; i < m_buffer->size(); ++i) {
    const bool created = exatn::createTensor(
        generateQubitTensorName(i), getExatnElementType(), exatn::TensorShape{2});
    assert(created);
  }

  // Initialize the qubit register tensor to zero state
  for (int i = 0; i < m_buffer->size(); ++i) {
    // Define the tensor body for a zero-state qubit
    const bool initialized = exatn::initTensorData(
        generateQubitTensorName(i),
        std::vector<TNQVM_COMPLEX_TYPE>{{1.0, 0.0}, {0.0, 0.0}});
    assert(initialized);
  }

  m_qubitNetwork = std::make_shared<exatn::TensorNetwork>("QubitReg");
  // Append the qubit tensors to the tensor network
  for (int i = 0; i < m_buffer->size(); ++i) {
    m_qubitNetwork->appendTensor(
        i + 1, exatn::getTensor(generateQubitTensorName(i)),
        std::vector<std::pair<unsigned int, unsigned int>>{});
  }
  exatn::TensorExpansion tensorEx("QuantumCircuit");
  tensorEx.appendComponent(m_qubitNetwork, 1.0);
  m_tensorExpansion = tensorEx;
  m_tensorExpansion.printIt();
  m_gateTensorBodies.clear();
  m_measuredBits.clear();
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::finalize() {
  m_tensorExpansion.printIt();
  exatn::TensorExpansion ketvector(m_tensorExpansion);
  auto obsTensorNetwork = std::make_shared<exatn::TensorNetwork>("Obs");
  std::vector<std::pair<unsigned int, unsigned int>> ketPairings, braPairings;
  for (int i = 0; i < m_buffer->size(); ++i) {
    ketPairings.emplace_back(
        std::make_pair((unsigned int)i, (unsigned int)(m_buffer->size() + i)));
    braPairings.emplace_back(std::make_pair((unsigned int)i, (unsigned int)i));
    if (xacc::container::contains(m_measuredBits, i)) {
      const auto tensorName = "MeasZ_" + std::to_string(i);
      const bool created = exatn::createTensor(
          tensorName, getExatnElementType(), exatn::TensorShape{2, 2});
      assert(created);
      exatn::initTensorData(
          tensorName, std::vector<TNQVM_COMPLEX_TYPE>{
                          {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}});
      obsTensorNetwork->appendTensor(
          i + 1, exatn::getTensor(tensorName),
          std::vector<std::pair<unsigned int, unsigned int>>{});
    } else {
      const auto tensorName = "MeasI_" + std::to_string(i);
      const bool created = exatn::createTensor(
          tensorName, getExatnElementType(), exatn::TensorShape{2, 2});
      assert(created);
      exatn::initTensorData(
          tensorName, std::vector<TNQVM_COMPLEX_TYPE>{
                          {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
      obsTensorNetwork->appendTensor(
          i + 1, exatn::getTensor(tensorName),
          std::vector<std::pair<unsigned int, unsigned int>>{});
    }
  }

  exatn::TensorOperator zHamOp("Hamiltonian");
  const bool appended = zHamOp.appendComponent(
      obsTensorNetwork, ketPairings, braPairings, std::complex<double>{1.0});
  assert(appended);
  std::cout << "Observable Op:\n";
  zHamOp.printIt();
  exatn::TensorExpansion ketWithObs(ketvector, zHamOp);
  std::cout << "Tensor Expansion:\n";
  ketWithObs.printIt();
  // Bra network:
  exatn::TensorExpansion bravector(ketvector);
  bravector.conjugate();
  bravector.printIt();
  exatn::TensorExpansion bratimesopertimesket(bravector, ketWithObs);
  bratimesopertimesket.printIt();
  const bool accumCreated =
      exatn::createTensorSync("ExpVal", getExatnElementType(), exatn::TensorShape{});
  assert(accumCreated);
  auto accumulator = exatn::getTensor("ExpVal");
  TNQVM_COMPLEX_TYPE result = 0.0;
  if (exatn::evaluateSync(bratimesopertimesket, accumulator)) {
    exatn::sync();
    auto talsh_tensor =
        exatn::getLocalTensor("ExpVal");
    assert(talsh_tensor->getVolume() == 1);
    const TNQVM_COMPLEX_TYPE *body_ptr;
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
      result = *body_ptr;
      std::cout << "Exp-val = " << result << "\n";
      m_buffer->addExtraInfo("exp-val-z", (double)(result.real()));
    }
  }
}

// === BEGIN: Gate Visitor Impls ===
template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Identity &in_IdentityGate) {
  appendGateTensor<CommonGates::I>(in_IdentityGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Hadamard &in_HadamardGate) {
  appendGateTensor<CommonGates::H>(in_HadamardGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(X &in_XGate) {
  appendGateTensor<CommonGates::X>(in_XGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Y &in_YGate) {
  appendGateTensor<CommonGates::Y>(in_YGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Z &in_ZGate) {
  appendGateTensor<CommonGates::Z>(in_ZGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Rx &in_RxGate) {
  assert(in_RxGate.nParameters() == 1);
  const double theta = in_RxGate.getParameter(0).as<double>();
  appendGateTensor<CommonGates::Rx>(in_RxGate, theta);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Ry &in_RyGate) {
  assert(in_RyGate.nParameters() == 1);
  const double theta = in_RyGate.getParameter(0).as<double>();
  appendGateTensor<CommonGates::Ry>(in_RyGate, theta);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Rz &in_RzGate) {
  assert(in_RzGate.nParameters() == 1);
  const double theta = in_RzGate.getParameter(0).as<double>();
  appendGateTensor<CommonGates::Rz>(in_RzGate, theta);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(T &in_TGate) {
  appendGateTensor<CommonGates::T>(in_TGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Tdg &in_TdgGate) {
  appendGateTensor<CommonGates::Tdg>(in_TdgGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(CPhase &in_CPhaseGate) {
  appendGateTensor<CommonGates::CPhase>(in_CPhaseGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(U &in_UGate) {
  assert(in_UGate.nParameters() == 3);
  const double theta = in_UGate.getParameter(0).as<double>();
  const double phi = in_UGate.getParameter(1).as<double>();
  const double lambda = in_UGate.getParameter(2).as<double>();
  appendGateTensor<CommonGates::U>(in_UGate, theta, phi, lambda);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(CNOT &in_CNOTGate) {
  appendGateTensor<CommonGates::CNOT>(in_CNOTGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Swap &in_SwapGate) {
  appendGateTensor<CommonGates::Swap>(in_SwapGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(CZ &in_CZGate) {
  appendGateTensor<CommonGates::CZ>(in_CZGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(iSwap &in_iSwapGate) {
  appendGateTensor<CommonGates::iSwap>(in_iSwapGate);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(fSim &in_fsimGate) {
  assert(in_fsimGate.nParameters() == 2);
  const double theta = in_fsimGate.getParameter(0).as<double>();
  const double phi = in_fsimGate.getParameter(1).as<double>();
  appendGateTensor<CommonGates::fSim>(in_fsimGate, theta, phi);
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::visit(Measure &in_MeasureGate) {
  m_measuredBits.emplace(in_MeasureGate.bits()[0]);
}
// === END: Gate Visitor Impls ===
template <typename TNQVM_COMPLEX_TYPE>
template <tnqvm::CommonGates GateType, typename... GateParams>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::appendGateTensor(
    const xacc::Instruction &in_gateInstruction, GateParams &&... in_params) {
  const auto gateName = GetGateName(GateType);
  const GateInstanceIdentifier gateInstanceId(gateName, in_params...);
  const std::string uniqueGateName = gateInstanceId.toNameString();
  // If the tensor data for this gate hasn't been initialized before,
  // then initialize it.
  if (m_gateTensorBodies.find(uniqueGateName) == m_gateTensorBodies.end()) {
    const auto gateMatrixRaw = GetGateMatrix<GateType>(in_params...);
    std::vector<std::vector<TNQVM_COMPLEX_TYPE>> gateMatrix;
    for (auto &row : gateMatrixRaw) {
      std::vector<TNQVM_COMPLEX_TYPE> rowConverted(row.begin(), row.end());
      gateMatrix.emplace_back(std::move(rowConverted));
    }
    m_gateTensorBodies[uniqueGateName] = flattenGateMatrix(gateMatrix);
    // Currently, we only support 2-qubit gates.
    assert(in_gateInstruction.nRequiredBits() > 0 &&
           in_gateInstruction.nRequiredBits() <= 2);
    const auto gateTensorShape =
        (in_gateInstruction.nRequiredBits() == 1 ? exatn::TensorShape{2, 2}
                                                 : exatn::TensorShape{2, 2, 2, 2});
    // Create the tensor
    const bool created = exatn::createTensor(
        uniqueGateName, getExatnElementType(), gateTensorShape);
    assert(created);
    // Init tensor body data
    exatn::initTensorData(uniqueGateName, flattenGateMatrix(gateMatrix));
    // Register tensor isometry:
    // For rank-2 gate isometric leg groups are: {0}, {1}.
    // For rank-4 gate isometric leg groups are: {0,1}, {2,3}.
    if (in_gateInstruction.nRequiredBits() == 1) {
      const bool registered =
          exatn::registerTensorIsometry(uniqueGateName, {0}, {1});
      assert(registered);
    } else if (in_gateInstruction.nRequiredBits() == 2) {
      const bool registered =
          exatn::registerTensorIsometry(uniqueGateName, {0, 1}, {2, 3});
      assert(registered);
    }
  }

  // Because the qubit location and gate pairing are of different integer types,
  // we need to reconstruct the qubit vector.
  std::vector<unsigned int> gatePairing;
  for (const auto &qbitLoc :
       const_cast<xacc::Instruction &>(in_gateInstruction).bits()) {
    gatePairing.emplace_back(qbitLoc);
  }

  // For control gates (e.g. CNOT), we need to reverse the leg pairing because
  // the (Control Index, Target Index) convention is the opposite of the
  // MSB->LSB bit order when the CNOT matrix is specified. e.g. the state vector
  // is indexed by q1q0.
  if (IsControlGate(GateType)) {
    std::reverse(gatePairing.begin(), gatePairing.end());
  }

  // Append the tensor for this gate to the network
  const bool appended = m_tensorExpansion.appendTensorGate(
      // Get the gate tensor data which must have been initialized.
      exatn::getTensor(uniqueGateName),
      // which qubits that the gate is acting on
      gatePairing);
  if (!appended) {
    const std::string gatePairingString = [&gatePairing]() {
      std::stringstream ss;
      ss << "{";
      for (const auto &pairIdx : gatePairing) {
        ss << pairIdx << ",";
      }
      ss << "}";
      return ss.str();
    }();
    xacc::error("Failed to append tensor for gate " +
                in_gateInstruction.name() + ", pairing = " + gatePairingString);
  }
}

template <typename TNQVM_COMPLEX_TYPE>
const double ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> in_function) {
  // TODO

  return 0.0;
}
} // end namespace tnqvm
// Register with CppMicroservices
REGISTER_PLUGIN(tnqvm::DefaultExatnGenVisitor, tnqvm::TNQVMVisitor);
#endif // TNQVM_HAS_EXATN
