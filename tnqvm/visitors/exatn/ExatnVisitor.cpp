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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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
 *   Implementation - Thien Nguyen 2019 - active;
 *
 **********************************************************************************/
#ifdef TNQVM_HAS_EXATN
#define _DEBUG_DIL

#include "ExatnVisitor.hpp"
#include "base/Gates.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "Instruction.hpp"
#include "talshxx.hpp"
#include <numeric>
#include <random>
#include <chrono>
#include <functional>
#include <unordered_set>
#include "utils/GateMatrixAlgebra.hpp"

#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

bool tnqvm_timing_log_enabled = true;

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

template<typename TNQVM_COMPLEX_TYPE>
std::vector<TNQVM_COMPLEX_TYPE> flattenGateMatrix(
    const std::vector<std::vector<TNQVM_COMPLEX_TYPE>> &in_gateMatrix) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  std::vector<TNQVM_COMPLEX_TYPE> resultVector;
  resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
  for (const auto &row : in_gateMatrix) {
    for (const auto &entry : row) {
      resultVector.emplace_back(entry);
    }
  }

  return resultVector;
}

template<typename TNQVM_COMPLEX_TYPE>
bool checkStateVectorNorm(
    const std::vector<TNQVM_COMPLEX_TYPE> &in_stateVec) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  const double norm = std::accumulate(
      in_stateVec.begin(), in_stateVec.end(), 0.0,
      [](double runningNorm, TNQVM_COMPLEX_TYPE vecComponent) {
        return runningNorm + std::norm(vecComponent);
      });

  return (std::abs(norm - 1.0) < 1e-12);
}

template<typename TNQVM_COMPLEX_TYPE>
double calcExpValueZ(const std::vector<int>& in_bits, const std::vector<TNQVM_COMPLEX_TYPE>& in_stateVec)
{
  TNQVM_TELEMETRY_ZONE("calcExpValueZ", __FILE__, __LINE__);
  const auto hasEvenParity = [](uint64_t x, const std::vector<int>& in_qubitIndices) -> bool {
    size_t count = 0;
    for (const auto& bitIdx : in_qubitIndices)
    {
        if (x & (1ULL << bitIdx))
        {
            count++;
        }
    }
    return (count % 2) == 0;
  };

  double result = 0.0;
  for(uint64_t i = 0; i < in_stateVec.size(); ++i)
  {
    result += (hasEvenParity(i, in_bits) ? 1.0 : -1.0) * std::norm(in_stateVec[i]);
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


std::string formatGateParamStr(const std::string& in_paramStr) {
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

template<typename TNQVM_COMPLEX_TYPE>
int TensorComponentPrintFunctor<TNQVM_COMPLEX_TYPE>::apply(talsh::Tensor &local_tensor) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  TNQVM_COMPLEX_TYPE *elements;
  const bool worked = local_tensor.getDataAccessHost(&elements);
  std::cout << "(Rank:" << local_tensor.getRank()
            << ", Volume: " << local_tensor.getVolume() << "): ";
  std::cout << "[";
  for (int i = 0; i < local_tensor.getVolume(); ++i) {
    const TNQVM_COMPLEX_TYPE element = elements[i];
    std::cout << element;
  }
  std::cout << "]\n";
  return 0;
}

template<typename TNQVM_COMPLEX_TYPE>
ReconstructStateVectorFunctor<TNQVM_COMPLEX_TYPE>::ReconstructStateVectorFunctor(
    const std::shared_ptr<AcceleratorBuffer> &buffer,
    std::vector<TNQVM_COMPLEX_TYPE> &io_stateVec)
    : m_qubits(buffer->size()), m_stateVec(io_stateVec) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  // Don't allow any attempts to reconstruct the state vector
  // if there are so many qubits.
  assert(m_qubits <= 32);
  m_stateVec.clear();
  m_stateVec.reserve(1 << m_qubits);
}

template<typename TNQVM_COMPLEX_TYPE>
int ReconstructStateVectorFunctor<TNQVM_COMPLEX_TYPE>::apply(talsh::Tensor &local_tensor) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  // Make sure we only call this on the final state tensor,
  // i.e. the rank must equal the number of qubits.
  assert(local_tensor.getRank() == m_qubits);
  TNQVM_COMPLEX_TYPE *elements;

  if (local_tensor.getDataAccessHost(&elements)) {
    m_stateVec.assign(elements, elements + local_tensor.getVolume());
  }

#ifdef _DEBUG_LOG_ENABLED
  const bool normOkay = checkStateVectorNorm(m_stateVec);
  assert(normOkay);
#endif

  return 0;
}

template<typename TNQVM_COMPLEX_TYPE>
CalculateExpectationValueFunctor<TNQVM_COMPLEX_TYPE>::CalculateExpectationValueFunctor(
    const std::vector<int> &qubitIndex)
    : m_qubitIndices(qubitIndex) {}

template<typename TNQVM_COMPLEX_TYPE>
int CalculateExpectationValueFunctor<TNQVM_COMPLEX_TYPE>::apply(talsh::Tensor &local_tensor) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  const auto hasEvenParity =
      [](size_t x, const std::vector<int> &in_qubitIndices) -> bool {
    size_t count = 0;
    for (const auto &bitIdx : in_qubitIndices) {
      if (x & (1ULL << bitIdx)) {
        count++;
      }
    }
    return (count % 2) == 0;
  };

  TNQVM_COMPLEX_TYPE *elements;
  const bool isOkay = local_tensor.getDataAccessHost(&elements);
  m_result = 0.0;
  if (isOkay) {
    for (uint64_t i = 0; i < local_tensor.getVolume(); ++i) {
      m_result += (hasEvenParity(i, m_qubitIndices) ? 1.0 : -1.0) *
                  std::norm(elements[i]);
    }
  }

  return 0;
}

template<typename TNQVM_COMPLEX_TYPE>
ApplyQubitMeasureFunctor<TNQVM_COMPLEX_TYPE>::ApplyQubitMeasureFunctor(int qubitIndex)
    : m_qubitIndex(qubitIndex) {}

template<typename TNQVM_COMPLEX_TYPE>
int ApplyQubitMeasureFunctor<TNQVM_COMPLEX_TYPE>::apply(talsh::Tensor &local_tensor) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  assert(local_tensor.getRank() > m_qubitIndex);
  TNQVM_COMPLEX_TYPE *elements;
  const auto N = local_tensor.getVolume();
  const bool isOkay = local_tensor.getDataAccessHost(&elements);
  if (isOkay) {
    const auto k_range = 1ULL << m_qubitIndex;
    const double randProbPick = generateRandomProbability();
    double cumulativeProb = 0.0;
    size_t stateSelect = 0;
    // select a state based on cumulative distribution
    while (cumulativeProb < randProbPick && stateSelect < N) {
      cumulativeProb += std::norm(elements[stateSelect++]);
    }
    stateSelect--;

    // take the value of the measured bit
    m_result = ((stateSelect >> m_qubitIndex) & 1);
    // Collapse the state vector according to the measurement result
    double measProb = 0.0;
    for (size_t g = 0; g < N; g += (k_range * 2)) {
      for (size_t i = 0; i < k_range; ++i) {
        if ((((i + g) >> m_qubitIndex) & 1) == m_result) {
          measProb += std::norm(elements[i + g]);
          elements[i + g + k_range] = 0.0;
        } else {
          measProb += std::norm(elements[i + g + k_range]);
          elements[i + g] = 0.0;
        }
      }
    }
    // Renormalize the state
    measProb = std::sqrt(measProb);
    for (size_t g = 0; g < N; g += (k_range * 2)) {
      for (size_t i = 0; i < k_range; i += 1) {
        if ((((i + g) >> m_qubitIndex) & 1) == m_result) {
          elements[i + g] /= measProb;
        } else {
          elements[i + g + k_range] /= measProb;
        }
      }
    }
  }

  return 0;
}

template<typename TNQVM_COMPLEX_TYPE>
ResetTensorDataFunctor<TNQVM_COMPLEX_TYPE>::ResetTensorDataFunctor(
    const std::vector<TNQVM_COMPLEX_TYPE> &in_stateVec)
    : m_stateVec(in_stateVec) {}

template<typename TNQVM_COMPLEX_TYPE>
int ResetTensorDataFunctor<TNQVM_COMPLEX_TYPE>::apply(talsh::Tensor &local_tensor) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  TNQVM_COMPLEX_TYPE *elements;

  if (local_tensor.getDataAccessHost(&elements)) {
    for (size_t i = 0; i < local_tensor.getVolume(); ++i) {
      elements[i] = m_stateVec[i];
    }
  }

  return 0;
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnDebugLogger<TNQVM_COMPLEX_TYPE>::preEvaluate(tnqvm::ExatnVisitor<TNQVM_COMPLEX_TYPE> *in_backEnd) {
  // If in Debug, print out tensor data using the Print Functor
  auto functor = std::make_shared<tnqvm::TensorComponentPrintFunctor<TNQVM_COMPLEX_TYPE>>();
  for (auto iter = in_backEnd->m_tensorNetwork.cbegin();
       iter != in_backEnd->m_tensorNetwork.cend(); ++iter) {
    const auto tensor = iter->second.getTensor();
    if (tensor->getName().front() != '_') {
      std::cout << tensor->getName();
      exatn::numericalServer->transformTensorSync(tensor->getName(), functor);
    }
  }
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnDebugLogger<TNQVM_COMPLEX_TYPE>::preMeasurement(tnqvm::ExatnVisitor<TNQVM_COMPLEX_TYPE> *in_backEnd,
                                      xacc::quantum::Measure &in_measureGate) {
  // Print out the state vector
  std::cout << "Applying " << in_measureGate.name() << " @ "
            << in_measureGate.bits()[0] << "\n";
  std::cout << "=========== BEFORE MEASUREMENT =================\n";
  std::cout << "State Vector: [";
  for (const auto &component : in_backEnd->retrieveStateVector()) {
    std::cout << component;
  }
  std::cout << "]\n";
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnDebugLogger<TNQVM_COMPLEX_TYPE>::postMeasurement(tnqvm::ExatnVisitor<TNQVM_COMPLEX_TYPE> *in_backEnd,
                                       xacc::quantum::Measure &in_measureGate,
                                       bool in_bitResult,
                                       double in_expectedValue) {
  // Print out the state vector
  std::cout << "=========== AFTER MEASUREMENT =================\n";
  std::cout << "Qubit measurement result (random binary): " << in_bitResult
            << "\n";
  std::cout << "Expected value (exp-val-z): " << in_expectedValue << "\n";
  std::cout << "State Vector: [";
  for (const auto &component : in_backEnd->retrieveStateVector()) {
    std::cout << component;
  }
  std::cout << "]\n";
  std::cout << "=============================================\n";
}

template<typename TNQVM_COMPLEX_TYPE>
ExatnVisitor<TNQVM_COMPLEX_TYPE>::ExatnVisitor()
    : m_tensorNetwork("Quantum Circuit"), m_tensorIdCounter(0),
      m_hasEvaluated(false), m_isAppendingCircuitGates(true) {}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::initialize(std::shared_ptr<AcceleratorBuffer> buffer,
                                                  int nbShots) {
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
    if (options.keyExists<int>("exatn-buffer-size-gb"))
    {
      int bufferSizeGb = options.get<int>("exatn-buffer-size-gb");
      if (bufferSizeGb < 1)
      {
        std::cout << "Minimum buffer size is 1 GB.\n";
        bufferSizeGb = 1;
      }
      // Set the memory buffer size:
      const int64_t memorySizeBytes = bufferSizeGb * (1ULL << 30);
      std::cout << "Set ExaTN host memory buffer to " << memorySizeBytes << " bytes.\n";
      const bool success = exatnParams.setParameter("host_memory_buffer_size", memorySizeBytes);
      assert(success);
      // Update buffer size.
      talshHostBufferSizeInBytes = memorySizeBytes;
    }
    else
    {
      // Use default buffer size
      const bool success = exatnParams.setParameter("host_memory_buffer_size", MAX_TALSH_MEMORY_BUFFER_SIZE_BYTES);
      assert(success);
    }
// This is a flag from ExaTN indicating that ExaTN was compiled
// w/ MPI enabled.
#ifdef MPI_ENABLED
    {
      if (options.keyExists<void*>("mpi-communicator"))
      {
        xacc::info("Setting ExaTN MPI_COMMUNICATOR...");
        auto communicator = options.get<void*>("mpi-communicator");
        exatn::MPICommProxy commProxy(communicator);
        exatn::initialize(commProxy, exatnParams);
      }
      else
      {
        // No specific communicator is specified,
        // exaTN will automatically use MPI_COMM_WORLD.
        exatn::initialize(exatnParams);
      }
      exatn::activateContrSeqCaching();
      //exatn::resetExecutionSerialization(true,true); //validation
    }
#else
    {
      exatn::initialize(exatnParams);
      exatn::activateContrSeqCaching();
      //exatn::resetExecutionSerialization(true,true); //validation
    }
#endif

    if (exatn::getDefaultProcessGroup().getSize() > 1)
    {
      // Multiple MPI processes:
      // if verbose is set, we must redirect log to files
      // with custom names for each process.
      if (xacc::verbose)
      {
        // Get the rank of this process.
        const int processRank = exatn::getProcessRank();
        const std::string fileNamePrefix = "process" + std::to_string(processRank);
        // Redirect log to files (one for each MPI process)
        xacc::logToFile(true, fileNamePrefix);
      }
    }

    if (options.stringExists("exatn-contract-seq-optimizer"))
    {
      const std::string optimizerName = options.getString("exatn-contract-seq-optimizer");
      std::cout << "Using '" << optimizerName << "' optimizer.\n";
      exatn::resetContrSeqOptimizer(optimizerName);
    }

    // ExaTN and XACC logging levels are always in-synced.
    // Note: If xacc::verbose is not set, we always set ExaTN logging level to 0.
    exatn::resetClientLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
    exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
    xacc::subscribeLoggingLevel([](int level) {
      exatn::resetClientLoggingLevel(xacc::verbose ? level : 0);
      exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
    });

    //Set up ExaTN computational backend:
    auto backends = exatn::queryComputationalBackends();
    if(std::find(backends.cbegin(),backends.cend(),"cuquantum") != backends.cend()) {
      exatn::switchComputationalBackend("cuquantum");
      //std::cout << "#MSG(TN-QVM:ExaTN): Switched computational backend to cuQuantum\n"; //debug
    }
  }

  m_hasEvaluated = false;
  m_buffer = std::move(buffer);
  m_shots = nbShots;
  // Generic kernel name:
  m_kernelName = "Quantum Circuit";

  // const int64_t exatnBufferSize = exatn::getMemoryBufferSize();
  // Note: exatn::getMemoryBufferSize() can cause potential deadlock if the
  // tensor runtime has not yet been created.
  const int64_t exatnBufferSize = talshHostBufferSizeInBytes;
  const int64_t maxVectorSize = exatnBufferSize / sizeof(TNQVM_COMPLEX_TYPE);
  const int64_t maxNbQubit = static_cast<int64_t>(log2(maxVectorSize));
  // Limit the number of qubits based on the host buffer size.
  // Note: there are other tensors (e.g. gate tensors, intermediate tensors),
  // hence we need to take those into account here.
  m_maxQubit = maxNbQubit - 4;

  // Note: this option is for *INTERNAL* use only.
  // e.g. purposely constraint the number of qubits to test all-reduce
  // wavefunction slices.
  if (options.keyExists<int>("max-qubit"))
  {
    m_maxQubit = options.get<int>("max-qubit");
    xacc::info("Set max qubit to " + m_maxQubit);
  }
  // Create the qubit register tensor
  for (int i = 0; i < m_buffer->size(); ++i) {
    const bool created = exatn::createTensor(
        generateQubitTensorName(i), getExatnElementType(),
        TensorShape{2});
    assert(created);
  }

  // Initialize the qubit register tensor to zero state
  for (int i = 0; i < m_buffer->size(); ++i) {
    // Define the tensor body for a zero-state qubit
    const bool initialized = exatn::initTensorData(generateQubitTensorName(i), std::vector<TNQVM_COMPLEX_TYPE>{{1.0, 0.0}, {0.0, 0.0}});
    assert(initialized);
  }

  // Append the qubit tensors to the tensor network
  for (int i = 0; i < m_buffer->size(); ++i) {
    m_tensorIdCounter++;
    m_tensorNetwork.appendTensor(
        m_tensorIdCounter, exatn::getTensor(generateQubitTensorName(i)),
        std::vector<std::pair<unsigned int, unsigned int>>{});
  }

  {
    // Copy the tensor network of qubit register
    m_qubitRegTensor = m_tensorNetwork;
    m_qubitRegTensor.rename("Qubit Register");
  }

// Add the Debug logging listener
#ifdef _DEBUG_LOG_ENABLED
  subscribe(ExatnDebugLogger::GetInstance());
#endif
}

template<typename TNQVM_COMPLEX_TYPE>
std::vector<TNQVM_COMPLEX_TYPE> ExatnVisitor<TNQVM_COMPLEX_TYPE>::retrieveStateVector() {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  std::vector<TNQVM_COMPLEX_TYPE> stateVec;
  auto stateVecFunctor =
      std::make_shared<ReconstructStateVectorFunctor<TNQVM_COMPLEX_TYPE>>(m_buffer, stateVec);
  exatn::numericalServer->transformTensorSync(
      m_tensorNetwork.getTensor(0)->getName(), stateVecFunctor);
  exatn::sync();
  return stateVec;
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::evaluateNetwork() {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  // Notify listeners
  {
    for (auto &listener : m_listeners) {
      listener->preEvaluate(this);
    }
  }

  // Evaluate the tensor network (quantum circuit):
  // For ExaTN, we only evaluate during finalization, i.e. after all gates have
  // been visited.
  if (m_buffer->size() <= MAX_NUMBER_QUBITS_FOR_STATE_VEC){
    TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
    m_tensorNetwork.rename(m_kernelName);
    const bool evaluated = exatn::evaluateSync(m_tensorNetwork);
    assert(evaluated);
    // Synchronize:
    exatn::sync();
    m_hasEvaluated = true;
    assert(m_tensorNetwork.getRank() == m_buffer->size());
  }
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::resetExaTN() {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  std::unordered_set<std::string> tensorList;
  for (auto iter = m_tensorNetwork.cbegin(); iter != m_tensorNetwork.cend();
       ++iter) {
    const auto &tensorName = iter->second.getTensor()->getName();
    // Not a root tensor
    if (!tensorName.empty() && tensorName[0] != '_') {
      tensorList.emplace(iter->second.getTensor()->getName());
    }
  }
  // Add any tensors which have been created but are not in the tensor network.
  // e.g. temporary tensors for expectation calculation.
  for (const auto &iter : m_gateTensorBodies) {
    const auto &tensorName = iter.first;
    if (tensorList.find(tensorName) == tensorList.end()) {
      tensorList.emplace(tensorName);
    }
  }

  for (const auto &tensorName : tensorList) {
    const bool destroyed = exatn::destroyTensor(tensorName);
    assert(destroyed);
  }
  m_gateTensorBodies.clear();
  m_appendedGateTensors.clear();
  m_tensorIdCounter = 0;
  TensorNetwork emptyTensorNet;
  m_tensorNetwork = emptyTensorNet;
  // Synchronize after tensor destroy
  exatn::sync();
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::resetNetwork() {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  // We must have evaluated the tensor network.
  assert(m_hasEvaluated);
  const auto stateVec = retrieveStateVector();
  // Re-initialize ExaTN
  resetExaTN();
  // The new qubit register tensor name will have name "RESET_"
  const std::string resetTensorName = "RESET_";
  // The qubit register tensor shape is {2, 2, 2, ...}, 1 leg for each qubit
  std::vector<int> qubitRegResetTensorShape(m_buffer->size(), 2);
  const bool created =
      exatn::createTensor(resetTensorName, getExatnElementType(),
                          qubitRegResetTensorShape);
  assert(created);
  // Initialize the tensor body with the state vector from the previous
  // evaluation.
  const bool initialized =
      exatn::initTensorData(resetTensorName, std::move(stateVec));
  assert(initialized);
  // Create a new tensor network
  m_tensorNetwork = TensorNetwork();
  // Reset counter
  m_tensorIdCounter = 1;

  // Use the root tensor from previous evaluation as the initial tensor
  m_tensorNetwork.appendTensor(
      m_tensorIdCounter, exatn::getTensor(resetTensorName),
      std::vector<std::pair<unsigned int, unsigned int>>{});
  // Reset the evaluation flag after initialization.
  m_hasEvaluated = false;
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::finalize() {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);

  // Calculate tensor network contraction FLOPS if requested:
  if (options.keyExists<bool>("calc-contract-cost-flops"))
  {
    auto bra = m_qubitRegTensor;
    // Conjugate the ket to get the bra (e.g. if it was initialized to a complex
    // superposition)
    bra.conjugate();
    auto combinedTensorNetwork = m_tensorNetwork;
    // Closing the tensor network with the bra
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (unsigned int i = 0; i < m_buffer->size(); ++i)
    {
      pairings.emplace_back(std::make_pair(i, i));
    }
    combinedTensorNetwork.appendTensorNetwork(std::move(bra), pairings);
    combinedTensorNetwork.collapseIsometries();

    // DEBUG:
    // combinedTensorNetwork.printIt();

    const std::string optimizerName = options.stringExists("exatn-contract-seq-optimizer") ? options.getString("exatn-contract-seq-optimizer") : "metis";
    const auto startOpt = std::chrono::system_clock::now();
    combinedTensorNetwork.getOperationList(optimizerName);
    const auto endOpt = std::chrono::system_clock::now();
    const int elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endOpt - startOpt).count();
    const double flops = combinedTensorNetwork.getFMAFlops();
    const double intermediatesVolume = combinedTensorNetwork.getMaxIntermediatePresenceVolume();
    const double sizeInBytes = intermediatesVolume * sizeof(TNQVM_COMPLEX_TYPE);
    // Note: we don't actually evaluate the tensor network when user requested this mode.
    m_hasEvaluated = true;

    // Add extra information: full tensor contraction (i.e. amplitude calculation)
    m_buffer->addExtraInfo("contract-flops", flops);
    m_buffer->addExtraInfo("max-node-bytes", sizeInBytes);
    m_buffer->addExtraInfo("optimizer-elapsed-time-ms", elapsedMs);

    // Calculate flops and memory for bit string generation:
    const auto flopsAndBytes = calcFlopsAndMemoryForSample(m_tensorNetwork);
    std::vector<double> flopsVec;
    std::vector<double> memBytesVec;
    for (auto it = std::make_move_iterator(flopsAndBytes.begin()),
          end = std::make_move_iterator(flopsAndBytes.end());
          it != end; ++it)
    {
      flopsVec.push_back(std::move(it->first));
      memBytesVec.push_back(std::move(it->second));
    }

    m_buffer->addExtraInfo("bitstring-contract-flops", flopsVec);
    m_buffer->addExtraInfo("bitstring-max-node-bytes", memBytesVec);
    m_buffer.reset();
    resetExaTN();
    return;
  }

  // Calculates the amplitude of a specific bitstring
  // or the partial (slice) wave function.
  // The open indices are denoted by "-1" value.
  if (options.keyExists<std::vector<int>>("bitstring"))
  {
    std::vector<int> bitString = options.get<std::vector<int>>("bitstring");
    if (bitString.size() != m_buffer->size())
    {
      xacc::error("Bitstring size must match the number of qubits.");
      return;
    }

    std::vector<TNQVM_COMPLEX_TYPE> waveFuncSlice = computeWaveFuncSlice(m_tensorNetwork, bitString, exatn::getDefaultProcessGroup());
    assert(!waveFuncSlice.empty());
    if (waveFuncSlice.size() == 1)
    {
      m_buffer->addExtraInfo("amplitude-real", waveFuncSlice[0].real());
      m_buffer->addExtraInfo("amplitude-imag", waveFuncSlice[0].imag());
    }
    else
    {
      const auto normalizeWaveFnSlice = [](std::vector<TNQVM_COMPLEX_TYPE>& io_waveFn){
        const double normVal = std::accumulate(io_waveFn.begin(), io_waveFn.end(), 0.0, [](double sumVal, const TNQVM_COMPLEX_TYPE& val){
          return sumVal + std::norm(val);
        });
        // The slice may have zero norm:
        if (normVal > 1e-12) {
          const TNQVM_COMPLEX_TYPE sqrtNorm = sqrt(normVal);
          for (auto& val : io_waveFn)
          {
            val = val / sqrtNorm;
          }
        }
      };

      normalizeWaveFnSlice(waveFuncSlice);
      std::vector<double> amplReal;
      std::vector<double> amplImag;
      amplReal.reserve(waveFuncSlice.size());
      amplImag.reserve(waveFuncSlice.size());
      for (const auto& val : waveFuncSlice)
      {
        amplReal.emplace_back(val.real());
        amplImag.emplace_back(val.imag());
      }
      m_buffer->addExtraInfo("amplitude-real-vec", amplReal);
      m_buffer->addExtraInfo("amplitude-imag-vec", amplImag);
    }

    m_buffer.reset();
    m_hasEvaluated = true;
    resetExaTN();
    return;
  }

  // Validates ExaTN numerical backend: contract tensor network and its conjugate
  if (options.keyExists<bool>("contract-with-conjugate"))
  {
    const bool validateOk = validateTensorNetworkContraction(m_tensorNetwork);
    assert(validateOk);
    m_buffer->addExtraInfo("contract-with-conjugate-result", validateOk);
    m_buffer.reset();
    m_hasEvaluated = true;
    resetExaTN();
    return;
  }

  if (m_buffer->size() > MAX_NUMBER_QUBITS_FOR_STATE_VEC && !m_measureQbIdx.empty() && m_shots > 0 && !m_hasEvaluated)
  {
    std::cout << "Simulating bit string by tensor contraction and projection \n";
    for (int i = 0; i < m_shots; ++i)
    {
      const auto convertToBitString = [](const std::vector<uint8_t>& in_bitVec){
          std::string result;
          for (const auto& bit : in_bitVec)
          {
              result.append(std::to_string(bit));
          }
          return result;
      };

      m_buffer->appendMeasurement(convertToBitString(generateMeasureSample(m_tensorNetwork, m_measureQbIdx)));
    }
  }
  else
  {
    if (m_buffer->size() > m_maxQubit) {
      m_buffer->addExtraInfo("exp-val-z", internalComputeExpectationValueZ());
    } else {
      if (!m_hasEvaluated) {
        // If we haven't evaluated the network, do it now (end of circuit).
        evaluateNetwork();
      }
      // Shots
      if (m_shots > 0) {
        const auto cachedStateVec = retrieveStateVector();
        for (int i = 0; i < m_shots; ++i) {
          auto stateVecCopy = cachedStateVec;
          for (const auto &idx : m_measureQbIdx) {
            // Append the boolean true/false as bit value
            m_resultBitString.append(
                std::to_string(ApplyMeasureOp(stateVecCopy, idx)));
          }
          // Finish measuring all qubits, append the bit-string measurement
          // result.
          m_buffer->appendMeasurement(m_resultBitString);
          // Clear the result bit string after appending (to be constructed in
          // the next shot)
          m_resultBitString.clear();
        }
      }
      // No-shots, just add expectation value:
      else {
        if (!m_measureQbIdx.empty()) {
          const auto tensorData = retrieveStateVector();
          const double exp_val_z = calcExpValueZ(m_measureQbIdx, tensorData);
          m_buffer->addExtraInfo("exp-val-z", exp_val_z);
        }
      }
    }

    // Notify listeners
    {
      for (auto &listener : m_listeners) {
        listener->onEvaluateComplete(this);
      }
    }
  }

  m_buffer.reset();
  resetExaTN();
}

// === BEGIN: Gate Visitor Impls ===
template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Identity &in_IdentityGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::I>(in_IdentityGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Hadamard &in_HadamardGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::H>(in_HadamardGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(X &in_XGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::X>(in_XGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Y &in_YGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::Y>(in_YGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Z &in_ZGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::Z>(in_ZGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Rx &in_RxGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  assert(in_RxGate.nParameters() == 1);
  const double theta = in_RxGate.getParameter(0).as<double>();
  appendGateTensor<CommonGates::Rx>(in_RxGate, theta);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Ry &in_RyGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  assert(in_RyGate.nParameters() == 1);
  const double theta = in_RyGate.getParameter(0).as<double>();
  appendGateTensor<CommonGates::Ry>(in_RyGate, theta);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Rz &in_RzGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  assert(in_RzGate.nParameters() == 1);
  const double theta = in_RzGate.getParameter(0).as<double>();
  appendGateTensor<CommonGates::Rz>(in_RzGate, theta);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(T &in_TGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::T>(in_TGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Tdg &in_TdgGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::Tdg>(in_TdgGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(CPhase &in_CPhaseGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::CPhase>(in_CPhaseGate, in_CPhaseGate.getParameter(0).as<double>());
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(U &in_UGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  assert(in_UGate.nParameters() == 3);
  const double theta = in_UGate.getParameter(0).as<double>();
  const double phi = in_UGate.getParameter(1).as<double>();
  const double lambda = in_UGate.getParameter(2).as<double>();
  appendGateTensor<CommonGates::U>(in_UGate, theta, phi, lambda);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(CNOT &in_CNOTGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::CNOT>(in_CNOTGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Swap &in_SwapGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::Swap>(in_SwapGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(CZ &in_CZGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::CZ>(in_CZGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(iSwap& in_iSwapGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  appendGateTensor<CommonGates::iSwap>(in_iSwapGate);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(fSim& in_fsimGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  assert(in_fsimGate.nParameters() == 2);
  const double theta = in_fsimGate.getParameter(0).as<double>();
  const double phi = in_fsimGate.getParameter(1).as<double>();
  appendGateTensor<CommonGates::fSim>(in_fsimGate, theta, phi);
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::visit(Measure &in_MeasureGate) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  if (m_buffer->size() > m_maxQubit)
  {
    // If the circuit contains many qubits, we can only
    // generate bit string samples by contracting tensors in the end.
    const int measQubit = in_MeasureGate.bits()[0];
    m_measureQbIdx.emplace_back(measQubit);
    return;
  }
  // When we visit a measure gate, evaluate the current tensor network (up to
  // this measurement) Note: currently, we cannot do gate operations post
  // measurement yet (i.e. multiple evaluateNetwork() calls). Multiple
  // measurement ops at the end is supported, i.e. can measure the entire qubit
  // register.
  // TODO: reset the tensor network and continue appending gate tensors.
  if (!m_hasEvaluated) {
    // If this is the first measure gate that we visit,
    // i.e. the tensor network hasn't been evaluate, do it now.
    evaluateNetwork();
  }

  // Notify listeners: before measurement
  {
    for (auto &listener : m_listeners) {
      listener->preMeasurement(this, in_MeasureGate);
    }
  }

  const int measQubit = in_MeasureGate.bits()[0];
  m_measureQbIdx.emplace_back(measQubit);

  // If multi-shot was requested, we skip measurement till the end.
  if (m_shots > 0) {
    return;
  }
}
// === END: Gate Visitor Impls ===
template <typename TNQVM_COMPLEX_TYPE>
template <tnqvm::CommonGates GateType, typename... GateParams>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::appendGateTensor(const xacc::Instruction &in_gateInstruction,
                                    GateParams &&... in_params) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  if (m_hasEvaluated) {
    // If we have evaluated the tensor network,
    // for example, because of measurement,
    // and now we want to append more quantum gates,
    // we need to reset the network before appending more gate tensors.
    resetNetwork();
  }

  const auto gateName = GetGateName(GateType);
  const GateInstanceIdentifier gateInstanceId(gateName, in_params...);
  const std::string uniqueGateName = gateInstanceId.toNameString();
  // If the tensor data for this gate hasn't been initialized before,
  // then initialize it.
  if (m_gateTensorBodies.find(uniqueGateName) == m_gateTensorBodies.end()) {
    const auto gateMatrixRaw= GetGateMatrix<GateType>(in_params...);
    std::vector<std::vector<TNQVM_COMPLEX_TYPE>> gateMatrix;
    for (auto& row : gateMatrixRaw)
    {
      std::vector<TNQVM_COMPLEX_TYPE> rowConverted(row.begin(), row.end());
      gateMatrix.emplace_back(std::move(rowConverted));
    }
    m_gateTensorBodies[uniqueGateName] = flattenGateMatrix(gateMatrix);
    // Currently, we only support 2-qubit gates.
    assert(in_gateInstruction.nRequiredBits() > 0 &&
           in_gateInstruction.nRequiredBits() <= 2);
    const auto gateTensorShape =
        (in_gateInstruction.nRequiredBits() == 1 ? TensorShape{2, 2}
                                                 : TensorShape{2, 2, 2, 2});
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

  // Helper to create unique tensor names in the format
  // <GateTypeName>_<Counter>, e.g. H2, CNOT5, etc. Note: this is different from
  // the gate instance unique name which is referencing *unique* gate matrices.
  // Multiple tensors can just refer to the same tensor body,
  // for example, all H_k (Hadamard gates) in the circuit will all refer to a
  // single *H* tensor body data.
  const auto generateTensorName = [&]() -> std::string {
    return GetGateName(GateType) + "_" + std::to_string(m_tensorIdCounter);
  };

  m_tensorIdCounter++;

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

  if (m_isAppendingCircuitGates) {
    // Append the gate tensor to the tracking list to apply inverse
    m_appendedGateTensors.emplace_back(
        std::make_pair(uniqueGateName, gatePairing));
  }

  // Append the tensor for this gate to the network
  const bool appended = m_tensorNetwork.appendTensorGate(
      m_tensorIdCounter,
      // Get the gate tensor data which must have been initialized.
      exatn::getTensor(uniqueGateName),
      // which qubits that the gate is acting on
      gatePairing);
  if (!appended) {
    const std::string gatePairingString = [&gatePairing](){
      std::stringstream ss;
      ss << "{";
      for (const auto& pairIdx : gatePairing) {
        ss << pairIdx << ",";
      }
      ss << "}";
      return ss.str();
    }();
    xacc::error("Failed to append tensor for gate " + in_gateInstruction.name() + ", pairing = " + gatePairingString);
  }
}

template<typename TNQVM_COMPLEX_TYPE>
ExatnVisitor<TNQVM_COMPLEX_TYPE>::ObservableTerm::ObservableTerm(
    const std::vector<std::shared_ptr<Instruction>> &in_operatorsInProduct,
    const TNQVM_COMPLEX_TYPE &in_coeff /*= 1.0*/)
    : coefficient(in_coeff), operators(in_operatorsInProduct) {}

template<typename TNQVM_COMPLEX_TYPE>
TNQVM_COMPLEX_TYPE ExatnVisitor<TNQVM_COMPLEX_TYPE>::observableExpValCalc(
    std::shared_ptr<AcceleratorBuffer> &in_buffer,
    std::shared_ptr<CompositeInstruction> &in_function,
    const std::vector<ObservableTerm> &in_observableExpression) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  if (!m_appendedGateTensors.empty() || !m_gateTensorBodies.empty()) {
    // We don't support mixing this *observable* mode of execution with the
    // regular mode.
    xacc::error("observableExpValCalc can only be called on an ExatnVisitor "
                "that is not executing a circuit.");
    return 0.0;
  }

  BaseInstructionVisitor *visitorCast =
      static_cast<BaseInstructionVisitor *>(this);
  this->initialize(in_buffer, -1);
  // Walk the IR tree, and visit each node
  InstructionIterator it(in_function);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled()) {
      nextInst->accept(visitorCast);
    }
  }

  const auto result = expVal(in_observableExpression);
  // Set Evaluated flag, hence no need to evaluate the original circuit anymore.
  // (we have calculated the expectation value by closing the entire tensor
  // network)
  m_hasEvaluated = true;
  exatn::sync();
  finalize();

  return result;
}

template<typename TNQVM_COMPLEX_TYPE>
TNQVM_COMPLEX_TYPE ExatnVisitor<TNQVM_COMPLEX_TYPE>::expVal(
    const std::vector<ObservableTerm> &in_observableExpression) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  TNQVM_COMPLEX_TYPE result = 0.0;
  for (const auto &term : in_observableExpression) {
    result += (term.coefficient * evaluateTerm(term.operators));
  }

  return result;
}

template<typename TNQVM_COMPLEX_TYPE>
TNQVM_COMPLEX_TYPE ExatnVisitor<TNQVM_COMPLEX_TYPE>::evaluateTerm(
    const std::vector<std::shared_ptr<Instruction>> &in_observableTerm) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  TNQVM_COMPLEX_TYPE result = 0.0;
  // Save/cache the tensor network
  const auto cachedTensor = m_tensorNetwork;
  const auto cachedIdCounter = m_tensorIdCounter;
  m_isAppendingCircuitGates = false;
  {
    for (auto &inst : in_observableTerm) {
      // Adding the Hamiltonian term to the tensor network
      inst->accept(static_cast<BaseInstructionVisitor *>(this));
    }
  }

  // Close the tensor network
  applyInverse();

  {
    TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);

    if (exatn::evaluateSync(m_tensorNetwork)) {
      exatn::sync();
      auto talsh_tensor =
          exatn::getLocalTensor(m_tensorNetwork.getTensor(0)->getName());
      assert(talsh_tensor->getVolume() == 1);
      const TNQVM_COMPLEX_TYPE *body_ptr;
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
        result = *body_ptr;
      }
    }
  }

  m_isAppendingCircuitGates = true;

  // Restore the tensor network after evaluate the exp-val for the term
  m_tensorNetwork = cachedTensor;
  m_tensorIdCounter = cachedIdCounter;
  return result;
}

template<typename TNQVM_COMPLEX_TYPE>
void ExatnVisitor<TNQVM_COMPLEX_TYPE>::applyInverse() {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  for (auto iter = m_appendedGateTensors.rbegin();
       iter != m_appendedGateTensors.rend(); ++iter) {
    m_tensorIdCounter++;
    m_tensorNetwork.appendTensorGate(
        m_tensorIdCounter,
        // Get the gate tensor data which must have been initialized.
        exatn::getTensor(iter->first),
        // which qubits that the gate is acting on
        iter->second,
        // Set conjugate flag
        true);
  }
  {
    auto bra = m_qubitRegTensor;
    // Conjugate the ket to get the bra (e.g. if it was initialized to a complex
    // superposition)
    bra.conjugate();
    // Closing the tensor network with the bra
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (unsigned int i = 0; i < m_buffer->size(); ++i) {
      pairings.emplace_back(std::make_pair(i, i));
    }
    m_tensorNetwork.appendTensorNetwork(std::move(bra), pairings);
  }

  { const bool collapsed = m_tensorNetwork.collapseIsometries(); }
}

template<typename TNQVM_COMPLEX_TYPE>
std::vector<TNQVM_COMPLEX_TYPE> ExatnVisitor<TNQVM_COMPLEX_TYPE>::getReducedDensityMatrix(
    std::shared_ptr<AcceleratorBuffer> &in_buffer,
    std::shared_ptr<CompositeInstruction> &in_function,
    const std::vector<size_t> &in_qubitIdx) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  if (!m_appendedGateTensors.empty() || !m_gateTensorBodies.empty()) {
    // We don't support mixing this *RDM* mode of execution with the regular
    // mode.
    // TODO: Adding runtime logic to determine if we need to use this mode on
    // *regular* circuit execution. e.g. if there are many qubits and we are
    // requesting *shots* samping from a small subset of qubits.
    xacc::error("getReducedDensityMatrix can only be called on an ExatnVisitor "
                "that is not executing a circuit.");
    return {};
  }

  std::vector<TNQVM_COMPLEX_TYPE> resultRDM;
  BaseInstructionVisitor *visitorCast =
      static_cast<BaseInstructionVisitor *>(this);
  this->initialize(in_buffer, -1);
  // Walk the IR tree, and visit each node
  InstructionIterator it(in_function);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled() && nextInst->name() != "Measure") {
      nextInst->accept(visitorCast);
    }
  }

  auto inverseTensorNetwork = m_tensorNetwork;
  inverseTensorNetwork.rename("Inverse Tensor Network");
  inverseTensorNetwork.conjugate();

  // Connect the original tensor network with its inverse
  // but leave those qubit lines that we want to get RDM open
  {
    TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
    auto combinedNetwork = m_tensorNetwork;
    combinedNetwork.rename("Combined Tensor Network");
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (size_t i = 0; i < m_buffer->size(); ++i) {
      if (std::find(in_qubitIdx.begin(), in_qubitIdx.end(), i) ==
          in_qubitIdx.end()) {
        // We want to close this leg (not one of those requested for RDM)
        pairings.emplace_back(std::make_pair(i, i));
      }
    }

    combinedNetwork.appendTensorNetwork(std::move(inverseTensorNetwork),
                                        pairings);
    const bool collapsed = combinedNetwork.collapseIsometries();

    if (exatn::evaluateSync(combinedNetwork)) {
      exatn::sync();
      auto talsh_tensor =
          exatn::getLocalTensor(combinedNetwork.getTensor(0)->getName());
      const auto tensorVolume = talsh_tensor->getVolume();
      // Double check the size of the RDM
      assert(tensorVolume == 1ULL << (2 * in_qubitIdx.size()));
      const TNQVM_COMPLEX_TYPE *body_ptr;
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
        resultRDM.assign(body_ptr, body_ptr + tensorVolume);
      }
    }
  }

  m_hasEvaluated = true;
  exatn::sync();
  finalize();
  return resultRDM;
}

template<typename TNQVM_COMPLEX_TYPE>
std::vector<uint8_t> ExatnVisitor<TNQVM_COMPLEX_TYPE>::getMeasureSample(
    std::shared_ptr<AcceleratorBuffer> &in_buffer,
    std::shared_ptr<CompositeInstruction> &in_function,
    const std::vector<size_t> &in_qubitIdx) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  if (!m_appendedGateTensors.empty() || !m_gateTensorBodies.empty()) {
    xacc::error("getMeasureSample can only be called on an ExatnVisitor that "
                "is not executing a circuit.");
    return {};
  }

  std::vector<uint8_t> resultBitString;
  std::vector<ExatnVisitor<TNQVM_COMPLEX_TYPE>::TNQVM_FLOAT_TYPE> resultProbs;
  for (const auto &qubitIdx : in_qubitIdx) {
    std::vector<TNQVM_COMPLEX_TYPE> resultRDM;

    BaseInstructionVisitor *visitorCast =
        static_cast<BaseInstructionVisitor *>(this);
    this->initialize(in_buffer, -1);
    // Walk the IR tree, and visit each node
    InstructionIterator it(in_function);
    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled() && nextInst->name() != "Measure") {
        nextInst->accept(visitorCast);
      }
    }

    auto inverseTensorNetwork = m_tensorNetwork;
    inverseTensorNetwork.rename("Inverse Tensor Network");
    inverseTensorNetwork.conjugate();

    {
      {
        // Adding collapse tensors based on previous measurement results.
        // i.e. condition/renormalize the tensor network to be consistent with
        // previous result.
        for (size_t measIdx = 0; measIdx < resultBitString.size(); ++measIdx) {
          const unsigned int qId = in_qubitIdx[measIdx];
          m_tensorIdCounter++;
          // If it was a "0":
          if (resultBitString[measIdx] == 0) {
            const std::vector<TNQVM_COMPLEX_TYPE> COLLAPSE_0{
                // Renormalize based on the probability of this outcome
                {1.0f / resultProbs[measIdx], 0.0},
                {0.0, 0.0},
                {0.0, 0.0},
                {0.0, 0.0}};

            const std::string tensorName =
                "COLLAPSE_0_" + std::to_string(measIdx);
            const bool created = exatn::createTensor(
                tensorName, getExatnElementType(),
                TensorShape{2, 2});
            assert(created);
            const bool registered =
                (exatn::registerTensorIsometry(tensorName, {0}, {1}));
            assert(registered);
            const bool initialized =
                exatn::initTensorData(tensorName, COLLAPSE_0);
            assert(initialized);
            const bool appended = m_tensorNetwork.appendTensorGate(
                m_tensorIdCounter, exatn::getTensor(tensorName), {qId});
            assert(appended);
          } else {
            assert(resultBitString[measIdx] == 1);
            // Renormalize based on the probability of this outcome
            const std::vector<TNQVM_COMPLEX_TYPE> COLLAPSE_1{
                {0.0, 0.0},
                {0.0, 0.0},
                {0.0, 0.0},
                {1.0f / resultProbs[measIdx], 0.0}};

            const std::string tensorName =
                "COLLAPSE_1_" + std::to_string(measIdx);
            const bool created = exatn::createTensor(
                tensorName, getExatnElementType(),
                TensorShape{2, 2});
            assert(created);
            const bool registered =
                (exatn::registerTensorIsometry(tensorName, {0}, {1}));
            assert(registered);
            const bool initialized =
                exatn::initTensorData(tensorName, COLLAPSE_1);
            assert(initialized);
            const bool appended = m_tensorNetwork.appendTensorGate(
                m_tensorIdCounter, exatn::getTensor(tensorName), {qId});
            assert(appended);
          }
        }
      }

      auto combinedNetwork = m_tensorNetwork;
      combinedNetwork.rename("Combined Tensor Network");
      {
        // Append the conjugate network to calculate the RDM of the measure
        // qubit
        std::vector<std::pair<unsigned int, unsigned int>> pairings;
        for (size_t i = 0; i < m_buffer->size(); ++i) {
          // Connect the original tensor network with its inverse
          // but leave the measure qubit line open.
          if (i != qubitIdx) {
            pairings.emplace_back(std::make_pair(i, i));
          }
        }

        combinedNetwork.appendTensorNetwork(std::move(inverseTensorNetwork),
                                            pairings);
        const bool collapsed = combinedNetwork.collapseIsometries();
      }

      // Evaluate
      {
        TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
        if (exatn::evaluateSync(combinedNetwork)) {
          exatn::sync();
          auto talsh_tensor =
              exatn::getLocalTensor(combinedNetwork.getTensor(0)->getName());
          const auto tensorVolume = talsh_tensor->getVolume();
          // Single qubit density matrix
          assert(tensorVolume == 4);
          const TNQVM_COMPLEX_TYPE *body_ptr;
          if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
            resultRDM.assign(body_ptr, body_ptr + tensorVolume);
          }
  #ifdef _DEBUG_LOG_ENABLED
          // Debug: print out RDM data
          {
            std::cout << "RDM @q" << qubitIdx << " = [";
            for (int i = 0; i < talsh_tensor->getVolume(); ++i) {
              const TNQVM_COMPLEX_TYPE element = body_ptr[i];
              std::cout << element;
            }
            std::cout << "]\n";
          }
  #endif
        }
      }
    }

    {
      // Perform the measurement
      assert(resultRDM.size() == 4);
      const double prob_0 = resultRDM.front().real();
      const double prob_1 = resultRDM.back().real();
      assert(prob_0 >= 0.0 && prob_1 >= 0.0);
      assert(std::fabs(1.0 - prob_0 - prob_1) < 1e-12);

      // Generate a random number
      const double randProbPick = generateRandomProbability();
      // If radom number < probability of 0 state -> pick zero, and vice versa.
      resultBitString.emplace_back(randProbPick <= prob_0 ? 0 : 1);
      resultProbs.emplace_back(randProbPick <= prob_0 ? prob_0 : prob_1);
#ifdef _DEBUG_LOG_ENABLED
      {
        std::cout << ">> Measure @q" << qubitIdx << " prob(0) = " << prob_0
                  << "\n";
        std::cout << ">> Measure @q" << qubitIdx << " prob(1) = " << prob_1
                  << "\n";
        std::cout << ">> Measure @q" << qubitIdx
                  << " random number = " << randProbPick << "\n";
        std::cout << ">> Measure @q" << qubitIdx << " pick "
                  << std::to_string(resultBitString.back()) << "\n";
      }
#endif
    }

    {
      m_hasEvaluated = true;
      finalize();
    }
  }

  return resultBitString;
}

template<typename TNQVM_COMPLEX_TYPE>
const double ExatnVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> in_function) {
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  if (!m_buffer) {
    xacc::error("Please initialize the visitor backend before calling "
                "getExpectationValueZ()!");
    return 0.0;
  }
  // The number of qubits exceed the limit for full wave-function contraction,
  // hence we cannot cache the wavefunction.
  if (m_buffer->size() > m_maxQubit) {
    // Need to slice.
    m_kernelName = in_function->name();
    return internalComputeExpectationValueZ(in_function);
  }

  // The new qubit register tensor name will have name "RESET_"
  const std::string resetTensorName = "RESET_";
  if (!m_hasEvaluated)
  {
    {
      TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
      const bool evaluated = exatn::evaluateSync(m_tensorNetwork);
      assert(evaluated);
      // Synchronize:
      exatn::sync();
      m_cacheStateVec = retrieveStateVector();
    }

    // State vector after the base ansatz
    assert(m_cacheStateVec.size() == (1ULL << m_buffer->size()));

    // The qubit register tensor shape is {2, 2, 2, ...}, 1 leg for each qubit
    std::vector<int> qubitRegResetTensorShape(m_buffer->size(), 2);
    const bool created = exatn::createTensor(resetTensorName, getExatnElementType(), qubitRegResetTensorShape);
    assert(created);
    // Initialize the tensor body with the state vector from the previous
    // evaluation.
    const bool initialized = exatn::initTensorData(resetTensorName, m_cacheStateVec);
    assert(initialized);
    for (auto iter = m_qubitRegTensor.cbegin(); iter != m_qubitRegTensor.cend(); ++iter)
    {
      const auto& tensorName = iter->second.getTensor()->getName();
      // Not a root tensor
      if (!tensorName.empty() && tensorName[0] != '_')
      {
        const bool destroyed = exatn::destroyTensorSync(tensorName);
        assert(destroyed);
      }
    }
  }

  // Create a new tensor network
  m_tensorNetwork = TensorNetwork(in_function->name());
  // Reset counter
  m_tensorIdCounter = 1;
  m_measureQbIdx.clear();
  // Use the root tensor from previous evaluation as the initial tensor
  m_tensorNetwork.appendTensor(m_tensorIdCounter, exatn::getTensor(resetTensorName), std::vector<std::pair<unsigned int, unsigned int>>{});
  // Walk the remaining circuit and visit all gates
  InstructionIterator it(in_function);
  m_hasEvaluated = false;
  size_t nbBasisChangeInsts = 0;
  while (it.hasNext())
  {
    auto nextInst = it.next();
    if (nextInst->isEnabled() && !nextInst->isComposite())
    {
      if (nextInst->name() != "Measure")
      {
        nextInst->accept(this);
        nbBasisChangeInsts++;
      }
      else
      {
        m_measureQbIdx.emplace_back(nextInst->bits()[0]);
      }
    }
  }
  assert(!m_measureQbIdx.empty());
  // If there are basis change instructions:
  // i.e. not Z basis
  if (nbBasisChangeInsts > 0)
  {
    TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
    const bool evaluated = exatn::evaluateSync(m_tensorNetwork);
    assert(evaluated);
  }
  m_hasEvaluated = true;
  const double exp_val_z = (nbBasisChangeInsts > 0) ? calcExpValueZ(m_measureQbIdx, retrieveStateVector()) :  calcExpValueZ(m_measureQbIdx, m_cacheStateVec);
  exatn::destroyTensorSync(m_tensorNetwork.getTensor(0)->getName());
  m_measureQbIdx.clear();
  return exp_val_z;
}

template <typename TNQVM_COMPLEX_TYPE>
double ExatnVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZBySlicing(
    std::shared_ptr<CompositeInstruction> in_function) {
  // Cache the current tensor network:
  exatn::TensorNetwork cacheTensorNet = m_tensorNetwork;
  InstructionIterator it(in_function);
  // Add remaining instructions:
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled() && !nextInst->isComposite()) {
      nextInst->accept(this);
    }
  }
  assert(!m_measureQbIdx.empty());
  const double result = getExpectationValueZBySlicing();
  m_measureQbIdx.clear();
  // Restore the base tensor network
  m_tensorNetwork = cacheTensorNet;
  return result;
}

template <typename TNQVM_COMPLEX_TYPE>
double ExatnVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZBySlicing() {
  // Number of qubits we need to project.
  const size_t nbProjectedQubits = m_buffer->size() - m_maxQubit;
  // The number of paths we need to reduce.
  const int64_t nbProjectedPaths = (1ULL << nbProjectedQubits);
  // Strategy:
  // Q0 -> Q(m_maxQubit - 1): compute slice
  // The rest (nbProjectedQubits): we sequence through nbProjectedPaths to
  // compute partial expectations for all slices then reduce.
  // Loop to be parallelized
  if (getNumMpiProcs() <= 1) {
    std::vector<double> partialExpectationValues(nbProjectedPaths);
    for (int i = 0; i < nbProjectedPaths; ++i) {
      bool evenParity = true;
      // Open legs: 0-m_maxQubit
      std::vector<int> bitString(m_maxQubit, -1);
      for (int64_t bitIdx = 0; bitIdx < nbProjectedQubits; ++bitIdx) {
        const int globalQid = bitIdx + m_maxQubit;
        const int64_t bitMask = 1ULL << bitIdx;
        if ((i & bitMask) == bitMask) {
          bitString.emplace_back(1);
          if (xacc::container::contains(m_measureQbIdx, globalQid)) {
            // Flip even parity flag
            evenParity = !evenParity;
          }
        } else {
          bitString.emplace_back(0);
        }
      }
      std::vector<TNQVM_COMPLEX_TYPE> waveFuncSlice = computeWaveFuncSlice(
          m_tensorNetwork, bitString, exatn::getDefaultProcessGroup());
      const double exp_val_z = calcExpValueZ(m_measureQbIdx, waveFuncSlice);
      partialExpectationValues[i] = evenParity ? exp_val_z : -exp_val_z;
    }
    const auto finalExpVal = std::accumulate(
        partialExpectationValues.begin(), partialExpectationValues.end(), 0.0);
    return finalExpVal;
  } else {
    // Multiple MPI processes:
    // Note: if the number of MPI processes > total number of paths,
    // just use enough processes (each process handles 1 path), the rest is
    // unused.
    const auto nbMpiProcsToUse = (nbProjectedPaths > getNumMpiProcs())
                                     ? getNumMpiProcs()
                                     : nbProjectedPaths;
    const auto processRank = exatn::getProcessRank();
    const auto nbProjectedPathsPerProcess = nbProjectedPaths / nbMpiProcsToUse;
    // If not evenly divided, process 0 to get the remain.
    // i.e. process 0 always has the largest partial exp-val vectors.
    const auto nbProjectedPathsProcess0 =
        nbProjectedPathsPerProcess + nbProjectedPaths % nbMpiProcsToUse;
    assert((nbProjectedPathsProcess0 +
            (nbMpiProcsToUse - 1) * nbProjectedPathsPerProcess) ==
           nbProjectedPaths);
    std::vector<double> partialExpectationValues(nbProjectedPathsPerProcess);
    if (processRank == 0) {
      partialExpectationValues.resize(nbProjectedPathsProcess0);
    }
    // Name of the tensor to hold the accumulated exp-value of the process.
    const std::string accumulatedTensorName = "ExpVal";

    // These processes need to do work:
    if (processRank < nbMpiProcsToUse) {
      const int64_t processStartIdx =
          (processRank == 0) ? 0
                             : (nbProjectedPathsProcess0 +
                                (processRank - 1) * nbProjectedPathsPerProcess);
      const int64_t processEndIdx =
          (processRank == 0) ? nbProjectedPathsProcess0
                             : (processStartIdx + nbProjectedPathsPerProcess);
      std::stringstream ss;
      ss << "Process [" << processRank
                << "]: Start = " << processStartIdx
                << "; End = " << processEndIdx << "\n";
      xacc::info(ss.str());
      int64_t vectorIdx = 0;
      for (int64_t i = processStartIdx; i < processEndIdx; ++i) {
        bool evenParity = true;
        // Open legs: 0-m_maxQubit
        std::vector<int> bitString(m_maxQubit, -1);
        for (int64_t bitIdx = 0; bitIdx < nbProjectedQubits; ++bitIdx) {
          const int globalQid = bitIdx + m_maxQubit;
          const int64_t bitMask = 1ULL << bitIdx;
          if ((i & bitMask) == bitMask) {
            bitString.emplace_back(1);
            if (xacc::container::contains(m_measureQbIdx, globalQid)) {
              // Flip even parity flag
              evenParity = !evenParity;
            }
          } else {
            bitString.emplace_back(0);
          }
        }
        std::vector<TNQVM_COMPLEX_TYPE> waveFuncSlice = computeWaveFuncSlice(
            m_tensorNetwork, bitString, exatn::getCurrentProcessGroup());
        const double exp_val_z = calcExpValueZ(m_measureQbIdx, waveFuncSlice);
        assert(vectorIdx < partialExpectationValues.size());
        partialExpectationValues[vectorIdx] =
            evenParity ? exp_val_z : -exp_val_z;
        ++vectorIdx;
      }

      // Compute local accumulate:
      const double localAccumulateExpVal =
          std::accumulate(partialExpectationValues.begin(),
                          partialExpectationValues.end(), 0.0);

      // MPI Reduce: We don't want to explicitly use MPI API here,
      // hence using exatn::allreduceTensor API.
      // Each process will just construct an one-element tensor which
      // contains the local accumulated exp-val.
      const bool created = exatn::createTensor(accumulatedTensorName,
                                               exatn::TensorElementType::REAL64,
                                               exatn::TensorShape{1});
      assert(created);
      std::stringstream ssLog;
      ssLog << "Process [" << processRank
                << "]: Local accumulated exp-val = "
                << localAccumulateExpVal << "\n";
      xacc::info(ssLog.str());
      // Init tensor body data
      exatn::initTensorData(accumulatedTensorName,
                            std::vector<double>{localAccumulateExpVal});
    } else {
      // These processes are excess (in case there are too many processes)
      const bool created = exatn::createTensor(accumulatedTensorName,
                                               exatn::TensorElementType::REAL64,
                                               exatn::TensorShape{1});
      assert(created);
      // Init zero tensor body data
      exatn::initTensorData(accumulatedTensorName, std::vector<double>{0.0});
    }

    // All-reduce the accumulated tensor across all processes in the group.
    const bool allReduced = exatn::allreduceTensorSync(
        exatn::getDefaultProcessGroup(), accumulatedTensorName);
    assert(allReduced);

    // Done:
    auto talsh_tensor = exatn::getLocalTensor(accumulatedTensorName);
    assert(talsh_tensor->getVolume() == 1);
    const double *body_ptr;
    // Invalid value to detect any problems.
    double finalExpVal = -9999.99;
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
      finalExpVal = *body_ptr;
    }
    const bool destroyed = exatn::destroyTensorSync(accumulatedTensorName);
    assert(destroyed);
    return finalExpVal;
  }
}

template <typename TNQVM_COMPLEX_TYPE>
double
ExatnVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZByAppendingConjugate(
    std::shared_ptr<CompositeInstruction> in_function) {
  // Cache the current tensor network:
  exatn::TensorNetwork cacheTensorNet = m_tensorNetwork;
  const auto cachedIdCounter = m_tensorIdCounter;
  auto cachedTensorList = m_appendedGateTensors;
  InstructionIterator it(in_function);
  // Add remaining instructions:
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled() && !nextInst->isComposite()) {
      nextInst->accept(this);
    }
  }
  assert(!m_measureQbIdx.empty());
  const double result = getExpectationValueZByAppendingConjugate();
  m_measureQbIdx.clear();
  // Restore the base tensor network
  m_tensorNetwork = cacheTensorNet;
  m_tensorIdCounter = cachedIdCounter;
  m_appendedGateTensors = cachedTensorList;
  return result;
}

template <typename TNQVM_COMPLEX_TYPE>
double
ExatnVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZByAppendingConjugate() {
  // Inject Z tensors to measure location.
  //////////////////////////////////////////////////////
  // o------|---------|-------------|---------|------o
  // o------| Quantum |-------------| Inverse |------o
  // o------| Circuit |----o Z o----| Quantum |------o
  // o------|         |----o Z o----| Circuit |------o
  // o------|---------|-------------|---------|------o
  /////////////////////////////////////////////////////

  static auto gateRegistry = xacc::getIRProvider("quantum");
  std::vector<InstPtr> measureOps;
  for (const auto &bit : m_measureQbIdx) {
    measureOps.emplace_back(gateRegistry->createInstruction("Z", bit));
  }

  const auto result = evaluateTerm(measureOps);

  return static_cast<double>(result.real());
}

template<typename TNQVM_COMPLEX_TYPE>
std::vector<uint8_t> ExatnVisitor<TNQVM_COMPLEX_TYPE>::generateMeasureSample(const TensorNetwork& in_tensorNetwork, const std::vector<int>& in_qubitIdx)
{
    TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
    std::vector<uint8_t> resultBitString;
    std::vector<ExatnVisitor<TNQVM_COMPLEX_TYPE>::TNQVM_FLOAT_TYPE> resultProbs;
    for (const auto& qubitIdx : in_qubitIdx)
    {
        std::vector<std::string> tensorsToDestroy;
        std::vector<TNQVM_COMPLEX_TYPE> resultRDM;
        exatn::TensorNetwork ket(in_tensorNetwork);
        ket.rename("MPSket");

        exatn::TensorNetwork bra(ket);
        bra.conjugate();
        bra.rename("MPSbra");
        auto tensorIdCounter = ket.getMaxTensorId();
        // Adding collapse tensors based on previous measurement results.
        // i.e. condition/renormalize the tensor network to be consistent with
        // previous result.
        for (size_t measIdx = 0; measIdx < resultBitString.size(); ++measIdx)
        {
            const unsigned int qId = in_qubitIdx[measIdx];
            // If it was a "0":
            if (resultBitString[measIdx] == 0)
            {
                const std::vector<TNQVM_COMPLEX_TYPE> COLLAPSE_0{
                    // Renormalize based on the probability of this outcome
                    {1.0f / resultProbs[measIdx], 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0}};

                const std::string tensorName = "COLLAPSE_0_" + std::to_string(measIdx);
                const bool created = exatn::createTensor(tensorName, getExatnElementType(), exatn::TensorShape{2, 2});
                assert(created);
                tensorsToDestroy.emplace_back(tensorName);
                const bool registered = exatn::registerTensorIsometry(tensorName, {0}, {1});
                assert(registered);
                const bool initialized = exatn::initTensorData(tensorName, COLLAPSE_0);
                assert(initialized);
                tensorIdCounter++;
                const bool appended = ket.appendTensorGate(tensorIdCounter, exatn::getTensor(tensorName), {qId});
                assert(appended);
            }
            else
            {
                assert(resultBitString[measIdx] == 1);
                // Renormalize based on the probability of this outcome
                const std::vector<TNQVM_COMPLEX_TYPE> COLLAPSE_1{
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {1.0f / resultProbs[measIdx], 0.0}};

                const std::string tensorName = "COLLAPSE_1_" + std::to_string(measIdx);
                const bool created = exatn::createTensor(tensorName, getExatnElementType(), exatn::TensorShape{2, 2});
                assert(created);
                tensorsToDestroy.emplace_back(tensorName);
                const bool registered = exatn::registerTensorIsometry(tensorName, {0}, {1});
                assert(registered);
                const bool initialized = exatn::initTensorData(tensorName, COLLAPSE_1);
                assert(initialized);
                tensorIdCounter++;
                const bool appended = ket.appendTensorGate(tensorIdCounter, exatn::getTensor(tensorName), {qId});
                assert(appended);
            }
        }

        auto combinedNetwork = ket;
        combinedNetwork.rename("Combined Tensor Network");
        {
            // Append the conjugate network to calculate the RDM of the measure
            // qubit
            std::vector<std::pair<unsigned int, unsigned int>> pairings;
            for (size_t i = 0; i < m_buffer->size(); ++i)
            {
                // Connect the original tensor network with its inverse
                // but leave the measure qubit line open.
                if (i != qubitIdx)
                {
                    pairings.emplace_back(std::make_pair(i, i));
                }
            }

            combinedNetwork.appendTensorNetwork(std::move(bra), pairings);
        }

        const bool isoCollapsed = combinedNetwork.collapseIsometries();
        {
          // DEBUG:
          {
            const std::string optimizerName = options.stringExists("exatn-contract-seq-optimizer") ? options.getString("exatn-contract-seq-optimizer") : "metis";
            const auto startOpt = std::chrono::system_clock::now();
            combinedNetwork.getOperationList(optimizerName);
            const auto endOpt = std::chrono::system_clock::now();
              std::cout << "getOperationList() took: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endOpt - startOpt).count()
              << " ms\n";
          }

          const double flops = combinedNetwork.getFMAFlops();
          const double intermediatesVolume = combinedNetwork.getMaxIntermediatePresenceVolume();
          assert(intermediatesVolume >= 0.0);
          const int64_t sizeInBytes = static_cast<int64_t>(intermediatesVolume * sizeof(TNQVM_COMPLEX_TYPE));
          std::cout << "Combined circuit requires " << flops << " FMA flops and " << sizeInBytes << " bytes\n";

          if (sizeInBytes > MAX_TALSH_MEMORY_BUFFER_SIZE_BYTES)
          {
            xacc::error("ExaTN intermediate tensors require more memory than max allowed of " +
              std::to_string(MAX_TALSH_MEMORY_BUFFER_SIZE_BYTES) + " bytes.");
          }
        }

        // Evaluate
        {
          TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
          if (exatn::evaluateSync(combinedNetwork))
          {
              exatn::sync();
              auto talsh_tensor = exatn::getLocalTensor(combinedNetwork.getTensor(0)->getName());
              const auto tensorVolume = talsh_tensor->getVolume();
              // Single qubit density matrix
              assert(tensorVolume == 4);
              const TNQVM_COMPLEX_TYPE* body_ptr;
              if (talsh_tensor->getDataAccessHostConst(&body_ptr))
              {
                  resultRDM.assign(body_ptr, body_ptr + tensorVolume);
              }
              // Debug: print out RDM data
              {
                  std::cout << "RDM @q" << qubitIdx << " = [";
                  for (int i = 0; i < talsh_tensor->getVolume(); ++i)
                  {
                      const TNQVM_COMPLEX_TYPE element = body_ptr[i];
                      std::cout << element;
                  }
                  std::cout << "]\n";
              }
          }
        }
        {
            // Perform the measurement
            assert(resultRDM.size() == 4);
            const double prob_0 = resultRDM.front().real();
            const double prob_1 = resultRDM.back().real();
            assert(prob_0 >= 0.0 && prob_1 >= 0.0);
            assert(std::fabs(1.0 - prob_0 - prob_1) < 1e-12);

            // Generate a random number
            const double randProbPick = generateRandomProbability();
            // If radom number < probability of 0 state -> pick zero, and vice versa.
            resultBitString.emplace_back(randProbPick <= prob_0 ? 0 : 1);
            resultProbs.emplace_back(randProbPick <= prob_0 ? prob_0 : prob_1);

            std::cout << ">> Measure @q" << qubitIdx << " prob(0) = " << prob_0 << "\n";
            std::cout << ">> Measure @q" << qubitIdx << " prob(1) = " << prob_1 << "\n";
            std::cout << ">> Measure @q" << qubitIdx << " random number = " << randProbPick << "\n";
            std::cout << ">> Measure @q" << qubitIdx << " pick " << std::to_string(resultBitString.back()) << "\n";
        }

        for (const auto& tensorName : tensorsToDestroy)
        {
            const bool tensorDestroyed = exatn::destroyTensor(tensorName);
            assert(tensorDestroyed);
        }

        exatn::sync();
    }

    return resultBitString;
}

template<typename TNQVM_COMPLEX_TYPE>
std::vector<std::pair<double, double>> ExatnVisitor<TNQVM_COMPLEX_TYPE>::calcFlopsAndMemoryForSample(const TensorNetwork& in_tensorNetwork)
{
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  std::vector<std::pair<double, double>> resultData;
  resultData.reserve(m_buffer->size());
  // Create the collapse tensor:
  const std::vector<TNQVM_COMPLEX_TYPE> COLLAPSE_TEMP { {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0} };
  const std::string tensorName = "COLLAPSE_TENSOR_TEMP";
  const bool created = exatn::createTensor(tensorName, getExatnElementType(), exatn::TensorShape{2, 2});
  assert(created);
  const bool registered = exatn::registerTensorIsometry(tensorName, {0}, {1});
  assert(registered);
  const bool initialized = exatn::initTensorData(tensorName, COLLAPSE_TEMP);
  assert(initialized);
  exatn::sync(tensorName);

  for (int qubitIdx = 0; qubitIdx < m_buffer->size(); ++qubitIdx)
  {
    exatn::TensorNetwork ket(in_tensorNetwork);
    ket.rename("MPSket");
    exatn::TensorNetwork bra(ket);
    bra.conjugate();
    bra.rename("MPSbra");
    auto tensorIdCounter = ket.getMaxTensorId();
    // Adding collapse tensors for previously-measured qubits:
    for (unsigned int measIdx = 0; measIdx < qubitIdx; ++measIdx)
    {
      tensorIdCounter++;
      const bool appended = ket.appendTensorGate(tensorIdCounter, exatn::getTensor(tensorName), { measIdx });
      assert(appended);
    }

    auto combinedNetwork = ket;
    combinedNetwork.rename("Combined Tensor Network");
    // Append the conjugate network to calculate the RDM of the measure
    // qubit
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (size_t i = 0; i < m_buffer->size(); ++i)
    {
      // Connect the original tensor network with its inverse
      // but leave the measure qubit line open.
      if (i != qubitIdx)
      {
        pairings.emplace_back(std::make_pair(i, i));
      }
    }
    combinedNetwork.appendTensorNetwork(std::move(bra), pairings);
    const bool isoCollapsed = combinedNetwork.collapseIsometries();
    const std::string optimizerName = options.stringExists("exatn-contract-seq-optimizer") ? options.getString("exatn-contract-seq-optimizer") : "metis";
    // Get the tensor operation list, i.e. run the tensor optimizer.
    combinedNetwork.getOperationList(optimizerName);
    const double flops = combinedNetwork.getFMAFlops();
    const double intermediatesVolume = combinedNetwork.getMaxIntermediatePresenceVolume();
    const double sizeInBytes = intermediatesVolume * sizeof(TNQVM_COMPLEX_TYPE);
    // Save the data:
    resultData.emplace_back(flops, sizeInBytes);
  }
  // Destroy the temporary projection tensor
  const bool tensorDestroyed = exatn::destroyTensor(tensorName);
  assert(tensorDestroyed);
  exatn::sync();

  return resultData;
}

template<typename TNQVM_COMPLEX_TYPE>
bool ExatnVisitor<TNQVM_COMPLEX_TYPE>::validateTensorNetworkContraction(TensorNetwork in_network) const
{
  TNQVM_TELEMETRY_ZONE(__FUNCTION__, __FILE__, __LINE__);
  auto inverseTensorNetwork = in_network;
  inverseTensorNetwork.rename("Inverse Tensor Network");
  inverseTensorNetwork.conjugate();

  // Connect the original tensor network with its inverse
  {
    TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
    auto combinedNetwork = in_network;
    combinedNetwork.rename("Combined Tensor Network");
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (size_t i = 0; i < m_buffer->size(); ++i)
    {
      pairings.emplace_back(std::make_pair(i, i));
    }

    combinedNetwork.appendTensorNetwork(std::move(inverseTensorNetwork), pairings);
    // combinedNetwork.printIt();

    if (exatn::evaluateSync(combinedNetwork))
    {
      exatn::sync();
      auto talsh_tensor = exatn::getLocalTensor(combinedNetwork.getTensor(0)->getName());
      const auto tensorVolume = talsh_tensor->getVolume();
      // Double check the size of the RDM
      assert(tensorVolume == 1);
      const TNQVM_COMPLEX_TYPE *body_ptr;
      if (talsh_tensor->getDataAccessHostConst(&body_ptr))
      {
        const TNQVM_COMPLEX_TYPE normVal = *body_ptr;
        constexpr double TOLERANCE = 1e-9;
        xacc::info("Contract <Tensor Network + Conjugate> = " + std::to_string(normVal.real()) + " + i " + std::to_string(normVal.imag()));
        return std::abs(normVal.real() - 1) < TOLERANCE && std::abs(normVal.imag()) < TOLERANCE;
      }
    }
  }

  return false;
}

template <typename TNQVM_COMPLEX_TYPE>
std::vector<TNQVM_COMPLEX_TYPE>
ExatnVisitor<TNQVM_COMPLEX_TYPE>::computeWaveFuncSlice(
    const TensorNetwork &in_tensorNetwork, const std::vector<int> &bitString,
    const exatn::ProcessGroup &in_processGroup) const {
  // Closing the tensor network with the bra
  std::vector<std::pair<unsigned int, unsigned int>> pairings;
  int nbOpenLegs = 0;
  const auto constructBraNetwork = [&](const std::vector<int> &in_bitString) {
    int tensorIdCounter = 1;
    TensorNetwork braTensorNet("bra");
    // Create the qubit register tensor
    for (int i = 0; i < in_bitString.size(); ++i) {
      const auto bitVal = in_bitString[i];
      const std::string braQubitName = "QB" + std::to_string(i);
      if (bitVal == 0) {
        const bool created =
            exatn::createTensor(in_processGroup, braQubitName,
                                getExatnElementType(), TensorShape{2});
        assert(created);
        // Bit = 0
        const bool initialized = exatn::initTensorData(
            braQubitName,
            std::vector<TNQVM_COMPLEX_TYPE>{{1.0, 0.0}, {0.0, 0.0}});
        assert(initialized);
        pairings.emplace_back(std::make_pair(i, i + nbOpenLegs));
      } else if (bitVal == 1) {
        const bool created =
            exatn::createTensor(in_processGroup, braQubitName,
                                getExatnElementType(), TensorShape{2});
        assert(created);
        // Bit = 1
        const bool initialized = exatn::initTensorData(
            braQubitName,
            std::vector<TNQVM_COMPLEX_TYPE>{{0.0, 0.0}, {1.0, 0.0}});
        assert(initialized);
        pairings.emplace_back(std::make_pair(i, i + nbOpenLegs));
      } else if (bitVal == -1) {
        // Add an Id tensor
        const bool created =
            exatn::createTensor(in_processGroup, braQubitName,
                                getExatnElementType(), TensorShape{2, 2});
        assert(created);
        const bool initialized = exatn::initTensorData(
            braQubitName, std::vector<TNQVM_COMPLEX_TYPE>{
                              {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
        assert(initialized);
        pairings.emplace_back(std::make_pair(i, i + nbOpenLegs));
        nbOpenLegs++;
      } else {
        xacc::error("Unknown values of '" + std::to_string(bitVal) +
                    "' encountered.");
      }
      braTensorNet.appendTensor(
          tensorIdCounter, exatn::getTensor(braQubitName),
          std::vector<std::pair<unsigned int, unsigned int>>{});
      tensorIdCounter++;
    }

    return braTensorNet;
  };

  auto braTensors = constructBraNetwork(bitString);
  braTensors.conjugate();
  auto combinedTensorNetwork = in_tensorNetwork;
  assert(pairings.size() == m_buffer->size());
  combinedTensorNetwork.appendTensorNetwork(std::move(braTensors), pairings);
  combinedTensorNetwork.collapseIsometries();
  // combinedTensorNetwork.printIt();
  std::vector<TNQVM_COMPLEX_TYPE> waveFnSlice;
  {
    TNQVM_TELEMETRY_ZONE("exatn::evaluateSync", __FILE__, __LINE__);
    // std::cout << "SUBMIT TENSOR NETWORK FOR EVALUATION\n";
    // combinedTensorNetwork.printIt();
    combinedTensorNetwork.rename(m_kernelName);
    if (exatn::evaluateSync(in_processGroup, combinedTensorNetwork)) {
      exatn::sync();
      auto talsh_tensor =
          exatn::getLocalTensor(combinedTensorNetwork.getTensor(0)->getName());
      const TNQVM_COMPLEX_TYPE *body_ptr;
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
        waveFnSlice.assign(body_ptr, body_ptr + talsh_tensor->getVolume());
      }
    }
  }
  // Destroy bra tensors
  for (int i = 0; i < m_buffer->size(); ++i) {
    const std::string braQubitName = "QB" + std::to_string(i);
    const bool destroyed = exatn::destroyTensor(braQubitName);
    assert(destroyed);
  }
  return waveFnSlice;
}

template <typename TNQVM_COMPLEX_TYPE>
size_t ExatnVisitor<TNQVM_COMPLEX_TYPE>::getNumMpiProcs() const {
  auto &process_group = exatn::getDefaultProcessGroup();
  return process_group.getSize();
}
} // end namespace tnqvm

#endif // TNQVM_HAS_EXATN
