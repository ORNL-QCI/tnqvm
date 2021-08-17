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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Implementation - Thien Nguyen;
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
#include "IRUtils.hpp"
#include "base/Gates.hpp"
#include "utils/GateMatrixAlgebra.hpp"

#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

namespace {
// Helper to construct qubit tensor name:
std::string generateQubitTensorName(int qubitIndex) {
  return "Q" + std::to_string(qubitIndex);
};

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
    exatn::resetClientLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
    exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel()
                                                  : 0);

    xacc::subscribeLoggingLevel([](int level) {
      exatn::resetClientLoggingLevel(xacc::verbose ? level : 0);
      exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
    });
  }
  m_buffer = buffer;
  // Create the qubit register tensor
  for (int i = 0; i < m_buffer->size(); ++i) {
    const bool created =
        exatn::createTensor(generateQubitTensorName(i), getExatnElementType(),
                            exatn::TensorShape{2});
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

  // Default number of layers
  m_layersReconstruct = 4;
  m_countByGates = false;
  m_layerTracker.clear();
  if (options.keyExists<int>("reconstruct-gates")) {
    m_layersReconstruct = options.get<int>("reconstruct-gates");
    xacc::info("Reconstruct tensor network every " +
               std::to_string(m_layersReconstruct) + " 2-body gates.");
    m_countByGates = true;
  } else {
    if (options.keyExists<int>("reconstruct-layers")) {
      m_layersReconstruct = options.get<int>("reconstruct-layers");
      xacc::info("Reconstruct tensor network every " +
                 std::to_string(m_layersReconstruct) + " layers.");
    }
  }

  m_reconstructTol = 1e-3;
  m_maxBondDim = 512;
  m_reconstructionFidelity = 1.0;
  m_initReconstructionRandom = false;
  if (options.keyExists<bool>("init-random")) {
    m_initReconstructionRandom = options.get<bool>("init-random");
  }
  m_previousOptExpansion.reset();
  // Default builder: MPS
  m_reconstructBuilder = "MPS";
  if (m_layersReconstruct > 0) {
    if (options.keyExists<double>("reconstruct-tolerance")) {
      m_reconstructTol = options.get<double>("reconstruct-tolerance");
      xacc::info("Reconstruct tolerance = " + std::to_string(m_reconstructTol));
    }
    if (options.keyExists<int>("max-bond-dim")) {
      m_maxBondDim = options.get<int>("max-bond-dim");
      xacc::info("Reconstruct max bond dim = " + std::to_string(m_maxBondDim));
    }
    if (options.stringExists("reconstruct-builder")) {
      m_reconstructBuilder = options.getString("reconstruct-builder");
      xacc::info("Reconstruct with: " + m_reconstructBuilder + " builder.");
    }
  }

  m_qubitNetwork = std::make_shared<exatn::TensorNetwork>("QubitReg");
  // Append the qubit tensors to the tensor network
  for (int i = 0; i < m_buffer->size(); ++i) {
    m_qubitNetwork->appendTensor(
        i + 1, exatn::getTensor(generateQubitTensorName(i)),
        std::vector<std::pair<unsigned int, unsigned int>>{});
  }
  exatn::TensorExpansion tensorEx("QuantumCircuit");
  tensorEx.appendComponent(m_qubitNetwork, TNQVM_COMPLEX_TYPE{1.0});
  m_tensorExpansion = tensorEx;
  m_gateTensorBodies.clear();
  m_measuredBits.clear();
  m_obsTensorOperator.reset();
  m_compositeNameToComponentId.clear();
  m_evaluatedExpansion.reset();
  m_layerCounter = 0;
  {
    const auto tensorName = "MeasX";
    const std::vector<TNQVM_COMPLEX_TYPE> tensorBody{
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    m_gateTensorBodies[tensorName] = tensorBody;
    const bool created = exatn::createTensor(tensorName, getExatnElementType(),
                                             exatn::TensorShape{2, 2});
    assert(created);
    exatn::initTensorData(tensorName, tensorBody);
  }
  {
    const auto tensorName = "MeasY";
    const std::vector<TNQVM_COMPLEX_TYPE> tensorBody{
        {0.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {0.0, 0.0}};
    m_gateTensorBodies[tensorName] = tensorBody;
    const bool created = exatn::createTensor(tensorName, getExatnElementType(),
                                             exatn::TensorShape{2, 2});
    assert(created);
    exatn::initTensorData(tensorName, tensorBody);
  }
  {
    const auto tensorName = "MeasZ";
    const std::vector<TNQVM_COMPLEX_TYPE> tensorBody{
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};
    m_gateTensorBodies[tensorName] = tensorBody;
    const bool created = exatn::createTensor(tensorName, getExatnElementType(),
                                             exatn::TensorShape{2, 2});
    assert(created);
    exatn::initTensorData(tensorName, tensorBody);
  }
  {
    const auto tensorName = "MeasI";
    const std::vector<TNQVM_COMPLEX_TYPE> tensorBody{
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    m_gateTensorBodies[tensorName] = tensorBody;
    const bool created = exatn::createTensor(tensorName, getExatnElementType(),
                                             exatn::TensorShape{2, 2});
    assert(created);
    exatn::initTensorData(tensorName, tensorBody);
  }

  if (options.keyExists<xacc::quantum::ObservedAnsatz>(
          "vqe-execution-context")) {
    // std::cout << "Has vqe-execution-context \n";
    auto observedAnsatz =
        options.get<xacc::quantum::ObservedAnsatz>("vqe-execution-context");
    m_obsTensorOperator =
        std::make_shared<exatn::TensorOperator>("ObservableSum");
    for (auto &obsSubCirc : observedAnsatz.getObservedSubCircuits()) {
      auto obsOps = analyzeObsSubCircuit(obsSubCirc);
      assert(obsOps.size() == m_buffer->size());
      for (const auto &op : obsOps) {
        if (op == ObsOpType::NA) {
          xacc::error("Not a valid kernel observe sub-circuit.");
        }
      }
      auto obsTensorOp = constructObsTensorOperator(obsOps);
      // std::cout << "Observable tensor operator for: " << obsSubCirc->name()
      //           << "\n";
      // obsTensorOp.printIt();
      assert(obsTensorOp.getNumComponents() == 1);
      auto component = obsTensorOp.getComponent(0);
      m_obsTensorOperator->appendComponent(
          component.network, component.ket_legs, component.bra_legs,
          component.coefficient);
      m_compositeNameToComponentId.emplace(
          obsSubCirc->name(), m_obsTensorOperator->getNumComponents() - 1);
    }
    assert(m_obsTensorOperator->getNumComponents() ==
           observedAnsatz.getObservedSubCircuits().size());
    // std::cout << "Total observable operator: \n";
    // m_obsTensorOperator->printIt();
  }
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::finalize() {
  m_buffer->addExtraInfo("reconstruction-fidelity", m_reconstructionFidelity);
  // This is a single-circuit execution.
  // Do the evaluation now.
  if (!m_obsTensorOperator && !m_measuredBits.empty()) {
    std::vector<ObsOpType> obsOps(m_buffer->size(), ObsOpType::I);
    for (size_t i = 0; i < m_buffer->size(); ++i) {
      if (xacc::container::contains(m_measuredBits, i)) {
        obsOps[i] = ObsOpType::Z;
      }
    }
    auto zHamOp = constructObsTensorOperator(obsOps);
    exatn::TensorExpansion ketvector(m_tensorExpansion);
    const bool success =
        exatn::balanceNormalizeNorm2Sync(ketvector, 1.0, 1.0, false);
    assert(success);
    exatn::TensorExpansion ketWithObs(ketvector, zHamOp);
    // Bra network:
    exatn::TensorExpansion bravector(ketvector);
    bravector.conjugate();
    // bravector.printIt();
    exatn::TensorExpansion bratimesopertimesket(bravector, ketWithObs);
    // bratimesopertimesket.printIt();
    const bool accumCreated = exatn::createTensorSync(
        "ExpVal", getExatnElementType(), exatn::TensorShape{});
    assert(accumCreated);
    const bool accumInitialized = exatn::initTensorSync("ExpVal", 0.0);
    assert(accumInitialized);
    auto accumulator = exatn::getTensor("ExpVal");
    if (exatn::evaluateSync(bratimesopertimesket, accumulator)) {
      auto talsh_tensor = exatn::getLocalTensor("ExpVal");
      assert(talsh_tensor->getVolume() == 1);
      const TNQVM_COMPLEX_TYPE *body_ptr;
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
        std::cout << "Exp-val = " << *body_ptr << "\n";
        m_buffer->addExtraInfo("exp-val-z", (double)(body_ptr->real()));
      }
    }
    const bool destroyed = exatn::destroyTensorSync("ExpVal");
    assert(destroyed);
  }

  if (options.keyExists<std::vector<int>>("bitstring")) {
    std::vector<int> bitString = options.get<std::vector<int>>("bitstring");
    if (bitString.size() != m_buffer->size()) {
      xacc::error("Bitstring size must match the number of qubits.");
      return;
    }

    auto tensorNetwork = m_tensorExpansion.getComponent(0).network;
    // m_tensorExpansion.printIt();
    const bool success =
        exatn::balanceNormalizeNorm2Sync(m_tensorExpansion, 1.0, 1.0, false);
    assert(success);
    // m_tensorExpansion.printIt();
    std::vector<TNQVM_COMPLEX_TYPE> waveFuncSlice = computeWaveFuncSlice(
        *tensorNetwork, bitString, exatn::getDefaultProcessGroup());
    assert(!waveFuncSlice.empty());
    if (waveFuncSlice.size() == 1) {
      const std::complex<double> renormalized_val =
          static_cast<std::complex<double>>(waveFuncSlice[0]) *
          m_tensorExpansion.getComponent(0).coefficient;
      m_buffer->addExtraInfo("amplitude-real", renormalized_val.real());
      m_buffer->addExtraInfo("amplitude-imag", renormalized_val.imag());
    } else {
      const auto normalizeWaveFnSlice =
          [](std::vector<TNQVM_COMPLEX_TYPE> &io_waveFn) {
            const double normVal = std::accumulate(
                io_waveFn.begin(), io_waveFn.end(), 0.0,
                [](double sumVal, const TNQVM_COMPLEX_TYPE &val) {
                  return sumVal + std::norm(val);
                });
            // The slice may have zero norm:
            if (normVal > 1e-12) {
              const TNQVM_COMPLEX_TYPE sqrtNorm = sqrt(normVal);
              for (auto &val : io_waveFn) {
                val = val / sqrtNorm;
              }
            }
          };

      normalizeWaveFnSlice(waveFuncSlice);
      std::vector<double> amplReal;
      std::vector<double> amplImag;
      amplReal.reserve(waveFuncSlice.size());
      amplImag.reserve(waveFuncSlice.size());
      for (const auto &val : waveFuncSlice) {
        amplReal.emplace_back(val.real());
        amplImag.emplace_back(val.imag());
      }
      m_buffer->addExtraInfo("amplitude-real-vec", amplReal);
      m_buffer->addExtraInfo("amplitude-imag-vec", amplImag);
    }
  }

  // Clean-up tensors
  for (size_t i = 0; i < m_buffer->size(); ++i) {
    const bool destroyed = exatn::destroyTensorSync(generateQubitTensorName(i));
    assert(destroyed);
  }
  for (const auto &[tensorName, tensorBody] : m_gateTensorBodies) {
    const bool destroyed = exatn::destroyTensorSync(tensorName);
    assert(destroyed);
  }
  m_gateTensorBodies.clear();
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
  // Count gate layer if this is a multi-qubit gate.
  if (in_gateInstruction.nRequiredBits() > 1) {
    updateLayerCounter(in_gateInstruction);
  }
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
    const auto gateTensorShape = (in_gateInstruction.nRequiredBits() == 1
                                      ? exatn::TensorShape{2, 2}
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

  reconstructCircuitTensor();
}

template <typename TNQVM_COMPLEX_TYPE>
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::reconstructCircuitTensor() {
  if (m_layersReconstruct <= 0) {
    return;
  }
  if (m_layerCounter >= m_layersReconstruct) {
    xacc::info("Reconstruct Tensor Expansion");
    auto target = std::make_shared<exatn::TensorExpansion>(m_tensorExpansion);
    // List of Approximate tensors to delete:
    static std::vector<std::string> TENSORS_TO_DESTROY;
    std::vector approxTensorsToDelete = TENSORS_TO_DESTROY;
    TENSORS_TO_DESTROY.clear();

    const std::vector<int> qubitTensorDim(m_buffer->size(), 2);
    auto rootTensor = std::make_shared<exatn::Tensor>("ROOT", qubitTensorDim);
    auto &networkBuildFactory = *(exatn::numerics::NetworkBuildFactory::get());
    auto builder =
        networkBuildFactory.createNetworkBuilderShared(m_reconstructBuilder);
    builder->setParameter("max_bond_dim", m_maxBondDim);
    auto approximant = [&]() {
      if (m_initReconstructionRandom || !m_previousOptExpansion) {
        auto approximantTensorNetwork =
            exatn::makeSharedTensorNetwork("Approx", rootTensor, *builder);
        for (auto iter = approximantTensorNetwork->cbegin();
             iter != approximantTensorNetwork->cend(); ++iter) {
          const auto &tensorName = iter->second.getTensor()->getName();
          if (tensorName != "ROOT") {
            auto tensor = iter->second.getTensor();
            const bool created =
                exatn::createTensorSync(tensor, getExatnElementType());
            assert(created);
            const bool initialized = exatn::initTensorRnd(tensor->getName());
            assert(initialized);
            // Keeps track of these approximate tensors to delete the next
            // round.
            TENSORS_TO_DESTROY.emplace_back(tensor->getName());
          }
        }
        approximantTensorNetwork->markOptimizableAllTensors();
        auto approximant_expansion =
            std::make_shared<exatn::TensorExpansion>("Approx");
        approximant_expansion->appendComponent(approximantTensorNetwork,
                                               TNQVM_COMPLEX_TYPE{1.0, 0.0});
        approximant_expansion->conjugate();
        return approximant_expansion;
      } else {
        // Re-init to the previous:
        return m_previousOptExpansion;
      }
    }();

    bool success = exatn::balanceNormalizeNorm2Sync(*target, 1.0, 1.0, false);
    assert(success);
    success = exatn::balanceNormalizeNorm2Sync(*approximant, 1.0, 1.0, true);
    assert(success);
    exatn::TensorNetworkReconstructor reconstructor(target, approximant,
                                                    m_reconstructTol);
    // std::cout << "Target: \n";
    // target->printIt();
    // std::cout << "approximant: \n";
    // approximant->printIt();
    // Run the reconstructor:
    bool reconstructSuccess = exatn::sync();
    assert(reconstructSuccess);
    // exatn::TensorNetworkReconstructor::resetDebugLevel(2); //debug
    reconstructor.resetLearningRate(1.0);
    double residual_norm, fidelity;
    const auto startOpt = std::chrono::system_clock::now();
    bool reconstructed =
        reconstructor.reconstruct(&residual_norm, &fidelity, true);
    reconstructSuccess = exatn::sync();
    assert(reconstructSuccess);
    if (reconstructed) {
      const auto endOpt = std::chrono::system_clock::now();
      const int elapsedMs =
          std::chrono::duration_cast<std::chrono::milliseconds>(endOpt -
                                                                startOpt)
              .count();
      std::stringstream ss;
      ss << "Reconstruction succeeded: Residual norm = " << residual_norm
         << "; Fidelity = " << fidelity << "; elapsed time = " << elapsedMs
         << "[ms]";
      xacc::info(ss.str());
      m_reconstructionFidelity *= fidelity;
      m_previousOptExpansion = exatn::duplicateSync(*approximant);
    } else {
      xacc::error("Reconstruction FAILED!");
    }

    approximant->conjugate();
    // std::cout << "After Reconstruct: \n";
    // approximant->printIt();

    // Assign tensor expansion:
    m_tensorExpansion = *approximant;

    for (const auto &tensorName : approxTensorsToDelete) {
      const bool destroyed = exatn::destroyTensorSync(tensorName);
      assert(destroyed);
    }
    // Reset the counter
    m_layerCounter = 0;
  }
}

template <typename TNQVM_COMPLEX_TYPE>
std::vector<ObsOpType>
ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::analyzeObsSubCircuit(
    std::shared_ptr<CompositeInstruction> in_function) const {
  std::vector<ObsOpType> result(m_buffer->size(), ObsOpType::I);
  InstructionIterator it(in_function);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled()) {
      if (nextInst->name() == "Measure") {
        const auto bitId = nextInst->bits()[0];
        // Measure Z
        if (result[bitId] == ObsOpType::I) {
          result[bitId] = ObsOpType::Z;
        }
      } else if (nextInst->name() == "H") {
        const auto bitId = nextInst->bits()[0];
        // Measure X
        if (result[bitId] == ObsOpType::I) {
          result[bitId] = ObsOpType::X;
        } else {
          // Illegal: this bit has been set to a measurement basis.
          result[bitId] = ObsOpType::NA;
        }
      } else if (nextInst->name() == "Rx") {
        const auto bitId = nextInst->bits()[0];
        const double angle =
            InstructionParameterToDouble(nextInst->getParameter(0));
        if (std::abs(angle - M_PI_2) > 1e-3) {
          result[bitId] = ObsOpType::NA;
        }
        // Measure Y
        if (result[bitId] == ObsOpType::I) {
          result[bitId] = ObsOpType::Y;
        } else {
          // Illegal: this bit has been set to a measurement basis.
          result[bitId] = ObsOpType::NA;
        }
      }
    }
  }
  return result;
}

template <typename TNQVM_COMPLEX_TYPE>
exatn::TensorOperator
ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::constructObsTensorOperator(
    const std::vector<ObsOpType> &in_obsOps) const {
  auto obsTensorNetwork = std::make_shared<exatn::TensorNetwork>("Obs");
  std::vector<std::pair<unsigned int, unsigned int>> ketPairings, braPairings;
  std::string opName;
  unsigned int counter = 0;
  for (const auto &opType : in_obsOps) {
    ketPairings.emplace_back(std::make_pair(counter, 2 * counter));
    braPairings.emplace_back(std::make_pair(counter, 2 * counter + 1));
    ++counter;
    switch (opType) {
    case ObsOpType::X:
      obsTensorNetwork->appendTensor(
          counter, exatn::getTensor("MeasX"),
          std::vector<std::pair<unsigned int, unsigned int>>{});
      opName += "X";
      break;
    case ObsOpType::Y:
      obsTensorNetwork->appendTensor(
          counter, exatn::getTensor("MeasY"),
          std::vector<std::pair<unsigned int, unsigned int>>{});
      opName += "Y";
      break;
    case ObsOpType::Z:
      obsTensorNetwork->appendTensor(
          counter, exatn::getTensor("MeasZ"),
          std::vector<std::pair<unsigned int, unsigned int>>{});
      opName += "Z";
      break;
    case ObsOpType::I:
      obsTensorNetwork->appendTensor(
          counter, exatn::getTensor("MeasI"),
          std::vector<std::pair<unsigned int, unsigned int>>{});
      opName += "I";
      break;
    case ObsOpType::NA:
      // fall-through
    default:
      xacc::error("Invalid Operator.");
    }
  }

  obsTensorNetwork->rename(opName);
  exatn::TensorOperator hamOp(opName);
  const bool appended = hamOp.appendComponent(
      obsTensorNetwork, ketPairings, braPairings, TNQVM_COMPLEX_TYPE{1.0});
  assert(appended);
  return hamOp;
}

template <typename TNQVM_COMPLEX_TYPE>
const double ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> in_function) {
  if (!m_evaluatedExpansion) {
    exatn::TensorExpansion ketvector(m_tensorExpansion);
    // std::cout << "Before renormalize:\n";
    // ketvector.printIt();
    const bool success =
        exatn::balanceNormalizeNorm2Sync(ketvector, 1.0, 1.0, false);
    assert(success);
    // std::cout << "After renormalize:\n";
    // ketvector.printIt();

    exatn::TensorExpansion ketWithObs(ketvector, *m_obsTensorOperator);
    // std::cout << "Tensor Expansion:\n";
    // ketWithObs.printIt();
    // Bra network:
    exatn::TensorExpansion bravector(ketvector);
    bravector.conjugate();
    // bravector.printIt();
    exatn::TensorExpansion bratimesopertimesket(bravector, ketWithObs);
    // bratimesopertimesket.printIt();
    const bool accumCreated = exatn::createTensorSync(
        "ExpVal", getExatnElementType(), exatn::TensorShape{});
    assert(accumCreated);
    const bool accumInitialized = exatn::initTensorSync("ExpVal", 0.0);
    assert(accumInitialized);
    m_gateTensorBodies["ExpVal"] = std::vector<TNQVM_COMPLEX_TYPE>{{0.0, 0.0}};
    auto accumulator = exatn::getTensor("ExpVal");
    const bool evaluated =
        exatn::evaluateSync(bratimesopertimesket, accumulator);
    assert(evaluated);
    m_evaluatedExpansion =
        std::make_shared<exatn::TensorExpansion>(bratimesopertimesket);
  }
  // std::cout << "There are: " << m_evaluatedExpansion->getNumComponents() << "
  // components in the expansion.\n";
  auto iter = m_compositeNameToComponentId.find(in_function->name());
  if (iter != m_compositeNameToComponentId.end()) {
    const auto componentId = iter->second;
    // std::cout << "Component Id = " << componentId << "\n";
    auto component = m_evaluatedExpansion->getComponent(componentId);
    auto talsh_tensor =
        exatn::getLocalTensor(component.network->getTensor(0)->getName());
    const TNQVM_COMPLEX_TYPE *body_ptr;
    auto access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr);
    assert(access_granted);
    // std::cout << "Component " << component.network->getTensor(0)->getName()
    //           << " expectation value = " << *body_ptr << "\n";
    const std::complex<double> tensor_body_val = *body_ptr;
    // std::cout << "Component coeff: " << component.coefficient << "\n";
    const std::complex<double> renormalizedComponentExpVal =
        tensor_body_val * component.coefficient;
    // std::cout << "renormalizedComponentExpVal: " <<
    // renormalizedComponentExpVal << "\n";
    return renormalizedComponentExpVal.real();
  }
  xacc::error("Unable to map execution data for sub-composite: " +
              in_function->name());
  return 0.0;
}

template <typename TNQVM_COMPLEX_TYPE>
std::vector<TNQVM_COMPLEX_TYPE>
ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::computeWaveFuncSlice(
    const exatn::TensorNetwork &in_tensorNetwork,
    const std::vector<int> &bitString,
    const exatn::ProcessGroup &in_processGroup) const {
  // Closing the tensor network with the bra
  std::vector<std::pair<unsigned int, unsigned int>> pairings;
  int nbOpenLegs = 0;
  const auto constructBraNetwork = [&](const std::vector<int> &in_bitString) {
    int tensorIdCounter = 1;
    exatn::TensorNetwork braTensorNet("bra");
    // Create the qubit register tensor
    for (int i = 0; i < in_bitString.size(); ++i) {
      const auto bitVal = in_bitString[i];
      const std::string braQubitName = "QB" + std::to_string(i);
      if (bitVal == 0) {
        const bool created =
            exatn::createTensor(in_processGroup, braQubitName,
                                getExatnElementType(), exatn::TensorShape{2});
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
                                getExatnElementType(), exatn::TensorShape{2});
        assert(created);
        // Bit = 1
        const bool initialized = exatn::initTensorData(
            braQubitName,
            std::vector<TNQVM_COMPLEX_TYPE>{{0.0, 0.0}, {1.0, 0.0}});
        assert(initialized);
        pairings.emplace_back(std::make_pair(i, i + nbOpenLegs));
      } else if (bitVal == -1) {
        // Add an Id tensor
        const bool created = exatn::createTensor(in_processGroup, braQubitName,
                                                 getExatnElementType(),
                                                 exatn::TensorShape{2, 2});
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
  // combinedTensorNetwork.printIt();
  std::vector<TNQVM_COMPLEX_TYPE> waveFnSlice;
  {
    // std::cout << "SUBMIT TENSOR NETWORK FOR EVALUATION\n";
    // combinedTensorNetwork.printIt();
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
void ExatnGenVisitor<TNQVM_COMPLEX_TYPE>::updateLayerCounter(
    const xacc::Instruction &in_gateInstruction) {
  auto &gate = const_cast<xacc::Instruction &>(in_gateInstruction);
  assert(gate.bits().size() == 2);
  if (m_countByGates) {
    ++m_layerCounter;
  } else {
    bool canCombine = true;
    const auto q1 = gate.bits()[0];
    const auto q2 = gate.bits()[1];

    for (const auto& [bit1, bit2]: m_layerTracker) {
      if ((q1 == bit1 || q1 == bit2) || (q2 == bit1 || q2 == bit2)) {
        canCombine = false;
        break;
      } 
    }
    if (canCombine) {
      m_layerTracker.emplace(std::make_pair(q1, q2));
    } else {
      ++m_layerCounter;
      m_layerTracker.clear();
    }
  }
}
} // end namespace tnqvm
#endif // TNQVM_HAS_EXATN
