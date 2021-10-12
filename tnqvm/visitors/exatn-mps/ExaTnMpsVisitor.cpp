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
 *   Implementation - Thien Nguyen
 *
*/

#include "ExaTnMpsVisitor.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "ExatnUtils.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include <map>
#include <unistd.h>

#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

namespace {
const std::vector<std::complex<double>> Q_ZERO_TENSOR_BODY{{1.0, 0.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> Q_ONE_TENSOR_BODY{{0.0, 0.0}, {1.0, 0.0}};
const std::string ROOT_TENSOR_NAME = "Root";
// The max number of qubits that we allow full state vector contraction.
// Above this limit, only tensor-based calculation is allowed.
// e.g. simulating bit-string measurement by tensor contraction.
// Note: the reason we don't rely solely on tensor contraction because
// for small circuits, where the full state-vector can be stored in the memory,
// it's faster to just run bit-string simulation on the state vector.
const int MAX_NUMBER_QUBITS_FOR_STATE_VEC = 20;

void printTensorData(const std::string& in_tensorName)
{
    auto talsh_tensor = exatn::getLocalTensor(in_tensorName);
    if (talsh_tensor)
    {
        const std::complex<double>* body_ptr;
        const bool access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr);
        if (!access_granted)
        {
            std::cout << "Failed to retrieve tensor data!!!\n";
        }
        else
        {
            for (int i = 0; i < talsh_tensor->getVolume(); ++i)
            {
                const auto& elem = body_ptr[i];
                std::cout << elem << "\n";
            }
        }

    }
    else
    {
        std::cout << "Failed to retrieve tensor data!!!\n";
    }
}

std::vector<std::complex<double>> getTensorData(const std::string& in_tensorName)
{
    std::vector<std::complex<double>> result;
    auto talsh_tensor = exatn::getLocalTensor(in_tensorName);

    if (talsh_tensor)
    {
        std::complex<double>* body_ptr;
        const bool access_granted = talsh_tensor->getDataAccessHost(&body_ptr);

        if (access_granted)
        {
            result.assign(body_ptr, body_ptr + talsh_tensor->getVolume());
        }
    }
    return result;
}

std::unordered_map<std::string, tnqvm::Stat::FunctionCallStat>& getStatRegistry()
{
    static std::unordered_map<std::string, tnqvm::Stat::FunctionCallStat> statMap;
    return statMap;
}

tnqvm::Stat::FunctionCallStat& getStatInstance(const std::string& in_name)
{
    auto& statMap = getStatRegistry();
    auto iter = statMap.find(in_name);
    if (iter != statMap.end())
    {
        return iter->second;
    }
    // Create new stat
    tnqvm::Stat::FunctionCallStat newStat(in_name);
    auto result = statMap.emplace(in_name, newStat);
    return result.first->second;
}

void printAllStats()
{
    for (auto& stat: getStatRegistry())
    {
        std::cout << stat.second.toString(true) << "\n";
    }
}

size_t getNumberOfThreads()
{
    static const size_t NB_THREADS = std::thread::hardware_concurrency();
    return NB_THREADS;
}

inline bool indexInRange(size_t in_idx, const std::pair<size_t, size_t>& in_range)
{
    return (in_idx >= in_range.first) && (in_idx <= in_range.second);
}

void stabilizeTensorBody(const std::string &in_tensorName) {
  // Extremely small values that we will remove from the tensor body.
  // It will cause numerical instability during SVD.
  std::function<int(talsh::Tensor & in_tensor)> updateFunc =
      [](talsh::Tensor &in_tensor) {
        constexpr double EPS_TRIM = 1e-100;
        std::complex<double> *elements;
        if (in_tensor.getDataAccessHost(&elements)) {
          for (int i = 0; i < in_tensor.getVolume(); ++i) {
            elements[i] = std::abs(elements[i]) < EPS_TRIM ? 0.0 : elements[i];
          }
        }

        return 0;
      };

  exatn::numericalServer->transformTensorSync(
      in_tensorName,
      std::make_shared<tnqvm::ExatnMpsVisitor::ExaTnTensorFunctor>(updateFunc));
  exatn::sync();
}
} // namespace
namespace tnqvm {
ExatnMpsVisitor::ExatnMpsVisitor():
    m_aggregator(this),
    // By default, don't enable aggregation, i.e. simply running gate-by-gate first.
    // TODO: implement aggreation processing with ExaTN.
    m_aggregateEnabled(false)
{
    // TODO
}

void ExatnMpsVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots)
{
    const auto initializeStart = std::chrono::system_clock::now();

    // Check if we have any specific config for the gate aggregator
    if (m_aggregateEnabled && options.keyExists<int>("agg-width"))
    {
        const int aggregatorWidth = options.get<int>("agg-width");
        AggregatorConfigs configs(aggregatorWidth);
        TensorAggregator newAggr(configs, this);
        m_aggregator = newAggr;
    }

    // Initialize ExaTN
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

// This is a flag from ExaTN indicating that ExaTN was compiled
// w/ MPI enabled.
    #ifdef MPI_ENABLED
    {
      exatn::ParamConf exatnParams;
      if (options.keyExists<int>("exatn-buffer-size-gb")) {
        int bufferSizeGb = options.get<int>("exatn-buffer-size-gb");
        if (bufferSizeGb < 1) {
          std::cout << "Minimum buffer size is 1 GB.\n";
          bufferSizeGb = 1;
        }
        // Set the memory buffer size:
        const int64_t memorySizeBytes = bufferSizeGb * (1ULL << 30);
        const bool success = exatnParams.setParameter("host_memory_buffer_size",
                                                      memorySizeBytes);
      } else {
        // Use default buffer size
        const int64_t memorySizeBytes = 8 * (1ULL << 30);

        const bool success = exatnParams.setParameter("host_memory_buffer_size",
                                                      memorySizeBytes);
        assert(success);
      }
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
    }
#else
    {
        exatn::initialize();
        exatn::activateContrSeqCaching();
    }
    #endif
        // ExaTN and XACC logging levels are always in-synced.
        exatn::resetClientLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
        exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);

        xacc::subscribeLoggingLevel([](int level) {
            exatn::resetClientLoggingLevel(xacc::verbose ? level : 0);
            exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
        });
    }

    // Default SVD cut-off is the numerical limit, i.e. technically, no cut-off.
    m_svdCutoff = std::numeric_limits<double>::min();
    if (options.keyExists<double>("svd-cutoff"))
    {
        m_svdCutoff = options.get<double>("svd-cutoff");
        std::cout << "[DEBUG] SVD Cut-off = " << m_svdCutoff << "\n";
    }

    // Max bond dimension: take precedent over svd-cutoff
    m_maxBondDim = std::numeric_limits<int>::max() - 1;
    if (options.keyExists<int>("max-bond-dim"))
    {
        m_maxBondDim = options.get<int>("max-bond-dim");
        std::cout << "[DEBUG] Max bond dimension = " << m_maxBondDim << "\n";
    }

    m_buffer = std::move(buffer);
    m_qubitTensorNames.clear();
    m_tensorIdCounter = 0;
    m_aggregatedGroupCounter = 0;
    m_registeredGateTensors.clear();
    m_measureQubits.clear();
    m_shotCount = nbShots;
#ifndef TNQVM_MPI_ENABLED
    const std::vector<int> qubitTensorDim(m_buffer->size(), 2);
    m_rootTensor = std::make_shared<exatn::Tensor>(ROOT_TENSOR_NAME, qubitTensorDim);
    // Build MPS tensor network
    if (m_buffer->size() > 2)
    {
        auto& networkBuildFactory = *(exatn::numerics::NetworkBuildFactory::get());
        auto builder = networkBuildFactory.createNetworkBuilderShared("MPS");
        // Initially, all bond dimensions are 1
        const bool success = builder->setParameter("max_bond_dim", 1);
        assert(success);

        m_tensorNetwork = exatn::makeSharedTensorNetwork("Qubit Register", m_rootTensor, *builder);
    }
    else if (m_buffer->size() == 2)
    {
        //Declare MPS tensors:
        auto t1 = std::make_shared<exatn::Tensor>("T1", exatn::TensorShape{2,1});
        auto t2 = std::make_shared<exatn::Tensor>("T2", exatn::TensorShape{1,2});
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>("Qubit Register",
                 "Root(i0,i1)+=T1(i0,j0)*T2(j0,i1)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"Root", m_rootTensor}, {"T1",t1}, {"T2",t2}});
    }
    else if (m_buffer->size() == 1)
    {
        auto t1 = std::make_shared<exatn::Tensor>("T1", exatn::TensorShape{2});
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>("Qubit Register",
                 "Root(i0)+=T1(i0)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"Root", m_rootTensor}, {"T1",t1}});
    }

    for (auto iter = m_tensorNetwork->cbegin(); iter != m_tensorNetwork->cend(); ++iter)
    {
        const auto& tensorName = iter->second.getTensor()->getName();
        if (tensorName != ROOT_TENSOR_NAME)
        {
            auto tensor = iter->second.getTensor();
            const auto newTensorName = "Q" + std::to_string(iter->first - 1);
            iter->second.getTensor()->rename(newTensorName);
            const bool created = exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
            assert(created);
            const bool initialized = exatn::initTensorDataSync(newTensorName, Q_ZERO_TENSOR_BODY);
            assert(initialized);
        }
    }
    // DEBUG:
    // printStateVec();

    // ExaTN Logging:
    if (options.keyExists<int>("exatn-logging-level"))
    {
        const int loggingLevel = options.get<int>("exatn-logging-level");
        // Valid level: 0, 1(short), 2 (long)
        // Default is 0, hence just update if the requested level is either 1 or 2
        if (loggingLevel > 0 && loggingLevel < 3)
        {
            std::cout << "[DEBUG]: Set ExaTN runtime logging level to " << loggingLevel << "\n";
            exatn::resetClientLoggingLevel(loggingLevel);
            exatn::resetRuntimeLoggingLevel(loggingLevel);
        }
    }

    const auto initializeEnd = std::chrono::system_clock::now();
    getStatInstance("Initialize").addSample(initializeStart, initializeEnd);
#else
    // MPI
    auto& process_group = exatn::getDefaultProcessGroup();
    m_qubitIdxToRank.clear();
    // Get the rank of the process
    int process_rank = exatn::getProcessRank();
    static bool printOnce = false;
    if (process_rank == 0 && !printOnce)
    {
        std::stringstream ss;
        ss << "============== MPI Info ============================\n";
        ss << "Process " << ::getpid() << ": rank = " << process_rank << "\n";
        ss << "Number of MPI processes = " << process_group.getSize() << "\n";
        ss << "Memory limit per process = " << process_group.getMemoryLimitPerProcess() << "\n";
        ss << "====================================================\n";
        xacc::info(ss.str());
        printOnce = true;
    }
    m_rank = process_rank;
    // Split processes into groups:
    if (process_group.getSize() > 1)
    {
        m_selfProcessGroup = process_group.split(process_rank);
    }
    else
    {
        // IMPORTANT:
        // Handled an edge case where the sub-communicator (e.g. from XACC HPC Virt Decorator)
        // is rank 1 (i.e. 1 process).
        // In that case, we must not split the process group since ExaTN will create destroy
        // the original COMMUNICATOR if we split the group w/ itself.
        m_selfProcessGroup = xacc::as_shared_ptr(&(const_cast<exatn::ProcessGroup&>(exatn::getDefaultProcessGroup())));
    }
    // std::cout << "Process [" << process_rank << "]: Number of MPI processes in sub-group = " << m_selfProcessGroup->getSize() << "\n";
    // std::cout << "Process [" << process_rank << "]: Memory limit per process in sub-group = " << m_selfProcessGroup->getMemoryLimitPerProcess() << "\n";

    // Establish shared process groups:
    // Pairwise split
    m_rightSharedProcessGroup.reset();
    m_leftSharedProcessGroup.reset();
    if(process_group.getSize() > 1)
    {
        const int color = process_rank / 2;
        auto splitProcessGroup = process_group.split(color);
        const int splitDir = process_rank % 2;
        if (splitDir == 0 && (process_rank != (process_group.getSize() - 1)))
        {
            m_rightSharedProcessGroup = splitProcessGroup;
        }
        else
        {
            m_leftSharedProcessGroup = splitProcessGroup;
        }
    }

    if(process_group.getSize() > 1)
    {
        if (process_rank != 0)
        {
            const int rankShift = process_rank - 1;
            const int color = rankShift / 2;
            auto splitProcessGroup = process_group.split(color);
            const int splitDir = rankShift % 2;
            if (splitDir == 0)
            {
                m_rightSharedProcessGroup = splitProcessGroup;
            }
            else
            {
                m_leftSharedProcessGroup = splitProcessGroup;
            }
        }
        else
        {
            auto splitProcessGroup = process_group.split(-1);
        }
    }

    const auto processGroupToString = [](const std::shared_ptr<const exatn::ProcessGroup>& in_processGroup) -> std::string {
        if (in_processGroup) {
            std::string result = "[ ";
            for (const auto& rank : in_processGroup->getProcessRanks())
            {
                result.append(std::to_string(rank) + " ");
            }
            result.append("]");
            return result;
        }
        else
        {
            return "NULL";
        }
    };

    // std::cout << "Process [" << process_rank << "]: Left group = "
    //     << processGroupToString(m_leftSharedProcessGroup)
    //     << "; Right group = " << processGroupToString(m_rightSharedProcessGroup) << "\n";

    if (process_group.getSize() < m_buffer->size())
    {
        const size_t lRange = (process_rank * m_buffer->size()) / process_group.getSize();
        const size_t hRange = (process_rank != (process_group.getSize() - 1)) ?
            ((process_rank + 1) * m_buffer->size()) / process_group.getSize() - 1 :
            m_buffer->size() - 1;

        m_qubitRange = std::make_pair(lRange, hRange);
    }
    else
    {
        // Each qubit to one process
        m_qubitRange = std::make_pair(process_rank, process_rank);
    }

    // std::cout << "Process [" << process_rank << "]: handles qubit " << m_qubitRange.first << " to " << m_qubitRange.second << "\n";

    if (m_buffer->size() > process_group.getSize())
    {
        for (int rank = 0; rank < process_group.getSize(); ++rank)
        {
            const size_t lRange = rank * (m_buffer->size() / process_group.getSize());
            const size_t hRange = (rank != (process_group.getSize() - 1)) ?
                (rank + 1) * (m_buffer->size() / process_group.getSize()) - 1 :
                m_buffer->size() - 1;
            for (int i = lRange; i <= hRange; ++i)
            {
                if (i < m_buffer->size())
                {
                    m_qubitIdxToRank.emplace(i, rank);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            m_qubitIdxToRank.emplace(i, i);
        }
    }

    const std::vector<int> qubitTensorDim(m_buffer->size(), 2);
    m_rootTensor = std::make_shared<exatn::Tensor>(ROOT_TENSOR_NAME, qubitTensorDim);
    // Build MPS tensor network
    if (m_buffer->size() > 2)
    {
        auto& networkBuildFactory = *(exatn::numerics::NetworkBuildFactory::get());
        auto builder = networkBuildFactory.createNetworkBuilderShared("MPS");
        // Initially, all bond dimensions are 1
        const bool success = builder->setParameter("max_bond_dim", 1);
        assert(success);

        m_tensorNetwork = exatn::makeSharedTensorNetwork("Qubit Register", m_rootTensor, *builder);
    }
    else if (m_buffer->size() == 2)
    {
        //Declare MPS tensors:
        auto t1 = std::make_shared<exatn::Tensor>("T1", exatn::TensorShape{2,1});
        auto t2 = std::make_shared<exatn::Tensor>("T2", exatn::TensorShape{1,2});
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>("Qubit Register",
                 "Root(i0,i1)+=T1(i0,j0)*T2(j0,i1)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"Root", m_rootTensor}, {"T1",t1}, {"T2",t2}});
    }
    else if (m_buffer->size() == 1)
    {
        auto t1 = std::make_shared<exatn::Tensor>("T1", exatn::TensorShape{2});
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>("Qubit Register",
                 "Root(i0)+=T1(i0)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"Root", m_rootTensor}, {"T1",t1}});
    }

    for (auto iter = m_tensorNetwork->cbegin(); iter != m_tensorNetwork->cend(); ++iter)
    {
        const auto& tensorName = iter->second.getTensor()->getName();
        if (tensorName != ROOT_TENSOR_NAME)
        {
            auto tensor = iter->second.getTensor();
            const auto newTensorName = "Q" + std::to_string(iter->first - 1);
            iter->second.getTensor()->rename(newTensorName);
            const bool created = exatn::createTensorSync(*m_selfProcessGroup, tensor, exatn::TensorElementType::COMPLEX64);
            assert(created);
            const bool initialized = exatn::initTensorDataSync(newTensorName, Q_ZERO_TENSOR_BODY);
            assert(initialized);
        }
    }
#endif
}

void ExatnMpsVisitor::printStateVec()
{
    assert(m_buffer->size() < MAX_NUMBER_QUBITS_FOR_STATE_VEC);
    std::cout << "MPS Tensor Network: \n";
    m_tensorNetwork->printIt();
    std::cout << "State Vector: \n";
    exatn::TensorNetwork ket(*m_tensorNetwork);
    ket.rename("MPSket");
#ifndef TNQVM_MPI_ENABLED
    const bool evaledOk = exatn::evaluateSync(ket);
    assert(evaledOk);
#else
    const bool evaledOk = exatn::evaluateSync(*m_selfProcessGroup, ket);
    assert(evaledOk);
#endif

    auto talsh_tensor = exatn::getLocalTensor(ket.getTensor(0)->getName());
    if (talsh_tensor)
    {
        const std::complex<double>* body_ptr;
        const bool access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr);
        if (!access_granted)
        {
            std::cout << "Failed to retrieve tensor data!!!\n";
        }
        else
        {
            for (int i = 0; i < talsh_tensor->getVolume(); ++i)
            {
                const auto& elem = body_ptr[i];
                std::cout << elem << "\n";
            }
        }

    }
    else
    {
        std::cout << "Failed to retrieve tensor data!!!\n";
    }
}

void ExatnMpsVisitor::finalize()
{
#ifndef TNQVM_MPI_ENABLED
    const auto finalizeStart = std::chrono::system_clock::now();

    // Always reset the logging level back to 0 when finished.
    exatn::resetClientLoggingLevel(0);
    exatn::resetRuntimeLoggingLevel(0);

    if (m_aggregateEnabled)
    {
        m_aggregator.flushAll();
        evaluateTensorNetwork(*m_tensorNetwork, m_stateVec);
    }

    if (m_buffer->size() < MAX_NUMBER_QUBITS_FOR_STATE_VEC)
    {
        exatn::TensorNetwork ket(*m_tensorNetwork);
        ket.rename("MPSket");
        const bool evaledOk = exatn::evaluateSync(ket);
        assert(evaledOk);
        const auto tensorData = getTensorData(ket.getTensor(0)->getName());
        // Simulate measurement by full tensor contraction (to get the state vector)
        // we can also implement repetitive bit count sampling
        // (will require many contractions but don't require large memory allocation)
        // DEBUG:
        // printStateVec();
        // Print state vector norm:
        const double norm = [&](){
            double sum = 0;
            for (const auto& val : tensorData)
            {
                sum = sum + std::norm(val);
            }
            return sum;
        }();
        m_buffer->addExtraInfo("norm", norm);

        if (!m_measureQubits.empty())
        {
            const auto calcExpValueZ = [](const std::vector<size_t>& in_bits, const std::vector<std::complex<double>>& in_stateVec) {
                const auto hasEvenParity = [](size_t x, const std::vector<size_t>& in_qubitIndices) -> bool {
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
            };

            // No shots, just add exp-val-z
            if (m_shotCount < 1)
            {
                const double exp_val_z = calcExpValueZ(m_measureQubits, tensorData);
                m_buffer->addExtraInfo("exp-val-z", exp_val_z);
            }
            else
            {
                addMeasureBitStringProbability(m_measureQubits, tensorData, m_shotCount);
            }
        }
    }
    else
    {
        if (!m_measureQubits.empty())
        {
            xacc::info("Simulating bit string by MPS tensor contraction");
            for (int i = 0; i < m_shotCount; ++i)
            {
                const auto convertToBitString = [](const std::vector<uint8_t>& in_bitVec){
                    std::string result;
                    for (const auto& bit : in_bitVec)
                    {
                        result.append(std::to_string(bit));
                    }
                    return result;
                };

                m_buffer->appendMeasurement(convertToBitString(getMeasureSample(m_measureQubits)));
            }
        }
    }


    for (int i = 0; i < m_buffer->size(); ++i)
    {
        const bool qTensorDestroyed = exatn::destroyTensor("Q" + std::to_string(i));
        assert(qTensorDestroyed);
    }

    const auto finalizeEnd = std::chrono::system_clock::now();
    getStatInstance("Finalize").addSample(finalizeStart, finalizeEnd);

    // Debug:
    // printAllStats();
#else
    for (const auto& [qubitIdx, rank] : m_qubitIdxToRank)
    {
        const std::string qubitTensorName = "Q" + std::to_string(qubitIdx);
        if (rank != m_rank)
        {
            const bool qTensorDestroyed = exatn::destroyTensor(qubitTensorName);
            assert(qTensorDestroyed);
        }

        const bool broadcastOk = exatn::replicateTensorSync(exatn::getDefaultProcessGroup(), qubitTensorName, rank);
        assert(broadcastOk);
    }

    // Only run bitstring sampling on root
    if (m_rank == 0)
    {
        // Update the tensor network to take into
        // account the updated tensors.
        rebuildTensorNetwork();

        // const auto stateVecNorm = computeStateVectorNorm(*m_tensorNetwork, exatn::getCurrentProcessGroup());
        // Small-circuit case: just reconstruct the full wavefunction
        if (m_buffer->size() < MAX_NUMBER_QUBITS_FOR_STATE_VEC && !options.keyExists<std::vector<int>>("bitstring"))
        {
            // DEBUG:
            // printStateVec();
            exatn::TensorNetwork ket(*m_tensorNetwork);
            ket.rename("MPSket");
            const bool evaledOk = exatn::evaluateSync(*m_selfProcessGroup, ket);
            assert(evaledOk);
            const auto tensorData = getTensorData(ket.getTensor(0)->getName());
            // Simulate measurement by full tensor contraction (to get the state vector)
            // we can also implement repetitive bit count sampling
            // (will require many contractions but don't require large memory allocation)
            // DEBUG:
            // printStateVec();
            // Print state vector norm:
            const double norm = [&](){
                double sum = 0;
                for (const auto& val : tensorData)
                {
                    sum = sum + std::norm(val);
                }
                return sum;
            }();
            m_buffer->addExtraInfo("norm", norm);

            if (!m_measureQubits.empty())
            {
                const auto calcExpValueZ = [](const std::vector<size_t>& in_bits, const std::vector<std::complex<double>>& in_stateVec) {
                    const auto hasEvenParity = [](size_t x, const std::vector<size_t>& in_qubitIndices) -> bool {
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
                };

                // No shots, just add exp-val-z
                if (m_shotCount < 1)
                {
                    const double exp_val_z = calcExpValueZ(m_measureQubits, tensorData);
                    m_buffer->addExtraInfo("exp-val-z", exp_val_z);
                }
                else
                {
                    addMeasureBitStringProbability(m_measureQubits, tensorData, m_shotCount);
                }
            }
        }
        else
        {
            // Large circuit
            // std::cout << "Final norm: " << stateVecNorm << "\n";
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

                std::vector<std::complex<double>> waveFuncSlice = computeWaveFuncSlice(*m_tensorNetwork, bitString, exatn::getCurrentProcessGroup());
                assert(!waveFuncSlice.empty());
                if (waveFuncSlice.size() == 1)
                {
                    m_buffer->addExtraInfo("amplitude-real", waveFuncSlice[0].real());
                    m_buffer->addExtraInfo("amplitude-imag", waveFuncSlice[0].imag());
                }
                else
                {
                    const auto normalizeWaveFnSlice =
                      [](std::vector<std::complex<double>> &io_waveFn) {
                        const double normVal = std::accumulate(
                            io_waveFn.begin(), io_waveFn.end(), 0.0,
                            [](double sumVal, const std::complex<double> &val) {
                              return sumVal + std::norm(val);
                            });
                        // The slice may have zero norm:
                        if (normVal > 1e-12) {
                          const std::complex<double> sqrtNorm = sqrt(normVal);
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
            else if (!m_measureQubits.empty())
            {
                xacc::info("Simulating bit string by MPS tensor contraction");
                m_shotCount = (m_shotCount < 1) ? 1 : m_shotCount;
                for (int i = 0; i < m_shotCount; ++i)
                {
                    const auto convertToBitString = [](const std::vector<uint8_t>& in_bitVec){
                        std::string result;
                        for (const auto& bit : in_bitVec)
                        {
                            result.append(std::to_string(bit));
                        }
                        return result;
                    };

                    m_buffer->appendMeasurement(convertToBitString(getMeasureSample(m_measureQubits)));
                }
            }
        }
    }

    for (const auto& [qubitIdx, rank] : m_qubitIdxToRank)
    {
        const std::string qubitTensorName = "Q" + std::to_string(qubitIdx);
        if (rank != m_rank)
        {
            const bool qTensorDestroyed = exatn::destroyTensor(qubitTensorName);
            assert(qTensorDestroyed);
        }

        const bool broadcastOk = exatn::replicateTensorSync(exatn::getDefaultProcessGroup(), qubitTensorName, rank);
        assert(broadcastOk);
    }

    for (int i = 0; i < m_buffer->size(); ++i)
    {
        const bool qTensorDestroyed = exatn::destroyTensor("Q" + std::to_string(i));
        assert(qTensorDestroyed);
    }
    // Clean up
    m_selfProcessGroup.reset();
    m_leftSharedProcessGroup.reset();
    m_rightSharedProcessGroup.reset();
#endif
}

void ExatnMpsVisitor::visit(Identity& in_IdentityGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_IdentityGate);
    }
    else
    {
        // Skip Identity gate
    }
}

void ExatnMpsVisitor::visit(Hadamard& in_HadamardGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_HadamardGate);
    }
    else
    {
        applyGate(in_HadamardGate);
    }
}

void ExatnMpsVisitor::visit(X& in_XGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_XGate);
    }
    else
    {
        applyGate(in_XGate);
    }
}

void ExatnMpsVisitor::visit(Y& in_YGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_YGate);
    }
    else
    {
        applyGate(in_YGate);
    }
}

void ExatnMpsVisitor::visit(Z& in_ZGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_ZGate);
    }
    else
    {
        applyGate(in_ZGate);
    }
}

void ExatnMpsVisitor::visit(Rx& in_RxGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_RxGate);
    }
    else
    {
        applyGate(in_RxGate);
    }
}

void ExatnMpsVisitor::visit(Ry& in_RyGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_RyGate);
    }
    else
    {
        applyGate(in_RyGate);
    }
}

void ExatnMpsVisitor::visit(Rz& in_RzGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_RzGate);
    }
    else
    {
        applyGate(in_RzGate);
    }
}

void ExatnMpsVisitor::visit(T& in_TGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_TGate);
    }
    else
    {
        applyGate(in_TGate);
    }
}

void ExatnMpsVisitor::visit(Tdg& in_TdgGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_TdgGate);
    }
    else
    {
        applyGate(in_TdgGate);
    }
}

// others
void ExatnMpsVisitor::visit(Measure& in_MeasureGate)
{
   m_measureQubits.emplace_back(in_MeasureGate.bits()[0]);
}

void ExatnMpsVisitor::visit(U& in_UGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_UGate);
    }
    else
    {
        applyGate(in_UGate);
    }
}

// two-qubit gates:
// NOTE: these gates are IMPORTANT for gate clustering consideration
void ExatnMpsVisitor::visit(CNOT& in_CNOTGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_CNOTGate);
    }
    else
    {
        applyGate(in_CNOTGate);
    }
}

void ExatnMpsVisitor::visit(Swap& in_SwapGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_SwapGate);
    }
    else
    {
        if (in_SwapGate.bits()[0] < in_SwapGate.bits()[1])
        {
            in_SwapGate.setBits({in_SwapGate.bits()[1], in_SwapGate.bits()[0]});
        }

        applyGate(in_SwapGate);
    }
}

void ExatnMpsVisitor::visit(CZ& in_CZGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_CZGate);
    }
    else
    {
        applyGate(in_CZGate);
    }
}

void ExatnMpsVisitor::visit(CPhase& in_CPhaseGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_CPhaseGate);
    }
    else
    {
        applyGate(in_CPhaseGate);
    }
}

void ExatnMpsVisitor::visit(iSwap& in_iSwapGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_iSwapGate);
    }
    else
    {
        applyGate(in_iSwapGate);
    }
}

void ExatnMpsVisitor::visit(fSim& in_fsimGate)
{
    if (m_aggregateEnabled)
    {
        m_aggregator.addGate(&in_fsimGate);
    }
    else
    {
        applyGate(in_fsimGate);
    }
}

const double ExatnMpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function)
{
    // Walk the circuit and visit all gates
    InstructionIterator it(in_function);
    while (it.hasNext())
    {
        auto nextInst = it.next();
        if (nextInst->isEnabled())
        {
            nextInst->accept(this);
        }
    }

    exatn::TensorNetwork ket(*m_tensorNetwork);
    ket.rename("MPSket");
    const bool evaledOk = exatn::evaluateSync(ket);
    assert(evaledOk);
    const auto tensorData = getTensorData(ket.getTensor(0)->getName());

    if (!m_measureQubits.empty())
    {
        // Just uses shot count estimation
        const int nbShotsToEstExpZ = 100000;
        addMeasureBitStringProbability(m_measureQubits, tensorData, nbShotsToEstExpZ);
        return m_buffer->getExpectationValueZ();
    }
    else
    {
        return 0.0;
    }
}

void ExatnMpsVisitor::onFlush(const AggregatedGroup& in_group)
{
    if (!m_aggregateEnabled)
    {
        return;
    }
    // DEBUG:
    // std::cout << "Flushing qubit line ";
    // for (const auto& id : in_group.qubitIdx)
    // {
    //     std::cout << id << ", ";
    // }
    // std::cout << "|| Number of gates = " << in_group.instructions.size() << "\n";

    // for (const auto& inst : in_group.instructions)
    // {
    //     std::cout << inst->toString() << "\n";
    // }
    // std::cout << "=============================================\n";

    // Construct tensor network
    // exatn::numerics::TensorNetwork aggregatedGateTensor("AggregatedTensorNetwork" + std::to_string(m_aggregatedGroupCounter));
    auto& aggregatedGateTensor = *m_tensorNetwork;

    m_aggregatedGroupCounter++;
    GateTensorConstructor tensorConstructor;
    for (const auto& inst : in_group.instructions)
    {
        const auto gateTensor = tensorConstructor.getGateTensor(*inst);
        const std::string& uniqueGateTensorName = gateTensor.uniqueName;
        if (m_registeredGateTensors.find(uniqueGateTensorName) == m_registeredGateTensors.end())
        {
            m_registeredGateTensors.emplace(uniqueGateTensorName);
            // Create the tensor
            const bool created = exatn::createTensor(
                uniqueGateTensorName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
            assert(created);
            // Init tensor body data
            exatn::initTensorData(uniqueGateTensorName, gateTensor.tensorData);
            const bool registered = exatn::registerTensorIsometry(uniqueGateTensorName, gateTensor.tensorIsometry.first, gateTensor.tensorIsometry.second);
        }

        // Because the qubit location and gate pairing are of different integer types,
        // we need to reconstruct the qubit vector.
        std::vector<unsigned int> gatePairing;
        for (const auto& qbitLoc : inst->bits())
        {
            gatePairing.emplace_back(qbitLoc);
        }
        std::reverse(gatePairing.begin(), gatePairing.end());
        m_tensorIdCounter++;
        // Append the tensor for this gate to the network
        aggregatedGateTensor.appendTensorGate(
            m_tensorIdCounter,
            // Get the gate tensor data which must have been initialized.
            exatn::getTensor(uniqueGateTensorName),
            // which qubits that the gate is acting on
            gatePairing
        );
    }

    // DEBUG:
    // std::cout << "AGGREGATED TENSOR NETWORK:\n";
    // aggregatedGateTensor.printIt();
}

void ExatnMpsVisitor::applyGate(xacc::Instruction& in_gateInstruction)
{
    const auto gateStart = std::chrono::system_clock::now();
    if (in_gateInstruction.bits().size() == 2)
    {
        return applyTwoQubitGate(in_gateInstruction);
    }

#ifndef TNQVM_MPI_ENABLED
    // Single qubit only in this path
    assert(in_gateInstruction.bits().size() == 1);
    const auto gateTensor = GateTensorConstructor::getGateTensor(in_gateInstruction);
    const std::string& uniqueGateTensorName = in_gateInstruction.name();
    // Create the tensor
    const bool created = exatn::createTensorSync(uniqueGateTensorName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
    assert(created);
    // Init tensor body data
    const bool initialized = exatn::initTensorDataSync(uniqueGateTensorName, gateTensor.tensorData);
    assert(initialized);
    // m_tensorNetwork->printIt();
    // Contract gate tensor to the qubit tensor
    const auto contractGateTensor = [](int in_qIdx, const std::string& in_gateTensorName){
        // Pattern:
        // (1) Boundary qubits (2 legs): Result(a, b) = Qi(a, i) * G (i, b)
        // (2) Middle qubits (3 legs): Result(a, b, c) = Qi(a, b, i) * G (i, c)
        // (3) Single qubit (1 leg):  Result(a) = Q0(i) * G (i, a)
        const std::string qubitTensorName = "Q" + std::to_string(in_qIdx);
        auto qubitTensor =  exatn::getTensor(qubitTensorName);
        assert(qubitTensor->getRank() == 1 || qubitTensor->getRank() == 2 || qubitTensor->getRank() == 3);
        auto gateTensor =  exatn::getTensor(in_gateTensorName);
        assert(gateTensor->getRank() == 2);

        const std::string RESULT_TENSOR_NAME = "Result";
        // Result tensor always has the same shape as the qubit tensor
        const bool resultTensorCreated = exatn::createTensorSync(RESULT_TENSOR_NAME,
                                                                exatn::TensorElementType::COMPLEX64,
                                                                qubitTensor->getShape());
        assert(resultTensorCreated);
        const bool resultTensorInitialized = exatn::initTensorSync(RESULT_TENSOR_NAME, 0.0);
        assert(resultTensorInitialized);

        std::string patternStr;
        if (qubitTensor->getRank() == 1)
        {
            // Single qubit
            assert(in_qIdx == 0);
            patternStr = RESULT_TENSOR_NAME + "(a)=" + qubitTensorName + "(i)*" + in_gateTensorName + "(i,a)";
        }
        else if (qubitTensor->getRank() == 2)
        {
            if (in_qIdx == 0)
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b)=" + qubitTensorName + "(i,b)*" + in_gateTensorName + "(i,a)";
            }
            else
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b)=" + qubitTensorName + "(a,i)*" + in_gateTensorName + "(i,b)";
            }
        }
        else if (qubitTensor->getRank() == 3)
        {
            patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + qubitTensorName + "(a,i,c)*" + in_gateTensorName + "(i,b)";
        }

        assert(!patternStr.empty());
        // std::cout << "Pattern string: " << patternStr << "\n";

        auto start = std::chrono::system_clock::now();
        const bool contractOk = exatn::contractTensorsSync(patternStr, 1.0);
        assert(contractOk);
        auto end = std::chrono::system_clock::now();
        getStatInstance("Contract Single-Qubit Gate Tensor").addSample(start, end);

        std::vector<std::complex<double>> resultTensorData =  getTensorData(RESULT_TENSOR_NAME);
        std::function<int(talsh::Tensor& in_tensor)> updateFunc = [&resultTensorData](talsh::Tensor& in_tensor){
            std::complex<double> *elements;

            if (in_tensor.getDataAccessHost(&elements) && (in_tensor.getVolume() == resultTensorData.size()))
            {
                for (int i = 0; i < in_tensor.getVolume(); ++i)
                {
                    elements[i] = resultTensorData[i];
                }
            }

            return 0;
        };

        exatn::numericalServer->transformTensorSync(qubitTensorName, std::make_shared<ExatnMpsVisitor::ExaTnTensorFunctor>(updateFunc));
        const bool resultTensorDestroyed = exatn::destroyTensor(RESULT_TENSOR_NAME);
        assert(resultTensorDestroyed);
    };


    {
        // Single-qubit gate contraction
        contractGateTensor(in_gateInstruction.bits()[0], uniqueGateTensorName);
    }

    // DEBUG:
    // printStateVec();

    const bool destroyed = exatn::destroyTensor(uniqueGateTensorName);
    assert(destroyed);
    exatn::sync();

    const auto gateEnd = std::chrono::system_clock::now();
    getStatInstance("One-qubit Gate Total").addSample(gateStart, gateEnd);
#else
    // MPI path:
    const size_t bitIdx = in_gateInstruction.bits()[0];
    if (indexInRange(bitIdx, m_qubitRange))
    {
        xacc::info("Process [" + std::to_string(m_rank) + "]: Process gate: " + in_gateInstruction.toString());
        const auto gateTensor = GateTensorConstructor::getGateTensor(in_gateInstruction);
        const std::string& uniqueGateTensorName = in_gateInstruction.name();
        // Create the tensor
        const bool created = exatn::createTensorSync(*m_selfProcessGroup, uniqueGateTensorName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
        assert(created);
        // Init tensor body data
        const bool initialized = exatn::initTensorDataSync(uniqueGateTensorName, gateTensor.tensorData);
        assert(initialized);
        // m_tensorNetwork->printIt();
        // Contract gate tensor to the qubit tensor
        const auto contractGateTensor = [](int in_qIdx, const std::string& in_gateTensorName, exatn::ProcessGroup& in_processGroup){
            // Pattern:
            // (1) Boundary qubits (2 legs): Result(a, b) = Qi(a, i) * G (i, b)
            // (2) Middle qubits (3 legs): Result(a, b, c) = Qi(a, b, i) * G (i, c)
            // (3) Single qubit (1 leg):  Result(a) = Q0(i) * G (i, a)
            const std::string qubitTensorName = "Q" + std::to_string(in_qIdx);
            auto qubitTensor =  exatn::getTensor(qubitTensorName);
            assert(qubitTensor->getRank() == 1 || qubitTensor->getRank() == 2 || qubitTensor->getRank() == 3);
            auto gateTensor =  exatn::getTensor(in_gateTensorName);
            assert(gateTensor->getRank() == 2);

            const std::string RESULT_TENSOR_NAME = "Result";
            // Result tensor always has the same shape as the qubit tensor
            const bool resultTensorCreated = exatn::createTensorSync(in_processGroup, RESULT_TENSOR_NAME,
                                                                    exatn::TensorElementType::COMPLEX64,
                                                                    qubitTensor->getShape());
            assert(resultTensorCreated);
            const bool resultTensorInitialized = exatn::initTensorSync(RESULT_TENSOR_NAME, 0.0);
            assert(resultTensorInitialized);

            std::string patternStr;
            if (qubitTensor->getRank() == 1)
            {
                // Single qubit
                assert(in_qIdx == 0);
                patternStr = RESULT_TENSOR_NAME + "(a)=" + qubitTensorName + "(i)*" + in_gateTensorName + "(i,a)";
            }
            else if (qubitTensor->getRank() == 2)
            {
                if (in_qIdx == 0)
                {
                    patternStr = RESULT_TENSOR_NAME + "(a,b)=" + qubitTensorName + "(i,b)*" + in_gateTensorName + "(i,a)";
                }
                else
                {
                    patternStr = RESULT_TENSOR_NAME + "(a,b)=" + qubitTensorName + "(a,i)*" + in_gateTensorName + "(i,b)";
                }
            }
            else if (qubitTensor->getRank() == 3)
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + qubitTensorName + "(a,i,c)*" + in_gateTensorName + "(i,b)";
            }

            assert(!patternStr.empty());
            // std::cout << "Pattern string: " << patternStr << "\n";
            const bool contractOk = exatn::contractTensorsSync(patternStr, 1.0);
            assert(contractOk);
            std::vector<std::complex<double>> resultTensorData =  getTensorData(RESULT_TENSOR_NAME);
            std::function<int(talsh::Tensor& in_tensor)> updateFunc = [&resultTensorData](talsh::Tensor& in_tensor){
                std::complex<double> *elements;

                if (in_tensor.getDataAccessHost(&elements) && (in_tensor.getVolume() == resultTensorData.size()))
                {
                    for (int i = 0; i < in_tensor.getVolume(); ++i)
                    {
                        elements[i] = resultTensorData[i];
                    }
                }

                return 0;
            };

            exatn::numericalServer->transformTensorSync(qubitTensorName, std::make_shared<ExatnMpsVisitor::ExaTnTensorFunctor>(updateFunc));
            const bool resultTensorDestroyed = exatn::destroyTensor(RESULT_TENSOR_NAME);
            assert(resultTensorDestroyed);
        };

        {
            // Single-qubit gate contraction
            contractGateTensor(in_gateInstruction.bits()[0], uniqueGateTensorName, *m_selfProcessGroup);
        }

        const bool destroyed = exatn::destroyTensor(uniqueGateTensorName);
        assert(destroyed);
    }
#endif
}

void ExatnMpsVisitor::applyTwoQubitGate(xacc::Instruction& in_gateInstruction)
{
#ifndef TNQVM_MPI_ENABLED
    exatn::sync();

    const auto gateStart = std::chrono::system_clock::now();

    const int q1 = in_gateInstruction.bits()[0];
    const int q2 = in_gateInstruction.bits()[1];
    // Neighbors only
    assert(std::abs(q1 - q2) == 1);
    const std::string q1TensorName = "Q" + std::to_string(q1);
    const std::string q2TensorName = "Q" + std::to_string(q2);

    // Step 1: merge two tensor together
    auto q1Tensor = exatn::getTensor(q1TensorName);
    auto q2Tensor = exatn::getTensor(q2TensorName);
    // std::cout << "Before merge:\n";
    // m_tensorNetwork->printIt();
    const auto getQubitTensorId = [&](const std::string& in_tensorName) {
        const auto idsVec = m_tensorNetwork->getTensorIdsInNetwork(in_tensorName);
        assert(idsVec.size() == 1);
        return idsVec.front();
    };

    // Merge Q2 into Q1
    const auto mergedTensorId = m_tensorNetwork->getMaxTensorId() + 1;
    std::string mergeContractionPattern;
    if (q1 < q2)
    {
        m_tensorNetwork->mergeTensors(getQubitTensorId(q1TensorName), getQubitTensorId(q2TensorName), mergedTensorId, &mergeContractionPattern);
        mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, q1TensorName);
        mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, q2TensorName);
    }
    else
    {
        m_tensorNetwork->mergeTensors(getQubitTensorId(q2TensorName), getQubitTensorId(q1TensorName), mergedTensorId, &mergeContractionPattern);
        mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, q2TensorName);
        mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, q1TensorName);
    }

    // std::cout << "After merge:\n";
    // m_tensorNetwork->printIt();

    auto mergedTensor =  m_tensorNetwork->getTensor(mergedTensorId);
    mergedTensor->rename("D");

    // std::cout << "Contraction Pattern: " << mergeContractionPattern << "\n";
    const bool mergedTensorCreated = exatn::createTensorSync(mergedTensor, exatn::TensorElementType::COMPLEX64);
    assert(mergedTensorCreated);
    const bool mergedTensorInitialized = exatn::initTensorSync(mergedTensor->getName(), 0.0);
    assert(mergedTensorInitialized);
    const bool mergedContractionOk = exatn::contractTensorsSync(mergeContractionPattern, 1.0);
    assert(mergedContractionOk);

    // Step 2: contract the merged tensor with the gate
    const auto gateTensor = GateTensorConstructor::getGateTensor(in_gateInstruction);
    const std::string uniqueGateTensorName = in_gateInstruction.name();

    // Create the tensor
    const bool created = exatn::createTensorSync(uniqueGateTensorName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
    assert(created);
    // Init tensor body data
    const bool initialized = exatn::initTensorDataSync(uniqueGateTensorName, gateTensor.tensorData);
    assert(initialized);

    assert(mergedTensor->getRank() >=2 && mergedTensor->getRank() <= 4);
    const std::string RESULT_TENSOR_NAME = "Result";
    // Result tensor always has the same shape as the *merged* qubit tensor
    const bool resultTensorCreated = exatn::createTensorSync(RESULT_TENSOR_NAME,
                                                            exatn::TensorElementType::COMPLEX64,
                                                            mergedTensor->getShape());
    assert(resultTensorCreated);
    const bool resultTensorInitialized = exatn::initTensorSync(RESULT_TENSOR_NAME, 0.0);
    assert(resultTensorInitialized);

    std::string patternStr;
    if (mergedTensor->getRank() == 3)
    {
        // Pattern: Result(a,b,c) = D(i,j,c)*Gate(i,j,a,b)
        if (q1 < q2)
        {
            if (q1 == 0)
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(i,j,c)*" + uniqueGateTensorName + "(j,i,b,a)";
            }
            else
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(a,i,j)*" + uniqueGateTensorName + "(j,i,c,b)";
            }
        }
        else
        {
            if (q2 == 0)
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(i,j,c)*" + uniqueGateTensorName + "(i,j,a,b)";
            }
            else
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(a,i,j)*" + uniqueGateTensorName + "(i,j,b,c)";
            }
        }
    }
    else if (mergedTensor->getRank() == 4)
    {
        if (q1 < q2)
        {
            patternStr = RESULT_TENSOR_NAME + "(a,b,c,d)=" + mergedTensor->getName() + "(a,i,j,d)*" + uniqueGateTensorName + "(j,i,c,b)";
        }
        else
        {
            patternStr = RESULT_TENSOR_NAME + "(a,b,c,d)=" + mergedTensor->getName() + "(a,i,j,d)*" + uniqueGateTensorName + "(i,j,b,c)";
        }
    }
    else if (mergedTensor->getRank() == 2)
    {
        // Only two-qubit in the qubit register
        assert(m_buffer->size() == 2);
        if (q1 < q2)
        {
            patternStr = RESULT_TENSOR_NAME + "(a,b)=" + mergedTensor->getName() + "(i,j)*" + uniqueGateTensorName + "(j,i,b,a)";
        }
        else
        {
            patternStr = RESULT_TENSOR_NAME + "(a,b)=" + mergedTensor->getName() + "(i,j)*" + uniqueGateTensorName + "(i,j,a,b)";
        }
    }

    assert(!patternStr.empty());
    // std::cout << "Gate contraction pattern: " << patternStr << "\n";

    {
        auto start = std::chrono::system_clock::now();
        const bool gateContractionOk = exatn::contractTensorsSync(patternStr, 1.0);
        assert(gateContractionOk);
        auto end = std::chrono::system_clock::now();
        getStatInstance("Contract Two-Qubit Gate Tensor").addSample(start, end);
    }

    const std::vector<std::complex<double>> resultTensorData =  getTensorData(RESULT_TENSOR_NAME);
    std::function<int(talsh::Tensor& in_tensor)> updateFunc = [&resultTensorData](talsh::Tensor& in_tensor){
        std::complex<double>* elements;

        if (in_tensor.getDataAccessHost(&elements) && (in_tensor.getVolume() == resultTensorData.size()))
        {
            for (int i = 0; i < in_tensor.getVolume(); ++i)
            {
                elements[i] = resultTensorData[i];
            }
        }

        return 0;
    };

    exatn::numericalServer->transformTensorSync(mergedTensor->getName(), std::make_shared<ExatnMpsVisitor::ExaTnTensorFunctor>(updateFunc));

    // Destroy the temp. result tensor
    const bool resultTensorDestroyed = exatn::destroyTensor(RESULT_TENSOR_NAME);
    assert(resultTensorDestroyed);

    // Destroy gate tensor
    const bool destroyed = exatn::destroyTensor(uniqueGateTensorName);
    assert(destroyed);


    const auto beforeSvd = std::chrono::system_clock::now();
    getStatInstance("Two-qubit Gate: Before SVD").addSample(gateStart, beforeSvd);

    // Step 3: SVD the merged tensor back into two MPS qubit tensor
    // Delete the two original qubit tensors
    const auto getBondLegId = [&](int in_qubitIdx, int in_otherQubitIdx){
        if (in_qubitIdx == 0)
        {
            return 1;
        }
        if(in_qubitIdx == (m_buffer->size() - 1))
        {
            return 0;
        }

        return (in_qubitIdx < in_otherQubitIdx) ? 2 : 0;
    };

    auto q1Shape = q1Tensor->getDimExtents();
    auto q2Shape = q2Tensor->getDimExtents();

    const auto q1BondLegId = getBondLegId(q1, q2);
    const auto q2BondLegId = getBondLegId(q2, q1);
    assert(q1Shape[q1BondLegId] == q2Shape[q2BondLegId]);
    int volQ1 = 1;
    int volQ2 = 1;
    for (int i = 0; i < q1Shape.size(); ++i)
    {
        if (i != q1BondLegId)
        {
            volQ1 *= q1Shape[i];
        }
    }

    for (int i = 0; i < q2Shape.size(); ++i)
    {
        if (i != q2BondLegId)
        {
            volQ2 *= q2Shape[i];
        }
    }

    const int newBondDim = std::min(volQ1, volQ2);
    // Update bond dimension
    q1Shape[q1BondLegId] = newBondDim;
    q2Shape[q2BondLegId] = newBondDim;

    // Destroy old qubit tensors
    const bool q1Destroyed = exatn::destroyTensor(q1TensorName);
    assert(q1Destroyed);
    const bool q2Destroyed = exatn::destroyTensor(q2TensorName);
    assert(q2Destroyed);
    exatn::sync();
    exatn::sync(mergedTensor->getName());

    // Create two new tensors:
    const bool q1Created = exatn::createTensorSync(q1TensorName, exatn::TensorElementType::COMPLEX64, q1Shape);
    assert(q1Created);

    const bool q2Created = exatn::createTensorSync(q2TensorName, exatn::TensorElementType::COMPLEX64, q2Shape);
    assert(q2Created);
    exatn::sync(q1TensorName);
    exatn::sync(q2TensorName);
    exatn::sync(mergedTensor->getName());

    exatn::sync();

    {
        auto start = std::chrono::system_clock::now();
        stabilizeTensorBody(mergedTensor->getName());
        // SVD decomposition using the same pattern that was used to merge two tensors
        const bool svdOk = exatn::decomposeTensorSVDLRSync(mergeContractionPattern);
        assert(svdOk);
        auto end = std::chrono::system_clock::now();
        getStatInstance("Decompose Tensor SVD").addSample(start, end);
    }

    exatn::sync(q1TensorName);
    exatn::sync(q2TensorName);

    // Validate SVD tensors
    // TODO: this should be eventually removed once we are confident with the ExaTN numerical backend.
    {
        // Validate SVD tensors
        // TODO: this should be eventually removed once we are confident with the ExaTN numerical backend.
        const auto calcMpsTensorNorm = [](const std::string& in_tensorName) {
            double sumNorm = 0.0;
            const bool normOk = exatn::computeNorm2Sync(in_tensorName, sumNorm);
            return sumNorm;
        };

        const double q1NormAfter = calcMpsTensorNorm(q1TensorName);
        const double q2NormAfter = calcMpsTensorNorm(q2TensorName);
        if (std::fabs(q1NormAfter) < 1e-3 || std::fabs(q2NormAfter) < 1e-3)
        {
            std::cout << "[ERROR] Tensor norm validation failed!\n";
            std::cout << in_gateInstruction.toString() << "\n";
            std::cout << q1TensorName << " norm = " << q1NormAfter << "\n";
            std::cout << q2TensorName << " norm = " << q2NormAfter << "\n";
            std::cout << "Tensor SVD Pattern: " <<  mergeContractionPattern << "\n";
            std::cout << "Merged Tensor: \n";
            printTensorData(mergedTensor->getName());
            std::cout << q1TensorName << "\n";
            printTensorData(q1TensorName);
            std::cout << q2TensorName << "\n";
            printTensorData(q2TensorName);
            // Crash in DEBUG to aid debugging.
            assert(false);
        }
    }

    const auto buildTensorMap = [&](){
        std::map<std::string, std::shared_ptr<exatn::Tensor>> tensorMap;
        tensorMap.emplace(ROOT_TENSOR_NAME, m_rootTensor);
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            const std::string qTensorName = "Q" + std::to_string(i);
            tensorMap.emplace(qTensorName, exatn::getTensor(qTensorName));
        }
        return tensorMap;
    };

    const std::string rootVarNameList = [&](){
        std::string result = "(";
        for (int i = 0; i < m_buffer->size() - 1; ++i)
        {
            result += ("i" + std::to_string(i) + ",");
        }
        result += ("i" + std::to_string(m_buffer->size() - 1) + ")");
        return result;
    }();

    const auto qubitTensorVarNameList = [&](int in_qIdx) -> std::string {
        if (in_qIdx == 0)
        {
            return "(i0,j0)";
        }
        if (in_qIdx == m_buffer->size() - 1)
        {
            return "(j" + std::to_string(m_buffer->size() - 2) + ",i" + std::to_string(m_buffer->size() - 1) + ")";
        }

        return "(j" + std::to_string(in_qIdx-1) + ",i" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx)+ ")";
    };

    const std::string mpsString = [&](){
        std::string result = ROOT_TENSOR_NAME + rootVarNameList + "=";
        for (int i = 0; i < m_buffer->size() - 1; ++i)
        {
            result += ("Q" + std::to_string(i) + qubitTensorVarNameList(i) + "*");
        }
        result += ("Q" + std::to_string(m_buffer->size() - 1) + qubitTensorVarNameList(m_buffer->size() - 1));
        return result;
    }();

    // std::cout << "MPS: " << mpsString << "\n";
    m_tensorNetwork = std::make_shared<exatn::TensorNetwork>(m_tensorNetwork->getName(), mpsString, buildTensorMap());

    const auto afterSvd = std::chrono::system_clock::now();
    getStatInstance("Two-qubit Gate: After SVD").addSample(gateStart, afterSvd);

    {
        auto start = std::chrono::system_clock::now();
        // Truncate SVD tensors:
        truncateSvdTensors(q1TensorName, q2TensorName, m_svdCutoff);
        auto end = std::chrono::system_clock::now();
        getStatInstance("Truncate SVD Tensor").addSample(start, end);
    }

    // Rebuild the tensor network since the qubit tensors have been changed after SVD truncation
    // e.g. we destroy the original tensors and replace with smaller dimension ones
    m_tensorNetwork = std::make_shared<exatn::TensorNetwork>(m_tensorNetwork->getName(), mpsString, buildTensorMap());

    const bool mergedTensorDestroyed = exatn::destroyTensor(mergedTensor->getName());
    assert(mergedTensorDestroyed);

    const auto gateEnd = std::chrono::system_clock::now();
    getStatInstance("Two-qubit Gate Total").addSample(gateStart, gateEnd);
    exatn::sync();
#else
    // MPI
    // !! The two qubit tensors must exist in this process group !!
    const auto processTwoQubitGate = [&](){
        const int q1 = in_gateInstruction.bits()[0];
        const int q2 = in_gateInstruction.bits()[1];
        // Neighbors only
        assert(std::abs(q1 - q2) == 1);
        const std::string q1TensorName = "Q" + std::to_string(q1);
        const std::string q2TensorName = "Q" + std::to_string(q2);

        // Step 1: merge two tensor together
        auto q1Tensor = exatn::getTensor(q1TensorName);
        auto q2Tensor = exatn::getTensor(q2TensorName);
        // std::cout << "Before merge:\n";
        // m_tensorNetwork->printIt();
        const auto getQubitTensorId = [&](const std::string& in_tensorName) {
            const auto idsVec = m_tensorNetwork->getTensorIdsInNetwork(in_tensorName);
            assert(idsVec.size() == 1);
            return idsVec.front();
        };

        // Merge Q2 into Q1
        const auto mergedTensorId = m_tensorNetwork->getMaxTensorId() + 1;
        std::string mergeContractionPattern;
        if (q1 < q2)
        {
            m_tensorNetwork->mergeTensors(getQubitTensorId(q1TensorName), getQubitTensorId(q2TensorName), mergedTensorId, &mergeContractionPattern);
            mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, q1TensorName);
            mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, q2TensorName);
        }
        else
        {
            m_tensorNetwork->mergeTensors(getQubitTensorId(q2TensorName), getQubitTensorId(q1TensorName), mergedTensorId, &mergeContractionPattern);
            mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, q2TensorName);
            mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, q1TensorName);
        }

        // std::cout << "After merge:\n";
        // m_tensorNetwork->printIt();

        auto mergedTensor =  m_tensorNetwork->getTensor(mergedTensorId);
        mergedTensor->rename("D");

        // std::cout << "Contraction Pattern: " << mergeContractionPattern << "\n";
        const bool mergedTensorCreated = exatn::createTensorSync(*m_selfProcessGroup, mergedTensor, exatn::TensorElementType::COMPLEX64);
        assert(mergedTensorCreated);
        const bool mergedTensorInitialized = exatn::initTensorSync(mergedTensor->getName(), 0.0);
        assert(mergedTensorInitialized);
        const bool mergedContractionOk = exatn::contractTensorsSync(mergeContractionPattern, 1.0);
        assert(mergedContractionOk);

        // Step 2: contract the merged tensor with the gate
        const auto gateTensor = GateTensorConstructor::getGateTensor(in_gateInstruction);
        const std::string uniqueGateTensorName = in_gateInstruction.name();

        // Create the tensor
        const bool created = exatn::createTensorSync(*m_selfProcessGroup, uniqueGateTensorName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
        assert(created);
        // Init tensor body data
        const bool initialized = exatn::initTensorDataSync(uniqueGateTensorName, gateTensor.tensorData);
        assert(initialized);

        assert(mergedTensor->getRank() >=2 && mergedTensor->getRank() <= 4);
        const std::string RESULT_TENSOR_NAME = "Result";
        // Result tensor always has the same shape as the *merged* qubit tensor
        const bool resultTensorCreated = exatn::createTensorSync(*m_selfProcessGroup, RESULT_TENSOR_NAME,
                                                                exatn::TensorElementType::COMPLEX64,
                                                                mergedTensor->getShape());
        assert(resultTensorCreated);
        const bool resultTensorInitialized = exatn::initTensorSync(RESULT_TENSOR_NAME, 0.0);
        assert(resultTensorInitialized);

        std::string patternStr;
        if (mergedTensor->getRank() == 3)
        {
            // Pattern: Result(a,b,c) = D(i,j,c)*Gate(i,j,a,b)
            if (q1 < q2)
            {
                if (q1 == 0)
                {
                    patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(i,j,c)*" + uniqueGateTensorName + "(j,i,b,a)";
                }
                else
                {
                    patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(a,i,j)*" + uniqueGateTensorName + "(j,i,c,b)";
                }
            }
            else
            {
                if (q2 == 0)
                {
                    patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(i,j,c)*" + uniqueGateTensorName + "(i,j,a,b)";
                }
                else
                {
                    patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(a,i,j)*" + uniqueGateTensorName + "(i,j,b,c)";
                }
            }
        }
        else if (mergedTensor->getRank() == 4)
        {
            if (q1 < q2)
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c,d)=" + mergedTensor->getName() + "(a,i,j,d)*" + uniqueGateTensorName + "(j,i,c,b)";
            }
            else
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c,d)=" + mergedTensor->getName() + "(a,i,j,d)*" + uniqueGateTensorName + "(i,j,b,c)";
            }
        }
        else if (mergedTensor->getRank() == 2)
        {
            // Only two-qubit in the qubit register
            assert(m_buffer->size() == 2);
            if (q1 < q2)
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b)=" + mergedTensor->getName() + "(i,j)*" + uniqueGateTensorName + "(j,i,b,a)";
            }
            else
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b)=" + mergedTensor->getName() + "(i,j)*" + uniqueGateTensorName + "(i,j,a,b)";
            }
        }

        assert(!patternStr.empty());
        // std::cout << "Gate contraction pattern: " << patternStr << "\n";

        {
            const bool gateContractionOk = exatn::contractTensorsSync(patternStr, 1.0);
            assert(gateContractionOk);
        }

        const std::vector<std::complex<double>> resultTensorData =  getTensorData(RESULT_TENSOR_NAME);
        std::function<int(talsh::Tensor& in_tensor)> updateFunc = [&resultTensorData](talsh::Tensor& in_tensor){
            std::complex<double>* elements;

            if (in_tensor.getDataAccessHost(&elements) && (in_tensor.getVolume() == resultTensorData.size()))
            {
                for (int i = 0; i < in_tensor.getVolume(); ++i)
                {
                    elements[i] = resultTensorData[i];
                }
            }

            return 0;
        };

        exatn::numericalServer->transformTensorSync(mergedTensor->getName(), std::make_shared<ExatnMpsVisitor::ExaTnTensorFunctor>(updateFunc));

        // Destroy the temp. result tensor
        const bool resultTensorDestroyed = exatn::destroyTensor(RESULT_TENSOR_NAME);
        assert(resultTensorDestroyed);

        // Destroy gate tensor
        const bool destroyed = exatn::destroyTensor(uniqueGateTensorName);
        assert(destroyed);

        // Step 3: SVD the merged tensor back into two MPS qubit tensor
        // Delete the two original qubit tensors
        const auto getBondLegId = [&](int in_qubitIdx, int in_otherQubitIdx){
            if (in_qubitIdx == 0)
            {
                return 1;
            }
            if(in_qubitIdx == (m_buffer->size() - 1))
            {
                return 0;
            }

            return (in_qubitIdx < in_otherQubitIdx) ? 2 : 0;
        };

        auto q1Shape = q1Tensor->getDimExtents();
        auto q2Shape = q2Tensor->getDimExtents();

        const auto q1BondLegId = getBondLegId(q1, q2);
        const auto q2BondLegId = getBondLegId(q2, q1);
        assert(q1Shape[q1BondLegId] == q2Shape[q2BondLegId]);
        int volQ1 = 1;
        int volQ2 = 1;
        for (int i = 0; i < q1Shape.size(); ++i)
        {
            if (i != q1BondLegId)
            {
                volQ1 *= q1Shape[i];
            }
        }

        for (int i = 0; i < q2Shape.size(); ++i)
        {
            if (i != q2BondLegId)
            {
                volQ2 *= q2Shape[i];
            }
        }

        const int newBondDim = std::min(volQ1, volQ2);
        // Update bond dimension
        q1Shape[q1BondLegId] = newBondDim;
        q2Shape[q2BondLegId] = newBondDim;

        // Destroy old qubit tensors
        const bool q1Destroyed = exatn::destroyTensor(q1TensorName);
        assert(q1Destroyed);
        const bool q2Destroyed = exatn::destroyTensor(q2TensorName);
        assert(q2Destroyed);
        exatn::sync(mergedTensor->getName());

        // Create two new tensors:
        const bool q1Created = exatn::createTensorSync(*m_selfProcessGroup, q1TensorName, exatn::TensorElementType::COMPLEX64, q1Shape);
        assert(q1Created);

        const bool q2Created = exatn::createTensorSync(*m_selfProcessGroup, q2TensorName, exatn::TensorElementType::COMPLEX64, q2Shape);
        assert(q2Created);
        exatn::sync(q1TensorName);
        exatn::sync(q2TensorName);
        exatn::sync(mergedTensor->getName());

        {
            // SVD decomposition using the same pattern that was used to merge two tensors
            // xacc::info("SVD: " + mergeContractionPattern);
            // mergedTensor->printIt();
            // exatn::getTensor(q1TensorName)->printIt();
            // exatn::getTensor(q2TensorName)->printIt();
            // printTensorData(mergedTensor->getName());
            stabilizeTensorBody(mergedTensor->getName());
            const bool svdOk = exatn::decomposeTensorSVDLRSync(mergeContractionPattern);
            assert(svdOk);
        }

        exatn::sync(q1TensorName);
        exatn::sync(q2TensorName);

        // Validate SVD tensors
        // TODO: this should be eventually removed once we are confident with the ExaTN numerical backend.
        {
            // Validate SVD tensors
            // TODO: this should be eventually removed once we are confident with the ExaTN numerical backend.
            const auto calcMpsTensorNorm = [](const std::string& in_tensorName) {
                double sumNorm = 0.0;
                const bool normOk = exatn::computeNorm2Sync(in_tensorName, sumNorm);
                return sumNorm;
            };

            const double q1NormAfter = calcMpsTensorNorm(q1TensorName);
            const double q2NormAfter = calcMpsTensorNorm(q2TensorName);
            if (std::fabs(q1NormAfter) < 1e-3 || std::fabs(q2NormAfter) < 1e-3)
            {
                std::cout << "[ERROR] Tensor norm validation failed!\n";
                std::cout << in_gateInstruction.toString() << "\n";
                std::cout << q1TensorName << " norm = " << q1NormAfter << "\n";
                std::cout << q2TensorName << " norm = " << q2NormAfter << "\n";
                std::cout << "Tensor SVD Pattern: " <<  mergeContractionPattern << "\n";
                std::cout << "Merged Tensor: \n";
                printTensorData(mergedTensor->getName());
                std::cout << q1TensorName << "\n";
                printTensorData(q1TensorName);
                std::cout << q2TensorName << "\n";
                printTensorData(q2TensorName);
                // Crash in DEBUG to aid debugging.
                assert(false);
            }
        }

        const auto buildTensorMap = [&](){
            std::map<std::string, std::shared_ptr<exatn::Tensor>> tensorMap;
            tensorMap.emplace(ROOT_TENSOR_NAME, m_rootTensor);
            for (int i = 0; i < m_buffer->size(); ++i)
            {
                const std::string qTensorName = "Q" + std::to_string(i);
                tensorMap.emplace(qTensorName, exatn::getTensor(qTensorName));
            }
            return tensorMap;
        };

        const std::string rootVarNameList = [&](){
            std::string result = "(";
            for (int i = 0; i < m_buffer->size() - 1; ++i)
            {
                result += ("i" + std::to_string(i) + ",");
            }
            result += ("i" + std::to_string(m_buffer->size() - 1) + ")");
            return result;
        }();

        const auto qubitTensorVarNameList = [&](int in_qIdx) -> std::string {
            if (in_qIdx == 0)
            {
                return "(i0,j0)";
            }
            if (in_qIdx == m_buffer->size() - 1)
            {
                return "(j" + std::to_string(m_buffer->size() - 2) + ",i" + std::to_string(m_buffer->size() - 1) + ")";
            }

            return "(j" + std::to_string(in_qIdx-1) + ",i" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx)+ ")";
        };

        const std::string mpsString = [&](){
            std::string result = ROOT_TENSOR_NAME + rootVarNameList + "=";
            for (int i = 0; i < m_buffer->size() - 1; ++i)
            {
                result += ("Q" + std::to_string(i) + qubitTensorVarNameList(i) + "*");
            }
            result += ("Q" + std::to_string(m_buffer->size() - 1) + qubitTensorVarNameList(m_buffer->size() - 1));
            return result;
        }();

        // std::cout << "MPS: " << mpsString << "\n";
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>(m_tensorNetwork->getName(), mpsString, buildTensorMap());
        {
            // Truncate SVD tensors:
            truncateSvdTensors(q1TensorName, q2TensorName, m_svdCutoff, m_selfProcessGroup.get());
        }

        // Rebuild the tensor network since the qubit tensors have been changed after SVD truncation
        // e.g. we destroy the original tensors and replace with smaller dimension ones
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>(m_tensorNetwork->getName(), mpsString, buildTensorMap());

        const bool mergedTensorDestroyed = exatn::destroyTensor(mergedTensor->getName());
        assert(mergedTensorDestroyed);

        // Validate norm:
        // const auto normVec = computeStateVectorNorm(*m_tensorNetwork, *m_selfProcessGroup);
        // xacc::info("Norm = " + std::to_string(normVec));
    };

    const int q1 = in_gateInstruction.bits()[0];
    const int q2 = in_gateInstruction.bits()[1];
    const int qMin = q1 < q2 ? q1 : q2;
    const int qMax = q1 > q2 ? q1 : q2;

    if (indexInRange(q1, m_qubitRange) && indexInRange(q2, m_qubitRange))
    {
        // Both qubits in range: process the gate
        xacc::info("Process [" + std::to_string(m_rank) + "]: Process gate: " + in_gateInstruction.toString());
        processTwoQubitGate();
    }
    else if (indexInRange(qMin, m_qubitRange))
    {
        // Left qubit in range -> this process will compute the gate
        // Receive q2 data from next subgroup
        xacc::info("Process [" + std::to_string(m_rank) + "]: Wait data to process gate: " + in_gateInstruction.toString());
        // MPI sync here:
        // Get tensor data:
        // Tensor that needs to be received:
        const std::string qubitTensorName = "Q" + std::to_string(qMax);

        const bool qMaxDestroyed = exatn::destroyTensor(qubitTensorName);
        assert(qMaxDestroyed);
        // Must have right shared group
        assert(m_rightSharedProcessGroup);
        {
            unsigned int neighborRank;
            const bool checkRankPre = m_rightSharedProcessGroup->rankIsIn(m_rank + 1, &neighborRank);
            assert(checkRankPre);
            const bool preBroadCastOk = exatn::replicateTensorSync(*m_rightSharedProcessGroup, qubitTensorName, neighborRank);
            assert(preBroadCastOk);
        }

        // Update the tensor network to take into
        // account the updated tensor.
        rebuildTensorNetwork();
        // Now, the *remote* tensor has been initialized, process gate as normal:
        // Apply gate
        xacc::info("Process [" + std::to_string(m_rank) + "]: Process gate: " + in_gateInstruction.toString());
        processTwoQubitGate();
        // Done: Send tensor to the neighbor process
        // Send tensor forward
        auto updatedTensor = exatn::getTensor(qubitTensorName);
        {
            unsigned int myLocalRank;
            const bool checkRankPost = m_rightSharedProcessGroup->rankIsIn(m_rank, &myLocalRank);
            assert(checkRankPost);
            const bool postBroadCastOk = exatn::replicateTensorSync(*m_rightSharedProcessGroup, qubitTensorName, 0);
            assert(postBroadCastOk);
        }
        {
          unsigned int neighborRank;
          const bool checkRank =
              m_rightSharedProcessGroup->rankIsIn(m_rank + 1, &neighborRank);
          assert(checkRank);
          const auto tensorShape = exatn::getTensor(qubitTensorName)->getShape();
          const bool dereplicateTensorOk = exatn::dereplicateTensorSync(
              *m_rightSharedProcessGroup, qubitTensorName, neighborRank);
          assert(dereplicateTensorOk);
          auto recreated = exatn::createTensorSync(
              *m_selfProcessGroup, qubitTensorName,
              exatn::TensorElementType::COMPLEX64, tensorShape);
          assert(recreated);
        }
    }
    else if (indexInRange(qMax, m_qubitRange))
    {
        // Right qubit in range, delegate the compute:
        const std::string qubitTensorName = "Q" + std::to_string(qMax);
        // Send tensor data to the left
        xacc::info("Process [" + std::to_string(m_rank) + "]: Send tensor data to process gate: " + in_gateInstruction.toString());
        // Must have a left shared sub-group
        assert(m_leftSharedProcessGroup);
        {
            unsigned int myLocalRank;
            const bool checkRankPre = m_leftSharedProcessGroup->rankIsIn(m_rank, &myLocalRank);
            assert(checkRankPre);
            const bool preBroadCastOk = exatn::replicateTensorSync(*m_leftSharedProcessGroup, qubitTensorName, myLocalRank);
            assert(preBroadCastOk);
        }

        // Update tensor data in this process
        const bool qMaxDestroyed = exatn::destroyTensor(qubitTensorName);
        assert(qMaxDestroyed);

        // Wait to receive tensor back (sync)
        {
            unsigned int neighborLocalRank;
            const bool checkRankPost = m_leftSharedProcessGroup->rankIsIn(m_rank - 1, &neighborLocalRank);
            assert(checkRankPost);
            const bool postBroadCastOk = exatn::replicateTensorSync(*m_leftSharedProcessGroup, qubitTensorName, 0);
            assert(postBroadCastOk);
        }

        {
          unsigned int myLocalRank;
          const bool checkRank =
              m_leftSharedProcessGroup->rankIsIn(m_rank, &myLocalRank);
          assert(checkRank);
          const bool dereplicateTensorOk = exatn::dereplicateTensorSync(
              *m_leftSharedProcessGroup, qubitTensorName, myLocalRank);
          assert(dereplicateTensorOk);
        }
        // Update the tensor network to take into
        // account the updated tensor.
        rebuildTensorNetwork();
    }
    else
    {
        // Don't care: both tensors are not in range
        xacc::info("Process [" + std::to_string(m_rank) + "]: Ignore gate: " + in_gateInstruction.toString());
    }
#endif
}

void ExatnMpsVisitor::evaluateTensorNetwork(exatn::numerics::TensorNetwork& io_tensorNetwork, std::vector<std::complex<double>>& out_stateVec)
{
    out_stateVec.clear();
    const bool evaluated = exatn::evaluateSync(io_tensorNetwork);
    assert(evaluated);
    // Synchronize:
    exatn::sync();

    std::function<int(talsh::Tensor& in_tensor)> accessFunc = [&out_stateVec](talsh::Tensor& in_tensor){
        out_stateVec.reserve(in_tensor.getVolume());
        std::complex<double> *elements;
        if (in_tensor.getDataAccessHost(&elements))
        {
            out_stateVec.assign(elements, elements + in_tensor.getVolume());
        }

        return 0;
    };

    auto functor = std::make_shared<ExatnMpsVisitor::ExaTnTensorFunctor>(accessFunc);

    exatn::numericalServer->transformTensorSync(io_tensorNetwork.getTensor(0)->getName(), functor);
    // DEBUG: 
    // for (const auto& elem : out_stateVec)
    // {
    //     std::cout << elem.real() <<  " + i " <<  elem.imag() << "\n";
    // }
}

void ExatnMpsVisitor::addMeasureBitStringProbability(const std::vector<size_t>& in_bits, const std::vector<std::complex<double>>& in_stateVec, int in_shotCount)
{
    // Factor to determine if we should spawn threads to simulate bitstring sampling.
    const int MULT_FACTOR = 100;
    if (getNumberOfThreads() < 2 || in_shotCount < MULT_FACTOR * getNumberOfThreads())
    {
        // Sequential execution
        for (int i = 0; i < in_shotCount; ++i)
        {
            std::string bitString;
            auto stateVecCopy = in_stateVec;
            for (const auto& bit : in_bits)
            {
                bitString.append(std::to_string(ApplyMeasureOp(stateVecCopy, bit)));
            }

            m_buffer->appendMeasurement(bitString);
        }
    }
    else
    {
        // Parallel execution
        std::vector<std::string> bitStringArray(in_shotCount);
        assert(bitStringArray.size() == in_shotCount);
        std::vector<std::thread> threads(getNumberOfThreads());
        std::mutex critical;
        for(int t = 0; t < getNumberOfThreads(); ++t)
        {
            threads[t] = std::thread(std::bind([&](int beginIdx, int endIdx, int threadIdx) {
                for(int i = beginIdx; i < endIdx; ++i)
                {
                    std::string bitString;
                    auto stateVecCopy = in_stateVec;
                    for (const auto& bit : in_bits)
                    {
                        bitString.append(std::to_string(ApplyMeasureOp(stateVecCopy, bit)));
                    }
                    bitStringArray[i] = bitString;
                }
                {
                    // Add measurement bitstring to the buffer:
                    std::lock_guard<std::mutex> lock(critical);
                    for(int i = beginIdx; i < endIdx; ++i)
                    {
                        m_buffer->appendMeasurement(bitStringArray[i]);
                    }
                }
            },
            t * in_shotCount / getNumberOfThreads(),
            (t+1) == getNumberOfThreads() ? in_shotCount: (t+1) * in_shotCount/getNumberOfThreads(),
            t));
        }
        std::for_each(threads.begin(),threads.end(),[](std::thread& x){
            x.join();
        });
    }
}

std::vector<uint8_t> ExatnMpsVisitor::getMeasureSample(const std::vector<size_t>& in_qubitIdx)
{
    std::vector<uint8_t> resultBitString;
    std::vector<double> resultProbs;
    for (const auto& qubitIdx : in_qubitIdx)
    {
        std::vector<std::string> tensorsToDestroy;
        std::vector<std::complex<double>> resultRDM;
        exatn::TensorNetwork ket(*m_tensorNetwork);
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
                const std::vector<std::complex<double>> COLLAPSE_0{
                    // Renormalize based on the probability of this outcome
                    {1.0 / resultProbs[measIdx], 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0}};

                const std::string tensorName = "COLLAPSE_0_" + std::to_string(measIdx);
                const bool created = exatn::createTensor(tensorName, exatn::TensorElementType::COMPLEX64, exatn::TensorShape{2, 2});
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
                const std::vector<std::complex<double>> COLLAPSE_1{
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {1.0 / resultProbs[measIdx], 0.0}};

                const std::string tensorName = "COLLAPSE_1_" + std::to_string(measIdx);
                const bool created = exatn::createTensor(tensorName, exatn::TensorElementType::COMPLEX64, exatn::TensorShape{2, 2});
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

        // Evaluate
        if (exatn::evaluateSync(combinedNetwork))
        {
            exatn::sync();
            auto talsh_tensor = exatn::getLocalTensor(combinedNetwork.getTensor(0)->getName());
            const auto tensorVolume = talsh_tensor->getVolume();
            // Single qubit density matrix
            assert(tensorVolume == 4);
            const std::complex<double>* body_ptr;
            if (talsh_tensor->getDataAccessHostConst(&body_ptr))
            {
                resultRDM.assign(body_ptr, body_ptr + tensorVolume);
            }
            // Debug: print out RDM data
            {
                std::stringstream logSs;
                logSs << "RDM @q" << qubitIdx << " = [";
                for (int i = 0; i < talsh_tensor->getVolume(); ++i)
                {
                    const std::complex<double> element = body_ptr[i];
                    logSs << element;
                }
                logSs << "]\n";
                xacc::info(logSs.str());
            }
        }

        {
            // Due to numerical stability, zero value may become an extremely-small negative number:
            // e.g. -1.234e-34, etc.
            // We'll skip any probability values less than this epsilon:
            constexpr double PROB_EPS = 1e-12;
            // Perform the measurement
            assert(resultRDM.size() == 4);
            const double prob_0 = std::abs(resultRDM.front().real()) < PROB_EPS
                                      ? 0.0
                                      : resultRDM.front().real();
            const double prob_1 = std::abs(resultRDM.back().real()) < PROB_EPS
                                      ? 0.0
                                      : resultRDM.back().real();
            assert(prob_0 >= 0.0 && prob_1 >= 0.0);
            assert(std::fabs(1.0 - prob_0 - prob_1) < 1e-12);

            // Generate a random number
            const double randProbPick = generateRandomProbability();
            // If radom number < probability of 0 state -> pick zero, and vice versa.
            resultBitString.emplace_back(randProbPick <= prob_0 ? 0 : 1);
            resultProbs.emplace_back(randProbPick <= prob_0 ? prob_0 : prob_1);
            std::stringstream logSs;
            logSs << ">> Measure @q" << qubitIdx << " prob(0) = " << prob_0 << "\n";
            logSs << ">> Measure @q" << qubitIdx << " prob(1) = " << prob_1 << "\n";
            logSs << ">> Measure @q" << qubitIdx << " random number = " << randProbPick << "\n";
            logSs << ">> Measure @q" << qubitIdx << " pick " << std::to_string(resultBitString.back()) << "\n";
            xacc::info(logSs.str());
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

void ExatnMpsVisitor::truncateSvdTensors(const std::string& in_leftTensorName, const std::string& in_rightTensorName, double in_eps, exatn::ProcessGroup *in_processGroup)
{
    int lhsTensorId = -1;
    int rhsTensorId = -1;

    for (auto iter = m_tensorNetwork->cbegin(); iter != m_tensorNetwork->cend(); ++iter)
    {
        const auto& tensorName = iter->second.getTensor()->getName();
        if (tensorName == in_leftTensorName)
        {
            lhsTensorId = iter->first;
        }
        if (tensorName == in_rightTensorName)
        {
            rhsTensorId = iter->first;
        }
    }

    assert(lhsTensorId > 0 && rhsTensorId > 0 && rhsTensorId != lhsTensorId);
    auto lhsConnections = m_tensorNetwork->getTensorConnections(lhsTensorId);
    auto rhsConnections = m_tensorNetwork->getTensorConnections(rhsTensorId);
    assert(lhsConnections && rhsConnections);

    exatn::TensorLeg lhsBondLeg;
    exatn::TensorLeg rhsBondLeg;
    for (const auto& leg : *lhsConnections)
    {
        if (leg.getTensorId() == rhsTensorId)
        {
            lhsBondLeg = leg;
            break;
        }
    }

    for (const auto& leg : *rhsConnections)
    {
        if (leg.getTensorId() == lhsTensorId)
        {
            rhsBondLeg = leg;
            break;
        }
    }

    // std::cout << in_leftTensorName << " leg " << rhsBondLeg.getDimensionId() << "--" << in_rightTensorName << " leg " << lhsBondLeg.getDimensionId() << "\n";

    const int lhsBondId = rhsBondLeg.getDimensionId();
    const int rhsBondId = lhsBondLeg.getDimensionId();
    auto lhsTensor = exatn::getTensor(in_leftTensorName);
    auto rhsTensor = exatn::getTensor(in_rightTensorName);

    assert(lhsBondId < lhsTensor->getRank());
    assert(rhsBondId < rhsTensor->getRank());
    assert(lhsTensor->getDimExtent(lhsBondId) == rhsTensor->getDimExtent(rhsBondId));
    const auto bondDim = lhsTensor->getDimExtent(lhsBondId);

    // Algorithm:
    // Lnorm(k) = Sum_ab [L(a,b,k)^2] and Rnorm(k) = Sum_c [R(k,c)]
    // Truncate k dimension by to k_opt where Lnorm(k_opt) < eps &&  Rnorm(k_opt) < eps
    std::vector<double> leftNorm;
    const bool leftNormOk = exatn::computePartialNormsSync(in_leftTensorName, lhsBondId, leftNorm);
    assert(leftNormOk);
    std::vector<double> rightNorm;
    const bool rightNormOk = exatn::computePartialNormsSync(in_rightTensorName, rhsBondId, rightNorm);
    assert(rightNormOk);

    assert(leftNorm.size() == rightNorm.size());
    assert(leftNorm.size() == bondDim);

    const auto findCutoffDim = [&]() -> int {
        for (int i = 0; i < bondDim; ++i)
        {
            if (leftNorm[i] < in_eps && rightNorm[i] < in_eps)
            {
                return i + 1;
            }
        }
        return bondDim;
    };

    const auto newBondDim = std::min(findCutoffDim(), m_maxBondDim);
    assert(newBondDim > 0);
    if (newBondDim < bondDim)
    {
      // xacc::info("Truncate SVD bond.");
      auto leftShape = lhsTensor->getDimExtents();
      auto rightShape = rhsTensor->getDimExtents();
      leftShape[lhsBondId] = newBondDim;
      rightShape[rhsBondId] = newBondDim;

      // Create two new tensors:
      const std::string newLhsTensorName =
          in_leftTensorName + "_" + std::to_string(lhsTensor->getTensorHash());
      const bool newLhsCreated =
          in_processGroup
              ? exatn::createTensorSync(*in_processGroup, newLhsTensorName,
                                        exatn::TensorElementType::COMPLEX64,
                                        leftShape)
              : exatn::createTensorSync(newLhsTensorName,
                                        exatn::TensorElementType::COMPLEX64,
                                        leftShape);
      assert(newLhsCreated);

      const std::string newRhsTensorName =
          in_rightTensorName + "_" + std::to_string(rhsTensor->getTensorHash());
      const bool newRhsCreated =
          in_processGroup
              ? exatn::createTensorSync(*in_processGroup, newRhsTensorName,
                                        exatn::TensorElementType::COMPLEX64,
                                        rightShape)
              : exatn::createTensorSync(newRhsTensorName,
                                        exatn::TensorElementType::COMPLEX64,
                                        rightShape);
      assert(newRhsCreated);

      // Take the slices:
      const bool lhsSliceOk =
          exatn::extractTensorSliceSync(in_leftTensorName, newLhsTensorName);
      assert(lhsSliceOk);
      const bool rhsSliceOk =
          exatn::extractTensorSliceSync(in_rightTensorName, newRhsTensorName);
      assert(rhsSliceOk);

      // Destroy the two original tensors:
      const bool lhsDestroyed = exatn::destroyTensorSync(in_leftTensorName);
      assert(lhsDestroyed);
      const bool rhsDestroyed = exatn::destroyTensorSync(in_rightTensorName);
      assert(rhsDestroyed);

      // Rename new tensors to the old name
      const auto renameNumericTensor = [&in_processGroup](
                                           const std::string &oldTensorName,
                                           const std::string &newTensorName) {
        auto tensor = exatn::getTensor(oldTensorName);
        assert(tensor);
        auto talsh_tensor = exatn::getLocalTensor(oldTensorName);
        assert(talsh_tensor);
        const std::complex<double> *body_ptr;
        const bool access_granted =
            talsh_tensor->getDataAccessHostConst(&body_ptr);
        assert(access_granted);
        std::vector<std::complex<double>> newData;
        newData.assign(body_ptr, body_ptr + talsh_tensor->getVolume());
        const bool newTensorCreated =
            in_processGroup
                ? exatn::createTensorSync(*in_processGroup, newTensorName,
                                          exatn::TensorElementType::COMPLEX64,
                                          tensor->getShape())
                : exatn::createTensorSync(newTensorName,
                                          exatn::TensorElementType::COMPLEX64,
                                          tensor->getShape());
        assert(newTensorCreated);
        const bool newTensorInitialized =
            exatn::initTensorDataSync(newTensorName, newData);
        assert(newTensorInitialized);
        // Destroy the two original tensor:
        const bool tensorDestroyed = exatn::destroyTensorSync(oldTensorName);
        assert(tensorDestroyed);
      };

      renameNumericTensor(newLhsTensorName, in_leftTensorName);
      renameNumericTensor(newRhsTensorName, in_rightTensorName);

      // Debug:
      // std::cout << "[DEBUG] Bond dim (" << in_leftTensorName << ", " <<
      // in_rightTensorName << "): " << bondDim << " -> " << newBondDim << "\n";
      std::stringstream logSs;
      logSs << "[SVD] Bond dim (" << in_leftTensorName << ", "
            << in_rightTensorName << "): " << bondDim << " -> " << newBondDim;
      xacc::info(logSs.str());
    }
}

#ifdef TNQVM_MPI_ENABLED
void ExatnMpsVisitor::rebuildTensorNetwork()
{
    const auto buildTensorMap = [&](){
        std::map<std::string, std::shared_ptr<exatn::Tensor>> tensorMap;
        tensorMap.emplace(ROOT_TENSOR_NAME, m_rootTensor);
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            const std::string qTensorName = "Q" + std::to_string(i);
            tensorMap.emplace(qTensorName, exatn::getTensor(qTensorName));
        }
        return tensorMap;
    };

    const std::string rootVarNameList = [&](){
        std::string result = "(";
        for (int i = 0; i < m_buffer->size() - 1; ++i)
        {
            result += ("i" + std::to_string(i) + ",");
        }
        result += ("i" + std::to_string(m_buffer->size() - 1) + ")");
        return result;
    }();

    const auto qubitTensorVarNameList = [&](int in_qIdx) -> std::string {
        if (in_qIdx == 0)
        {
            return "(i0,j0)";
        }
        if (in_qIdx == m_buffer->size() - 1)
        {
            return "(j" + std::to_string(m_buffer->size() - 2) + ",i" + std::to_string(m_buffer->size() - 1) + ")";
        }

        return "(j" + std::to_string(in_qIdx-1) + ",i" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx)+ ")";
    };

    const std::string mpsString = [&](){
        std::string result = ROOT_TENSOR_NAME + rootVarNameList + "=";
        for (int i = 0; i < m_buffer->size() - 1; ++i)
        {
            result += ("Q" + std::to_string(i) + qubitTensorVarNameList(i) + "*");
        }
        result += ("Q" + std::to_string(m_buffer->size() - 1) + qubitTensorVarNameList(m_buffer->size() - 1));
        return result;
    }();
    m_tensorNetwork = std::make_shared<exatn::TensorNetwork>(m_tensorNetwork->getName(), mpsString, buildTensorMap());
}
#endif

std::vector<std::complex<double>> ExatnMpsVisitor::computeWaveFuncSlice(
    const exatn::TensorNetwork& in_tensorNetwork, const std::vector<int>& bitString,
    const exatn::ProcessGroup& in_processGroup) const {
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
        const bool created = exatn::createTensor(
            in_processGroup, braQubitName, exatn::TensorElementType::COMPLEX64,
            exatn::TensorShape{2});
        assert(created);
        // Bit = 0
        const bool initialized = exatn::initTensorData(
            braQubitName,
            std::vector<std::complex<double>>{{1.0, 0.0}, {0.0, 0.0}});
        assert(initialized);
        pairings.emplace_back(std::make_pair(i, i + nbOpenLegs));
      } else if (bitVal == 1) {
        const bool created = exatn::createTensor(
            in_processGroup, braQubitName, exatn::TensorElementType::COMPLEX64,
            exatn::TensorShape{2});
        assert(created);
        // Bit = 1
        const bool initialized = exatn::initTensorData(
            braQubitName,
            std::vector<std::complex<double>>{{0.0, 0.0}, {1.0, 0.0}});
        assert(initialized);
        pairings.emplace_back(std::make_pair(i, i + nbOpenLegs));
      } else if (bitVal == -1) {
        // Add an Id tensor
        const bool created = exatn::createTensor(
            in_processGroup, braQubitName, exatn::TensorElementType::COMPLEX64,
            exatn::TensorShape{2, 2});
        assert(created);
        const bool initialized = exatn::initTensorData(
            braQubitName, std::vector<std::complex<double>>{
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
  std::vector<std::complex<double>> waveFnSlice;
  {
    // std::cout << "SUBMIT TENSOR NETWORK FOR EVALUATION\n";
    // combinedTensorNetwork.printIt();
    if (exatn::evaluateSync(in_processGroup, combinedTensorNetwork)) {
      exatn::sync();
      auto talsh_tensor =
          exatn::getLocalTensor(combinedTensorNetwork.getTensor(0)->getName());
      const std::complex<double> *body_ptr;
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

double ExatnMpsVisitor::computeStateVectorNorm(const exatn::numerics::TensorNetwork& in_tensorNetwork, const exatn::ProcessGroup& in_processGroup) const
{
    auto braTensors = in_tensorNetwork;
    braTensors.rename("Bra_MPS");
    braTensors.conjugate();
    auto combinedTensorNetwork = in_tensorNetwork;
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (unsigned int i = 0; i < m_buffer->size(); ++i)
    {
        pairings.emplace_back(std::make_pair(i, i));
    }
    combinedTensorNetwork.appendTensorNetwork(std::move(braTensors), pairings);
    std::complex<double> norm;
    if (exatn::evaluateSync(in_processGroup, combinedTensorNetwork)) {
      exatn::sync();
      auto talsh_tensor =
          exatn::getLocalTensor(combinedTensorNetwork.getTensor(0)->getName());
      assert(talsh_tensor->getVolume() ==  1);
      const std::complex<double> *body_ptr;
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
        norm = *body_ptr;
      }
    }
    // std::cout << "Norm: " << norm.real() << " , " << norm.imag() << "\n";
    return norm.real();
}
}
