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

#include "ExaTnPmpsVisitor.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include "base/Gates.hpp"
#include "NoiseModel.hpp"
#include "xacc_service.hpp"
#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

#define INITIAL_BOND_DIM 1
#define INITIAL_KRAUS_DIM 1
#define QUBIT_DIM 2

namespace {
std::vector<std::complex<double>> Q_ZERO_TENSOR_BODY(size_t in_volume)
{
    std::vector<std::complex<double>> body(in_volume, {0.0, 0.0});
    body[0] = std::complex<double>(1.0, 0.0);
    return body;
}

const std::string ROOT_TENSOR_NAME = "Root";

// Prints the density matrix represented by a locally-purified MPS tensor network.
void printDensityMatrix(const exatn::TensorNetwork& in_pmpsNet, size_t in_nbQubit)
{
    exatn::TensorNetwork tempNetwork(in_pmpsNet);
    tempNetwork.rename("__TEMP__" + in_pmpsNet.getName());
    const bool evaledOk = exatn::evaluateSync(tempNetwork);
    assert(evaledOk);
    auto talsh_tensor = exatn::getLocalTensor(tempNetwork.getTensor(0)->getName());
    const auto expectedDensityMatrixVolume = (1ULL << in_nbQubit) * (1ULL << in_nbQubit);
    const auto nbRows = 1ULL << in_nbQubit;
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
            assert(expectedDensityMatrixVolume == talsh_tensor->getVolume());
            std::complex<double> traceVal { 0.0, 0.0 };
            int rowId = -1;
            for (int i = 0; i < talsh_tensor->getVolume(); ++i)
            {
                if (i % nbRows == 0)
                {
                    std::cout << "\n";
                    rowId++;
                }
                if (i == (rowId + rowId * nbRows))
                {
                    traceVal += body_ptr[i];
                }
                const auto& elem = body_ptr[i];
                std::cout << elem << " ";
            }
            std::cout << "\n";
            // Verify trace(density matrix) == 1.0
            assert(std::abs(traceVal.real() - 1.0) < 1e-12);
            assert(std::abs(traceVal.imag()) < 1e-12);
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

std::vector<std::complex<double>> calculateDensityMatrix(const exatn::TensorNetwork& in_pmpsNet, size_t in_nbQubit)
{
    exatn::TensorNetwork tempNetwork(in_pmpsNet);
    tempNetwork.rename("__TEMP__" + in_pmpsNet.getName());
    const bool evaledOk = exatn::evaluateSync(tempNetwork);
    assert(evaledOk);
    return getTensorData(tempNetwork.getTensor(0)->getName());
}

std::string generateResultBitString(const std::vector<std::complex<double>>& in_dmDiagonalElems, const std::vector<size_t>& in_measureQubits, size_t in_nbQubits, xacc::NoiseModel* in_noiseModel = nullptr)
{
    static auto randomProbFunc = std::bind(std::uniform_real_distribution<double>(0, 1), std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    // Pick a random probability
    const auto probPick = randomProbFunc();
    const size_t N = in_dmDiagonalElems.size();
    // Random state selection
    double cumulativeProb = 0.0;
    size_t stateSelect = 0;
    // select a state based on cumulative distribution (diagonal elements)
    while (cumulativeProb < probPick && stateSelect < N)
    {
        cumulativeProb += in_dmDiagonalElems[stateSelect++].real();
    }

    // back one step
    stateSelect--;
    std::string result;
    for (const auto& qubit : in_measureQubits)
    {
        const auto qubitIdx = in_nbQubits - qubit - 1;
        const bool bit = ((stateSelect >> qubitIdx) & 1);
        if (in_noiseModel)
        {
            // Apply Readout error:
            const auto roErrorProb = randomProbFunc();
            const auto [meas0Prep1, meas1Prep0] = in_noiseModel->readoutError(qubit);
            const double flipProb = bit ? meas0Prep1 : meas1Prep0;
            const bool measBit = (roErrorProb < flipProb) ? !bit : bit;
            result.push_back(measBit ? '1' : '0');
        }
        else
        {
            result.push_back(bit ? '1' : '0');
        }
    }

    return result;
}

std::vector<std::complex<double>> getGateMatrix(const xacc::Instruction& in_gate)
{
    using namespace tnqvm;
    const auto gateEnum = GetGateType(in_gate.name());
    const auto getMatrix = [&](){
        switch (gateEnum)
        {
            case CommonGates::Rx: return GetGateMatrix<CommonGates::Rx>(in_gate.getParameter(0).as<double>());
            case CommonGates::Ry: return GetGateMatrix<CommonGates::Ry>(in_gate.getParameter(0).as<double>());
            case CommonGates::Rz: return GetGateMatrix<CommonGates::Rz>(in_gate.getParameter(0).as<double>());
            case CommonGates::I: return GetGateMatrix<CommonGates::I>();
            case CommonGates::H: return GetGateMatrix<CommonGates::H>();
            case CommonGates::X: return GetGateMatrix<CommonGates::X>();
            case CommonGates::Y: return GetGateMatrix<CommonGates::Y>();
            case CommonGates::Z: return GetGateMatrix<CommonGates::Z>();
            case CommonGates::T:
              return GetGateMatrix<CommonGates::T>();
            case CommonGates::U:
              return GetGateMatrix<CommonGates::U>(
                  in_gate.getParameter(0).as<double>(),
                  in_gate.getParameter(1).as<double>(),
                  in_gate.getParameter(2).as<double>());
            case CommonGates::Tdg:
              return GetGateMatrix<CommonGates::Tdg>();
            case CommonGates::CNOT: return GetGateMatrix<CommonGates::CNOT>();
            case CommonGates::CY:
              return GetGateMatrix<CommonGates::CY>();
            case CommonGates::CZ:
              return GetGateMatrix<CommonGates::CZ>();
            case CommonGates::CH:
              return GetGateMatrix<CommonGates::CH>();
            case CommonGates::CRZ:
              return GetGateMatrix<CommonGates::CRZ>(in_gate.getParameter(0).as<double>());
            case CommonGates::CPhase:
              return GetGateMatrix<CommonGates::CPhase>(in_gate.getParameter(0).as<double>());
            case CommonGates::Swap: return GetGateMatrix<CommonGates::Swap>();
            case CommonGates::iSwap: return GetGateMatrix<CommonGates::iSwap>();
            case CommonGates::fSim: return GetGateMatrix<CommonGates::fSim>(in_gate.getParameter(0).as<double>(), in_gate.getParameter(1).as<double>());
            default: return GetGateMatrix<CommonGates::I>();
        }
    };

    const auto flattenGateMatrix = [](const std::vector<std::vector<std::complex<double>>>& in_gateMatrix){
        std::vector<std::complex<double>> resultVector;
        resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
        for (const auto &row : in_gateMatrix)
        {
            for (const auto &entry : row)
            {
                resultVector.emplace_back(entry);
            }
        }

        return resultVector;
    };

    return flattenGateMatrix(getMatrix());
}

void contractSingleQubitGateTensor(const std::string& qubitTensorName, const std::string& in_gateTensorName)
{
    auto qubitTensor =  exatn::getTensor(qubitTensorName);
    assert(qubitTensor->getRank() == 2 || qubitTensor->getRank() == 3 || qubitTensor->getRank() == 4);
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
    if (qubitTensor->getRank() == 2)
    {
        patternStr = RESULT_TENSOR_NAME + "(a,b)=" + qubitTensorName + "(i,b)*" + in_gateTensorName + "(i,a)";
    }
    else if (qubitTensor->getRank() == 3)
    {
        patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + qubitTensorName + "(i,b,c)*" + in_gateTensorName + "(i,a)";
    }
    else if (qubitTensor->getRank() == 4)
    {
        patternStr = RESULT_TENSOR_NAME + "(a,b,c,d)=" + qubitTensorName + "(i,b,c,d)*" + in_gateTensorName + "(i,a)";
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

    exatn::numericalServer->transformTensorSync(qubitTensorName, std::make_shared<tnqvm::ExaTnPmpsVisitor::ExaTnTensorFunctor>(updateFunc));
    const bool resultTensorDestroyed = exatn::destroyTensor(RESULT_TENSOR_NAME);
    assert(resultTensorDestroyed);
}

// Retrieve the leg Id of the connection b/w two tensors.
std::pair<size_t, size_t> getBondLegId(const exatn::TensorNetwork& in_tensorNetwork, const std::string& in_leftTensorName, const std::string& in_rightTensorName)
{
    int lhsTensorId = -1;
    int rhsTensorId = -1;

    for (auto iter = in_tensorNetwork.cbegin(); iter != in_tensorNetwork.cend(); ++iter)
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
    auto lhsConnections = in_tensorNetwork.getTensorConnections(lhsTensorId);
    auto rhsConnections = in_tensorNetwork.getTensorConnections(rhsTensorId);
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

    const auto lhsBondId = rhsBondLeg.getDimensionId();
    const auto rhsBondId = lhsBondLeg.getDimensionId();
    return std::make_pair(lhsBondId, rhsBondId);
}

void contractTwoQubitGateTensor(const exatn::TensorNetwork& in_tensorNetwork, const std::vector<size_t>& in_bits, const std::string& in_gateTensorName)
{
    exatn::TensorNetwork tempNetwork(in_tensorNetwork);
    const std::string q1TensorName = "Q" + std::to_string(in_bits[0]);
    const std::string q2TensorName = "Q" + std::to_string(in_bits[1]);

    const auto mergedTensorId = tempNetwork.getMaxTensorId() + 1;
    std::string mergeContractionPattern;
    const auto getTensorId = [&](const std::string& in_tensorName){
        const auto ids = tempNetwork.getTensorIdsInNetwork(in_tensorName);
        assert(ids.size() == 1);
        return ids[0];
    };
    const bool shouldFlipOrder = in_bits[0] > in_bits[1];
    // Merge qubit tensors
    if (!shouldFlipOrder)
    {
        tempNetwork.mergeTensors(getTensorId(q1TensorName), getTensorId(q2TensorName), mergedTensorId, &mergeContractionPattern);
        mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, q1TensorName);
        mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, q2TensorName);
    }
    else
    {
        tempNetwork.mergeTensors(getTensorId(q2TensorName), getTensorId(q1TensorName), mergedTensorId, &mergeContractionPattern);
        mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, q2TensorName);
        mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, q1TensorName);
    }
    auto qubitsMergedTensor =  tempNetwork.getTensor(mergedTensorId);
    mergeContractionPattern.replace(mergeContractionPattern.find("D"), 1, qubitsMergedTensor->getName());
    // std::cout << "Merged: " << mergeContractionPattern << "\n";

    const auto contractMergePattern = [](std::shared_ptr<exatn::Tensor>& mergedTensor, const std::string& in_mergePattern) {
        const bool mergedTensorCreated = exatn::createTensorSync(mergedTensor, exatn::TensorElementType::COMPLEX64);
        assert(mergedTensorCreated);
        const bool mergedTensorInitialized = exatn::initTensorSync(mergedTensor->getName(), 0.0);
        assert(mergedTensorInitialized);
        const bool mergedContractionOk = exatn::contractTensorsSync(in_mergePattern, 1.0);
        assert(mergedContractionOk);
    };

    contractMergePattern(qubitsMergedTensor, mergeContractionPattern);
    std::string svdPattern = mergeContractionPattern;
    std::shared_ptr<exatn::Tensor> gateMergeTensor = std::make_shared<exatn::Tensor>("Result", qubitsMergedTensor->getShape());
    // Merge with gate:
    std::string patternStr;
    if (qubitsMergedTensor->getRank() == 4)
    {
        if (!shouldFlipOrder)
        {
            patternStr = "Result(u0,u1,u2,u3)=" + qubitsMergedTensor->getName() + "(c0,u1,c1,u3)*" + in_gateTensorName + "(c1,c0,u0,u2)";
        }
        else
        {
            patternStr = "Result(u0,u1,u2,u3)=" + qubitsMergedTensor->getName() + "(c0,u1,c1,u3)*" + in_gateTensorName + "(c0,c1,u0,u2)";
        }
    }
    else if (qubitsMergedTensor->getRank() == 5)
    {
        // qubitsMergedTensor->printIt();
        if (!shouldFlipOrder)
        {
            if (in_bits[0] == 0)
            {
                patternStr = "Result(u0,u1,u2,u3,u4)=" + qubitsMergedTensor->getName() + "(c0,u1,c1,u3,u4)*" + in_gateTensorName + "(c1,c0,u0,u2)";
            }
            else
            {
                patternStr = "Result(u0,u1,u2,u3,u4)=" + qubitsMergedTensor->getName() + "(c0,u1,u2,c1,u4)*" + in_gateTensorName + "(c1,c0,u0,u3)";
            }
        }
        else
        {
            if (in_bits[1] == 0)
            {
                patternStr = "Result(u0,u1,u2,u3,u4)=" + qubitsMergedTensor->getName() + "(c0,u1,c1,u3,u4)*" + in_gateTensorName + "(c0,c1,u0,u2)";
            }
            else
            {
                patternStr = "Result(u0,u1,u2,u3,u4)=" + qubitsMergedTensor->getName() + "(c0,u1,u2,c1,u4)*" + in_gateTensorName + "(c0,c1,u0,u3)";
            }
        }
    }
    else if (qubitsMergedTensor->getRank() == 6)
    {
        if (!shouldFlipOrder)
        {
            patternStr = "Result(u0,u1,u2,u3,u4,u5)=" + qubitsMergedTensor->getName() + "(c0,u1,u2,c1,u4,u5)*" + in_gateTensorName + "(c1,c0,u0,u3)";
        }
        else
        {
            patternStr = "Result(u0,u1,u2,u3,u4,u5)=" + qubitsMergedTensor->getName() + "(c0,u1,u2,c1,u4,u5)*" + in_gateTensorName + "(c0,c1,u0,u3)";
        }
    }
    else
    {
        xacc::error("Invalid tensor network structure encountered!");
    }

    contractMergePattern(gateMergeTensor, patternStr);

    // SVD:
    {
        svdPattern.replace(svdPattern.find(qubitsMergedTensor->getName()), qubitsMergedTensor->getName().size(), gateMergeTensor->getName());
        // std::cout << "SVD: " << svdPattern << "\n";
        auto q1Tensor = exatn::getTensor(q1TensorName);
        auto q2Tensor = exatn::getTensor(q2TensorName);
        const auto bondIds = getBondLegId(in_tensorNetwork, q1TensorName, q2TensorName);
        auto q1Shape = q1Tensor->getDimExtents();
        auto q2Shape = q2Tensor->getDimExtents();
        const auto q1BondDim = std::max(2 * q1Shape[bondIds.first], q1Tensor->getVolume()/q1Shape[bondIds.first]);
        const auto q2BondDim = std::max(2 * q2Shape[bondIds.second], q2Tensor->getVolume()/q2Shape[bondIds.second]);
        const auto entangledBondDim = std::min(q1BondDim, q2BondDim);
        q1Shape[bondIds.first] = entangledBondDim;
        q2Shape[bondIds.second] = entangledBondDim;

        // Destroy old qubit tensors
        const bool q1Destroyed = exatn::destroyTensor(q1TensorName);
        assert(q1Destroyed);
        const bool q2Destroyed = exatn::destroyTensor(q2TensorName);
        assert(q2Destroyed);
        exatn::sync();

        // Create two new tensors:
        const bool q1Created = exatn::createTensorSync(q1TensorName, exatn::TensorElementType::COMPLEX64, q1Shape);
        assert(q1Created);

        const bool q2Created = exatn::createTensorSync(q2TensorName, exatn::TensorElementType::COMPLEX64, q2Shape);
        assert(q2Created);
        exatn::sync(q1TensorName);
        exatn::sync(q2TensorName);
        // SVD decomposition using the same pattern that was used to merge two tensors
        const bool svdOk = exatn::decomposeTensorSVDLRSync(svdPattern);
        assert(svdOk);
    }

    const auto destroyed = exatn::destroyTensorSync("Result");
    assert(destroyed);
}
}
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
        exatn::resetClientLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
        exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);

        xacc::subscribeLoggingLevel([](int level) {
            exatn::resetClientLoggingLevel(xacc::verbose ? level : 0);
            exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
        });
    }

    m_buffer = buffer;
    m_pmpsTensorNetwork = buildInitialNetwork(buffer->size(), true);
    m_noiseConfig.reset();
    if (options.pointerLikeExists<xacc::NoiseModel>("noise-model")) {
      m_noiseConfig = xacc::as_shared_ptr(
          options.getPointerLike<xacc::NoiseModel>("noise-model"));
    } else {
      // Backend JSON was provided as a full string
      if (options.stringExists("backend-json")) {
        m_noiseConfig = xacc::getService<xacc::NoiseModel>("IBM");
        m_noiseConfig->initialize(
            {{"backend-json", options.getString("backend-json")}});
      }
      // Backend was referred to by name
      // e.g. ibmq_ourense
      if (options.stringExists("backend")) {
        m_noiseConfig = xacc::getService<xacc::NoiseModel>("IBM");
        m_noiseConfig->initialize({{"backend", options.getString("backend")}});
      }
    }
    // DEBUG
    // printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
    // Since this is a noisy simulation, always run shots by default.
    m_nbShots = (nbShots < 1) ? 1024 : nbShots;
}

exatn::TensorNetwork ExaTnPmpsVisitor::buildInitialNetwork(size_t in_nbQubits, bool in_createQubitTensors) const
{
    if (in_createQubitTensors)
    {
        for (int i = 0; i < in_nbQubits; ++i)
        {
            const std::string tensorName = "Q" + std::to_string(i);
            auto tensor = [&](){
                if (in_nbQubits == 1)
                {
                    assert(tensorName == "Q0");
                    return std::make_shared<exatn::Tensor>(tensorName, exatn::TensorShape{QUBIT_DIM, INITIAL_BOND_DIM});
                }
                if ((i == 0) || (i == (in_nbQubits - 1)))
                {
                    return std::make_shared<exatn::Tensor>(tensorName, exatn::TensorShape{QUBIT_DIM, INITIAL_BOND_DIM, INITIAL_KRAUS_DIM});
                }

                return std::make_shared<exatn::Tensor>(tensorName, exatn::TensorShape{QUBIT_DIM, INITIAL_BOND_DIM, INITIAL_KRAUS_DIM, INITIAL_BOND_DIM });
            }();

            const bool created = exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
            assert(created);
            const bool initialized = exatn::initTensorDataSync(tensorName, Q_ZERO_TENSOR_BODY(tensor->getVolume()));
            assert(initialized);
        }
    }

    const auto buildTensorMap = [](size_t in_nbQubits) {
        const std::vector<int> qubitTensorDim(in_nbQubits, QUBIT_DIM);
        const std::vector<int> ancTensorDim(in_nbQubits, INITIAL_KRAUS_DIM);
        // Root tensor dimension: 2 .. 2 (upper legs/system dimensions) 1 ... 1 (lower legs/anc dimension)
        std::vector<int> rootTensorDim;
        rootTensorDim.insert(rootTensorDim.end(), qubitTensorDim.begin(), qubitTensorDim.end());
        rootTensorDim.insert(rootTensorDim.end(), ancTensorDim.begin(), ancTensorDim.end());
        auto rootTensor = std::make_shared<exatn::Tensor>(ROOT_TENSOR_NAME, rootTensorDim);
        std::map<std::string, std::shared_ptr<exatn::Tensor>> tensorMap;
        tensorMap.emplace(ROOT_TENSOR_NAME, rootTensor);
        for (int i = 0; i < in_nbQubits; ++i)
        {
            const std::string qTensorName = "Q" + std::to_string(i);
            tensorMap.emplace(qTensorName, exatn::getTensor(qTensorName));
        }
        return tensorMap;
    };

    const std::string rootVarNameList = [](size_t in_nbQubits){
        std::string result = "(";
        // Primary qubit legs
        for (int i = 0; i < in_nbQubits; ++i)
        {
            result += ("i" + std::to_string(i) + ",");
        }

        // Anc legs:
        for (int i = 0; i < in_nbQubits; ++i)
        {
            result += ("k" + std::to_string(i) + ",");
        }
        assert(result.back() == ',');
        result.back() = ')';
        return result;
    }(in_nbQubits);

    const auto qubitTensorVarNameList = [](int in_qIdx, int in_nbQubits) -> std::string {
        if (in_nbQubits == 1)
        {
            assert(in_qIdx == 0);
            return "(i0,k0)";
        }
        if (in_qIdx == 0)
        {
            return "(i0,j0,k0)";
        } 
        if (in_qIdx == in_nbQubits - 1)
        {
            return "(i" + std::to_string(in_nbQubits - 1) + ",j" + std::to_string(in_nbQubits - 2) + ",k" + std::to_string(in_nbQubits - 1) + ")";
        }

        return "(i" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx-1) + ",k" + std::to_string(in_qIdx) + ",j" + std::to_string(in_qIdx) + ")";
    };

    const std::string pmpsString = [&]() {
        std::string result = ROOT_TENSOR_NAME + rootVarNameList + "=";
        for (int i = 0; i < in_nbQubits - 1; ++i)
        {
            result += ("Q" + std::to_string(i) + qubitTensorVarNameList(i, in_nbQubits) + "*");
        }
        result += ("Q" + std::to_string(m_buffer->size() - 1) + qubitTensorVarNameList(in_nbQubits - 1, in_nbQubits));
        return result;
    }();

    // std::cout << "Purified MPS: \n" << pmpsString << "\n";
    exatn::TensorNetwork purifiedMps("PMPS_Network", pmpsString, buildTensorMap(in_nbQubits));
    // purifiedMps.printIt();

    // Conjugate of the network:
    exatn::TensorNetwork conjugate(purifiedMps);
    conjugate.rename("Conjugate");
    conjugate.conjugate();
    // conjugate.printIt();
    // Pair all ancilla legs
    std::vector<std::pair<unsigned int, unsigned int>> pairings;
    for (size_t i = 0; i < in_nbQubits; ++i)
    {
        pairings.emplace_back(std::make_pair(i + in_nbQubits, i + in_nbQubits));
    }

    purifiedMps.appendTensorNetwork(std::move(conjugate), pairings);
    // std::cout << "Purified MPS:\n";
    // purifiedMps.printIt();
    return purifiedMps;
}

void ExaTnPmpsVisitor::finalize()
{
    exatn::sync();
    // If there are measurements:
    if (!m_measuredBits.empty())
    {
        // Retrieve the density matrix:
        const auto flattenedDm = calculateDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
        const std::vector<std::complex<double>> diagElems = [&](){
            const auto dim = 1ULL << m_buffer->size();
            std::vector<std::complex<double>> result(dim);
            for (size_t i = 0; i < dim; ++i)
            {
                result[i] = flattenedDm[ i * dim + i];
                assert(std::abs(result[i].imag()) < 1e-12);
            }
            return result;
        }();

        std::vector<std::pair<double, double>> flattenDmPairs;
        for (const auto &elem : flattenedDm) {
          flattenDmPairs.emplace_back(std::make_pair(elem.real(), elem.imag()));
        }
        m_buffer->addExtraInfo("density_matrix", flattenDmPairs);
        const auto sumDiag = [](const std::vector<std::complex<double>>& in_diag){
            double sum = 0.0;
            for (const auto& x : in_diag)
            {
                sum += x.real();
            }
            return sum;
        }(diagElems);
        // Validate trace = 1.0
        assert(std::abs(sumDiag - 1.0) < 1e-3);
        for (int i = 0; i < m_nbShots; ++i)
        {
            m_buffer->appendMeasurement(generateResultBitString(diagElems, m_measuredBits, m_buffer->size(), m_noiseConfig.get()));
        }

        m_measuredBits.clear();
    }

    for (size_t i = 0; i < m_buffer->size(); ++i)
    {
        const bool destroyed = exatn::destroyTensorSync("Q" + std::to_string(i));
        assert(destroyed);
    }
}

std::vector<KrausOp> ExaTnPmpsVisitor::convertNoiseChannel(
    const std::vector<NoiseChannelKraus> &in_channels) const {
  std::vector<KrausOp> result;
  const auto noiseUtils = xacc::getService<NoiseModelUtils>("default");
  for (const auto &channel : in_channels) {
    // Note: we don't support multi-qubit channels in the PMPS simulator.
    // Hence, just ignore it.
    if (channel.noise_qubits.size() == 1) {
      KrausOp newOp;
      newOp.qubit = channel.noise_qubits[0];
      newOp.mats = noiseUtils->krausToChoi(channel.mats);
      result.emplace_back(newOp);
    } else {
      static bool warnOnce = false;
      if (!warnOnce) {
        std::cout << "Multi-qubit channels are not supported. Ignoring.\n";
        warnOnce = true;
      }
    }
  }
  return result;
}

void ExaTnPmpsVisitor::applySingleQubitGate(xacc::quantum::Gate& in_gateInstruction)
{
    assert(in_gateInstruction.bits().size() == 1);
    const auto gateMatrix = getGateMatrix(in_gateInstruction);
    assert(gateMatrix.size() == 4);
    const std::string& gateTensorName = in_gateInstruction.name();
    // Create the tensor
    const bool created = exatn::createTensorSync(gateTensorName, exatn::TensorElementType::COMPLEX64, exatn::TensorShape{ 2, 2 });
    assert(created);
    // Init tensor body data
    const bool initialized = exatn::initTensorDataSync(gateTensorName, gateMatrix);
    assert(initialized);

    const size_t bitIdx = in_gateInstruction.bits()[0];
    const std::string qubitTensorName = "Q" + std::to_string(bitIdx);
    contractSingleQubitGateTensor(qubitTensorName, gateTensorName);
    const bool destroyed = exatn::destroyTensorSync(gateTensorName);
    assert(destroyed);

    // Apply noise (Kraus) Op
    if (m_noiseConfig)
    {
        const auto noiseOps = convertNoiseChannel(m_noiseConfig->getNoiseChannels(in_gateInstruction));
        for (const auto& op : noiseOps)
        {
            applyKrausOp(op);
        }
    }
}

void ExaTnPmpsVisitor::applyTwoQubitGate(xacc::quantum::Gate& in_gateInstruction)
{
    assert(in_gateInstruction.bits().size() == 2);
    // Must be a nearest-neighbor gate
    assert(std::abs((int)in_gateInstruction.bits()[0] - (int)in_gateInstruction.bits()[1]) == 1);
    const auto gateMatrix = getGateMatrix(in_gateInstruction);
    assert(gateMatrix.size() == 16);

    const std::string& gateTensorName = in_gateInstruction.name();
    // Create the tensor
    const bool created = exatn::createTensorSync(gateTensorName, exatn::TensorElementType::COMPLEX64, exatn::TensorShape{ 2, 2, 2, 2 });
    assert(created);
    // Init tensor body data
    const bool initialized = exatn::initTensorDataSync(gateTensorName, gateMatrix);
    assert(initialized);
    contractTwoQubitGateTensor(m_pmpsTensorNetwork, in_gateInstruction.bits(), gateTensorName);
    const bool destroyed = exatn::destroyTensorSync(gateTensorName);
    assert(destroyed);
    m_pmpsTensorNetwork = buildInitialNetwork(m_buffer->size(), false);
    // Truncate SVD:
    const std::string q1TensorName = "Q" + std::to_string(in_gateInstruction.bits()[0]);
    const std::string q2TensorName = "Q" + std::to_string(in_gateInstruction.bits()[1]);
    truncateSvdTensors(q1TensorName, q2TensorName);
    m_pmpsTensorNetwork = buildInitialNetwork(m_buffer->size(), false);

    // Apply noise (Kraus) Op
    if (m_noiseConfig)
    {
        const auto noiseOps = convertNoiseChannel(m_noiseConfig->getNoiseChannels(in_gateInstruction));
        for (const auto& op : noiseOps)
        {
            applyKrausOp(op);
        }
    }
}

void ExaTnPmpsVisitor::applyKrausOp(const KrausOp& in_op)
{
    static size_t krausTensorCounter = 0;
    ++krausTensorCounter;

    auto krausTensor = std::make_shared<exatn::Tensor>("__KRAUS__" + std::to_string(krausTensorCounter), exatn::TensorShape{2, 2, 2, 2});
    const bool created = exatn::createTensorSync(krausTensor, exatn::TensorElementType::COMPLEX64);
    assert(created);
    std::vector<std::complex<double>> krausVec;
    krausVec.reserve(in_op.mats.size() * in_op.mats.size());
    for (const auto &row : in_op.mats)
    {
      for (const auto &entry : row)
      {
        krausVec.emplace_back(entry);
      }
    }
    const bool initialized = exatn::initTensorDataSync(krausTensor->getName(), krausVec);
    assert(initialized);
    applyLocalKrausOp(in_op.qubit, krausTensor->getName());
}

void ExaTnPmpsVisitor::applyLocalKrausOp(size_t in_siteId, const std::string& in_opTensorName)
{
   // std::cout << "Apply noise op " << in_opTensorName << "\n";
   auto opTensor = exatn::getTensor(in_opTensorName);
    // Must be a 4-leg tensor
    assert(opTensor->getRank() == 4);
    const auto qubitTensorName = "Q" + std::to_string(in_siteId);
    // Step 1: Merge Q - Q-dagger to form a 2-leg tensor
    std::string mergeContractionPattern;
    const auto mergedTensorId = m_pmpsTensorNetwork.getMaxTensorId() + 1;
    // m_pmpsTensorNetwork.printIt();
    const auto tensorId = in_siteId + 1;
    const auto conjTensorId = m_buffer->size() + in_siteId + 1;
    m_pmpsTensorNetwork.mergeTensors(tensorId,  conjTensorId, mergedTensorId, &mergeContractionPattern);
    mergeContractionPattern.replace(mergeContractionPattern.find("L"), 1, qubitTensorName);
    mergeContractionPattern.replace(mergeContractionPattern.find("R"), 1, qubitTensorName);
    auto mergedTensor = m_pmpsTensorNetwork.getTensor(mergedTensorId);
    mergedTensor->rename("D");
    // std::cout << mergeContractionPattern << "\n";
    const bool mergedTensorCreated = exatn::createTensorSync(mergedTensor, exatn::TensorElementType::COMPLEX64);
    assert(mergedTensorCreated);
    const bool mergedTensorInitialized = exatn::initTensorSync(mergedTensor->getName(), 0.0);
    assert(mergedTensorInitialized);
    const bool mergedContractionOk = exatn::contractTensorsSync(mergeContractionPattern, 1.0);
    assert(mergedContractionOk);
    // const auto dmTensorData = getTensorData("D");
    // std::cout << "Density Matrix: \n";
    // for (const auto& elem : dmTensorData)
    // {
    //     std::cout << elem << "\n";
    // }
    // Step 2: Append Kraus tensor as a 2-qubit gate
    static size_t counter = 0;
    const std::string RESULT_TENSOR_NAME = "Result_Kraus_" + std::to_string(counter++);
    const std::string patternStr = [&]() -> std::string {
        if (mergedTensor->getRank() == 2)
        {
            assert(m_buffer->size() == 1);
            return RESULT_TENSOR_NAME + "(u0,u1)=D(c0,c1)*" + opTensor->getName() + "(u0,c0,u1,c1)";
        }
        else if (mergedTensor->getRank() == 4)
        {
            return RESULT_TENSOR_NAME + "(u0,u1,u2,u3)=D(c0,u1,c1,u3)*" + opTensor->getName() + "(u0,c0,u2,c1)";
        }
        else if (mergedTensor->getRank() == 6)
        {
            return RESULT_TENSOR_NAME + "(u0,u1,u2,u3,u4,u5)=D(c0,u1,u2,c1,u4,u5)*" + opTensor->getName() + "(u0,c0,u3,c1)";
        }
        else
        {
            xacc::error("Internal error: " + mergeContractionPattern);
            return "";
        }
    }();

    // Result tensor always has the same shape as the *merged* qubit tensor
    const bool resultTensorCreated = exatn::createTensorSync(RESULT_TENSOR_NAME,
                                                            exatn::TensorElementType::COMPLEX64,
                                                            mergedTensor->getShape());
    assert(resultTensorCreated);
    const bool resultTensorInitialized = exatn::initTensorSync(RESULT_TENSOR_NAME, 0.0);
    assert(resultTensorInitialized);

    // Step 3: Contract the tensor network to form a new tensor
    const bool contractionOk = exatn::contractTensorsSync(patternStr, 1.0);
    assert(contractionOk);

    // const auto tensorData = getTensorData(RESULT_TENSOR_NAME);

    // std::cout << "Data: \n";
    // for (const auto& elem : tensorData)
    // {
    //     std::cout << elem << "\n";
    // }


    // Step 4: SVD back
    auto qubitTensor = exatn::getTensor(qubitTensorName);
    // qubitTensor->printIt();
    auto tensorShape = qubitTensor->getShape();
    if (tensorShape.getRank() == 2)
    {
        const auto oldKrausBondDim = tensorShape.getDimExtent(1);
        const auto newKrausBondDim = std::min(tensorShape.getVolume() / oldKrausBondDim,  oldKrausBondDim * 2);
        tensorShape.resetDimension(1, newKrausBondDim);
    }
    else
    {
        const auto oldKrausBondDim = tensorShape.getDimExtent(2);
        const auto newKrausBondDim = std::min(tensorShape.getVolume() / oldKrausBondDim,  oldKrausBondDim * 2);
        tensorShape.resetDimension(2, newKrausBondDim);
    }
    // std::cout << "\nNew shape: ";
    // tensorShape.printIt();
    // std::cout << "\n";

    bool destroyed = exatn::destroyTensorSync(qubitTensorName);
    assert(destroyed);

    auto svdTensor1 = std::make_shared<exatn::Tensor>(qubitTensorName, tensorShape);
    auto svdTensor2 = std::make_shared<exatn::Tensor>("__SVD__", tensorShape);
    bool created = exatn::createTensorSync(svdTensor1, exatn::TensorElementType::COMPLEX64);
    assert(created);
    created = exatn::createTensorSync(svdTensor2, exatn::TensorElementType::COMPLEX64);
    assert(created);

    const std::string svdPattern = [&]() -> std::string {
        if (mergedTensor->getRank() == 2)
        {
            assert(m_buffer->size() == 1);
            return RESULT_TENSOR_NAME + "(u0,u1)=" + svdTensor1->getName() + "(u0,c0)*" + svdTensor2->getName()+ "(u1,c0)";
        }
        else if (mergedTensor->getRank() == 4)
        {
            return RESULT_TENSOR_NAME + "(u0,u1,u2,u3)=" + svdTensor1->getName() + "(u0,u1,c0)*" + svdTensor2->getName() + "(u2,u3,c0)";
        }
        else if (mergedTensor->getRank() == 6)
        {
            return RESULT_TENSOR_NAME + "(u0,u1,u2,u3,u4,u5)=" + svdTensor1->getName() + "(u0,u1,c0,u2)*" + svdTensor2->getName() + "(u3,u4,c0,u5)";
        }
        else
        {
            xacc::error("Internal error: " + mergeContractionPattern);
            return "";
        }
    }();
    // std::cout << "SVD: " << svdPattern << "\n";
    // svdTensor1->printIt();
    // svdTensor2->printIt();
    // exatn::getTensor(RESULT_TENSOR_NAME)->printIt();
    // exatn::sync(RESULT_TENSOR_NAME);
    // exatn::sync(svdTensor1->getName());
    // exatn::sync(svdTensor2->getName());
    const bool svdOk = exatn::decomposeTensorSVDLRSync(svdPattern);
    assert(svdOk);
    // const auto tensorData1 = getTensorData(svdTensor1->getName());
    // std::cout << "Data1: \n";
    // for (const auto& elem : tensorData1)
    // {
    //     std::cout << elem << "\n";
    // }
    // const auto tensorData2 = getTensorData(svdTensor2->getName());
    // std::cout << "Data2: \n";
    // for (const auto& elem : tensorData2)
    // {
    //     std::cout << elem << "\n";
    // }
    destroyed = exatn::destroyTensorSync(svdTensor2->getName());
    assert(destroyed);
    destroyed = exatn::destroyTensorSync(mergedTensor->getName());
    assert(destroyed);
    destroyed = exatn::destroyTensorSync(RESULT_TENSOR_NAME);
    assert(destroyed);
    destroyed = exatn::destroyTensorSync(in_opTensorName);
    assert(destroyed);
    const auto krausBondId = (tensorShape.getRank() == 2) ? 1 : 2;
    std::vector<double> krausBondNorm;
    const bool normOk = exatn::computePartialNormsSync(qubitTensorName, krausBondId, krausBondNorm);
    assert(normOk);
    // std::cout << qubitTensorName << ":\n";
    // for (const auto& el: krausBondNorm)
    // {
    //     std::cout << el << "\n";
    // }
    const auto findCutoffDim = [&](double in_eps) -> int {
        for (int i = 0; i < krausBondNorm.size(); ++i)
        {
            if (krausBondNorm[i] < in_eps)
            {
                return i;
            }
        }
        return krausBondNorm.size();
    };

    const auto newBondDim = findCutoffDim(1e-9);
    assert(newBondDim > 0);
    // std::cout << "New dim = " << newBondDim << "\n";
    if (newBondDim < krausBondNorm.size())
    {
        auto oldShape = svdTensor1->getDimExtents();
        oldShape[krausBondId] = newBondDim;
        const std::string newSliceTensorName = svdTensor1->getName() + "_" + std::to_string(svdTensor1->getTensorHash());
        const bool sliceTensorCreated = exatn::createTensorSync(newSliceTensorName, exatn::TensorElementType::COMPLEX64, oldShape);
        assert(sliceTensorCreated);

        // Take the slices:
        const bool sliceOk = exatn::extractTensorSliceSync(svdTensor1->getName(), newSliceTensorName);
        assert(sliceOk);

        // Destroy the original tensor:
        const bool origDestroyed = exatn::destroyTensorSync(svdTensor1->getName());
        assert(origDestroyed);

        // Rename new tensor to the old name
        const auto renameNumericTensor = [](const std::string& oldTensorName, const std::string& newTensorName){
            auto tensor = exatn::getTensor(oldTensorName);
            assert(tensor);
            auto talsh_tensor = exatn::getLocalTensor(oldTensorName);
            assert(talsh_tensor);
            const std::complex<double>* body_ptr;
            const bool access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr);
            assert(access_granted);
            std::vector<std::complex<double>> newData;
            newData.assign(body_ptr, body_ptr + talsh_tensor->getVolume());
            const bool newTensorCreated = exatn::createTensorSync(newTensorName, exatn::TensorElementType::COMPLEX64, tensor->getShape());
            assert(newTensorCreated);
            const bool newTensorInitialized = exatn::initTensorDataSync(newTensorName, newData);
            assert(newTensorInitialized);
            // Destroy the two original tensor:
            const bool tensorDestroyed = exatn::destroyTensorSync(oldTensorName);
            assert(tensorDestroyed);
        };

        renameNumericTensor(newSliceTensorName, qubitTensorName);
        exatn::sync();
        //std::cout << "Change bond dim of " << qubitTensorName << " = " << newBondDim << "\n";
    }

    m_pmpsTensorNetwork = buildInitialNetwork(m_buffer->size(), false);
    //m_pmpsTensorNetwork.printIt();
}

void ExaTnPmpsVisitor::visit(Identity& in_IdentityGate)
{
    applySingleQubitGate(in_IdentityGate);
}

void ExaTnPmpsVisitor::visit(Hadamard& in_HadamardGate)
{
    applySingleQubitGate(in_HadamardGate);
    // DEBUG:
    // std::cout << "Apply: " << in_HadamardGate.toString() << "\n";
    // printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
}

void ExaTnPmpsVisitor::visit(X& in_XGate)
{
    applySingleQubitGate(in_XGate);
    // DEBUG:
    // std::cout << "Apply: " << in_XGate.toString() << "\n";
    // printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
}

void ExaTnPmpsVisitor::visit(Y& in_YGate)
{
    applySingleQubitGate(in_YGate);
}

void ExaTnPmpsVisitor::visit(Z& in_ZGate)
{
    applySingleQubitGate(in_ZGate);
}

void ExaTnPmpsVisitor::visit(Rx& in_RxGate)
{
    applySingleQubitGate(in_RxGate);
}

void ExaTnPmpsVisitor::visit(Ry& in_RyGate)
{
    applySingleQubitGate(in_RyGate);
}

void ExaTnPmpsVisitor::visit(Rz& in_RzGate)
{
    applySingleQubitGate(in_RzGate);
}

void ExaTnPmpsVisitor::visit(T& in_TGate)
{
    applySingleQubitGate(in_TGate);
}

void ExaTnPmpsVisitor::visit(Tdg& in_TdgGate)
{
    applySingleQubitGate(in_TdgGate);
}

// others
void ExaTnPmpsVisitor::visit(Measure& in_MeasureGate)
{
    m_measuredBits.emplace_back(in_MeasureGate.bits()[0]);
}

void ExaTnPmpsVisitor::visit(U& in_UGate)
{
    applySingleQubitGate(in_UGate);
}

// two-qubit gates:
void ExaTnPmpsVisitor::visit(CNOT& in_CNOTGate)
{
    applyTwoQubitGate(in_CNOTGate);
    // DEBUG:
    // std::cout << "Apply: " << in_CNOTGate.toString() << "\n";
    // printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
}

void ExaTnPmpsVisitor::visit(Swap& in_SwapGate)
{
    applyTwoQubitGate(in_SwapGate);
}

void ExaTnPmpsVisitor::visit(CZ& in_CZGate)
{
    applyTwoQubitGate(in_CZGate);
}

void ExaTnPmpsVisitor::visit(CPhase& in_CPhaseGate)
{
    applyTwoQubitGate(in_CPhaseGate);
}

void ExaTnPmpsVisitor::visit(iSwap& in_iSwapGate)
{
    applyTwoQubitGate(in_iSwapGate);
}

void ExaTnPmpsVisitor::visit(fSim& in_fsimGate)
{
    applyTwoQubitGate(in_fsimGate);
}

const double ExaTnPmpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function)
{
    return 0.0;
}

void ExaTnPmpsVisitor::truncateSvdTensors(const std::string& in_leftTensorName, const std::string& in_rightTensorName, double in_eps)
{
    int lhsTensorId = -1;
    int rhsTensorId = -1;

    for (auto iter = m_pmpsTensorNetwork.cbegin(); iter != m_pmpsTensorNetwork.cend(); ++iter)
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
    auto lhsConnections = m_pmpsTensorNetwork.getTensorConnections(lhsTensorId);
    auto rhsConnections = m_pmpsTensorNetwork.getTensorConnections(rhsTensorId);
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

    const auto newBondDim = findCutoffDim();
    assert(newBondDim > 0);
    if (newBondDim < bondDim)
    {
        auto leftShape = lhsTensor->getDimExtents();
        auto rightShape = rhsTensor->getDimExtents();
        leftShape[lhsBondId] = newBondDim;
        rightShape[rhsBondId] = newBondDim;

        // Create two new tensors:
        const std::string newLhsTensorName = in_leftTensorName + "_" + std::to_string(lhsTensor->getTensorHash());
        const bool newLhsCreated = exatn::createTensorSync(newLhsTensorName, exatn::TensorElementType::COMPLEX64, leftShape);
        assert(newLhsCreated);

        const std::string newRhsTensorName = in_rightTensorName + "_" + std::to_string(rhsTensor->getTensorHash());
        const bool newRhsCreated = exatn::createTensorSync(newRhsTensorName, exatn::TensorElementType::COMPLEX64, rightShape);
        assert(newRhsCreated);

        // Take the slices:
        const bool lhsSliceOk = exatn::extractTensorSliceSync(in_leftTensorName, newLhsTensorName);
        assert(lhsSliceOk);
        const bool rhsSliceOk = exatn::extractTensorSliceSync(in_rightTensorName, newRhsTensorName);
        assert(rhsSliceOk);

        // Destroy the two original tensors:
        const bool lhsDestroyed = exatn::destroyTensorSync(in_leftTensorName);
        assert(lhsDestroyed);
        const bool rhsDestroyed = exatn::destroyTensorSync(in_rightTensorName);
        assert(rhsDestroyed);

        // Rename new tensors to the old name
        const auto renameNumericTensor = [](const std::string& oldTensorName, const std::string& newTensorName){
            auto tensor = exatn::getTensor(oldTensorName);
            assert(tensor);
            auto talsh_tensor = exatn::getLocalTensor(oldTensorName);
            assert(talsh_tensor);
            const std::complex<double>* body_ptr;
            const bool access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr);
            assert(access_granted);
            std::vector<std::complex<double>> newData;
            newData.assign(body_ptr, body_ptr + talsh_tensor->getVolume());
            const bool newTensorCreated = exatn::createTensorSync(newTensorName, exatn::TensorElementType::COMPLEX64, tensor->getShape());
            assert(newTensorCreated);
            const bool newTensorInitialized = exatn::initTensorDataSync(newTensorName, newData);
            assert(newTensorInitialized);
            // Destroy the two original tensor:
            const bool tensorDestroyed = exatn::destroyTensorSync(oldTensorName);
            assert(tensorDestroyed);
        };

        renameNumericTensor(newLhsTensorName, in_leftTensorName);
        renameNumericTensor(newRhsTensorName, in_rightTensorName);
        exatn::sync();

        // Debug:
        // std::cout << "[DEBUG] Bond dim (" << in_leftTensorName << ", " << in_rightTensorName << "): " << bondDim << " -> " << newBondDim << "\n";
    }
}
}
