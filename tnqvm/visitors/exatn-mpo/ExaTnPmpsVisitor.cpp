#include "ExaTnPmpsVisitor.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include "base/Gates.hpp"
#include "Kraus.hpp"

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
            case CommonGates::T: return GetGateMatrix<CommonGates::T>();
            case CommonGates::Tdg: return GetGateMatrix<CommonGates::Tdg>();
            case CommonGates::CNOT: return GetGateMatrix<CommonGates::CNOT>();
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
    std::cout << "Pattern string: " << patternStr << "\n";
    
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
    std::cout << "Merged: " << mergeContractionPattern << "\n";
    
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
        qubitsMergedTensor->printIt();
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
        std::cout << "SVD: " << svdPattern << "\n";
        auto q1Tensor = exatn::getTensor(q1TensorName);
        auto q2Tensor = exatn::getTensor(q2TensorName);
        const auto bondIds = getBondLegId(in_tensorNetwork, q1TensorName, q2TensorName);
        auto q1Shape = q1Tensor->getDimExtents();
        auto q2Shape = q2Tensor->getDimExtents();
        q1Shape[bondIds.first] = 2 * q1Shape[bondIds.first];
        q2Shape[bondIds.second] = 2 * q2Shape[bondIds.second];

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
        exatn::resetRuntimeLoggingLevel(xacc::verbose ? xacc::getLoggingLevel() : 0);
        xacc::subscribeLoggingLevel([](int level) {
            exatn::resetRuntimeLoggingLevel(xacc::verbose ? level : 0);
        });
    }

    m_buffer = buffer;
    m_pmpsTensorNetwork = buildInitialNetwork(buffer->size(), true);
    // DEBUG
    printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
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

    std::cout << "Purified MPS: \n" << pmpsString << "\n";
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
    std::cout << "Purified MPS:\n";
    purifiedMps.printIt();

    return purifiedMps;
}

void ExaTnPmpsVisitor::finalize() 
{ 
    // TODO
}

void ExaTnPmpsVisitor::applySingleQubitGate(xacc::Instruction& in_gateInstruction)
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
}

void ExaTnPmpsVisitor::applyTwoQubitGate(xacc::Instruction& in_gateInstruction)
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
}

void applyLocalKrausOp(size_t in_siteId, const std::string& in_opTensorName)
{
    auto opTensor = exatn::getTensor(in_opTensorName);
    // Must be a 3-leg tensor
    assert(opTensor->getRank() == 3);
}

void ExaTnPmpsVisitor::visit(Identity& in_IdentityGate) 
{ 
    
}

void ExaTnPmpsVisitor::visit(Hadamard& in_HadamardGate) 
{ 
   applySingleQubitGate(in_HadamardGate);
   // DEBUG:
   std::cout << "Apply: " << in_HadamardGate.toString() << "\n";
   printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
}

void ExaTnPmpsVisitor::visit(X& in_XGate) 
{ 
   applySingleQubitGate(in_XGate);
   // DEBUG:
   std::cout << "Apply: " << in_XGate.toString() << "\n";
   printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
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
    applyTwoQubitGate(in_CNOTGate);
    // DEBUG:
   std::cout << "Apply: " << in_CNOTGate.toString() << "\n";
   printDensityMatrix(m_pmpsTensorNetwork, m_buffer->size());
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
