#include "ExaTnMpsVisitor.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "ExatnUtils.hpp"
#include "utils/GateMatrixAlgebra.hpp"

#ifdef TNQVM_EXATN_USES_MKL_BLAS
#include <dlfcn.h>
#endif

namespace {
const std::vector<std::complex<double>> Q_ZERO_TENSOR_BODY{{1.0, 0.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> Q_ONE_TENSOR_BODY{{0.0, 0.0}, {1.0, 0.0}};  
const std::string ROOT_TENSOR_NAME = "Root";

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
}
namespace tnqvm {
ExatnMpsVisitor::ExatnMpsVisitor():
    m_aggrerator(this),
    // By default, don't enable aggreration, i.e. simply running gate-by-gate first. 
    // TODO: implement aggreation processing with ExaTN.
    m_aggrerateEnabled(false)
{
    // TODO
}

void ExatnMpsVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) 
{ 
    // Check if we have any specific config for the gate aggregator
    if (m_aggrerateEnabled && options.keyExists<int>("agg-width"))
    {
        const int aggregatorWidth = options.get<int>("agg-width");
        AggreratorConfigs configs(aggregatorWidth);
        TensorAggrerator newAggr(configs, this);
        m_aggrerator = newAggr;
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

        // If exaTN has not been initialized, do it now.
        exatn::initialize();
    }

    m_buffer = std::move(buffer);
    m_qubitTensorNames.clear();
    m_tensorIdCounter = 0;
    m_aggregatedGroupCounter = 0;
    m_registeredGateTensors.clear();
    m_measureQubits.clear();
    m_shotCount = nbShots;
    // Build MPS tensor network
    const std::vector<int> qubitTensorDim(m_buffer->size(), 2);
    auto& networkBuildFactory = *(exatn::numerics::NetworkBuildFactory::get());
    auto builder = networkBuildFactory.createNetworkBuilderShared("MPS");
    // Initially, all bond dimensions are 1
    const bool success = builder->setParameter("max_bond_dim", 1); 
    assert(success);
    const bool rootCreated = exatn::createTensorSync(ROOT_TENSOR_NAME, exatn::TensorElementType::COMPLEX64, qubitTensorDim);
    assert(rootCreated);

    const bool rootInitialized = exatn::initTensorSync(ROOT_TENSOR_NAME, 0.0); 
    assert(rootInitialized);

    m_rootTensor = exatn::getTensor(ROOT_TENSOR_NAME);
    m_tensorNetwork = exatn::makeSharedTensorNetwork("Qubit Register", m_rootTensor, *builder);
    
    for (auto iter = m_tensorNetwork->cbegin(); iter != m_tensorNetwork->cend(); ++iter) 
    {
        const auto& tensorName = iter->second.getTensor()->getName();
        if (tensorName != ROOT_TENSOR_NAME)
        {
            const auto renameTensor = [](const std::string& in_name){
                const auto idxStr = in_name.substr(1);
                const int idx = std::stoi(idxStr);
                return "Q" + std::to_string(idx - 1);
                return in_name;
            };

            auto tensor = iter->second.getTensor();
            const auto newTensorName = renameTensor(tensorName);
            iter->second.getTensor()->rename(newTensorName);
            const bool created = exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
            assert(created);
            const bool initialized = exatn::initTensorDataSync(newTensorName, Q_ZERO_TENSOR_BODY);
            assert(initialized);
        }
    }
    // DEBUG:
    // printStateVec();
}

void ExatnMpsVisitor::printStateVec()
{
    std::cout << "MPS Tensor Network: \n";
    m_tensorNetwork->printIt();
    const bool evaluated = exatn::evaluateSync(*m_tensorNetwork);    
    assert(evaluated);   
    std::cout << "State Vector: \n";

    auto talsh_tensor = exatn::getLocalTensor(ROOT_TENSOR_NAME); 
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
    if (m_aggrerateEnabled)
    {
        m_aggrerator.flushAll();    
        evaluateTensorNetwork(*m_tensorNetwork, m_stateVec);
    }

    if (!m_measureQubits.empty())
    {
        addMeasureBitStringProbability(m_measureQubits, getTensorData(ROOT_TENSOR_NAME), m_shotCount);
    }

    for (int i = 0; i < m_buffer->size(); ++i)
    {
        const bool qTensorDestroyed = exatn::destroyTensor("Q" + std::to_string(i));
        assert(qTensorDestroyed);
    }
    const bool rootDestroyed = exatn::destroyTensor(ROOT_TENSOR_NAME);
    assert(rootDestroyed);
}

void ExatnMpsVisitor::visit(Identity& in_IdentityGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_IdentityGate);
    }
    else
    {
        // Skip Identity gate
    }
}

void ExatnMpsVisitor::visit(Hadamard& in_HadamardGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_HadamardGate);
    }
    else
    {
        applyGate(in_HadamardGate);
    }
}

void ExatnMpsVisitor::visit(X& in_XGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_XGate); 
    }
    else
    {
        applyGate(in_XGate);
    }
}

void ExatnMpsVisitor::visit(Y& in_YGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_YGate); 
    }
    else
    {
        applyGate(in_YGate);
    }
}

void ExatnMpsVisitor::visit(Z& in_ZGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_ZGate); 
    }
    else
    {
        applyGate(in_ZGate);
    }
}

void ExatnMpsVisitor::visit(Rx& in_RxGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_RxGate); 
    }
    else
    {
        applyGate(in_RxGate);
    }
}

void ExatnMpsVisitor::visit(Ry& in_RyGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_RyGate); 
    }
    else
    {
        applyGate(in_RyGate);
    }
}

void ExatnMpsVisitor::visit(Rz& in_RzGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_RzGate); 
    }
    else
    {
        applyGate(in_RzGate);
    }
}

void ExatnMpsVisitor::visit(T& in_TGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_TGate); 
    }
    else
    {
        applyGate(in_TGate);
    }
}

void ExatnMpsVisitor::visit(Tdg& in_TdgGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_TdgGate); 
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
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_UGate); 
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
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_CNOTGate); 
    }
    else
    {
        applyGate(in_CNOTGate);
    }
}

void ExatnMpsVisitor::visit(Swap& in_SwapGate) 
{ 
    if (m_aggrerateEnabled)
    {   
        m_aggrerator.addGate(&in_SwapGate); 
    }
    else
    {
        applyGate(in_SwapGate);
    }
}

void ExatnMpsVisitor::visit(CZ& in_CZGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_CZGate); 
    }
    else
    {
        applyGate(in_CZGate);
    }
}

void ExatnMpsVisitor::visit(CPhase& in_CPhaseGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_CPhaseGate); 
    }
    else
    {
        applyGate(in_CPhaseGate);
    }
}

const double ExatnMpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
{ 
    // TODO
    return 0.0;
}

void ExatnMpsVisitor::onFlush(const AggreratedGroup& in_group)
{
    if (!m_aggrerateEnabled)
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
        const std::string& uniqueGateName = gateTensor.uniqueName;
        if (m_registeredGateTensors.find(uniqueGateName) == m_registeredGateTensors.end())
        {
            m_registeredGateTensors.emplace(uniqueGateName);
            // Create the tensor
            const bool created = exatn::createTensor(
                uniqueGateName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
            assert(created);
            // Init tensor body data
            exatn::initTensorData(uniqueGateName, gateTensor.tensorData);
            const bool registered = exatn::registerTensorIsometry(uniqueGateName, gateTensor.tensorIsometry.first, gateTensor.tensorIsometry.second);
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
            exatn::getTensor(uniqueGateName),
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
    // Single qubit only atm
    assert(in_gateInstruction.bits().size() == 1);
    const auto gateTensor = GateTensorConstructor::getGateTensor(in_gateInstruction);
    const std::string& uniqueGateName = gateTensor.uniqueName;
    // Create the tensor
    const bool created = exatn::createTensorSync(uniqueGateName, exatn::TensorElementType::COMPLEX64, gateTensor.tensorShape);
    assert(created);
    // Init tensor body data
    const bool initialized = exatn::initTensorDataSync(uniqueGateName, gateTensor.tensorData);
    assert(initialized);
    m_tensorNetwork->printIt();
    // Contract gate tensor to the qubit tensor
    const auto contractGateTensor = [](int in_qIdx, const std::string& in_gateTensorName){
        // Pattern: 
        // (1) Boundary qubits (2 legs): Result(a, b) = Qi(a, i) * G (i, b)
        // (2) Middle qubits (3 legs): Result(a, b, c) = Qi(a, b, i) * G (i, c)
        const std::string qubitTensorName = "Q" + std::to_string(in_qIdx); 
        auto qubitTensor =  exatn::getTensor(qubitTensorName);
        assert(qubitTensor->getRank() == 2 || qubitTensor->getRank() == 3);
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

        exatn::numericalServer->transformTensorSync(qubitTensorName, std::make_shared<ExatnMpsVisitor::ExaTnTensorFunctor>(updateFunc));
        const bool resultTensorDestroyed = exatn::destroyTensor(RESULT_TENSOR_NAME);
        assert(resultTensorDestroyed);
    };
    contractGateTensor(in_gateInstruction.bits()[0], uniqueGateName);
    // DEBUG:
    printStateVec();

    const bool destroyed = exatn::destroyTensor(uniqueGateName);
    assert(destroyed);
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
    exatn::sync();
    // DEBUG: 
    // for (const auto& elem : out_stateVec)
    // {
    //     std::cout << elem.real() <<  " + i " <<  elem.imag() << "\n";
    // }
}

void ExatnMpsVisitor::addMeasureBitStringProbability(const std::vector<size_t>& in_bits, const std::vector<std::complex<double>>& in_stateVec, int in_shotCount)
{
    for (const auto& elem : in_stateVec)
    {
        std::cout << elem.real() <<  " + i " <<  elem.imag() << "\n";
    }

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
}
