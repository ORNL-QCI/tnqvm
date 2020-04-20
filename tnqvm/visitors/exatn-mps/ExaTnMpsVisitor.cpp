#include "ExaTnMpsVisitor.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "talshxx.hpp"
#include "ExatnUtils.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include <map>
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
    const std::vector<int> qubitTensorDim(m_buffer->size(), 2);
    const bool rootCreated = exatn::createTensorSync(ROOT_TENSOR_NAME, exatn::TensorElementType::COMPLEX64, qubitTensorDim);
    assert(rootCreated);

    const bool rootInitialized = exatn::initTensorSync(ROOT_TENSOR_NAME, 0.0); 
    assert(rootInitialized);

    m_rootTensor = exatn::getTensor(ROOT_TENSOR_NAME);

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
                  {"Root", exatn::getTensor(ROOT_TENSOR_NAME)}, {"T1",t1}, {"T2",t2}});
    }
    else if (m_buffer->size() == 1)
    {
        auto t1 = std::make_shared<exatn::Tensor>("T1", exatn::TensorShape{2});
        m_tensorNetwork = std::make_shared<exatn::TensorNetwork>("Qubit Register",
                 "Root(i0)+=T1(i0)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"Root", exatn::getTensor(ROOT_TENSOR_NAME)}, {"T1",t1}});
    }
    
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

     
    // Note: currently, we simulate measurement by full tensor contraction (to get the state vector)
    // we can also implement repetitive bit count sampling 
    // (will require many contractions but don't require large memory allocation) 
    const bool evaledOk = exatn::evaluateSync(*m_tensorNetwork);
    assert(evaledOk); 
    // printStateVec();

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
        if (in_SwapGate.bits()[0] < in_SwapGate.bits()[1])
        {
            in_SwapGate.setBits({in_SwapGate.bits()[1], in_SwapGate.bits()[0]});
        }

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
    if (in_gateInstruction.bits().size() == 2)
    {
        return applyTwoQubitGate(in_gateInstruction);
    }

    // Single qubit only in this path
    assert(in_gateInstruction.bits().size() == 1);
    const auto gateTensor = GateTensorConstructor::getGateTensor(in_gateInstruction);
    const std::string& uniqueGateTensorName = gateTensor.uniqueName;
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
    contractGateTensor(in_gateInstruction.bits()[0], uniqueGateTensorName);

    // DEBUG:
    // printStateVec();

    const bool destroyed = exatn::destroyTensor(uniqueGateTensorName);
    assert(destroyed);
}

void ExatnMpsVisitor::applyTwoQubitGate(xacc::Instruction& in_gateInstruction)
{
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
    const std::string& uniqueGateTensorName = gateTensor.uniqueName;
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
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(i,j,c)*" + uniqueGateTensorName + "(j,i,a,b)";
            }
            else
            {
                patternStr = RESULT_TENSOR_NAME + "(a,b,c)=" + mergedTensor->getName() + "(a,i,j)*" + uniqueGateTensorName + "(j,i,b,c)";
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
            patternStr = RESULT_TENSOR_NAME + "(a,b,c,d)=" + mergedTensor->getName() + "(a,i,j,d)*" + uniqueGateTensorName + "(j,i,b,c)";
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
            patternStr = RESULT_TENSOR_NAME + "(a,b)=" + mergedTensor->getName() + "(i,j)*" + uniqueGateTensorName + "(j,i,a,b)";
        }
        else
        {
            patternStr = RESULT_TENSOR_NAME + "(a,b)=" + mergedTensor->getName() + "(i,j)*" + uniqueGateTensorName + "(i,j,a,b)";
        }
    }
    
    assert(!patternStr.empty());
    // std::cout << "Gate contraction pattern: " << patternStr << "\n";

    const bool gateContractionOk = exatn::contractTensorsSync(patternStr, 1.0);
    assert(gateContractionOk);
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
    exatn::sync();

    // Create two new tensors:
    const bool q1Created = exatn::createTensorSync(q1TensorName, exatn::TensorElementType::COMPLEX64, q1Shape);
    assert(q1Created);
    const bool q1Initialized = exatn::initTensorSync(q1TensorName, 0.0);
    assert(q1Initialized);
    
    const bool q2Created = exatn::createTensorSync(q2TensorName, exatn::TensorElementType::COMPLEX64, q2Shape);
    assert(q2Created);
    const bool q2Initialized = exatn::initTensorSync(q2TensorName, 0.0);
    assert(q2Initialized);

    // SVD decomposition using the same pattern that was used to merge two tensors
    const bool svdOk = exatn::decomposeTensorSVDLRSync(mergeContractionPattern);
    assert(svdOk);
        
    const bool mergedTensorDestroyed = exatn::destroyTensor(mergedTensor->getName());
    assert(mergedTensorDestroyed);
    
    std::map<std::string, std::shared_ptr<exatn::Tensor>> tensorMap;
    tensorMap.emplace(ROOT_TENSOR_NAME, exatn::getTensor(ROOT_TENSOR_NAME));
    for (int i = 0; i < m_buffer->size(); ++i)
    {
        const std::string qTensorName = "Q" + std::to_string(i);
        tensorMap.emplace(qTensorName, exatn::getTensor(qTensorName));
    }
    
    const bool rootReInitialized = exatn::initTensorSync("Root", 0.0); 
    assert(rootReInitialized);
    
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
    m_tensorNetwork = std::make_shared<exatn::TensorNetwork>(m_tensorNetwork->getName(), mpsString, tensorMap);  
    const bool evaledOk = exatn::evaluateSync(*m_tensorNetwork);
    assert(evaledOk);    
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
