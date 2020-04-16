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
    exatn::numerics::TensorNetwork emptyTensorNet("TensorNetwork");
    m_tensorNetwork = emptyTensorNet;
    m_aggregatedGroupCounter = 0;
    m_registeredGateTensors.clear();
    m_measureQubits.clear();
    m_shotCount = nbShots;

    for (int i = 0; i < m_buffer->size(); ++i) 
    {   
        m_qubitTensorNames.emplace_back("Q" + std::to_string(i));
    }

    // Create the qubit register tensor
    for (const auto& qTensor : m_qubitTensorNames) 
    {
        const bool created = exatn::createTensor(
            qTensor, exatn::TensorElementType::COMPLEX64,
            exatn::numerics::TensorShape{2});
        assert(created);
    }

    // Initialize the qubit register tensor to zero state
    for (const auto& qTensor : m_qubitTensorNames) 
    {
        const bool initialized = exatn::initTensorData(qTensor, Q_ZERO_TENSOR_BODY);
        assert(initialized);
    }

    // Append the qubit tensors to the tensor network
    for (const auto& qTensor : m_qubitTensorNames) 
    {
        m_tensorIdCounter++;
        m_tensorNetwork.appendTensor(m_tensorIdCounter, exatn::getTensor(qTensor), std::vector<std::pair<unsigned int, unsigned int>>{});
    }
}

void ExatnMpsVisitor::finalize() 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.flushAll();    
        evaluateTensorNetwork(m_tensorNetwork, m_stateVec);
        if (!m_measureQubits.empty())
        {
            addMeasureBitStringProbability(m_measureQubits, m_stateVec, m_shotCount);
        }
    }
    
}


void ExatnMpsVisitor::visit(Identity& in_IdentityGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_IdentityGate);
    }
}

void ExatnMpsVisitor::visit(Hadamard& in_HadamardGate) 
{ 
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_HadamardGate);
    }
}

void ExatnMpsVisitor::visit(X& in_XGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_XGate); 
    }
}

void ExatnMpsVisitor::visit(Y& in_YGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_YGate); 
    }
}

void ExatnMpsVisitor::visit(Z& in_ZGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_ZGate); 
    }
}

void ExatnMpsVisitor::visit(Rx& in_RxGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_RxGate); 
    }
}

void ExatnMpsVisitor::visit(Ry& in_RyGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_RyGate); 
    }
}

void ExatnMpsVisitor::visit(Rz& in_RzGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_RzGate); 
    }
}

void ExatnMpsVisitor::visit(T& in_TGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_TGate); 
    }
}

void ExatnMpsVisitor::visit(Tdg& in_TdgGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_TdgGate); 
    }
}

// others
void ExatnMpsVisitor::visit(Measure& in_MeasureGate) 
{ 
   m_measureQubits.emplace_back(in_MeasureGate.bits()[0]);
}

void ExatnMpsVisitor::visit(U& in_UGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_UGate); 
    }
}

// two-qubit gates: 
// NOTE: these gates are IMPORTANT for gate clustering consideration
void ExatnMpsVisitor::visit(CNOT& in_CNOTGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_CNOTGate); 
    }
}

void ExatnMpsVisitor::visit(Swap& in_SwapGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {   
        m_aggrerator.addGate(&in_SwapGate); 
    }
}

void ExatnMpsVisitor::visit(CZ& in_CZGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_CZGate); 
    }
}

void ExatnMpsVisitor::visit(CPhase& in_CPhaseGate) 
{ 
    // TODO
    if (m_aggrerateEnabled)
    {
        m_aggrerator.addGate(&in_CPhaseGate); 
    }
}

const double ExatnMpsVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
{ 
    // TODO
    return 0.0;
}

void ExatnMpsVisitor::onFlush(const AggreratedGroup& in_group)
{
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
    auto& aggregatedGateTensor = m_tensorNetwork;
    
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
