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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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

namespace {
    // Helper to construct qubit tensor name:
    std::string generateQubitTensorName(int qubitIndex) 
    {
        return "Q" + std::to_string(qubitIndex);
    };

    std::vector<std::complex<double>> flattenGateMatrix(const std::vector<std::vector<std::complex<double>>>& in_gateMatrix)
    {
        std::vector<std::complex<double>> resultVector;
        resultVector.reserve(in_gateMatrix.size() * in_gateMatrix.size());
        for (const auto& row: in_gateMatrix)
        {
            for (const auto& entry: row)
            {
                resultVector.emplace_back(entry);
            }
        }

        return resultVector;      
    }

    bool checkStateVectorNorm(const std::vector<std::complex<double>>& in_stateVec) 
    {
        const double norm = std::accumulate(in_stateVec.begin(), in_stateVec.end(), 0.0, [](double runningNorm, std::complex<double> vecComponent){
            return runningNorm + std::norm(vecComponent);
        });

        return (std::abs(norm - 1.0) < 1e-12);
    }

    inline double generateRandomProbability() 
    {
        auto randFunc = std::bind(std::uniform_real_distribution<double>(0, 1), std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        return randFunc();
    }
}

namespace tnqvm {    
    GateInstanceIdentifier::GateInstanceIdentifier(const std::string& in_gateName):
        m_gateName(in_gateName)
    {}

    template<typename... GateParams>
    GateInstanceIdentifier::GateInstanceIdentifier(const std::string& in_gateName, const GateParams&... in_params):
        GateInstanceIdentifier(in_gateName)
    {
        addParam(in_params...);
    }
    
    template<typename GateParam>
    void GateInstanceIdentifier::addParam(const GateParam& in_param)
    {
        m_gateParams.emplace_back(std::to_string(in_param));
    }

    template<typename GateParam, typename... MoreParams>
    void GateInstanceIdentifier::addParam(const GateParam& in_param, const MoreParams&... in_moreParams)
    {
        m_gateParams.emplace_back(std::to_string(in_param));
        addParam(in_moreParams...);
    }

    std::string GateInstanceIdentifier::toNameString() const 
    {
        if (m_gateParams.empty())
        {
            return m_gateName;
        }
        else 
        {
            return m_gateName + 
                "(" + 
                [&]() -> std::string {
                    std::string paramList;
                    for (size_t i = 0; i < m_gateParams.size() - 1; ++i)
                    {
                        paramList.append(m_gateParams[i] + ",");
                    }
                    paramList.append(m_gateParams.back());

                    return paramList;
                }() +
                ")";
        }
    }
    // Define the tensor body for a zero-state qubit
    const std::vector<std::complex<double>> ExatnVisitor::Q_ZERO_TENSOR_BODY{ {1.0,0.0}, {0.0,0.0} };
    const std::vector<std::complex<double>> ExatnVisitor::Q_ONE_TENSOR_BODY{ {0.0,0.0}, {1.0,0.0} };
    
    int TensorComponentPrintFunctor::apply(talsh::Tensor& local_tensor) 
    {        
        std::complex<double>* elements;
        const bool worked = local_tensor.getDataAccessHost(&elements);
        std::cout << "(Rank:" << local_tensor.getRank() << ", Volume: " << local_tensor.getVolume() << "): ";
        std::cout << "[";
        for (int i = 0; i < local_tensor.getVolume(); ++i)
        {
           const std::complex<double> element = elements[i];
           std::cout << element;
        }
        std::cout << "]\n";
        return 0;
    }

    ReconstructStateVectorFunctor::ReconstructStateVectorFunctor(const std::shared_ptr<AcceleratorBuffer>& buffer, std::vector<std::complex<double>>& io_stateVec):
        m_qubits(buffer->size()),
        m_stateVec(io_stateVec)
    {
        // Don't allow any attempts to reconstruct the state vector
        // if there are so many qubits.
        assert(m_qubits <= 32);
        m_stateVec.clear();
        m_stateVec.reserve(1 << m_qubits);
    }
    
    int ReconstructStateVectorFunctor::apply(talsh::Tensor& local_tensor) 
    {        
        // Make sure we only call this on the final state tensor, 
        // i.e. the rank must equal the number of qubits.
        assert(local_tensor.getRank() == m_qubits);
        std::complex<double>* elements;
       
        if (local_tensor.getDataAccessHost(&elements))
        {
            m_stateVec.assign(elements, elements + local_tensor.getVolume());
        }

        #ifdef _DEBUG
        const bool normOkay = checkStateVectorNorm(m_stateVec);
        assert(normOkay);
        #endif
        
        return 0;
    }

    CalculateExpectationValueFunctor::CalculateExpectationValueFunctor(const std::vector<int>& qubitIndex):
        m_qubitIndices(qubitIndex)
    {}
    
    int CalculateExpectationValueFunctor::apply(talsh::Tensor& local_tensor) 
    {        
        const auto hasEvenParity = [](size_t x, const std::vector<int>& in_qubitIndices) -> bool {
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
 
        std::complex<double>* elements;
        const bool isOkay = local_tensor.getDataAccessHost(&elements);
        m_result = 0.0; 
        if (isOkay)
        {
            for(uint64_t i = 0; i < local_tensor.getVolume(); ++i)
            {
                m_result += (hasEvenParity(i, m_qubitIndices) ? 1.0 : -1.0) * std::norm(elements[i]);
            }
        }
       
        return 0;
    }
    
    ApplyQubitMeasureFunctor::ApplyQubitMeasureFunctor(int qubitIndex):
        m_qubitIndex(qubitIndex)
    {}

    int ApplyQubitMeasureFunctor::apply(talsh::Tensor& local_tensor) 
    {        
        assert(local_tensor.getRank() > m_qubitIndex);
        std::complex<double>* elements;
        const auto N = local_tensor.getVolume();
        const bool isOkay = local_tensor.getDataAccessHost(&elements);
        if (isOkay)
        {
            const auto k_range = 1ULL << m_qubitIndex;        
            const double randProbPick = generateRandomProbability();
            double cumulativeProb = 0.0;
            size_t stateSelect = 0;
            //select a state based on cumulative distribution
            while (cumulativeProb < randProbPick && stateSelect < N)
            {
                cumulativeProb += std::norm(elements[stateSelect++]);
            }
            stateSelect--;

            //take the value of the measured bit
	        m_result = ((stateSelect >> m_qubitIndex) & 1);
            // Collapse the state vector according to the measurement result
            double measProb = 0.0;
            for (size_t g = 0; g < N; g += (k_range * 2))
            {
                for (size_t i = 0; i < k_range; ++i)
                {
                    if ((((i + g) >> m_qubitIndex) & 1) == m_result)
                    {
                        measProb += std::norm(elements[i + g]);
				        elements[i + g + k_range] = 0.0; 
                    }
			        else
			        {
				        measProb += std::norm(elements[i + g + k_range]);
				        elements[i + g] = 0.0;
                    }
                }
            }
            // Renormalize the state
	        measProb = std::sqrt(measProb);
            for (size_t g = 0; g < N; g += (k_range * 2))
	        {
		        for (size_t i = 0; i < k_range; i += 1)
		        {
			        if ((((i + g) >> m_qubitIndex) & 1) == m_result)
			        {
				        elements[i + g] /= measProb;
			        }
			        else
			        {
				        elements[i + g + k_range] /= measProb;
                    }    
		        }
	        }
        }

        return 0;
    }

    ResetTensorDataFunctor::ResetTensorDataFunctor(const std::vector<std::complex<double>>& in_stateVec):
        m_stateVec(in_stateVec)
    {}

    int ResetTensorDataFunctor::apply(talsh::Tensor& local_tensor) 
    {
        std::complex<double>* elements;
       
        if (local_tensor.getDataAccessHost(&elements))
        {
            for (size_t i = 0; i < local_tensor.getVolume(); ++i)
            {
                elements[i] = m_stateVec[i];
            }
        }
        
        return 0;        
    }
    
    void ExatnDebugLogger::preEvaluate(tnqvm::ExatnVisitor* in_backEnd)  
    {
        // If in Debug, print out tensor data using the Print Functor
        auto functor = std::make_shared<tnqvm::TensorComponentPrintFunctor>();
        for (auto iter = in_backEnd->m_tensorNetwork.cbegin(); iter != in_backEnd->m_tensorNetwork.cend(); ++iter)
        {
            const auto tensor = iter->second.getTensor();
            if (tensor->getName().front() != '_')
            {
                std::cout << tensor->getName();
                exatn::numericalServer->transformTensorSync(tensor->getName(), functor);
            }
        }
    }
    
    void ExatnDebugLogger::preMeasurement(tnqvm::ExatnVisitor* in_backEnd, xacc::quantum::Measure& in_measureGate)  
    {
        // Print out the state vector
        std::cout << "Applying " << in_measureGate.name() << " @ " << in_measureGate.bits()[0] << "\n";
        std::cout << "=========== BEFORE MEASUREMENT =================\n";
        std::cout << "State Vector: [";
        for (const auto& component : in_backEnd->retrieveStateVector())
        {
            std::cout << component;
        }
        std::cout << "]\n";
    }
    
    void ExatnDebugLogger::postMeasurement(tnqvm::ExatnVisitor* in_backEnd, xacc::quantum::Measure& in_measureGate, bool in_bitResult, double in_expectedValue)  
    {
        // Print out the state vector
        std::cout << "=========== AFTER MEASUREMENT =================\n";
        std::cout << "Qubit measurement result (random binary): " << in_bitResult << "\n";
        std::cout << "Expected value (exp-val-z): " << in_expectedValue << "\n";
        std::cout << "State Vector: [";
        for (const auto& component : in_backEnd->retrieveStateVector())
        {
            std::cout << component;
        }
        std::cout << "]\n";
        std::cout << "=============================================\n";
    }


    ExatnVisitor::ExatnVisitor():
        m_tensorNetwork("Quantum Circuit"),
        m_tensorIdCounter(0),
        m_hasEvaluated(false),
        m_isAppendingCircuitGates(true)
    {}
    
    void ExatnVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) 
    {
        // Initialize ExaTN
        exatn::initialize();
        assert(exatn::isInitialized());
        m_hasEvaluated = false;
        m_buffer = std::move(buffer);
        m_shots = nbShots;
    
        // Create the qubit register tensor
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            const bool created = exatn::createTensor(generateQubitTensorName(i), exatn::TensorElementType::COMPLEX64, TensorShape{2}); 
            assert(created);
        }
        
        // Initialize the qubit register tensor to zero state
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            const bool initialized = exatn::initTensorData(generateQubitTensorName(i), Q_ZERO_TENSOR_BODY);
            assert(initialized);
        }
        
        // Append the qubit tensors to the tensor network
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            m_tensorIdCounter++;
            m_tensorNetwork.appendTensor(m_tensorIdCounter, exatn::getTensor(generateQubitTensorName(i)), std::vector<std::pair<unsigned int, unsigned int>>{});
        }

        {
            // Copy the tensor network of qubit register
            m_qubitRegTensor = m_tensorNetwork;
            m_qubitRegTensor.rename("Qubit Register");
        }
        
        // Add the Debug logging listener
        #ifdef _DEBUG
        subscribe(ExatnDebugLogger::GetInstance());
        #endif
    }

    std::vector<std::complex<double>> ExatnVisitor::retrieveStateVector()
    {
        std::vector<std::complex<double>> stateVec;
        auto stateVecFunctor = std::make_shared<ReconstructStateVectorFunctor>(m_buffer, stateVec);
        exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), stateVecFunctor);
        exatn::sync();
        return stateVec;   
    }

    void ExatnVisitor::evaluateNetwork() 
    {
        // Notify listeners
        {
            for (auto& listener: m_listeners)
            {
                listener->preEvaluate(this);
            }
        }
        
        // Evaluate the tensor network (quantum circuit):
        // For ExaTN, we only evaluate during finalization, i.e. after all gates have been visited. 
        {
            const bool evaluated = exatn::evaluateSync(m_tensorNetwork); 
            assert(evaluated);
            // Synchronize:
            exatn::sync();
            m_hasEvaluated = true;
        }

        assert(m_tensorNetwork.getRank() == m_buffer->size());
    }


    void ExatnVisitor::resetExaTN()
    {
        std::unordered_set<std::string> tensorList;
        for (auto iter = m_tensorNetwork.cbegin(); iter != m_tensorNetwork.cend(); ++iter)
        {
            const auto& tensorName = iter->second.getTensor()->getName();
            // Not a root tensor
            if (!tensorName.empty() && tensorName[0] != '_')
            {
                tensorList.emplace(iter->second.getTensor()->getName());
            }            
        }
       
        for (const auto& tensorName : tensorList)
        {
            const bool destroyed = exatn::destroyTensor(tensorName);
            assert(destroyed);
        }
        m_gateTensorBodies.clear();
        m_tensorIdCounter = 0;
        TensorNetwork emptyTensorNet;
        m_tensorNetwork = emptyTensorNet;
        // Synchronize after tensor destroy
        exatn::sync();
        exatn::finalize();
    }

    void ExatnVisitor::resetNetwork()
    {
        // We must have evaluated the tensor network.
        assert(m_hasEvaluated);
        const auto stateVec = retrieveStateVector();
        // Re-initialize ExaTN
        resetExaTN();
        exatn::initialize();
        // The new qubit register tensor name will have name "RESET_"
        const std::string resetTensorName = "RESET_";
        // The qubit register tensor shape is {2, 2, 2, ...}, 1 leg for each qubit
        std::vector<int> qubitRegResetTensorShape(m_buffer->size(), 2);        
        const bool created = exatn::createTensor(resetTensorName, exatn::TensorElementType::COMPLEX64, qubitRegResetTensorShape); 
        assert(created);
        // Initialize the tensor body with the state vector from the previous evaluation.
        const bool initialized = exatn::initTensorData(resetTensorName, std::move(stateVec));
        assert(initialized);
        // Create a new tensor network
        m_tensorNetwork = TensorNetwork();
        // Reset counter
        m_tensorIdCounter = 1;
        
        // Use the root tensor from previous evaluation as the initial tensor
        m_tensorNetwork.appendTensor(m_tensorIdCounter, exatn::getTensor(resetTensorName), std::vector<std::pair<unsigned int, unsigned int>>{});    
        // Reset the evaluation flag after initialization.
        m_hasEvaluated = false;
    }
    
    void ExatnVisitor::finalize() 
    {
        if (!m_hasEvaluated)
        {
            // If we haven't evaluated the network, do it now (end of circuit).
            evaluateNetwork();
        }

        if (m_shots > 1)
        {
            const auto cachedStateVec = retrieveStateVector();            
            for (int i = 0; i < m_shots; ++i)
            {
                for (const auto& idx: m_measureQbIdx)
                {           
                    // Apply the measurement logic 
                    auto measurementFunctor = std::make_shared<ApplyQubitMeasureFunctor>(idx);
                    exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), measurementFunctor);
                    exatn::sync();         
                    // Append the boolean true/false as bit value
                    m_resultBitString.append(measurementFunctor->getResult() ? "1" : "0");    

                }
                // Finish measuring all qubits, append the bit-string measurement result.
                m_buffer->appendMeasurement(m_resultBitString);
                // Clear the result bit string after appending (to be constructed in the next shot)
                m_resultBitString.clear();

                // Restore state vector for the next shot
                exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), std::make_shared<ResetTensorDataFunctor>(cachedStateVec));
                exatn::sync();
            }     
        }

        // Notify listeners
        {
            for (auto& listener: m_listeners)
            {
                listener->onEvaluateComplete(this);
            }
        }

        // Set the bit string from measurement to the buffer
        // Note: if m_shots is not specified (-1), don't need to set the measurement bit string.
        if (!m_resultBitString.empty() && m_shots > 0)
        {
            m_buffer->appendMeasurement(m_resultBitString);
            // Clear the result bit string after appending.
            m_resultBitString.clear();
        }        
        
        m_buffer.reset();
        resetExaTN();
    }

    // === BEGIN: Gate Visitor Impls ===
    void ExatnVisitor::visit(Identity& in_IdentityGate) 
    {  
       appendGateTensor<CommonGates::I>(in_IdentityGate);
    }
    
    void ExatnVisitor::visit(Hadamard& in_HadamardGate) 
    { 
        appendGateTensor<CommonGates::H>(in_HadamardGate);
    }
    
    void ExatnVisitor::visit(X& in_XGate) 
    { 
        appendGateTensor<CommonGates::X>(in_XGate);    
    }

    void ExatnVisitor::visit(Y& in_YGate) 
    { 
        appendGateTensor<CommonGates::Y>(in_YGate);
    }
    
    void ExatnVisitor::visit(Z& in_ZGate) 
    { 
        appendGateTensor<CommonGates::Z>(in_ZGate); 
    }
    
    void ExatnVisitor::visit(Rx& in_RxGate) 
    { 
       assert(in_RxGate.nParameters() == 1);
       const double theta = in_RxGate.getParameter(0).as<double>();
       appendGateTensor<CommonGates::Rx>(in_RxGate, theta);
    }
    
    void ExatnVisitor::visit(Ry& in_RyGate) 
    { 
       assert(in_RyGate.nParameters() == 1);
       const double theta = in_RyGate.getParameter(0).as<double>();
       appendGateTensor<CommonGates::Ry>(in_RyGate, theta);
    }
    
    void ExatnVisitor::visit(Rz& in_RzGate) 
    { 
        assert(in_RzGate.nParameters() == 1);
        const double theta = in_RzGate.getParameter(0).as<double>();
        appendGateTensor<CommonGates::Rz>(in_RzGate, theta);
    }
    
    void ExatnVisitor::visit(T& in_TGate) 
    {
        appendGateTensor<CommonGates::T>(in_TGate); 
    }

    void ExatnVisitor::visit(Tdg& in_TdgGate)
    {
        appendGateTensor<CommonGates::Tdg>(in_TdgGate); 
    }


    void ExatnVisitor::visit(CPhase& in_CPhaseGate) 
    { 
        appendGateTensor<CommonGates::CPhase>(in_CPhaseGate);
    }
    
    void ExatnVisitor::visit(U& in_UGate) 
    { 
        appendGateTensor<CommonGates::U>(in_UGate);
    }
    
    void ExatnVisitor::visit(CNOT& in_CNOTGate) 
    { 
       appendGateTensor<CommonGates::CNOT>(in_CNOTGate);
    }
    
    void ExatnVisitor::visit(Swap& in_SwapGate) 
    { 
        appendGateTensor<CommonGates::Swap>(in_SwapGate);
    }
    
    void ExatnVisitor::visit(CZ& in_CZGate) 
    { 
        appendGateTensor<CommonGates::CZ>(in_CZGate);
    }
    
    void ExatnVisitor::visit(Measure& in_MeasureGate) 
    { 
        // When we visit a measure gate, evaluate the current tensor network (up to this measurement)
        // Note: currently, we cannot do gate operations post measurement yet (i.e. multiple evaluateNetwork() calls).
        // Multiple measurement ops at the end is supported, i.e. can measure the entire qubit register.
        // TODO: reset the tensor network and continue appending gate tensors.
        if (!m_hasEvaluated)
        {
           // If this is the first measure gate that we visit,
           // i.e. the tensor network hasn't been evaluate, do it now.
           evaluateNetwork();
        }

        // Notify listeners: before measurement
        {
            for (auto& listener: m_listeners)
            {
                listener->preMeasurement(this, in_MeasureGate);
            }
        }

        const int measQubit = in_MeasureGate.bits()[0];
        m_measureQbIdx.emplace_back(measQubit);
       
        // If multi-shot was requested, we skip measurement till the end.
        if (m_shots > 1) 
        {
            return;
        }

        // Calculate the expected value
        auto expectationFunctor = std::make_shared<CalculateExpectationValueFunctor>(m_measureQbIdx);
        exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), expectationFunctor);
                    
        // Apply the measurement logic 
        auto measurementFunctor = std::make_shared<ApplyQubitMeasureFunctor>(measQubit);
        exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), measurementFunctor);
        exatn::sync();

        m_buffer->addExtraInfo("exp-val-z", expectationFunctor->getResult());
        // Append the boolean true/false as bit value
        m_resultBitString.append(measurementFunctor->getResult() ? "1" : "0");
        // Notify listeners: after measurement
        {
            for (auto& listener: m_listeners)
            {
                listener->postMeasurement(this, in_MeasureGate, measurementFunctor->getResult(), expectationFunctor->getResult());
            }
        }            
    }
    // === END: Gate Visitor Impls ===
    
    template<tnqvm::CommonGates GateType, typename... GateParams>
    void ExatnVisitor::appendGateTensor(const xacc::Instruction& in_gateInstruction, GateParams&&... in_params)
    { 
        if (m_hasEvaluated)
        {
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
        if (m_gateTensorBodies.find(uniqueGateName) == m_gateTensorBodies.end())
        {
            const auto gateMatrix = GetGateMatrix<GateType>(in_params...);
            m_gateTensorBodies[uniqueGateName] = flattenGateMatrix(gateMatrix);
            // Currently, we only support 2-qubit gates.
            assert(in_gateInstruction.nRequiredBits() > 0 && in_gateInstruction.nRequiredBits() <= 2);
            const auto gateTensorShape = (in_gateInstruction.nRequiredBits() == 1 ? TensorShape{2, 2} : TensorShape{2, 2, 2, 2});      
            // Create the tensor
            const bool created = exatn::createTensor(uniqueGateName, exatn::TensorElementType::COMPLEX64, gateTensorShape); 
            assert(created);
            // Init tensor body data
            exatn::initTensorData(uniqueGateName, flattenGateMatrix(gateMatrix));
            // Register tensor isometry: 
            // For rank-2 gate isometric leg groups are: {0}, {1}. 
            // For rank-4 gate isometric leg groups are: {0,1}, {2,3}.
            if (in_gateInstruction.nRequiredBits() == 1)
            {
                const bool registered = exatn::registerTensorIsometry(uniqueGateName, {0}, {1}); 
                assert(registered);
            }
            else if (in_gateInstruction.nRequiredBits() == 2)
            {
                const bool registered = exatn::registerTensorIsometry(uniqueGateName, {0,1}, {2,3}); 
                assert(registered);
            }
        }

        // Helper to create unique tensor names in the format <GateTypeName>_<Counter>, e.g. H2, CNOT5, etc.
        // Note: this is different from the gate instance unique name which is referencing *unique* gate matrices.
        // Multiple tensors can just refer to the same tensor body,
        // for example, all H_k (Hadamard gates) in the circuit will all refer to a single *H* tensor body data.   
        const auto generateTensorName = [&]() -> std::string {
            return GetGateName(GateType) + "_" + std::to_string(m_tensorIdCounter);
        };
        
        m_tensorIdCounter++;

        // Because the qubit location and gate pairing are of different integer types,
        // we need to reconstruct the qubit vector.
        std::vector<unsigned int> gatePairing;
        for (const auto& qbitLoc: const_cast<xacc::Instruction&>(in_gateInstruction).bits())
        {
            gatePairing.emplace_back(qbitLoc);
        }

        // For control gates (e.g. CNOT), we need to reverse the leg pairing because the (Control Index, Target Index)
        // convention is the opposite of the MSB->LSB bit order when the CNOT matrix is specified.
        // e.g. the state vector is indexed by q1q0.
        if (IsControlGate(GateType))
        {
            std::reverse(gatePairing.begin(), gatePairing.end());
        }
        
        if (m_isAppendingCircuitGates)
        {
            // Append the gate tensor to the tracking list to apply inverse
            m_appendedGateTensors.emplace_back(std::make_pair(uniqueGateName, gatePairing));
        }

        // Append the tensor for this gate to the network
        m_tensorNetwork.appendTensorGate(
            m_tensorIdCounter,
            // Get the gate tensor data which must have been initialized.
            exatn::getTensor(uniqueGateName),
            // which qubits that the gate is acting on
            gatePairing
        ); 
    }

    ExatnVisitor::ObservableTerm::ObservableTerm(const std::vector<std::shared_ptr<Instruction>>& in_operatorsInProduct, const std::complex<double>& in_coeff /*= 1.0*/):
        coefficient(in_coeff),
        operators(in_operatorsInProduct)
    {}

    std::complex<double> ExatnVisitor::observableExpValCalc(std::shared_ptr<AcceleratorBuffer>& in_buffer, std::shared_ptr<CompositeInstruction>& in_function, const std::vector<ObservableTerm>& in_observableExpression)
    {
        if (!m_appendedGateTensors.empty() || !m_gateTensorBodies.empty())
        {
            // We don't support mixing this *observable* mode of execution with the regular mode.
            xacc::error("observableExpValCalc can only be called on an ExatnVisitor that is not executing a circuit.");
            return 0.0;
        }

        BaseInstructionVisitor* visitorCast = static_cast<BaseInstructionVisitor*>(this);
        this->initialize(in_buffer, -1);
        // Walk the IR tree, and visit each node
        InstructionIterator it(in_function);
        while (it.hasNext()) 
        {
            auto nextInst = it.next();
            if (nextInst->isEnabled()) 
            {
                nextInst->accept(visitorCast);
            }
        }

        const auto result = expVal(in_observableExpression);
        // Set Evaluated flag, hence no need to evaluate the original circuit anymore.
        // (we have calculated the expectation value by closing the entire tensor network)
        m_hasEvaluated = true;
        exatn::sync();
        finalize();

        return result;
    }

    std::complex<double> ExatnVisitor::expVal(const std::vector<ObservableTerm>& in_observableExpression)
    {
        std::complex<double> result = 0.0;
        for (const auto& term: in_observableExpression)
        {
            result += (term.coefficient * evaluateTerm(term.operators));
        }

        return result;
    }

    std::complex<double> ExatnVisitor::evaluateTerm(const std::vector<std::shared_ptr<Instruction>>& in_observableTerm) 
    {
        std::complex<double> result = 0.0;
        // Save/cache the tensor network
        const auto cachedTensor = m_tensorNetwork;
        const auto cachedIdCounter = m_tensorIdCounter;
        m_isAppendingCircuitGates = false;
        {
            for (auto& inst: in_observableTerm)
            {
                // Adding the Hamiltonian term to the tensor network
                inst->accept(static_cast<BaseInstructionVisitor*>(this));
            }           
        }
        
        // Close the tensor network
        applyInverse();        
        
        {
            if(exatn::evaluateSync(m_tensorNetwork))
            {
                exatn::sync();
                auto talsh_tensor = exatn::getLocalTensor(m_tensorNetwork.getTensor(0)->getName());
                assert(talsh_tensor->getVolume() == 1);
                const std::complex<double>* body_ptr;
                if (talsh_tensor->getDataAccessHostConst(&body_ptr))
                {
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
    
    void ExatnVisitor::applyInverse() 
    {
        for (auto iter =  m_appendedGateTensors.rbegin(); iter != m_appendedGateTensors.rend(); ++iter)
        {
            m_tensorIdCounter++;
            m_tensorNetwork.appendTensorGate(
                m_tensorIdCounter,
                // Get the gate tensor data which must have been initialized.
                exatn::getTensor(iter->first),
                // which qubits that the gate is acting on
                iter->second,
                // Set conjugate flag
                true
            );
        }
        {
            auto bra = m_qubitRegTensor;    
            // Conjugate the ket to get the bra (e.g. if it was initialized to a complex superposition)
            bra.conjugate();    
            // Closing the tensor network with the bra
            std::vector<std::pair<unsigned int, unsigned int>> pairings;
            for (unsigned int i = 0; i < m_buffer->size(); ++i)
            {
                pairings.emplace_back(std::make_pair(i, i));
            }
            m_tensorNetwork.appendTensorNetwork(std::move(bra), pairings);
        }       
           
        {
            const bool collapsed = m_tensorNetwork.collapseIsometries(); 
        }
    }

    std::vector<std::complex<double>> ExatnVisitor::getReducedDensityMatrix(std::shared_ptr<AcceleratorBuffer>& in_buffer, std::shared_ptr<CompositeInstruction>& in_function, const std::vector<size_t>& in_qubitIdx)
    {
        if (!m_appendedGateTensors.empty() || !m_gateTensorBodies.empty())
        {
            // We don't support mixing this *RDM* mode of execution with the regular mode.
            // TODO: Adding runtime logic to determine if we need to use this mode on *regular* circuit execution.
            // e.g. if there are many qubits and we are requesting *shots* samping from a small subset of qubits.
            xacc::error("getReducedDensityMatrix can only be called on an ExatnVisitor that is not executing a circuit.");
            return {};
        }

        std::vector<std::complex<double>> resultRDM;               
        BaseInstructionVisitor* visitorCast = static_cast<BaseInstructionVisitor*>(this);
        this->initialize(in_buffer, -1);
        // Walk the IR tree, and visit each node
        InstructionIterator it(in_function);
        while (it.hasNext()) 
        {
            auto nextInst = it.next();
            if (nextInst->isEnabled() && nextInst->name() != "Measure") 
            {
                nextInst->accept(visitorCast);
            }
        }

        auto inverseTensorNetwork = m_tensorNetwork;
        inverseTensorNetwork.rename("Inverse Tensor Network");
        inverseTensorNetwork.conjugate();

        // Connect the original tensor network with its inverse 
        // but leave those qubit lines that we want to get RDM open
        {
            auto combinedNetwork = m_tensorNetwork;
            combinedNetwork.rename("Combined Tensor Network");
            std::vector<std::pair<unsigned int, unsigned int>> pairings;
            for (size_t i = 0; i < m_buffer->size(); ++i)
            {
               if (std::find(in_qubitIdx.begin(), in_qubitIdx.end(), i) == in_qubitIdx.end())
               {
                    // We want to close this leg (not one of those requested for RDM)
                    pairings.emplace_back(std::make_pair(i, i));
               }
            }

            combinedNetwork.appendTensorNetwork(std::move(inverseTensorNetwork), pairings);
            const bool collapsed = combinedNetwork.collapseIsometries();                  
            
            if(exatn::evaluateSync(combinedNetwork))
            {
                exatn::sync();
                auto talsh_tensor = exatn::getLocalTensor(combinedNetwork.getTensor(0)->getName());
                const auto tensorVolume = talsh_tensor->getVolume();
                // Double check the size of the RDM  
                assert(tensorVolume == 1ULL << (2 * in_qubitIdx.size()));    
                const std::complex<double>* body_ptr;
                if (talsh_tensor->getDataAccessHostConst(&body_ptr))
                {     
                    resultRDM.assign(body_ptr, body_ptr + tensorVolume);
                }                
            }                 
        }      
               
        m_hasEvaluated = true;
        exatn::sync();
        finalize();        
        return resultRDM;         
    }

    std::vector<uint8_t> ExatnVisitor::getMeasureSample(std::shared_ptr<AcceleratorBuffer>& in_buffer, std::shared_ptr<CompositeInstruction>& in_function, const std::vector<size_t>& in_qubitIdx)
    {
        if (!m_appendedGateTensors.empty() || !m_gateTensorBodies.empty())
        {
            xacc::error("getMeasureSample can only be called on an ExatnVisitor that is not executing a circuit.");
            return {};
        }
        
        std::vector<uint8_t> resultBitString;
        std::vector<double> resultProbs;
        for (const auto& qubitIdx: in_qubitIdx)
        {
            std::vector<std::complex<double>> resultRDM;

            BaseInstructionVisitor* visitorCast = static_cast<BaseInstructionVisitor*>(this);
            this->initialize(in_buffer, -1);
            // Walk the IR tree, and visit each node
            InstructionIterator it(in_function);
            while (it.hasNext()) 
            {
                auto nextInst = it.next();
                if (nextInst->isEnabled() && nextInst->name() != "Measure") 
                {
                    nextInst->accept(visitorCast);
                }
            }

            auto inverseTensorNetwork = m_tensorNetwork;
            inverseTensorNetwork.rename("Inverse Tensor Network");
            inverseTensorNetwork.conjugate();
            
            {       
                {
                    // Adding collapse tensors based on previous measurement results.
                    // i.e. condition/renormalize the tensor network to be consistent with previous result.
                    for(size_t measIdx = 0; measIdx < resultBitString.size(); ++measIdx)
                    {
                        const unsigned int qId = in_qubitIdx[measIdx];
                        m_tensorIdCounter++;
                        // If it was a "0":
                        if (resultBitString[measIdx] == 0)
                        {
                            const std::vector<std::complex<double>> COLLAPSE_0 {
                                // Renormalize based on the probability of this outcome
                                {1.0/resultProbs[measIdx], 0.0}, {0.0, 0.0},
                                {0.0, 0.0}, {0.0, 0.0}
                            };              

                            const std::string tensorName = "COLLAPSE_0@" + std::to_string(measIdx);
                            const bool created = exatn::createTensor(tensorName, exatn::TensorElementType::COMPLEX64, TensorShape{2, 2});
                            assert(created); 
                            const bool registered =(exatn::registerTensorIsometry(tensorName, {0}, {1}));
                            assert(registered); 
                            const bool initialized = exatn::initTensorData(tensorName, COLLAPSE_0);
                            assert(initialized);
                            const bool appended = m_tensorNetwork.appendTensorGate(m_tensorIdCounter, exatn::getTensor(tensorName), {qId});
                            assert(appended);                         
                        }
                        else
                        {
                            assert(resultBitString[measIdx] == 1);
                            // Renormalize based on the probability of this outcome
                            const std::vector<std::complex<double>> COLLAPSE_1 {
                                {0.0, 0.0}, {0.0, 0.0},
                                {0.0, 0.0}, {1.0/resultProbs[measIdx], 0.0}
                            };              

                            const std::string tensorName = "COLLAPSE_1@" + std::to_string(measIdx);
                            const bool created = exatn::createTensor(tensorName, exatn::TensorElementType::COMPLEX64, TensorShape{2, 2});
                            assert(created); 
                            const bool registered =(exatn::registerTensorIsometry(tensorName, {0}, {1}));
                            assert(registered); 
                            const bool initialized = exatn::initTensorData(tensorName, COLLAPSE_1);
                            assert(initialized);
                            const bool appended = m_tensorNetwork.appendTensorGate(m_tensorIdCounter, exatn::getTensor(tensorName), {qId});
                            assert(appended);               
                        }
                    }
                }

                auto combinedNetwork = m_tensorNetwork;
                combinedNetwork.rename("Combined Tensor Network");
                {
                    // Append the conjugate network to calculate the RDM of the measure qubit
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

                    combinedNetwork.appendTensorNetwork(std::move(inverseTensorNetwork), pairings);
                    const bool collapsed = combinedNetwork.collapseIsometries(); 
                }
               
                // Evaluate                
                if(exatn::evaluateSync(combinedNetwork))
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
                    #ifdef _DEBUG
                    // Debug: print out RDM data
                    {
                        std::cout << "RDM @q" << qubitIdx << " = [";
                        for (int i = 0; i < talsh_tensor->getVolume(); ++i)
                        {
                            const std::complex<double> element = body_ptr[i];
                            std::cout << element;
                        }
                        std::cout << "]\n";
                    } 
                    #endif               
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
                #ifdef _DEBUG
                {
                    std::cout << ">> Measure @q" << qubitIdx << " prob(0) = " << prob_0 << "\n";
                    std::cout << ">> Measure @q" << qubitIdx << " prob(1) = " << prob_1 << "\n";
                    std::cout << ">> Measure @q" << qubitIdx << " random number = " << randProbPick << "\n";
                    std::cout << ">> Measure @q" << qubitIdx << " pick " << std::to_string(resultBitString.back()) << "\n";
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

    const double ExatnVisitor::getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) 
    {
        if (!m_buffer)
        {
            xacc::error("Please initialize the visitor backend before calling getExpectationValueZ()!");
            return 0.0;
        }
        
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

        if (!m_hasEvaluated)
        {
            // No measurement was specified in the circuit.
            xacc::warning("Expectation Value Z cannot be evaluated because there is no measurement.");
            return 0.0;
        }

        const auto expectation = (*m_buffer)["exp-val-z"].as<double>();
        return expectation;             
    } 
}  // end namespace tnqvm

#endif //TNQVM_HAS_EXATN