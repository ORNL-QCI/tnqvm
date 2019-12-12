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
#ifdef TNQVM_HAS_EXATENSOR
#define _DEBUG_DIL

#include "ExatnMPSVisitor.hpp"
#include "base/Gates.hpp"
#include "exatn.hpp"
#include "tensor_basic.hpp"
#include "Instruction.hpp"
#include "talshxx.hpp"
#include <numeric>

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
    const std::vector<std::complex<double>> ExatnMPSVisitor::Q_ZERO_TENSOR_BODY{ {1.0,0.0}, {0.0,0.0} };
    
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

    CalculateExpectationValueFunctor::CalculateExpectationValueFunctor(int qubitIndex):
        m_qubitIndex(qubitIndex)
    {}
    
    int CalculateExpectationValueFunctor::apply(talsh::Tensor& local_tensor) 
    {        
        assert(local_tensor.getRank() > m_qubitIndex);
        
        const auto N = local_tensor.getVolume();
	    const auto k_range = 1ULL << m_qubitIndex;
        m_result = 0.0;
        // The measurement result of a state
        double measureResult = 1.0;
        std::complex<double>* elements;
        
        const bool isOkay = local_tensor.getDataAccessHost(&elements);
        if (isOkay)
        {
            for(uint64_t i = 0; i < N; i += k_range)
            {
                for (uint64_t j = 0; j < k_range; ++j)
                {
                    m_result += measureResult * std::norm(elements[i + j]);
                }
                
                // The measurement result toggles between -1.0 and 1.0
                measureResult *= -1.0;
            }
        }
       
        return 0;
    }

    ExatnMPSVisitor::ExatnMPSVisitor():
        m_tensorNetwork(),
        m_tensorIdCounter(0),
        m_hasEvaluated(false)
    {
        // TODO
    }
    
    void ExatnMPSVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer) 
    {
        // Initialize ExaTN
        exatn::initialize();
        assert(exatn::isInitialized());

        m_buffer = std::move(buffer);
       

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
    }

    void ExatnMPSVisitor::evaluateNetwork() 
    {
        #ifdef _DEBUG
        m_tensorNetwork.printIt();
        #endif
        
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
        
        #ifdef _DEBUG
        {
            // If in Debug, print out tensor data
            auto functor = std::make_shared<TensorComponentPrintFunctor>();
            for (auto iter = m_tensorNetwork.cbegin(); iter != m_tensorNetwork.cend(); ++iter)
            {
                const auto tensor = iter->second.getTensor();
                std::cout << tensor->getName();
                exatn::numericalServer->transformTensorSync(tensor->getName(), functor);
                exatn::sync();
            }
        }      
        
        {
            std::cout << "Root tensor: " << m_tensorNetwork.getTensor(0)->getName() << "\n";
            std::vector<std::complex<double>> stateVec;
            // Print out the state vector
            auto stateVecFunctor = std::make_shared<ReconstructStateVectorFunctor>(m_buffer, stateVec);
            exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), stateVecFunctor);
            exatn::sync();
            std::cout << "[";
            for (const auto& component : stateVec)
            {
                std::cout << component;
            }
            std::cout << "]\n";
        }      
        #endif
    }
    
    void ExatnMPSVisitor::finalize() 
    {
        if (!m_hasEvaluated)
        {
            // If we haven't evaluated the network, do it now (end of circuit).
            evaluateNetwork();
        }
        // Destroy gate tensors
        for (const auto& gateTensor : m_gateTensorBodies)
        {
            const bool destroyed = exatn::destroyTensor(gateTensor.first);
            assert(destroyed);
        }
        m_gateTensorBodies.clear();

        // Destroy qubit tensors 
        for (int i = 0; i < m_buffer->size(); ++i)
        {
            const bool destroyed = exatn::destroyTensor(generateQubitTensorName(i)); 
            assert(destroyed);
        }

        // Synchronize after tensor destroy
        exatn::sync();

        m_buffer.reset();
        exatn::finalize();
    }

    // === BEGIN: Gate Visitor Impls ===
    void ExatnMPSVisitor::visit(Identity& in_IdentityGate) 
    {  
       appendGateTensor<CommonGates::I>(in_IdentityGate);
    }
    
    void ExatnMPSVisitor::visit(Hadamard& in_HadamardGate) 
    { 
        appendGateTensor<CommonGates::H>(in_HadamardGate);
    }
    
    void ExatnMPSVisitor::visit(X& in_XGate) 
    { 
        appendGateTensor<CommonGates::X>(in_XGate);    
    }

    void ExatnMPSVisitor::visit(Y& in_YGate) 
    { 
        appendGateTensor<CommonGates::Y>(in_YGate);
    }
    
    void ExatnMPSVisitor::visit(Z& in_ZGate) 
    { 
        appendGateTensor<CommonGates::Z>(in_ZGate); 
    }
    
    void ExatnMPSVisitor::visit(Rx& in_RxGate) 
    { 
       assert(in_RxGate.nParameters() == 1);
       const double theta = in_RxGate.getParameter(0).as<double>();
       appendGateTensor<CommonGates::Rx>(in_RxGate, theta);
    }
    
    void ExatnMPSVisitor::visit(Ry& in_RyGate) 
    { 
       assert(in_RyGate.nParameters() == 1);
       const double theta = in_RyGate.getParameter(0).as<double>();
       appendGateTensor<CommonGates::Ry>(in_RyGate, theta);
    }
    
    void ExatnMPSVisitor::visit(Rz& in_RzGate) 
    { 
        assert(in_RzGate.nParameters() == 1);
        const double theta = in_RzGate.getParameter(0).as<double>();
        appendGateTensor<CommonGates::Rz>(in_RzGate, theta);
    }
    
    void ExatnMPSVisitor::visit(CPhase& in_CPhaseGate) 
    { 
        appendGateTensor<CommonGates::CPhase>(in_CPhaseGate);
    }
    
    void ExatnMPSVisitor::visit(U& in_UGate) 
    { 
        appendGateTensor<CommonGates::U>(in_UGate);
    }
    
    void ExatnMPSVisitor::visit(CNOT& in_CNOTGate) 
    { 
       appendGateTensor<CommonGates::CNOT>(in_CNOTGate);
    }
    
    void ExatnMPSVisitor::visit(Swap& in_SwapGate) 
    { 
        appendGateTensor<CommonGates::Swap>(in_SwapGate);
    }
    
    void ExatnMPSVisitor::visit(CZ& in_CZGate) 
    { 
        appendGateTensor<CommonGates::CZ>(in_CZGate);
    }
    
    void ExatnMPSVisitor::visit(Measure& in_MeasureGate) 
    { 
        // When we visit a measure gate, evaluate the current tensor network (up to this measurement)
        // Note: currently, we cannot do multiple measurement operations yet.
        // TODO: need to be able to re-sync the tensor network after each measurement to continue.
        if (!m_hasEvaluated)
        {
            evaluateNetwork();
            const int measQubit = in_MeasureGate.bits()[0];       
            auto expectationFunctor = std::make_shared<CalculateExpectationValueFunctor>(measQubit);
            exatn::numericalServer->transformTensorSync(m_tensorNetwork.getTensor(0)->getName(), expectationFunctor);
            exatn::sync();
            m_buffer->addExtraInfo("exp-val-z", expectationFunctor->getResult());
            #ifdef _DEBUG       
            std::cout << "applying " << in_MeasureGate.name() << " @ " << measQubit << ", " << expectationFunctor->getResult() << "\n";
            #endif
        }
        else
        {
            xacc::error("Multiple measurement operations in one circuit is not supported yet.");
        }        
    }
    // === END: Gate Visitor Impls ===
    
    template<tnqvm::CommonGates GateType, typename... GateParams>
    void ExatnMPSVisitor::appendGateTensor(const xacc::Instruction& in_gateInstruction, GateParams&&... in_params)
    { 
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

        // Append the tensor for this gate to the network
        m_tensorNetwork.appendTensorGate(
            m_tensorIdCounter,
            // Get the gate tensor data which must have been initialized.
            exatn::getTensor(uniqueGateName),
            // which qubits that the gate is acting on
            gatePairing
        ); 
    }
//Life cycle:

// ExaTensorMPSVisitor::ExaTensorMPSVisitor():
//  EagerEval(false), InitialValence(INITIAL_VALENCE), QubitRange(std::make_pair(-1,-1))
// {
// }

// ExaTensorMPSVisitor::~ExaTensorMPSVisitor()
// {
// }

// void ExaTensorMPSVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer)
// {
//  assert(!(this->isInitialized()));
//  Buffer = buffer;
//  const auto numQubits = Buffer->size(); //total number of qubits in the system
// #ifdef _DEBUG_DIL
//  std::cout << "[ExaTensorMPSVisitor]: Constructing an MPS wavefunction for " << numQubits << " qubits ... "; //debug
// #endif
//  //Construct initial MPS tensors for all qubits:
//  const unsigned int rankMPS = 3; //MPS tensor rank
//  const std::size_t dimExts[] = {InitialValence,InitialValence,BASE_SPACE_DIM}; //initial MPS tensor shape: [virtual,virtual,real]
//  for(unsigned int i = 0; i < numQubits; ++i){
//   StateMPS.emplace_back(Tensor(rankMPS,dimExts)); //construct a bodyless MPS tensor
//   StateMPS[i].allocateBody(); //allocates MPS tensor body
//   this->initMPSTensor(StateMPS[i]); //initializes the MPS tensor to a pure state
//  }
// #ifdef _DEBUG_DIL
//  std::cout << "Done" << std::endl; //debug
// #endif
//  return;
// }

// void ExaTensorMPSVisitor::finalize()
// {
//  assert(this->isInitialized());
//  if(!(this->isEvaluated())){
//   int error_code = this->evaluate();
//   assert(error_code == 0);
//  }
//  return;
// }

// //Private member functions:

// /** Initializes an MPS tensor to a disentangled pure |0> state. **/
// void ExaTensorMPSVisitor::initMPSTensor(Tensor & tensor)
// {
//  assert(tensor.getRank() == 3);
//  tensor.nullifyBody();
//  tensor[{0,0,0}] = TensDataType(1.0,0.0);
//  return;
// }

// /** Builds a tensor network for the wavefunction representation for qubits [firstQubit:lastQubit].
//     Inside the constructed tensor network, qubit#0 is the real qubit#firstQubit, and so on. **/
// void ExaTensorMPSVisitor::buildWaveFunctionNetwork(int firstQubit, int lastQubit)
// {
//  assert(this->isInitialized());
//  assert(TensNet.isEmpty());
//  unsigned int numQubitsTotal = StateMPS.size(); //total number of MPS tensors = total number of qubits
//  if(lastQubit < 0) lastQubit = numQubitsTotal - 1; //default last qubit
//  assert(firstQubit >= 0 && lastQubit < numQubitsTotal && firstQubit <= lastQubit);
//  //Construct the output tensor:
//  unsigned int numQubits = lastQubit - firstQubit + 1; //rank of the output tensor
//  const std::size_t outDims[numQubits] = {BASE_SPACE_DIM}; //dimensions of the output tensor
//  std::vector<TensorLeg> legs;
//  for(unsigned int i = 1; i <= numQubits; ++i) legs.emplace_back(TensorLeg(i,2)); //leg #2 is the open leg of each MPS tensor
//  TensNet.appendTensor(Tensor(numQubits,outDims),legs); //output tensor
//  //Construct the input tensors (ring MPS topology):
//  for(unsigned int i = 1; i <= numQubits; ++i){
//   legs.clear();
//   unsigned int prevTensId = i - 1; if(prevTensId == 0) prevTensId = numQubits; //previous MPS tensor id: [1..numQubits]
//   unsigned int nextTensId = i + 1; if(nextTensId > numQubits) nextTensId = 1; //next MPS tensor id: [1..numQubits]
//   legs.emplace_back(TensorLeg(prevTensId,1)); //connection to the previous MPS tensor
//   legs.emplace_back(TensorLeg(nextTensId,0)); //connection to the next MPS tensor
//   legs.emplace_back(TensorLeg(0,i-1)); //connection to the output tensor
//   TensNet.appendTensor(StateMPS[firstQubit+i-1],legs); //append a wavefunction MPS tensor to the tensor network
//  }
//  return;
// }

// /** Appends gate tensors from the current gate sequence to the wavefunction tensor network. **/
// void ExaTensorMPSVisitor::appendGateSequence()
// {
//  assert(!(TensNet.isEmpty()));
//  assert(OptimizedTensors.empty());
//  std::vector<unsigned int> legIds;
//  for(auto & it: GateSequence){
//   const Tensor & tensor = it.first;
//   const unsigned int * qids = it.second;
//   for(unsigned int i = 0; i < tensor.getRank()/2; ++i) legIds.push_back(qids[i]-QubitRange.first);
//   TensNet.appendTensor(tensor,legIds); //append the unitary tensor to the tensor network
//   legIds.clear();
//   delete[] qids;
//  }
//  GateSequence.clear();
//  return;
// }

// /** Closes the circuit tensor network with output tensors (those to be optimized). **/
// void ExaTensorMPSVisitor::closeCircuitNetwork()
// {
//  assert(!(TensNet.isEmpty()));
//  assert(OptimizedTensors.empty());
//  const auto numOutLegs = TensNet.getTensor(0).getRank(); //total number of open legs in the tensor network
//  auto numTensors = TensNet.getNumTensors(); //number of the r.h.s. tensors in the tensor network
//  //Construct the tensor network for the output WaveFunction:
//  // Construct the output tensor:
//  TensorNetwork optNet;
//  const std::size_t outDims[numOutLegs] = {BASE_SPACE_DIM}; //dimensions of the output tensor
//  std::vector<TensorLeg> legs;
//  for(unsigned int i = 1; i <= numOutLegs; ++i) legs.emplace_back(TensorLeg(i,2)); //leg #2 is the open leg of each MPS tensor
//  optNet.appendTensor(Tensor(numOutLegs,outDims),legs); //output tensor
//  // Construct the input tensors (ring MPS topology):
//  for(unsigned int i = 1; i <= numOutLegs; ++i){
//   legs.clear();
//   unsigned int prevTensId = i - 1; if(prevTensId == 0) prevTensId = numOutLegs; //previous MPS tensor id: [1..numOutLegs]
//   unsigned int nextTensId = i + 1; if(nextTensId > numOutLegs) nextTensId = 1; //next MPS tensor id: [1..numOutLegs]
//   legs.emplace_back(TensorLeg(prevTensId,1)); //connection to the previous MPS tensor
//   legs.emplace_back(TensorLeg(nextTensId,0)); //connection to the next MPS tensor
//   legs.emplace_back(TensorLeg(0,i-1)); //connection to the output tensor
//   const auto tensRank = TensNet.getTensor(i).getRank(); //tensor i is the i-th input MPS tensor, i=[1..numMPSTensors]
//   const auto dimExts = TensNet.getTensor(i).getDimExtents(); //tensor i is the i-th input MPS tensor, i=[1..numMPSTensors]
//   optNet.appendTensor(Tensor(tensRank,dimExts),legs); //append a wavefunction MPS tensor to the optimized wavefunction tensor network
//  }
//  //Define output leg matching:
//  std::vector<std::pair<unsigned int, unsigned int>> legPairs;
//  for(unsigned int i = 0; i < numOutLegs; ++i) legPairs.emplace_back(std::pair<unsigned int, unsigned int>(i,i));
//  //Append the output wavefunction tensor network to the input+gates one:
//  TensNet.appendNetwork(optNet,legPairs);
//  //Mark the tensors to be optimized:
//  for(unsigned int i = 0; i < numOutLegs; ++i) OptimizedTensors.push_back(++numTensors);
//  assert(numTensors == TensNet.getNumTensors());
//  return;
// }

// int ExaTensorMPSVisitor::appendNBodyGate(const Tensor & gate, const unsigned int qubit_id[])
// {
//  int error_code = 0;
//  const unsigned int gateRank = gate.getRank(); //N-body gate (rank-2N)
//  assert(gateRank > 0 && gateRank%2 == 0);
//  const auto numQubits = gateRank/2;
//  if(QubitRange.first < 0) QubitRange.first = StateMPS.size();
//  if(QubitRange.second < 0) QubitRange.second = 0;
//  unsigned int * qids = new unsigned int [numQubits];
//  for(unsigned int i = 0; i < numQubits; ++i){
//   assert(qubit_id[i] < StateMPS.size());
//   qids[i] = qubit_id[i];
//   if(qids[i] < QubitRange.first) QubitRange.first = qids[i];
//   if(qids[i] > QubitRange.second) QubitRange.second = qids[i];
//  }
//  GateSequence.push_back(std::make_pair(gate,qids));
//  if(EagerEval || GateSequence.size() >= MAX_GATES) error_code = this->evaluate();
//  return error_code;
// }

// //Public visitor methods:

// void ExaTensorMPSVisitor::visit(Hadamard & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(X & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Y & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Z & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Rx & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Ry & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Rz & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(U & gate)
// {
//  // TODO
//  return;
// }
// void ExaTensorMPSVisitor::visit(CPhase & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(CNOT & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(CZ & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Swap & gate)
// {
//  unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
//  std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
//  const Tensor & gateTensor = GateTensors.getTensor(gate);
//  int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
//  return;
// }

// void ExaTensorMPSVisitor::visit(Measure & gate)
// {
//  //`Implement
//  return;
// }

// void ExaTensorMPSVisitor::visit(ConditionalFunction & condFunc)
// {
//  //`Implement
//  return;
// }

// void ExaTensorMPSVisitor::visit(GateFunction & gateFunc)
// {
//  //`Implement
//  return;
// }

// //Numerical evaluation:

// bool ExaTensorMPSVisitor::isInitialized()
// {
//  return !StateMPS.empty();
// }

// bool ExaTensorMPSVisitor::isEvaluated()
// {
//  return GateSequence.empty();
// }

// void ExaTensorMPSVisitor::setEvaluationStrategy(const bool eagerEval)
// {
//  assert(TensNet.isEmpty());
//  EagerEval = eagerEval;
//  return;
// }

// void ExaTensorMPSVisitor::setInitialMPSValence(const std::size_t initialValence)
// {
//  assert(TensNet.isEmpty());
//  assert(initialValence > 0);
//  InitialValence = initialValence;
//  return;
// }

// const double getExpectationValueZ(std::shared_ptr<CompositeInstruction> function) 
// {
//     //TODO
//     return 0.0;

// }

// int ExaTensorMPSVisitor::evaluate()
// {
//  int error_code = 0;
//  assert(this->isInitialized());
//  assert(TensNet.isEmpty());
//  if(!(GateSequence.empty())){
//   buildWaveFunctionNetwork(QubitRange.first,QubitRange.second); //build the wavefunction tensor network
//   appendGateSequence(); //apply gate tensors
//   closeCircuitNetwork(); //close the circuit tensor network with the output wavefunction tensors (those to be optimized)
//   std::vector<double> norms(OptimizedTensors.size(),1.0); //all optimized tensors are normalized to unity
//   error_code = exatensor::optimizeOverlapMax(TensNet,OptimizedTensors,norms); //optimize output MPS wavefunction tensors
//   //`Update the WaveFunction tensors with the optimized output tensors and destroy old wavefunction tensors
//   //`Destroy the tensor network object, optimized tensors
//   QubitRange.first = -1; QubitRange.second = -1;
//  }
//  return error_code;
// }

    

}  // end namespace tnqvm

#endif //TNQVM_HAS_EXATENSOR