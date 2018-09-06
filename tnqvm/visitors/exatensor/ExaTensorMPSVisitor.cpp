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

#include "ExaTensorMPSVisitor.hpp"

#define _DEBUG_DIL

namespace tnqvm {

//Life cycle:

ExaTensorMPSVisitor::ExaTensorMPSVisitor():
 EagerEval(false), InitialValence(INITIAL_VALENCE), QubitRange(std::make_pair(-1,-1))
{
}

ExaTensorMPSVisitor::~ExaTensorMPSVisitor()
{
}

void ExaTensorMPSVisitor::initialize(std::shared_ptr<AcceleratorBuffer> buffer)
{
 assert(!(this->isInitialized()));
 Buffer = buffer;
 const auto numQubits = Buffer->size(); //total number of qubits in the system
#ifdef _DEBUG_DIL
 std::cout << "[ExaTensorMPSVisitor]: Constructing an MPS wavefunction for " << numQubits << " qubits ... "; //debug
#endif
 //Construct initial MPS tensors for all qubits:
 const unsigned int rankMPS = 3; //MPS tensor rank
 const std::size_t dimExts[] = {InitialValence,InitialValence,BASE_SPACE_DIM}; //initial MPS tensor shape: [virtual,virtual,real]
 for(unsigned int i = 0; i < numQubits; ++i){
  StateMPS.emplace_back(Tensor(rankMPS,dimExts)); //construct a bodyless MPS tensor
  StateMPS[i].allocateBody(); //allocates MPS tensor body
  this->initMPSTensor(StateMPS[i]); //initializes the MPS tensor to a pure state
 }
#ifdef _DEBUG_DIL
 std::cout << "Done" << std::endl; //debug
#endif
 return;
}

void ExaTensorMPSVisitor::finalize()
{
 assert(this->isInitialized());
 if(!(this->isEvaluated())){
  int error_code = this->evaluate();
  assert(error_code == 0);
 }
 return;
}

//Private member functions:

/** Initializes an MPS tensor to a disentangled pure |0> state. **/
void ExaTensorMPSVisitor::initMPSTensor(Tensor & tensor)
{
 assert(tensor.getRank() == 3);
 tensor.nullifyBody();
 tensor[{0,0,0}] = TensDataType(1.0,0.0);
 return;
}

/** Builds a tensor network for the wavefunction representation for qubits [firstQubit:lastQubit].
    Inside the constructed tensor network, qubit#0 is the real qubit#firstQubit, and so on. **/
void ExaTensorMPSVisitor::buildWaveFunctionNetwork(int firstQubit, int lastQubit)
{
 assert(this->isInitialized());
 assert(TensNet.isEmpty());
 unsigned int numQubitsTotal = StateMPS.size(); //total number of MPS tensors = total number of qubits
 if(lastQubit < 0) lastQubit = numQubitsTotal - 1; //default last qubit
 assert(firstQubit >= 0 && lastQubit < numQubitsTotal && firstQubit <= lastQubit);
 //Construct the output tensor:
 unsigned int numQubits = lastQubit - firstQubit + 1; //rank of the output tensor
 const std::size_t outDims[numQubits] = {BASE_SPACE_DIM}; //dimensions of the output tensor
 std::vector<TensorLeg> legs;
 for(unsigned int i = 1; i <= numQubits; ++i) legs.emplace_back(TensorLeg(i,2)); //leg #2 is the open leg of each MPS tensor
 TensNet.appendTensor(Tensor(numQubits,outDims),legs); //output tensor
 //Construct the input tensors (ring MPS topology):
 for(unsigned int i = 1; i <= numQubits; ++i){
  legs.clear();
  unsigned int prevTensId = i - 1; if(prevTensId == 0) prevTensId = numQubits; //previous MPS tensor id: [1..numQubits]
  unsigned int nextTensId = i + 1; if(nextTensId > numQubits) nextTensId = 1; //next MPS tensor id: [1..numQubits]
  legs.emplace_back(TensorLeg(prevTensId,1)); //connection to the previous MPS tensor
  legs.emplace_back(TensorLeg(nextTensId,0)); //connection to the next MPS tensor
  legs.emplace_back(TensorLeg(0,i-1)); //connection to the output tensor
  TensNet.appendTensor(StateMPS[firstQubit+i-1],legs); //append a wavefunction MPS tensor to the tensor network
 }
 return;
}

/** Appends gate tensors from the current gate sequence to the wavefunction tensor network. **/
void ExaTensorMPSVisitor::appendGateSequence()
{
 assert(!(TensNet.isEmpty()));
 assert(OptimizedTensors.empty());
 std::vector<unsigned int> legIds;
 for(auto & it: GateSequence){
  const Tensor & tensor = it.first;
  const unsigned int * qids = it.second;
  for(unsigned int i = 0; i < tensor.getRank()/2; ++i) legIds.push_back(qids[i]-QubitRange.first);
  TensNet.appendTensor(tensor,legIds); //append the unitary tensor to the tensor network
  legIds.clear();
  delete[] qids;
 }
 GateSequence.clear();
 return;
}

/** Closes the circuit tensor network with output tensors (those to be optimized). **/
void ExaTensorMPSVisitor::closeCircuitNetwork()
{
 assert(!(TensNet.isEmpty()));
 assert(OptimizedTensors.empty());
 const auto numOutLegs = TensNet.getTensor(0).getRank(); //total number of open legs in the tensor network
 auto numTensors = TensNet.getNumTensors(); //number of the r.h.s. tensors in the tensor network
 //Construct the tensor network for the output WaveFunction:
 // Construct the output tensor:
 TensorNetwork optNet;
 const std::size_t outDims[numOutLegs] = {BASE_SPACE_DIM}; //dimensions of the output tensor
 std::vector<TensorLeg> legs;
 for(unsigned int i = 1; i <= numOutLegs; ++i) legs.emplace_back(TensorLeg(i,2)); //leg #2 is the open leg of each MPS tensor
 optNet.appendTensor(Tensor(numOutLegs,outDims),legs); //output tensor
 // Construct the input tensors (ring MPS topology):
 for(unsigned int i = 1; i <= numOutLegs; ++i){
  legs.clear();
  unsigned int prevTensId = i - 1; if(prevTensId == 0) prevTensId = numOutLegs; //previous MPS tensor id: [1..numOutLegs]
  unsigned int nextTensId = i + 1; if(nextTensId > numOutLegs) nextTensId = 1; //next MPS tensor id: [1..numOutLegs]
  legs.emplace_back(TensorLeg(prevTensId,1)); //connection to the previous MPS tensor
  legs.emplace_back(TensorLeg(nextTensId,0)); //connection to the next MPS tensor
  legs.emplace_back(TensorLeg(0,i-1)); //connection to the output tensor
  const auto tensRank = TensNet.getTensor(i).getRank(); //tensor i is the i-th input MPS tensor, i=[1..numMPSTensors]
  const auto dimExts = TensNet.getTensor(i).getDimExtents(); //tensor i is the i-th input MPS tensor, i=[1..numMPSTensors]
  optNet.appendTensor(Tensor(tensRank,dimExts),legs); //append a wavefunction MPS tensor to the optimized wavefunction tensor network
 }
 //Define output leg matching:
 std::vector<std::pair<unsigned int, unsigned int>> legPairs;
 for(unsigned int i = 0; i < numOutLegs; ++i) legPairs.emplace_back(std::pair<unsigned int, unsigned int>(i,i));
 //Append the output wavefunction tensor network to the input+gates one:
 TensNet.appendNetwork(optNet,legPairs);
 //Mark the tensors to be optimized:
 for(unsigned int i = 0; i < numOutLegs; ++i) OptimizedTensors.push_back(++numTensors);
 assert(numTensors == TensNet.getNumTensors());
 return;
}

int ExaTensorMPSVisitor::appendNBodyGate(const Tensor & gate, const unsigned int qubit_id[])
{
 int error_code = 0;
 const unsigned int gateRank = gate.getRank(); //N-body gate (rank-2N)
 assert(gateRank > 0 && gateRank%2 == 0);
 const auto numQubits = gateRank/2;
 if(QubitRange.first < 0) QubitRange.first = StateMPS.size();
 if(QubitRange.second < 0) QubitRange.second = 0;
 unsigned int * qids = new unsigned int [numQubits];
 for(unsigned int i = 0; i < numQubits; ++i){
  assert(qubit_id[i] < StateMPS.size());
  qids[i] = qubit_id[i];
  if(qids[i] < QubitRange.first) QubitRange.first = qids[i];
  if(qids[i] > QubitRange.second) QubitRange.second = qids[i];
 }
 GateSequence.push_back(std::make_pair(gate,qids));
 if(EagerEval || GateSequence.size() >= MAX_GATES) error_code = this->evaluate();
 return error_code;
}

//Public visitor methods:

void ExaTensorMPSVisitor::visit(Hadamard & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(X & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Y & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Z & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Rx & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Ry & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Rz & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(CPhase & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(CNOT & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(CZ & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Swap & gate)
{
 unsigned int qbits[] = {static_cast<unsigned int>(gate.bits()[0]), static_cast<unsigned int>(gate.bits()[1])};
 std::cout << "Applying " << gate.name() << " @ {" << qbits[0] << "," << qbits[1] << "}" << std::endl;
 const Tensor & gateTensor = GateTensors.getTensor(gate);
 int error_code = this->appendNBodyGate(gateTensor,qbits); assert(error_code == 0);
 return;
}

void ExaTensorMPSVisitor::visit(Measure & gate)
{
 //`Implement
 return;
}

void ExaTensorMPSVisitor::visit(ConditionalFunction & condFunc)
{
 //`Implement
 return;
}

void ExaTensorMPSVisitor::visit(GateFunction & gateFunc)
{
 //`Implement
 return;
}

//Numerical evaluation:

bool ExaTensorMPSVisitor::isInitialized()
{
 return !StateMPS.empty();
}

bool ExaTensorMPSVisitor::isEvaluated()
{
 return GateSequence.empty();
}

void ExaTensorMPSVisitor::setEvaluationStrategy(const bool eagerEval)
{
 assert(TensNet.isEmpty());
 EagerEval = eagerEval;
 return;
}

void ExaTensorMPSVisitor::setInitialMPSValence(const std::size_t initialValence)
{
 assert(TensNet.isEmpty());
 assert(initialValence > 0);
 InitialValence = initialValence;
 return;
}

int ExaTensorMPSVisitor::evaluate()
{
 int error_code = 0;
 assert(this->isInitialized());
 assert(TensNet.isEmpty());
 if(!(GateSequence.empty())){
  buildWaveFunctionNetwork(QubitRange.first,QubitRange.second); //build the wavefunction tensor network
  appendGateSequence(); //apply gate tensors
  closeCircuitNetwork(); //close the circuit tensor network with the output wavefunction tensors (those to be optimized)
  std::vector<double> norms(OptimizedTensors.size(),1.0); //all optimized tensors are normalized to unity
  error_code = exatensor::optimizeOverlapMax(TensNet,OptimizedTensors,norms); //optimize output MPS wavefunction tensors
  //`Update the WaveFunction tensors with the optimized output tensors and destroy old wavefunction tensors
  //`Destroy the tensor network object, optimized tensors
  QubitRange.first = -1; QubitRange.second = -1;
 }
 return error_code;
}

}  // end namespace tnqvm

#endif //TNQVM_HAS_EXATENSOR
