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
#ifndef TNQVM_EXATENSORMPSVISITOR_HPP_
#define TNQVM_EXATENSORMPSVISITOR_HPP_

#ifdef TNQVM_HAS_EXATENSOR

#include <cstdlib>
#include <complex>
#include <vector>
#include <utility>

#include "TNQVMVisitor.hpp"

#include "GateFactory.hpp"

#include "tensornet.hpp"

using namespace xacc;
using namespace xacc::quantum;

namespace tnqvm {

class ExaTensorMPSVisitor : public TNQVMVisitor {

public:

//Static constants:
 static const std::size_t INITIAL_VALENCE = 2; //default initial dimension extent for virtual MPS indices
 static const unsigned int MAX_GATES = 8; //max number of gates in the gate sequence before evaluation

//Life cycle:
 ExaTensorMPSVisitor();
 virtual ~ExaTensorMPSVisitor();

 void initialize(std::shared_ptr<AcceleratorBuffer> buffer); //accelerator buffer
 void finalize();

//Visitor methods:
 void visit(Identity& gate) {}
 void visit(Hadamard & gate);
 void visit(X & gate);
 void visit(Y & gate);
 void visit(Z & gate);
 void visit(Rx & gate);
 void visit(Ry & gate);
 void visit(Rz & gate);
 void visit(CPhase & gate);
 void visit(CNOT & gate);
 void visit(CZ & gate);
 void visit(Swap & gate);
 void visit(Measure & gate);
 void visit(ConditionalFunction & condFunc);
 void visit(GateFunction & gateFunc);

//Numerical evaluation:
 bool isInitialized(); //returns TRUE if the wavefunction has been initialized
 bool isEvaluated(); //returns TRUE if the wavefunction is fully evaluated and gate sequence is empty, FALSE otherwise
 void setEvaluationStrategy(const bool eagerEval); //sets tensor network evaluation strategy
 void setInitialMPSValence(const std::size_t initialValence); //initial dimension extent for virtual MPS indices
 int evaluate(); //evaluates the constructed tensor network (returns an error or 0)

private:

//Type aliases:
 using TensDataType = GateFactory::TensDataType;
 using Tensor = GateFactory::Tensor;
 using TensorLeg = exatensor::TensorLeg;
 using TensorNetwork = exatensor::TensorNetwork<TensDataType>;
 using WaveFunction = std::vector<Tensor>;

//Gate factory member:
 GateFactory GateTensors;

//Data members:
 std::shared_ptr<AcceleratorBuffer> Buffer;        //accelerator buffer
 WaveFunction StateMPS;                            //MPS wave-function of qubits (MPS tensors)
 TensorNetwork TensNet;                            //currently constructed tensor network
 std::vector<std::pair<Tensor, unsigned int *>> GateSequence; //sequence of visited quantum gates before evaluation
 std::pair<int, int> QubitRange;                   //range of involved qubits in the current tensor network
 std::vector<unsigned int> OptimizedTensors;       //IDs of the tensors to be optimized in the closed tensor network
 std::size_t InitialValence;                       //initial dimension extent for virtual dimensions of MPS tensors
 bool EagerEval;                                   //if TRUE each gate will be applied immediately (defaults to FALSE)

//Private member functions:
 void initMPSTensor(Tensor & tensor); //initializes an MPS tensor to a pure |0> state
 void buildWaveFunctionNetwork(int firstQubit = 0, int lastQubit = -1); //builds a TensorNetwork object for the wavefunction of qubits [first:last]
 void appendGateSequence(); //appends gate tensors from the current gate sequence to the wavefunction tensor network of qubits [first:last]
 void closeCircuitNetwork(); //closes the circuit TensorNetwork object with output tensors (those to be optimized)
 int appendNBodyGate(const Tensor & gate, const unsigned int qubit_id[]); //appends (applies) an N-body quantum gate to the wavefunction

}; //end class ExaTensorMPSVisitor

} //end namespace tnqvm

#endif //TNQVM_HAS_EXATENSOR

#endif //TNQVM_EXATENSORMPSVISITOR_HPP_
