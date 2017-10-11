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
 *   Implementation - Dmitry Lyakh 2017/10/05;
 *
 **********************************************************************************/
#ifdef TNQVM_HAS_EXATENSOR

#include "ExaTensorMPSVisitor.hpp"

namespace xacc {
namespace quantum {

//Life cycle:

ExaTensorMPSVisitor::ExaTensorMPSVisitor(std::shared_ptr<TNQVMBuffer> buffer, const std::size_t initialValence):
 Buffer(buffer)
{
 assert(initialValence > 0);
 assert(StateMPS.size() == 0);
 auto numQubits = buffer->size();
 std::cout << "[ExaTensorMPSVisitor]: Constructing an MPS wavefunction for " << numQubits << " qubits ... ";
 //Construct initial MPS tensors for all qubits:
 const unsigned int rankMPS = 3; //MPS tensor rank
 const std::size_t dimExts[] = {initialValence,initialValence,BASE_SPACE_DIM}; //initial MPS tensor shape
 for(decltype(numQubits) i = 0; i < numQubits; ++i){
  StateMPS.emplace_back(Tensor(rankMPS,dimExts)); //construct a bodyless MPS tensor
  StateMPS[i].allocateBody(); //allocates MPS tensor body
  StateMPS[i].nullifyBody(); //sets all MPS tensor elements to zero
 }
 std::cout << "Done" << std::endl;
}

ExaTensorMPSVisitor::~ExaTensorMPSVisitor()
{
}

//Private member functions:

int ExaTensorMPSVisitor::apply1BodyGate(const Tensor & gate, const unsigned int q0)
{
 return 0;
}

int ExaTensorMPSVisitor::apply2BodyGate(const Tensor & gate, const unsigned int q0, const unsigned int q1)
{
 return 0;
}

int ExaTensorMPSVisitor::applyNBodyGate(const Tensor & gate, const unsigned int q[])
{
 return 0;
}

//Visitor methods:

void ExaTensorMPSVisitor::visit(Hadamard & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(X & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Y & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Z & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Rx & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Ry & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Rz & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(CPhase & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(CNOT & gate)
{
 auto qbit0 = gate.bits()[0];
 auto qbit1 = gate.bits()[1];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "," << qbit1 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Swap & gate)
{
 auto qbit0 = gate.bits()[0];
 auto qbit1 = gate.bits()[1];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "," << qbit1 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(Measure & gate)
{
 auto qbit0 = gate.bits()[0];
 std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
 return;
}

void ExaTensorMPSVisitor::visit(ConditionalFunction & condFunc)
{
 return;
}

void ExaTensorMPSVisitor::visit(GateFunction & gateFunc)
{
 return;
}

//Numerical evaluation:

int ExaTensorMPSVisitor::evaluate()
{
 return 0;
}

}  // end namespace quantum
}  // end namespace xacc

#endif //TNQVM_HAS_EXATENSOR
