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
#ifndef QUANTUM_GATE_ACCELERATORS_TNQVM_EXATENSORMPSVISITOR_HPP_
#define QUANTUM_GATE_ACCELERATORS_TNQVM_EXATENSORMPSVISITOR_HPP_

#ifdef TNQVM_HAS_EXATENSOR

#include <cstdlib>
#include <complex>
#include <vector>
#include "AllGateVisitor.hpp"
#include "TNQVMBuffer.hpp"

#include "exatensor.hpp"

namespace xacc {
namespace quantum {

class ExaTensorMPSVisitor : public AllGateVisitor {

private:

//Type aliases:
 using TensDataType = std::complex<double>;
 using Tensor = exatensor::TensorDenseAdpt<TensDataType>;
 using TensorNetwork = exatensor::TensorNetwork<TensDataType>;
 using Bond = std::pair<unsigned int, unsigned int>;
 using WaveFunction = std::vector<Tensor>;

//Data members:
 std::shared_ptr<TNQVMBuffer> Buffer; //accelerator buffer
 WaveFunction StateMPS; //MPS wave-function of qubits

public:

//Life cycle:

    ExaTensorMPSVisitor(std::shared_ptr<TNQVMBuffer> buffer): Buffer(buffer)
    {
     auto numQubits = buffer->size();
    }

    virtual ~ExaTensorMPSVisitor(){}

//Visit methods:

    void visit(Hadamard & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(X & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(Y & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(Z & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(Rx & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(Ry & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(Rz & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(CPhase & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(CNOT & gate)
    {
     auto qbit0 = gate.bits()[0];
     auto qbit1 = gate.bits()[1];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "," << qbit1 << "}" << std::endl;
     return;
    }

    void visit(Swap & gate)
    {
     auto qbit0 = gate.bits()[0];
     auto qbit1 = gate.bits()[1];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "," << qbit1 << "}" << std::endl;
     return;
    }

    void visit(Measure & gate)
    {
     auto qbit0 = gate.bits()[0];
     std::cout << "Applying " << gate.getName() << " @ {" << qbit0 << "}" << std::endl;
     return;
    }

    void visit(ConditionalFunction & condFunc)
    {
     return;
    }

    void visit(GateFunction & gateFunc)
    {
     return;
    }

    void evaluate()
    {
     return;
    }

}; //end class ExaTensorMPSVisitor

}  // end namespace quantum
}  // end namespace xacc
#endif

//DEPRECATED:
#if 0
    Tensor nqbit_gate_tensor(unsigned int n_qbits,
                             std::shared_ptr<TensDataType> body) {
        unsigned int rank = 2 * n_qbits;
        std::size_t dims[n_qbits];
        for (unsigned int i = 0; i < rank; ++i) {
            dims[i] = 2;
        }
        std::size_t vol = 1;
        return Tensor(rank, dims, body);
    }

    /// init the wave function tensor
    void initWavefunc(unsigned int n_qbits) {
        unsigned int rank = n_qbits;
        std::size_t dims[n_qbits];
        for (unsigned int i = 0; i < rank; ++i) {
            dims[i] = 2;
        }
        std::size_t vol = std::pow(2, n_qbits);
        TensDataType* p = new TensDataType[vol];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        Tensor initTensor(rank, dims, body);
        std::vector<Bond> bonds;
        wavefunc.appendTensor(initTensor, bonds);
        printWavefunc();
    }

    void printWavefunc() const {
        std::cout<<"------wave function---->>\n";
        wavefunc.printIt();
        std::cout<<"<<----wave function------\n"<<std::endl;
    }

#endif //TNQVM_HAS_EXATENSOR

#endif //QUANTUM_GATE_ACCELERATORS_TNQVM_EXATENSORMPSVISITOR_HPP_
