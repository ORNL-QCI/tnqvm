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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
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
 *   Initial implementation - Mengsu Chen 2017/7/17
 *
 **********************************************************************************/
#ifndef QUANTUM_GATE_ACCELERATORS_TNQVM_EXATENSORVISITOR_HPP_
#define QUANTUM_GATE_ACCELERATORS_TNQVM_EXATENSORVISITOR_HPP_

#include <complex>
#include <cstdlib>
#include <ctime>
#include "AllGateVisitor.hpp"
#include "tensor_network.hpp"

namespace xacc {
namespace quantum {

class ExaTensorVisitor : public AllGateVisitor {
    using TensDataType = std::complex<double>;
    using Tensor = exatensor::TensorDenseAdpt<TensDataType>;
    using TensorNetwork = exatensor::TensorNetwork<TensDataType>;
    using Bond = std::pair<unsigned int, unsigned int>;

private:
    TensorNetwork wavefunc;

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

public:
    /// Constructor
    ExaTensorVisitor(int n_qbits) {
        initWavefunc(n_qbits);
    }

    void visit(Hadamard& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(CNOT& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[16];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(2, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(X& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(Y& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(Z& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(Measure& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(ConditionalFunction& c) {}

    void visit(Rx& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(Ry& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(Rz& gate) {
        auto iqbit = gate.bits()[0];
        std::cout << "applying " << gate.getName() << " @ " << iqbit
                  << std::endl;
        TensDataType* p = new TensDataType[4];
        std::shared_ptr<TensDataType> body(p,
                                           [](TensDataType* p) { delete[] p; });
        auto tGate = nqbit_gate_tensor(1, body);
        std::vector<Bond> bonds{Bond(0, iqbit)};
        wavefunc.appendTensor(tGate, bonds);
        printWavefunc();
    }

    void visit(CPhase& gate) {}

    void visit(Swap& s) {}

    void visit(GateFunction& f) {}

    void evaluate() {


    }
    virtual ~ExaTensorVisitor() {}
};

}  // end namespace quantum
}  // end namespace xacc
#endif
