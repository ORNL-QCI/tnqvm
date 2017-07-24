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
 *   Initial implementation - Mengsu Chen 2017/7/17
 *
 **********************************************************************************/
#ifndef QUANTUM_GATE_ACCELERATORS_TNQVM_TENSORVISITOR_HPP_
#define QUANTUM_GATE_ACCELERATORS_TNQVM_TENSORVISITOR_HPP_

#include "AllGateVisitor.hpp"
#include "tensor_expression.hpp"
#include "itensor/all.h"
#include <complex>

namespace xacc{
namespace quantum{

class TensorVisitor: public AllGateVisitor {
    using ITensor = itensor::ITensor;
    using Index = itensor::Index;
    using IndexVal = itensor::IndexVal;
private:
    itensor::ITensor wavefunc;
    std::vector<int> iqbit2iind;
    std::vector<int> cbits;

    /// init the wave function tensor
    void initWavefunc(int n_qbits){
        std::vector<ITensor> tInitQbits;
        for(int i=0; i<n_qbits; ++i){
            Index ind_qbit("qbit",2);
            ITensor tInitQbit(ind_qbit);
            tInitQbit.set(ind_qbit(1), 1.);
            tInitQbits.push_back(tInitQbit);
            iqbit2iind.push_back(i);
        }
        wavefunc = tInitQbits[0];
        for(int i=1; i<n_qbits; ++i){
            wavefunc = wavefunc / tInitQbits[i];
        }
        itensor::println("wavefunc=%s", wavefunc);
        itensor::PrintData(wavefunc);
    }

    void print_iqbit2iind() const{
        for(int i=0; i<iqbit2iind.size()-1; ++i){
            std::cout<<iqbit2iind[i]<<", ";
        }
        std::cout<<*(iqbit2iind.end()-1)<<std::endl;
    }

    Index getIndIn(int iqbit_in) const {
        print_iqbit2iind();
        auto wf_inds = wavefunc.inds();
        auto iind_in = iqbit2iind[iqbit_in];
        auto ind_in = wf_inds[iind_in];
        return ind_in;       
    }

    void endVisit(int iqbit_in) {
        auto iind = iqbit2iind[iqbit_in];
        for(int iqbit=0; iqbit<iqbit2iind.size(); ++iqbit){
            if (iqbit2iind[iqbit]>iind){
                iqbit2iind[iqbit]--;
            }
        }
        iqbit2iind[iqbit_in] = iqbit2iind.size()-1;
        // iqbit2iind.insert(iqbit2iind.begin()+iqbit_in, iqbit2iind.size()-1);
        // iqbit2iind.pop_back();
        print_iqbit2iind();
    }

public:

    /// Constructor
    TensorVisitor(){
        int n_qbits = 3;
        initWavefunc(n_qbits);
    }


	void visit(Hadamard& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = getIndIn(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        // 0 -> 0+1 where 0 is at position 1 of input axis(space)
        tGate.set(ind_in(1), ind_out(1), 1.);
        tGate.set(ind_in(1), ind_out(2), 1.);
        // 1 -> 0-1
        tGate.set(ind_in(2), ind_out(1), 1.);
        tGate.set(ind_in(2), ind_out(2), -1.);
        wavefunc *= tGate;
        endVisit(iqbit_in);
        itensor::PrintData(wavefunc);
	}

	void visit(CNOT& gate) {
        auto iqbit_in0 = gate.bits()[0];
        auto iqbit_in1 = gate.bits()[1];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in0<<" , "<<iqbit_in1<<std::endl;
        auto ind_in0 = getIndIn(iqbit_in0); // control
        auto ind_in1 = getIndIn(iqbit_in1);
        auto ind_out0 = itensor::Index(gate.getName(), 2);
        auto ind_out1 = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in0, ind_in1, ind_out0, ind_out1);
        tGate.set(ind_in0(1), ind_in1(1), ind_out0(1), ind_out1(1), 1.);
        tGate.set(ind_in0(1), ind_in1(2), ind_out0(1), ind_out1(2), 1.);
        tGate.set(ind_in0(2), ind_in1(2), ind_out0(2), ind_out1(1), 1.);
        tGate.set(ind_in0(2), ind_in1(1), ind_out0(2), ind_out1(2), 1.);
        wavefunc *= tGate;
        endVisit(iqbit_in0);
        endVisit(iqbit_in1);
        itensor::PrintData(wavefunc);
	}


	void visit(X& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = getIndIn(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        tGate.set(ind_in(1), ind_out(2), 1.);
        tGate.set(ind_in(2), ind_out(1), 1.);
        wavefunc *= tGate;
        endVisit(iqbit_in);
        itensor::PrintData(wavefunc);
	}

	void visit(Y& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = getIndIn(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        tGate.set(ind_in(1), ind_out(2), std::complex<double>(0,1.));
        tGate.set(ind_in(2), ind_out(1), std::complex<double>(0,-1.));
        wavefunc *= tGate;
        endVisit(iqbit_in);
        itensor::PrintData(wavefunc);
	}


	void visit(Z& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = getIndIn(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        tGate.set(ind_in(1), ind_out(1), 1.);
        tGate.set(ind_in(2), ind_out(2), -1.);
        wavefunc *= tGate;
        endVisit(iqbit_in);
        itensor::PrintData(wavefunc);
	}

	void visit(Measure& gate) {
        auto tmp = wavefunc.conj() * wavefunc;
        // auto iqbit_measured = gate.bits()[0];
        // auto ind_measured = getIndIn(iqbit_in);
        // auto tMeasurer0 = itensor::ITensor()
		// int classicalBitIdx = m.getClassicalBitIndex();
		// quilStr += "MEASURE " + std::to_string(m.bits()[0]) + " [" + std::to_string(classicalBitIdx) + "]\n";
		// classicalAddresses += std::to_string(classicalBitIdx) + ", ";
		// numAddresses++;
		// qubitToClassicalBitIndex.insert(std::make_pair(m.bits()[0], classicalBitIdx));
	}

	void visit(ConditionalFunction& c) {
		// auto visitor = std::make_shared<QuilVisitor>();
		// auto classicalBitIdx = qubitToClassicalBitIndex[c.getConditionalQubit()];
		// quilStr += "JUMP-UNLESS @" + c.getName() + " [" + std::to_string(classicalBitIdx) + "]\n";
		// for (auto inst : c.getInstructions()) {
		// 	inst->accept(visitor);
		// }
		// quilStr += visitor->getQuilString();
		// quilStr += "LABEL @" + c.getName() + "\n";
	}

	void visit(Rx& rx) {
		// auto angleStr = boost::lexical_cast<std::string>(rx.getParameter(0));
		// quilStr += "RX("
		// 		+ angleStr
		// 		+ ") " + std::to_string(rx.bits()[0]) + "\n";
	}

	void visit(Ry& ry) {
		// auto angleStr = boost::lexical_cast<std::string>(ry.getParameter(0));
		// quilStr += "RY("
		// 		+ angleStr
		// 		+ ") " + std::to_string(ry.bits()[0]) + "\n";
	}

	void visit(Rz& rz) {
		// auto angleStr = boost::lexical_cast<std::string>(rz.getParameter(0));
		// quilStr += "RZ("
		// 		+ angleStr
		// 		+ ") " + std::to_string(rz.bits()[0]) + "\n";
	}

	void visit(CPhase& cp) {
		// auto angleStr = boost::lexical_cast<std::string>(cp.getParameter(0));
		// quilStr += "CPHASE("
		// 		+ angleStr
		// 		+ ") " + std::to_string(cp.bits()[0]) + " " + std::to_string(cp.bits()[1]) + "\n";
	}

	void visit(Swap& s) {
		// quilStr += "SWAP " + std::to_string(s.bits()[0]) + " " + std::to_string(s.bits()[1]) + "\n";
	}

	void visit(GateFunction& f) {
		return;
	}

	virtual ~TensorVisitor() {}
};

} // end namespace quantum
} // end namespace xacc
#endif