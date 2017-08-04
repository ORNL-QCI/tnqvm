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
 *   Initial implementation - Mengsu Chen 2017.7
 *
 **********************************************************************************/

#include "ITensorMPSVisitor.hpp"
#include "AllGateVisitor.hpp"
#include "itensor/all.h"
#include <complex>
#include <cstdlib>
#include <ctime>

namespace xacc{
namespace quantum{

    /// Constructor
    ITensorMPSVisitor::ITensorMPSVisitor(std::shared_ptr<TNQVMBuffer> accbuffer_in)
        : n_qbits (accbuffer_in->size()),
          accbuffer (accbuffer_in) {
        initWavefunc(n_qbits);
        // printWavefunc();
        std::srand(std::time(0));
        cbits.resize(n_qbits);
    }

	void ITensorMPSVisitor::visit(Hadamard& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        // 0 -> 0+1 where 0 is at position 1 of input axis(space)
        tGate.set(ind_in(1), ind_out(1), 1.);
        tGate.set(ind_in(1), ind_out(2), 1.);
        // 1 -> 0-1
        tGate.set(ind_in(2), ind_out(1), 1.);
        tGate.set(ind_in(2), ind_out(2), -1.);
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}

	void ITensorMPSVisitor::visit(CNOT& gate) {
        auto iqbit_in0_ori = gate.bits()[0];
        auto iqbit_in1_ori = gate.bits()[1];
        int iqbit_in0, iqbit_in1;
        if (iqbit_in0_ori<iqbit_in1_ori-1){
            permute_to(iqbit_in0_ori, iqbit_in1_ori-1);
            iqbit_in0 = iqbit_in1_ori-1;
            iqbit_in1 = iqbit_in1_ori;
        }else if (iqbit_in1_ori<iqbit_in0_ori-1){
            permute_to(iqbit_in1_ori, iqbit_in0_ori-1);
            iqbit_in0 = iqbit_in0_ori;
            iqbit_in1 = iqbit_in0_ori-1;
        }else{
            iqbit_in0 = iqbit_in0_ori;
            iqbit_in1 = iqbit_in1_ori;
        }
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in0<<" , "<<iqbit_in1<<std::endl;
        auto ind_in0 = ind_for_qbit(iqbit_in0); // control
        auto ind_in1 = ind_for_qbit(iqbit_in1);
        auto ind_out0 = itensor::Index(gate.getName(), 2);
        auto ind_out1 = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in0, ind_in1, ind_out0, ind_out1);
        tGate.set(ind_out0(1), ind_out1(1), ind_in0(1), ind_in1(1), 1.);
        tGate.set(ind_out0(1), ind_out1(2), ind_in0(1), ind_in1(2), 1.);
        tGate.set(ind_out0(2), ind_out1(1), ind_in0(2), ind_in1(2), 1.);
        tGate.set(ind_out0(2), ind_out1(2), ind_in0(2), ind_in1(1), 1.);
        auto tobe_svd = tGate * legMats[iqbit_in0] * bondMats[std::min(iqbit_in0, iqbit_in1)] * legMats[iqbit_in1];
        ITensor legMat(legMats[iqbit_in0].inds()[1], ind_out0), bondMat, restTensor;
        itensor::svd(tobe_svd, legMat, bondMat, restTensor, {"Cutoff", 1E-4});
        legMats[iqbit_in0] = legMat;
        bondMats[std::min(iqbit_in0, iqbit_in1)] = bondMat;
        kickback_ind(restTensor, restTensor.inds()[1]);
        legMats[iqbit_in1] = restTensor;
        if (iqbit_in0_ori<iqbit_in1_ori-1){
            permute_to(iqbit_in1_ori-1, iqbit_in0_ori);
        }else if (iqbit_in1_ori<iqbit_in0_ori-1){
            permute_to(iqbit_in0_ori-1, iqbit_in1_ori);
        }
        printWavefunc();
	}


	void ITensorMPSVisitor::visit(X& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        tGate.set(ind_out(1), ind_in(2), 1.);
        tGate.set(ind_out(2), ind_in(1), 1.);
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}

	void ITensorMPSVisitor::visit(Y& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        tGate.set(ind_out(1), ind_in(2), std::complex<double>(0,-1.));
        tGate.set(ind_out(2), ind_in(1), std::complex<double>(0,1.));
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}


	void ITensorMPSVisitor::visit(Z& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        tGate.set(ind_out(1), ind_in(1), 1.);
        tGate.set(ind_out(2), ind_in(2), -1.);
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}

    /** The inner product is carried out in the following way 
    *   so that:
    *   1. no need to prime legs or bonds
    *   2. no high rank tensor (eg. a rank-nqbits tensor) pops out during the computation
    *
    *    /\              /\
    *   L--L            L--L
    *   |  |            |  |
    *   b      ---->    b  b
    *   |               |  |
    *                   L--L
    *                   |  |
    *                   b
    *                   |
    *   where L is a legMat, b is a bondMat
    */
    double ITensorMPSVisitor::wavefunc_inner(){
        ITensor inner = itensor::conj(legMats[0]*bondMats[0]) * legMats[0];
        for(int i=1; i<n_qbits-1; ++i){
            inner = inner*itensor::conj(legMats[i]*bondMats[i]) * bondMats[i-1] * legMats[i];
        }
        inner = inner * itensor::conj(legMats[n_qbits-1]) * bondMats[n_qbits-2] * legMats[n_qbits-1];
        return itensor::norm(inner);
    }

    double ITensorMPSVisitor::average(int iqbit, const ITensor& op_tensor){
        ITensor inner;
        if (iqbit==0){
            auto bra = itensor::conj(legMats[0]*bondMats[0]) * op_tensor;
            bra.noprime();
            inner = bra * legMats[0];
        }else{
            inner = itensor::conj(legMats[0]*bondMats[0]) * legMats[0];
        }
        for(int i=1; i<n_qbits-1; ++i){
            if (i==iqbit){
                inner = inner * itensor::conj(legMats[i]*bondMats[i]) * op_tensor * bondMats[i-1] * legMats[i];
            }else{
                inner = inner*itensor::conj(legMats[i]*bondMats[i]) * bondMats[i-1] * legMats[i];
            }
        }
        if (iqbit==n_qbits-1){
            inner = inner * itensor::conj(legMats[n_qbits-1]) * op_tensor * bondMats[n_qbits-2] * legMats[n_qbits-1];
        }else{
            inner = inner * itensor::conj(legMats[n_qbits-1]) * bondMats[n_qbits-2] * legMats[n_qbits-1];
        }
        return itensor::norm(inner);
    }

	void ITensorMPSVisitor::visit(Measure& gate) {
        double rv = (std::rand()%1000000)/1000000.;
        auto iqbit_measured = gate.bits()[0];
        auto ind_measured = ind_for_qbit(iqbit_measured);
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_measured<<std::endl;
        auto ind_measured_p = ind_for_qbit(iqbit_measured);
        ind_measured_p.prime();

        auto tMeasure0 = itensor::ITensor(ind_measured, ind_measured_p);
        tMeasure0.set(ind_measured_p(1), ind_measured(1), 1.);
        double p0 = average(iqbit_measured,tMeasure0) / wavefunc_inner();
        accbuffer->aver_from_wavefunc *= (2*p0-1);

        std::cout<<"rv= "<<rv<<"   p0= "<<p0<<std::endl;

        if(rv<p0){
            cbits[iqbit_measured] = 0;
            legMats[iqbit_measured] = tMeasure0 * legMats[iqbit_measured]; // collapse wavefunction
            legMats[iqbit_measured].prime(ind_measured_p,-1);
        }else{
            cbits[iqbit_measured] = 1;
            auto tMeasure1 = itensor::ITensor(ind_measured, ind_measured_p);
            tMeasure1.set(ind_measured_p(2), ind_measured(2), 1.);
            legMats[iqbit_measured] = tMeasure1 * legMats[iqbit_measured]; // collapse wavefunction
            legMats[iqbit_measured].prime(ind_measured_p,-1);
        }
        printWavefunc();
	}

	void ITensorMPSVisitor::visit(ConditionalFunction& c) {
		auto classicalBitIdx = c.getConditionalQubit();
        std::cout<<"applying "<<c.getName()<<" @ "<<classicalBitIdx<<std::endl;
        if (cbits[classicalBitIdx]==1){ // TODO: add else
    		for (auto inst : c.getInstructions()) {
	    		inst->accept(this);
		    }
        }
	}

	void ITensorMPSVisitor::visit(Rx& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        double theta = boost::get<double>(gate.getParameter(0));
        tGate.set(ind_out(1), ind_in(1), std::cos(.5*theta));
        tGate.set(ind_out(1), ind_in(2), std::complex<double>(0,-1)*std::sin(.5*theta));
        tGate.set(ind_out(2), ind_in(1), std::complex<double>(0,-1)*std::sin(.5*theta));
        tGate.set(ind_out(2), ind_in(2), std::cos(.5*theta));
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}

	void ITensorMPSVisitor::visit(Ry& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        double theta = boost::get<double>(gate.getParameter(0));
        tGate.set(ind_out(1), ind_in(1), std::cos(.5*theta));
        tGate.set(ind_out(1), ind_in(2), -std::sin(.5*theta));
        tGate.set(ind_out(2), ind_in(1), std::sin(.5*theta));
        tGate.set(ind_out(2), ind_in(2), std::cos(.5*theta));
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}

	void ITensorMPSVisitor::visit(Rz& gate) {
        auto iqbit_in = gate.bits()[0];
        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in<<std::endl;
        auto ind_in = ind_for_qbit(iqbit_in);
        auto ind_out = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in, ind_out);
        double theta = boost::get<double>(gate.getParameter(0));
        tGate.set(ind_out(1), ind_in(1), std::exp(std::complex<double>(0,-.5*theta)));
        tGate.set(ind_out(2), ind_in(2), std::exp(std::complex<double>(0,.5*theta)));
        legMats[iqbit_in] = tGate * legMats[iqbit_in];
        printWavefunc();
	}

	void ITensorMPSVisitor::visit(CPhase& cp) {
        std::cerr<<"UNIMPLEMENTED!"<<std::endl;
		// auto angleStr = boost::lexical_cast<std::string>(cp.getParameter(0));
		// quilStr += "CPHASE("
		// 		+ angleStr
		// 		+ ") " + std::to_string(cp.bits()[0]) + " " + std::to_string(cp.bits()[1]) + "\n";
	}

    void ITensorMPSVisitor::permute_to(int iqbit, int iqbit_to){
        std::cout<<"permute "<<iqbit<<" to "<<iqbit_to<<std::endl;
        int delta = iqbit<iqbit_to ? 1 : -1;
        while(iqbit!=iqbit_to){
            Swap gate(iqbit, iqbit+delta);
            visit(gate);
            iqbit = iqbit+delta;
        }
    }

	void ITensorMPSVisitor::visit(Swap& gate) {
        auto iqbit_in0_ori = gate.bits()[0];
        auto iqbit_in1_ori = gate.bits()[1];
        int iqbit_in0, iqbit_in1;
        if (iqbit_in0_ori<iqbit_in1_ori-1){
            permute_to(iqbit_in0_ori, iqbit_in1_ori-1);
            iqbit_in0 = iqbit_in1_ori-1;
            iqbit_in1 = iqbit_in1_ori;
        }else if (iqbit_in1_ori<iqbit_in0_ori-1){
            permute_to(iqbit_in1_ori, iqbit_in0_ori-1);
            iqbit_in0 = iqbit_in0_ori;
            iqbit_in1 = iqbit_in0_ori-1;
        }else{
            iqbit_in0 = iqbit_in0_ori;
            iqbit_in1 = iqbit_in1_ori;
        }

        std::cout<<"applying "<<gate.getName()<<" @ "<<iqbit_in0<<" , "<<iqbit_in1<<std::endl;
        auto ind_in0 = ind_for_qbit(iqbit_in0); // control
        auto ind_in1 = ind_for_qbit(iqbit_in1);
        auto ind_out0 = itensor::Index(gate.getName(), 2);
        auto ind_out1 = itensor::Index(gate.getName(), 2);
        auto tGate = itensor::ITensor(ind_in0, ind_in1, ind_out0, ind_out1);
        tGate.set(ind_out0(1), ind_out1(1), ind_in0(1), ind_in1(1), 1.);
        tGate.set(ind_out0(1), ind_out1(2), ind_in0(2), ind_in1(1), 1.);
        tGate.set(ind_out0(2), ind_out1(1), ind_in0(1), ind_in1(2), 1.);
        tGate.set(ind_out0(2), ind_out1(2), ind_in0(2), ind_in1(2), 1.);
        auto tobe_svd = tGate * legMats[iqbit_in0] * bondMats[std::min(iqbit_in0, iqbit_in1)] * legMats[iqbit_in1];
        ITensor legMat(legMats[iqbit_in0].inds()[1], ind_out0), bondMat, restTensor;
        itensor::svd(tobe_svd, legMat, bondMat, restTensor, {"Cutoff", 1E-4});
        legMats[iqbit_in0] = legMat;
        bondMats[std::min(iqbit_in0,iqbit_in1)] = bondMat;
        kickback_ind(restTensor, restTensor.inds()[1]);
        legMats[iqbit_in1] = restTensor;
        if (iqbit_in0_ori<iqbit_in1_ori-1){
            permute_to(iqbit_in1_ori-1, iqbit_in0_ori);
        }else if (iqbit_in1_ori<iqbit_in0_ori-1){
            permute_to(iqbit_in0_ori-1, iqbit_in1_ori);
        }
        printWavefunc();
	}

    void ITensorMPSVisitor::kickback_ind(ITensor& tensor, const Index& ind){
        auto ind_p = itensor::prime(ind);
        ITensor identity(ind,ind_p);
        for (int i=1; i<=ind.m(); ++i){
            identity.set(ind(i),ind_p(i),1.);
        }
        tensor *= identity;
        tensor.prime(ind_p,-1);
    }

	void ITensorMPSVisitor::visit(GateFunction& f) {
		return;
	}

	ITensorMPSVisitor::~ITensorMPSVisitor() {}

    /// init the wave function tensor
    void ITensorMPSVisitor::initWavefunc(int n_qbits){
        std::vector<ITensor> tInitQbits;
        for(int i=0; i<n_qbits; ++i){
            Index ind_qbit("qbit",2);
            ITensor tInitQbit(ind_qbit);
            tInitQbit.set(ind_qbit(1), 1.);
            tInitQbits.push_back(tInitQbit);
            iqbit2iind.push_back(i);
        }
        Index ind_head("head",1);
        ITensor head(ind_head);
        head.set(ind_head(1), 1.);
        tInitQbits.push_back(head);
        wavefunc = tInitQbits[0];
        for(int i=1; i<n_qbits+1; ++i){
            wavefunc = wavefunc / tInitQbits[i];
        }
        reduce_to_MPS();
    }

    itensor::Index ITensorMPSVisitor::ind_for_qbit(int iqbit) const {
        if (legMats.size()<=iqbit){
            return wavefunc.inds()[iqbit];
        }else{
            return legMats[iqbit].inds()[0];
        }
    }

    void ITensorMPSVisitor::printWavefunc() const {
        std::cout<<">>>>>>>>----------wf--------->>>>>>>>>>\n";
        auto mps = legMats[0];
        for(int i=1; i<n_qbits;++i){
            mps *= bondMats[i-1];
            mps *= legMats[i];
        }

        unsigned long giind = 0;
        const int n_qbits = iqbit2iind.size();
        auto print_nz = [&giind, n_qbits, this](itensor::Cplx c){
            if(std::norm(c)>0){
                for(int iind=0; iind<n_qbits; ++iind){
                    auto spin = (giind>>iind) & 1UL;
                    std::cout<<spin;
                }
                std::cout<<"    "<<c<<std::endl;
            }
            ++giind;
        };
        auto normed_wf = mps / itensor::norm(mps);
        normed_wf.visit(print_nz);
        // itensor::PrintData(mps);
        std::cout<<"<<<<<<<<---------------------<<<<<<<<<<\n"<<std::endl;
    }

    /** The process of SVD is to decompose a tensor,
     *  for example, a rank 3 tensor T
     *  |                    |
     *  |                    |
     *  T====  becomes    legMat---bondMat---restTensor===
     
     *                       |                  |
     *                       |                  |
     *         becomes    legMat---bondMat---legMat---bondMat---restTensor---
     
    */      
    void ITensorMPSVisitor::reduce_to_MPS(){
        ITensor tobe_svd = wavefunc;
        ITensor bondMat, restTensor;
        Index last_rbond = wavefunc.inds()[n_qbits];
        for(int i=0; i<n_qbits-1; ++i){
            std::cout<<"i= "<<i<<std::endl;
            ITensor legMat(last_rbond, ind_for_qbit(i));
            itensor::svd(tobe_svd, legMat, bondMat, restTensor, {"Cutoff", 1E-4});
            legMats.push_back(legMat); // the indeces of legMat in order: leg, last_rbond, lbond
            bondMats.push_back(bondMat);
            tobe_svd = restTensor;
            last_rbond = bondMat.inds()[1];
        }
        Index ind_tail("tail",1);
        ITensor tail(ind_tail);
        tail.set(ind_tail(1),1.);
        legMats.push_back(restTensor / tail);
        printWavefunc();
    }

} // end namespace quantum
} // end namespace xacc