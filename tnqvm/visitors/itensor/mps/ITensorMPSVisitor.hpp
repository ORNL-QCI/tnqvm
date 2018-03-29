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
#ifndef QUANTUM_GATE_ACCELERATORS_TNQVM_ITensorMPSVisitor_HPP_
#define QUANTUM_GATE_ACCELERATORS_TNQVM_ITensorMPSVisitor_HPP_

#include <cstdlib>
#include "XACC.hpp"
#include "TNQVMVisitor.hpp"
#include "Cloneable.hpp"
#include "itensor/all.h"

namespace tnqvm{

class ITensorMPSVisitor: public TNQVMVisitor, public xacc::Cloneable<TNQVMVisitor> {
    using ITensor = itensor::ITensor;
    using Index = itensor::Index;
    using IndexVal = itensor::IndexVal;
public:
    ITensorMPSVisitor();
    virtual ~ITensorMPSVisitor();

	virtual std::shared_ptr<TNQVMVisitor> clone() {
		return std::make_shared<ITensorMPSVisitor>();
	}

    virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer);
    virtual void finalize() {

    }

	virtual const std::string name() const {
		return "itensor-mps";
	}

	virtual const std::string description() const {
		return "";
	}

	/**
	 * Return the last execute call's execution time in seconds.
	 *
	 * @return runtime The execution time in seconds.
	 */
	virtual const double getExecutionTime() {
		return execTime;
	}

    // one-qubit gates
    void visit(Identity& gate) {}
    void visit(Hadamard& gate);
    void visit(X& gate);
    void visit(Y& gate);
    void visit(Z& gate);
    void visit(Rx& gate);
    void visit(Ry& gate);
    void visit(Rz& gate);
    void visit(CPhase& cp);
    // two-qubit gates
    void visit(CNOT& gate);
    void visit(Swap& gate);
    void visit(CZ& gate);
    // others
    void visit(Measure& gate);
    void visit(ConditionalFunction& c);
    void visit(GateFunction& f);

private:
    double execTime = 0.0;
    double singleQubitTime = 1e-8;
    double twoQubitTime = 1e-7;

    itensor::ITensor wavefunc;
    std::vector<int> iqbit2iind;
    std::vector<int> cbits;

    std::vector<ITensor> bondMats;    // singular matricies
    std::vector<ITensor> legMats;     // matricies with physical legs

    std::vector<ITensor> bondMats_m;  // the snapshot for measurement
    std::vector<ITensor> legMats_m;

    std::set<int> iqbits_m;           // indecies of qbits to measure

    itensor::IndexSet legs;           // physical degree of freedom
    int n_qbits;
    bool snapped;

    bool verbose = false;

    /// init the wave function tensor
    void initWavefunc(int n_qbits);
    void initWavefunc_bysvd(int n_qbits);
    void reduce_to_MPS();
    Index ind_for_qbit(int iqbit) const ;
    void printWavefunc() const ;
    void permute_to(int iqbit, int iqbit_to);
    void kickback_ind(ITensor& tensor, const Index& ind);
    double wavefunc_inner();
    double average(int iqbit, const ITensor& op_tensor);
    itensor::ITensor tZ_measure_on(int iqbit_measured);
    double averZs(std::set<int> iqbits);
    void snap_wavefunc();
};

} // end namespace xacc
#endif
