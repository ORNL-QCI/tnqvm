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
 *   Initial API and implementation - Alex McCaskey
 *
 **********************************************************************************/
#ifndef TNQVM_TNQVM_HPP_
#define TNQVM_TNQVM_HPP_

#include "XACC.hpp"

namespace xacc{

namespace tnqvm {

class TNQVM: public Accelerator {
public:

	virtual void initialize(){
		__verbose = 0;
	}

	/**
	 * Create, store, and return an AcceleratorBuffer with the given
	 * variable id string. This string serves as a unique identifier
	 * for future lookups and reuse of the AcceleratorBuffer.
	 *
	 * @param varId
	 * @return
	 */
	std::shared_ptr<AcceleratorBuffer> createBuffer(const std::string& varId);

	/**
	 * Create, store, and return an AcceleratorBuffer with the given
	 * variable id string and of the given number of bits.
	 * The string id serves as a unique identifier
	 * for future lookups and reuse of the AcceleratorBuffer.
	 *
	 * @param varId
	 * @param size
	 * @return
	 */
	std::shared_ptr<AcceleratorBuffer> createBuffer(const std::string& varId,
			const int size);

	/**
	 * Return true if this Accelerator can allocated
	 * NBits number of bits.
	 * @param NBits
	 * @return
	 */
	virtual bool isValidBufferSize(const int NBits);

	/**
	 * Execute the simulation. Requires both a valid SimulatedQubits buffer and
	 * XACC IR Function instance modeling the quantum circuit.
	 *
	 * @param ir
	 */
	virtual void execute(std::shared_ptr<AcceleratorBuffer> buffer,
			const std::shared_ptr<xacc::Function> kernel);

	/**
	 * Execute a set of kernels with one remote call. Return
	 * a list of AcceleratorBuffers that provide a new view
	 * of the given one AcceleratorBuffer. The ith AcceleratorBuffer
	 * contains the results of the ith kernel execution.
	 *
	 * @param buffer The AcceleratorBuffer to execute on
	 * @param functions The list of IR Functions to execute
	 * @return tempBuffers The list of new AcceleratorBuffers
	 */
	virtual std::vector<std::shared_ptr<AcceleratorBuffer>> execute(
			std::shared_ptr<AcceleratorBuffer> buffer,
			const std::vector<std::shared_ptr<Function>> functions);

	/**
	 * Return all relevant TNQVM runtime options.
	 */
	virtual std::shared_ptr<options_description> getOptions() {
		auto desc = std::make_shared<options_description>(
				"TNQVM Accelerator Options");
		desc->add_options()("tnqvm-visitor", value<std::string>(),
				"Provide visitor to be used in mapping IR to a Tensor Network.")
				("tnqvm-list-visitors", "List the available visitors.");
		return desc;
	}

	virtual bool handleOptions(variables_map& map) {
		if (map.count("tnqvm-list-visitors")) {
			xacc::info("Available TNQVM Visitor: itensor-mps");
			xacc::info("Available TNQVM Visitor: exatensor-mps");
			return true;
		}
		return false;
	}


	/**
	 * This Accelerator models QPU Gate accelerators.
	 * @return
	 */
	virtual AcceleratorType getType() {
		return AcceleratorType::qpu_gate;
	}

	virtual const std::string name() const {
		return "tnqvm";
	}

	virtual const std::string description() const {
		return "XACC tensor netowrk quantum virtual machine (TNQVM) Accelerator";
	}

	/**
	 * We have no need to transform the IR for this Accelerator,
	 * so return an empty list
	 * @return
	 */
	virtual std::vector<std::shared_ptr<xacc::IRTransformation>> getIRTransformations() {
		std::vector<std::shared_ptr<xacc::IRTransformation>> v;
		return v;
	}

	virtual bool isPhysical() {
		return false;
	}

	/**
	 * The destructor
	 */
	virtual ~TNQVM() {
	}

    int verbose() const {return __verbose; }
	void verbose(int level) {__verbose = level;}
	void set_verbose(int level) {__verbose = level;}
    void mute  () { __verbose = 0; }
    void unmute() { __verbose = 1;} // default to 1

private:
	int __verbose;
};
}
}

#endif
