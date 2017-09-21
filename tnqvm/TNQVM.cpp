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
#include "TNQVM.hpp"
#ifdef TNQVM_HAS_EXATENSOR
#include "ExaTensorMPSVisitor.hpp"
#endif
#include "ITensorVisitor.hpp"
#include "ITensorMPSVisitor.hpp"

namespace xacc{

namespace tnqvm {

std::shared_ptr<AcceleratorBuffer> TNQVM::createBuffer(
		const std::string& varId) {
	return createBuffer(varId, 100);
}

std::shared_ptr<AcceleratorBuffer> TNQVM::createBuffer(
		const std::string& varId, const int size) {
	if (!isValidBufferSize(size)) {
		XACCError("Invalid buffer size.");
	}
	auto derived_buffer = std::make_shared<TNQVMBuffer>(varId, size);
	derived_buffer->set_verbose(__verbose);
	std::shared_ptr<AcceleratorBuffer> buffer = derived_buffer;
	storeBuffer(varId, buffer);
	return buffer;
}

bool TNQVM::isValidBufferSize(const int NBits) {
	return NBits <= 1000;
}

void TNQVM::execute(std::shared_ptr<AcceleratorBuffer> buffer,
		const std::shared_ptr<xacc::Function> kernel) {

	auto options = RuntimeOptions::instance();
	std::string visitorType = "itensor-mps";
	if (options->exists("tnqvm-visitor")) {
		visitorType = (*options)["tnqvm-visitor"];
	}

	std::shared_ptr<BaseInstructionVisitor> visitor;
	if (visitorType == "itensor-mps") {

		visitor = std::make_shared<xacc::quantum::ITensorMPSVisitor>(
				std::dynamic_pointer_cast<TNQVMBuffer>(buffer));
	} else if (visitorType == "exatensor") {
#ifdef TNQVM_HAS_EXATENSOR
		visitor = std::make_shared<xacc::quantum::ExaTensorMPSVisitor>(
				std::dynamic_pointer_cast<TNQVMBuffer>(buffer));
#else
		XACCError("Cannot use ExaTensor MPS Visitor since TNQVM not built with ExaTensor.");
#endif
	} else {
		XACCError("Invalid TNQVM visitor string.");
	}

	InstructionIterator it(kernel);
	while (it.hasNext()) {
		auto nextInst = it.next();
		if (nextInst->isEnabled()) {
			nextInst->accept(visitor);
		}
	}

	// If we made it here with this exatensor string, then
	// we don't have to re-check that we built with ExaTensor
	if (visitorType == "exatensor") {
#ifdef TNQVM_HAS_EXATENSOR
		std::dynamic_pointer_cast<xacc::quantum::ExaTensorMPSVisitor>(visitor)->evaluate();
#endif
	}

}

}
}
