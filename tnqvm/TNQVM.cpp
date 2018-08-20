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
//#include "ExaTensorMPSVisitor.hpp"

namespace tnqvm {

std::shared_ptr<AcceleratorBuffer> TNQVM::createBuffer(
		const std::string& varId) {
	return createBuffer(varId, 100);
}

std::shared_ptr<AcceleratorBuffer> TNQVM::createBuffer(
		const std::string& varId, const int size) {
	if (!isValidBufferSize(size)) {
		xacc::error("TNQVM - Invalid buffer size.");
	}
	auto buffer = std::make_shared<TNQVMBuffer>(varId, size);
	storeBuffer(varId, buffer);
	return buffer;
}

bool TNQVM::isValidBufferSize(const int NBits) {
	return NBits <= 1000;
}

std::vector<std::shared_ptr<AcceleratorBuffer>> TNQVM::execute(
		std::shared_ptr<AcceleratorBuffer> buffer,
		const std::vector<std::shared_ptr<Function>> functions) {

	int counter = 0;
	std::vector<std::shared_ptr<AcceleratorBuffer>> tmpBuffers;
    
    if (xacc::optionExists("run-and-measure")) {
        // Here we assume we have one ansatz function, functions[0].
        // The rest are measurements to be made.
        // Get the visitor backend
	    visitor = xacc::getService<TNQVMVisitor>("itensor-mps");

	    // Initialize the visitor
	    visitor->initialize(buffer);

	    // Walk the IR tree, and visit each node
	    InstructionIterator it(functions[0]);
	    while (it.hasNext()) {
		    auto nextInst = it.next();
		    if (nextInst->isEnabled()) {
			    nextInst->accept(
					    visitor);
		    }
	    }

        // Now we have a wavefunction that represents 
        // execution of the ansatz. Make measurements
        for (int i = 1; i < functions.size(); i++) {
            double exp = 1.0;
            auto tmpBuffer = createBuffer(
                    buffer->name() + std::to_string(i-1), buffer->size());
                    
            if (functions[i]->nInstructions() > 0) {
                exp = visitor->getExpectationValueZ(functions[i]);
            }
            
            tmpBuffer->setExpectationValueZ(exp);
            tmpBuffers.push_back(tmpBuffer);
        }

    } else {
	    for (auto f : functions) {
		    auto tmpBuffer = createBuffer(
				    buffer->name() + std::to_string(counter), buffer->size());
		    execute(tmpBuffer, f);
		    tmpBuffers.push_back(tmpBuffer);
		    counter++;
	    }
    }

	return tmpBuffers;
}

void TNQVM::execute(std::shared_ptr<AcceleratorBuffer> buffer,
		const std::shared_ptr<xacc::Function> kernel) {

	if (!std::dynamic_pointer_cast<TNQVMBuffer>(buffer)) {
		xacc::error("Invalid AcceleratorBuffer, must be a TNQVMBuffer.");
	}

	std::string visitorType = "itensor-mps";
	if (xacc::optionExists("tnqvm-visitor")) {
		visitorType = xacc::getOption("tnqvm-visitor");
	}

	// Get the visitor backend
	visitor = xacc::getService<TNQVMVisitor>(visitorType);

	// Initialize the visitor
	visitor->initialize(buffer);

	// Walk the IR tree, and visit each node
	InstructionIterator it(kernel);
	while (it.hasNext()) {
		auto nextInst = it.next();
		if (nextInst->isEnabled()) {
			nextInst->accept(
					visitor);
		}
	}

	// Finalize the visitor
	visitor->finalize();
}

const std::vector<std::complex<double>> TNQVM::getAcceleratorState(std::shared_ptr<Function> program) {
    std::string visitorType = "itensor-mps";
	if (xacc::optionExists("tnqvm-visitor")) {
		visitorType = xacc::getOption("tnqvm-visitor");
	}

	// Get the visitor backend
	visitor = xacc::getService<TNQVMVisitor>(visitorType);

    int maxBit = 0;
	InstructionIterator it1(program);
	while (it1.hasNext()) {
		auto nextInst = it1.next();
		if (nextInst->isEnabled() && !nextInst->isComposite()) {
            for (auto& b : nextInst->bits()) if (b > maxBit) maxBit = b;
		}

    }

    // FIXME for bug #??
    if(maxBit == 0) maxBit++;
    
    auto buffer = std::make_shared<TNQVMBuffer>("q",maxBit+1);

	// Initialize the visitor
	visitor->initialize(buffer);

	// Walk the IR tree, and visit each node
	InstructionIterator it(program);
	while (it.hasNext()) {
		auto nextInst = it.next();
		if (nextInst->isEnabled()) {
			nextInst->accept(
					visitor);
		}
	}

	// Finalize the visitor
	visitor->finalize();
    
    return visitor->getState();
}

}
