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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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
#include "IRUtils.hpp"

namespace {
inline int getShotCountOption(const xacc::HeterogeneousMap &in_options) {
  int result = -1;
  if (in_options.keyExists<int>("shots")) {
    result = in_options.get<int>("shots");
  }
  return result;
}
} // namespace
namespace tnqvm {

const std::string TNQVM::DEFAULT_VISITOR_BACKEND = "itensor-mps";

void TNQVM::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::vector<std::shared_ptr<xacc::CompositeInstruction>> functions) {
  visitor = xacc::getService<TNQVMVisitor>(getVisitorName())->clone();
  // If in VQE mode and there are more than one kernels
  if (vqeMode && functions.size() > 1 && visitor->supportVqeMode()) {
    auto kernelDecomposed = ObservedAnsatz::fromObservedComposites(functions);
    // Always validate kernel decomposition in DEBUG
    assert(kernelDecomposed.validate(functions));
    visitor->setOptions(options);

    // Initialize the visitor
    visitor->initialize(buffer, getShotCountOption(options));
    visitor->setKernelName(kernelDecomposed.getBase()->name());
    // Walk the base IR tree, and visit each node
    InstructionIterator it(kernelDecomposed.getBase());
    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled() && !nextInst->isComposite()) {
        nextInst->accept(visitor);
      }
    }

    // Now we have a wavefunction that represents execution of the ansatz.
    // Run the observable sub-circuits (change of basis + measurements)
    auto obsCircuits = kernelDecomposed.getObservedSubCircuits();
    for (int i = 0; i < obsCircuits.size(); ++i) {
      auto tmpBuffer = std::make_shared<xacc::AcceleratorBuffer>(
          obsCircuits[i]->name(), buffer->size());
      double e = visitor->getExpectationValueZ(obsCircuits[i]);
      tmpBuffer->addExtraInfo("exp-val-z", e);
      buffer->appendChild(obsCircuits[i]->name(), tmpBuffer);
    }
    // Finalize the visitor
    visitor->finalize();
  }
  // Normal execution mode
  else {
    for (auto f : functions) {
      auto tmpBuffer =
          std::make_shared<xacc::AcceleratorBuffer>(f->name(), buffer->size());
      execute(tmpBuffer, f);
      buffer->appendChild(f->name(), tmpBuffer);
    }
  }

  return;
}

void TNQVM::execute(std::shared_ptr<xacc::AcceleratorBuffer> buffer,
                    const std::shared_ptr<xacc::CompositeInstruction> kernel) {
  // Get the visitor backend
  visitor = xacc::getService<TNQVMVisitor>(getVisitorName());
  visitor->setOptions(options);

  // Initialize the visitor
  visitor->initialize(buffer, getShotCountOption(options));
  visitor->setKernelName(kernel->name());
  // If this is an Exatn-MPS visitor, transform the kernel to nearest-neighbor
  // Note: currently, we don't support MPS aggregated blocks (multiple qubit MPS
  // tensors in one block). Hence, the circuit must always be transformed into
  // *nearest* neighbor only (distance = 1 for two-qubit gates).
  if (visitor->name() == "exatn-mps" || visitor->name() == "exatn-pmps") {
    auto opt = xacc::getService<xacc::IRTransformation>("lnn-transform");
    opt->apply(kernel, nullptr, {std::make_pair("max-distance", 1)});
    // std::cout << "After LNN transform: \n" << kernel->toString() << "\n";
  }

  // Walk the IR tree, and visit each node
  InstructionIterator it(kernel);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled()) {
      nextInst->accept(visitor);
    }
  }

  // Finalize the visitor
  visitor->finalize();
}

const std::vector<std::complex<double>>
TNQVM::getAcceleratorState(std::shared_ptr<CompositeInstruction> program) {
  // Get the visitor backend
  visitor = xacc::getService<TNQVMVisitor>(getVisitorName());

  int maxBit = 0;
  if (!xacc::optionExists("n-qubits")) {
    InstructionIterator it1(program);
    while (it1.hasNext()) {
      auto nextInst = it1.next();
      if (nextInst->isEnabled() && !nextInst->isComposite()) {
        for (auto &b : nextInst->bits())
          if (b > maxBit)
            maxBit = b;
      }
    }

    // FIXME for bug #??
    if (maxBit == 0)
      maxBit++;
  } else {
    maxBit = std::stoi(xacc::getOption("n-qubits")) - 1;
  }

  auto buffer = std::make_shared<xacc::AcceleratorBuffer>("q", maxBit + 1);

  // Initialize the visitor
  visitor->initialize(buffer, getShotCountOption(options));

  // Walk the IR tree, and visit each node
  InstructionIterator it(program);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled()) {
      nextInst->accept(visitor);
    }
  }

  // Finalize the visitor
  visitor->finalize();

  return visitor->getState();
}
} // namespace tnqvm
