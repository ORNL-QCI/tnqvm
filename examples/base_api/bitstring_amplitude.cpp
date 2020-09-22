/***********************************************************************************
 * Copyright (c) 2020, UT-Battelle
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
 *   Initial implementation - Thien Nguyen
 *
 **********************************************************************************/

// This example demonstrates bitstring amplitude calculation API.
#include "xacc.hpp"

int main(int argc, char **argv) {

  // Initialize the XACC Framework
  xacc::Initialize(argc, argv);

  // Allocate a register of 50 qubits
  // Note: since we only compute the amplitude of a single output state,
  // we can handle many qubits.
  auto qubitReg = xacc::qalloc(50);

  // Output state: all ones's
  const std::vector<int> bitstring(50, 1);
  // Using the ExaTN backend:
  // specify the bitstring to compute the amplitude.
  auto qpu = xacc::getAccelerator("tnqvm", {
                                               {"tnqvm-visitor", "exatn"},
                                               {"bitstring", bitstring},
                                           });
  // Create a Program: creare an entangled state:
  // |00000..00> + |11111....11>
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test(qbit q, double theta) {
    H(q[0]);
    for (int i = 1; i < 50; i++) {
      CX(q[0], q[i]); 
    }
  })",
                                  qpu);

  // Request the quantum kernel representing
  // the above source code
  auto program = ir->getComposite("test");
  // Execute!
  qpu->execute(qubitReg, program);
  // Print the result in the buffer:
  // i.e. the amplitude of the |1111..111> output state.
  // Expected = 1/sqrt(2) = 0.707
  qubitReg->print();

  // Finalize the XACC Framework
  xacc::Finalize();

  return 0;
}
