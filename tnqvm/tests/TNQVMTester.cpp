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
#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

using namespace xacc;
// using namespace xacc::quantum;

const std::string uccsdSrc = R"uccsdSrc(def foo(buffer, theta0,theta1):
   X(0)
   X(1)
   H(3)
   Rx(1.5708, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(1 * theta0, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   H(3)
   Rx(10.9956, 1)
   Rx(1.5708, 3)
   H(1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(-1 * theta0, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   Rx(10.9956, 3)
   H(1)
   H(3)
   Rx(1.5708, 2)
   H(1)
   H(0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   H(3)
   Rx(10.9956, 2)
   H(1)
   H(0)
   Rx(1.5708, 3)
   Rx(1.5708, 2)
   Rx(1.5708, 1)
   H(0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   Rx(10.9956, 3)
   Rx(10.9956, 2)
   Rx(10.9956, 1)
   H(0)
   H(3)
   H(2)
   H(1)
   Rx(1.5708, 0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(-0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   H(3)
   H(2)
   H(1)
   Rx(10.9956, 0)
   Rx(1.5708, 3)
   H(2)
   Rx(1.5708, 1)
   Rx(1.5708, 0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(-0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   Rx(10.9956, 3)
   H(2)
   Rx(10.9956, 1)
   Rx(10.9956, 0)
   Rx(1.5708, 3)
   H(2)
   H(1)
   H(0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   Rx(10.9956, 3)
   H(2)
   H(1)
   H(0)
   Rx(1.5708, 2)
   H(0)
   CNOT(0, 1)
   CNOT(1, 2)
   Rz(-1 * theta0, 2)
   CNOT(1, 2)
   CNOT(0, 1)
   Rx(10.9956, 2)
   H(0)
   H(2)
   Rx(1.5708, 0)
   CNOT(0, 1)
   CNOT(1, 2)
   Rz(1 * theta0, 2)
   CNOT(1, 2)
   CNOT(0, 1)
   H(2)
   Rx(10.9956, 0)
   Rx(1.5708, 3)
   Rx(1.5708, 2)
   H(1)
   Rx(1.5708, 0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   Rx(10.9956, 3)
   Rx(10.9956, 2)
   H(1)
   Rx(10.9956, 0)
   H(3)
   Rx(1.5708, 2)
   Rx(1.5708, 1)
   Rx(1.5708, 0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(-0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   H(3)
   Rx(10.9956, 2)
   Rx(10.9956, 1)
   Rx(10.9956, 0)
   H(3)
   H(2)
   Rx(1.5708, 1)
   H(0)
   CNOT(0, 1)
   CNOT(1, 2)
   CNOT(2, 3)
   Rz(-0.5 * theta1, 3)
   CNOT(2, 3)
   CNOT(1, 2)
   CNOT(0, 1)
   H(3)
   H(2)
   Rx(10.9956, 1)
   H(0)
)uccsdSrc";

TEST(TNQVMTester, checkKernelExecution) {
  auto acc = xacc::getAccelerator("tnqvm");//TNQVM acc;
  auto qreg1 = xacc::qalloc(3);

  auto provider = xacc::getIRProvider("quantum");
  auto f = provider->createComposite("foo", {});

  auto x = provider->createInstruction("X", 0);
  auto h = provider->createInstruction("H", 1);
  auto cn1 = provider->createInstruction("CNOT", { 1, 2 });
  auto cn2 = provider->createInstruction("CNOT", { 0, 1 });
  auto h2 = provider->createInstruction("H", 0);

  f->addInstruction(x);
  f->addInstruction(h);
  f->addInstruction(cn1);
  f->addInstruction(cn2);
  f->addInstruction(h2);
  acc->execute(qreg1, f);
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
