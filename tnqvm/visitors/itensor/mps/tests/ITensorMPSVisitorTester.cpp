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
#include "ITensorMPSVisitor.hpp"
#include "InstructionIterator.hpp"
#include <Eigen/Dense>
#include "xacc.hpp"
#include "IRProvider.hpp"
#include "xacc_service.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include "base/Gates.hpp"

using namespace xacc::quantum;
using namespace tnqvm;
using namespace xacc;

TEST(ITensorMPSVisitorTester, checkSimpleSimulation) {
  auto gateRegistry = xacc::getIRProvider("quantum");

  auto statePrep = gateRegistry->createComposite("statePrep", { "theta" });

  auto term0 = gateRegistry->createComposite("term0", { "theta" });
  
  const auto calculateExpectedResult = [](double in_theta) -> double {
    // Create a 2-qubit state vector for validation 
    auto expectedStateVector = AllocateStateVector(2);
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::Rx>(3.1415926));
    ApplySingleQubitGate(expectedStateVector, 1, GetGateMatrix<CommonGates::Ry>(3.1415926 / 2.0));
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::Rx>(7.8539752));
    ApplyCNOTGate(expectedStateVector, 1, 0);
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::Rz>(in_theta));
    ApplyCNOTGate(expectedStateVector, 1, 0);
    ApplySingleQubitGate(expectedStateVector, 1, GetGateMatrix<CommonGates::Ry>(7.8539752));
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::Rx>(3.1415926 / 2.0));

    // Return the expected-Z of the first qubit
    return -1.0 * (std::pow(std::abs(expectedStateVector[1]), 2) + std::pow(std::abs(expectedStateVector[3]), 2)) 
    + 1.0 * (std::pow(std::abs(expectedStateVector[0]), 2) + std::pow(std::abs(expectedStateVector[2]), 2));
  };  

  auto rx = gateRegistry->createInstruction("Rx", std::vector<std::size_t>{0});
  InstructionParameter p0(3.1415926);
  rx->setParameter(0, p0);
  
  auto ry = gateRegistry->createInstruction("Ry", std::vector<std::size_t>{1});
  InstructionParameter p1(3.1415926 / 2.0);
  ry->setParameter(0, p1);
  
  auto rx2 = gateRegistry->createInstruction("Rx", std::vector<std::size_t>{0});
  InstructionParameter p2(7.8539752);
  rx2->setParameter(0, p2);
 
  auto cnot1 = gateRegistry->createInstruction("CNOT", std::vector<std::size_t>{1, 0});
 
  auto rz = gateRegistry->createInstruction("Rz", std::vector<std::size_t>{0});
  InstructionParameter p3("theta");
  rz->setParameter(0, p3);
 
  auto cnot2 = gateRegistry->createInstruction("CNOT", std::vector<std::size_t>{1, 0});
  
  auto ry2 = gateRegistry->createInstruction("Ry", std::vector<std::size_t>{1});
  InstructionParameter p4(7.8539752);
  ry2->setParameter(0, p4);

  auto rx3 = gateRegistry->createInstruction("Rx", std::vector<std::size_t>{0});
  InstructionParameter p5(3.1415926 / 2.0);
  rx3->setParameter(0, p5);

  auto meas = gateRegistry->createInstruction("Measure", std::vector<std::size_t>{0});
  InstructionParameter p6(0);
  meas->setParameter(0, p6);

  term0->addInstruction(rx);
  term0->addInstruction(ry);
  term0->addInstruction(rx2);
  term0->addInstruction(cnot1);
  term0->addInstruction(rz);
  term0->addInstruction(cnot2);
  term0->addInstruction(ry2);
  term0->addInstruction(rx3);
  term0->addInstruction(meas);

  const auto run = [&term0](double theta) -> double {
    auto buffer = xacc::qalloc(2);
     // Get the visitor backend
    auto visitor = std::make_shared<ITensorMPSVisitor>();
    auto visCast = std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);
    
    // NOTE: there is bug in TNQVMBuffer::resetBuffer, it is currently doing *nothing*,
    // hence we cannot reuse the buffer between runs, i.e. must do xacc::qalloc() to guarantee initial state.
    buffer->resetBuffer();
    // Initialize the visitor
    visitor->initialize(buffer);

    std::vector<double> v{theta};
    auto evaled = term0->operator()(v);

    // Walk the IR tree, and visit each node
    InstructionIterator it(evaled);
    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled()) {
        nextInst->accept(visCast);
      }
    }

    // Finalize the visitor
    visitor->finalize();

    const auto result = buffer->getExpectationValueZ();
    return result;
  };

  const auto pi = 3.1415926;
  const auto epsilon = 1e-12;
  {
    const auto expectedValue = calculateExpectedResult(-pi);
    EXPECT_NEAR(expectedValue, run(-pi), epsilon);
  }
  {
    const auto expectedValue = calculateExpectedResult(-1.44159);
    EXPECT_NEAR(expectedValue, run(-1.44159), epsilon);
  }
  {
    const auto expectedValue = calculateExpectedResult(1.25841);
    EXPECT_NEAR(expectedValue, run(1.25841), epsilon);
  }
  {
    const auto expectedValue = calculateExpectedResult(1.85841);
    EXPECT_NEAR(expectedValue, run(1.85841), epsilon);
  }
  {
    const auto expectedValue = calculateExpectedResult(pi);
    EXPECT_NEAR(expectedValue,run(pi), epsilon);
  }  
}

TEST(ITensorMPSVisitorTester, checkOneQubitBug) {

  auto gateRegistry = xacc::getIRProvider("quantum");

  auto statePrep = gateRegistry->createComposite("statePrep", { "theta"});

  auto term0 = gateRegistry->createComposite("term0", { "theta" });

  auto rx = gateRegistry->createInstruction("Rx", std::vector<std::size_t>{0});
  InstructionParameter p0("theta");
  rx->setParameter(0, p0);

  auto ry = gateRegistry->createInstruction("Ry", std::vector<std::size_t>{1});
  InstructionParameter p1(3.1415926 / 2.0);
  ry->setParameter(0, p1);
  auto rz = gateRegistry->createInstruction("Rz", std::vector<std::size_t>{0});
  InstructionParameter p3("theta");
  rz->setParameter(0, p3);

  auto h = gateRegistry->createInstruction("H", std::vector<std::size_t>{0});

  auto meas = gateRegistry->createInstruction("Measure", std::vector<std::size_t>{0});
  InstructionParameter p6(0);
  meas->setParameter(0, p6);

  statePrep->addInstruction(rx);
  statePrep->addInstruction(rz);

  term0->addInstruction(statePrep);
  term0->addInstruction(h);
  term0->addInstruction(meas);

  auto buffer = xacc::qalloc(1);

  // Get the visitor backend
  auto visitor = std::make_shared<ITensorMPSVisitor>();
  auto visCast = std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

  auto run = [&](std::shared_ptr<ITensorMPSVisitor> visitor,
                 double theta) -> double {
    buffer->resetBuffer();
    // Initialize the visitor
    visitor->initialize(buffer);

    std::vector<double> v{theta};
    auto evaled = term0->operator()(v);

    // Walk the IR tree, and visit each node
    InstructionIterator it(evaled);

    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled()) {
        nextInst->accept(visCast);
      }
    }

    // Finalize the visitor
    visitor->finalize();
    return buffer->getExpectationValueZ();
  };

  auto pi = 3.14926;
  // UNCOMMENT TO SEE BUG: Looks like it's an edge case for one-qubit circuits:
  // in ITensorMPSVisitor::averZs, bondMats_m vector is empty.  
  // run(visitor, pi);
}

TEST(ITensorMPSVisitorTester, checkSampling) {

  auto gateRegistry = xacc::getIRProvider("quantum");

  auto term0 = gateRegistry->createComposite("term0");

  auto x1 = gateRegistry->createInstruction("X", std::vector<std::size_t>{0});
  auto x2 = gateRegistry->createInstruction("X", std::vector<std::size_t>{1});

  term0->addInstruction(x1);
  term0->addInstruction(x2);
  auto meas1 = gateRegistry->createInstruction("Measure", std::vector<std::size_t>{0});
  InstructionParameter p6(0);
  meas1->setParameter(0, p6);

  auto meas2 = gateRegistry->createInstruction("Measure", std::vector<std::size_t>{1});
  InstructionParameter p7(1);
  meas2->setParameter(0, p7);

  term0->addInstruction(meas1);
  term0->addInstruction(meas2);

  auto buffer = xacc::qalloc(2);

  // Get the visitor backend
  auto visitor = std::make_shared<ITensorMPSVisitor>();
  auto visCast = std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

  auto run = [&](std::shared_ptr<ITensorMPSVisitor> visitor,
                 double theta) -> double {
    buffer->resetBuffer();
    // Initialize the visitor
    visitor->initialize(buffer);

    std::vector<double> v{};
    auto evaled = term0->operator()(v);

    // Walk the IR tree, and visit each node
    InstructionIterator it(evaled);

    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled()) {
        nextInst->accept(visCast);
      }
    }

    // Finalize the visitor
    visitor->finalize();
    return buffer->getExpectationValueZ();
  };

  run(visitor, 0.0);

  auto mstrs = buffer->getMeasurementCounts();
  for (auto& kv : mstrs) {
    EXPECT_TRUE(kv.first == "11");
  }
}

TEST(ITensorMPSVisitorTester, checkDeuteuron) {
  auto accelerator = xacc::getAccelerator("tnqvm");
  auto xasmCompiler = xacc::getCompiler("xasm");
  
  const auto calculateExpectedResult = [](double in_theta) -> double {
    // Create a 2-qubit state vector for validation 
    auto expectedStateVector = AllocateStateVector(2);
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::X>());
    ApplySingleQubitGate(expectedStateVector, 1, GetGateMatrix<CommonGates::Ry>(in_theta));
    ApplyCNOTGate(expectedStateVector, 1, 0);
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::H>());
    ApplySingleQubitGate(expectedStateVector, 1, GetGateMatrix<CommonGates::H>());
    const bool result = ApplyMeasureOp(expectedStateVector, 0);
    if (result)
    {
      // Q0 is 1 => 01 and 11  
      EXPECT_NEAR(std::norm(expectedStateVector[0]), 0.0, 1e-12);
      EXPECT_NEAR(std::norm(expectedStateVector[2]), 0.0, 1e-12);
      return std::norm(expectedStateVector[1]) - std::norm(expectedStateVector[3]);     
    }
    else
    {
      // Q0 is 0 => 00 and 10
      EXPECT_NEAR(std::norm(expectedStateVector[1]), 0.0, 1e-12);
      EXPECT_NEAR(std::norm(expectedStateVector[3]), 0.0, 1e-12);
      return std::norm(expectedStateVector[0]) - std::norm(expectedStateVector[2]); 
    }
  };  
  
  auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
      H(q[0]);
      H(q[1]);
      Measure(q[0]);
      Measure(q[1]);
  })", accelerator);

  auto program = ir->getComposite("ansatz");

  const auto angles = linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  for (const auto &a : angles) {
    auto buffer = xacc::qalloc(2);
    auto evaled = program->operator()({a});
    accelerator->execute(buffer, evaled);
    EXPECT_NEAR(std::abs(buffer->getExpectationValueZ()), std::abs(calculateExpectedResult(a)), 1e-12);
  }
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
