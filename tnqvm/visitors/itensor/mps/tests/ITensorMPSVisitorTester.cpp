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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ITensorMPSVisitorTester

#include <memory>
#include <gtest/gtest.h>
#include "ITensorMPSVisitor.hpp"
#include "InstructionIterator.hpp"
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "XACC.hpp"
#include "IRProvider.hpp"

using namespace xacc::quantum;
using namespace tnqvm;
using namespace xacc;

TEST(ITensorMPSVisitorTester, checkSimpleSimulation) {

  auto gateRegistry = xacc::getService<IRProvider>("gate");

  auto statePrep = gateRegistry->createFunction(
      "statePrep", std::vector<int>{0, 1},
      std::vector<InstructionParameter>{InstructionParameter("theta")});

  auto term0 = gateRegistry->createFunction(
      "term0", std::vector<int>{0, 1},
      std::vector<InstructionParameter>{InstructionParameter("theta")});

  auto rx = gateRegistry->createInstruction("Rx", std::vector<int>{0});
  InstructionParameter p0(3.1415926);
  rx->setParameter(0, p0);

  auto ry = gateRegistry->createInstruction("Ry", std::vector<int>{1});
  InstructionParameter p1(3.1415926 / 2.0);
  ry->setParameter(0, p1);

  auto rx2 = gateRegistry->createInstruction("Rx", std::vector<int>{0});
  InstructionParameter p2(7.8539752);
  rx2->setParameter(0, p2);

  auto cnot1 = gateRegistry->createInstruction("CNOT", std::vector<int>{1, 0});

  auto rz = gateRegistry->createInstruction("Rz", std::vector<int>{0});
  InstructionParameter p3("theta");
  rz->setParameter(0, p3);

  auto cnot2 = gateRegistry->createInstruction("CNOT", std::vector<int>{1, 0});

  auto ry2 = gateRegistry->createInstruction("Ry", std::vector<int>{1});
  InstructionParameter p4(7.8539752);
  ry2->setParameter(0, p4);

  auto rx3 = gateRegistry->createInstruction("Rx", std::vector<int>{0});
  InstructionParameter p5(3.1415926 / 2.0);
  rx3->setParameter(0, p5);

  auto meas = gateRegistry->createInstruction("Measure", std::vector<int>{0});
  InstructionParameter p6(0);
  meas->setParameter(0, p6);

  statePrep->addInstruction(rx);
  statePrep->addInstruction(ry);
  statePrep->addInstruction(rx2);
  statePrep->addInstruction(cnot1);
  statePrep->addInstruction(rz);
  statePrep->addInstruction(cnot2);
  statePrep->addInstruction(ry2);
  statePrep->addInstruction(rx3);

  term0->addInstruction(statePrep);
  term0->addInstruction(meas);

  auto buffer = std::make_shared<TNQVMBuffer>("qreg", 2);

  // Get the visitor backend
  auto visitor = std::make_shared<ITensorMPSVisitor>();
  auto visCast = std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

  auto run = [&](std::shared_ptr<ITensorMPSVisitor> visitor,
                 double theta) -> double {
    buffer->resetBuffer();
    // Initialize the visitor
    visitor->initialize(buffer);

    Eigen::VectorXd v(1);
    v(0) = theta;
    auto evaled = term0->operator()(v);

    // Walk the IR tree, and visit each node
    InstructionIterator it(evaled);
    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled()) {
        std::cout << "HELLO: " << nextInst->name() << "\n";
        nextInst->accept(visCast);
      }
    }

    // Finalize the visitor
    visitor->finalize();
    return buffer->getExpectationValueZ();
  };

  auto pi = boost::math::constants::pi<double>();

  EXPECT_NEAR(-1, run(visitor, -pi), 1e-8); // < 1e-8);
  //	EXPECT_NEAR(-0.128844, run(visitor, -1.44159), 1e-5);
  //	EXPECT_NEAR(0.307333, run(visitor, 1.25841), 1e-8);
  //	EXPECT_NEAR(-.283662, run(visitor, 1.85841), 1e-8);
  //	EXPECT_NEAR(-1,run(visitor, pi), 1e-8);
}

TEST(ITensorMPSVisitorTester, checkOneQubitBug) {

  auto gateRegistry = xacc::getService<IRProvider>("gate");

  auto statePrep = gateRegistry->createFunction(
      "statePrep", std::vector<int>{0, 1},
      std::vector<InstructionParameter>{InstructionParameter("theta")});

  auto term0 = gateRegistry->createFunction(
      "term0", std::vector<int>{0, 1},
      std::vector<InstructionParameter>{InstructionParameter("theta")});

  auto rx = gateRegistry->createInstruction("Rx", std::vector<int>{0});
  InstructionParameter p0("theta");
  rx->setParameter(0, p0);

  auto ry = gateRegistry->createInstruction("Ry", std::vector<int>{1});
  InstructionParameter p1(3.1415926 / 2.0);
  ry->setParameter(0, p1);
  auto rz = gateRegistry->createInstruction("Rz", std::vector<int>{0});
  InstructionParameter p3("theta");
  rz->setParameter(0, p3);

  auto h = gateRegistry->createInstruction("H", std::vector<int>{0});

  auto meas = gateRegistry->createInstruction("Measure", std::vector<int>{0});
  InstructionParameter p6(0);
  meas->setParameter(0, p6);

  statePrep->addInstruction(rx);
  statePrep->addInstruction(rz);

  term0->addInstruction(statePrep);
  term0->addInstruction(h);
  term0->addInstruction(meas);

  auto buffer = std::make_shared<TNQVMBuffer>("qreg", 1);

  // Get the visitor backend
  auto visitor = std::make_shared<ITensorMPSVisitor>();
  auto visCast = std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

  auto run = [&](std::shared_ptr<ITensorMPSVisitor> visitor,
                 double theta) -> double {
    buffer->resetBuffer();
    // Initialize the visitor
    visitor->initialize(buffer);

    Eigen::VectorXd v(1);
    v(0) = theta;
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

  auto pi = boost::math::constants::pi<double>();

  // UNCOMMENT TO SEE BUG
  //	run(visitor, pi);
}

TEST(ITensorMPSVisitorTester, checkSampling) {

  auto gateRegistry = xacc::getService<IRProvider>("gate");

  auto term0 = gateRegistry->createFunction(
      "term0", std::vector<int>{0, 1}, std::vector<InstructionParameter>{});

  auto x1 = gateRegistry->createInstruction("X", std::vector<int>{0});
  auto x2 = gateRegistry->createInstruction("X", std::vector<int>{1});

  term0->addInstruction(x1);
  term0->addInstruction(x2);
  auto meas1 = gateRegistry->createInstruction("Measure", std::vector<int>{0});
  InstructionParameter p6(0);
  meas1->setParameter(0, p6);

  auto meas2 = gateRegistry->createInstruction("Measure", std::vector<int>{1});
  InstructionParameter p7(1);
  meas2->setParameter(0, p7);

  term0->addInstruction(meas1);
  term0->addInstruction(meas2);

  auto buffer = std::make_shared<TNQVMBuffer>("qreg", 2);

  // Get the visitor backend
  auto visitor = std::make_shared<ITensorMPSVisitor>();
  auto visCast = std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

  auto run = [&](std::shared_ptr<ITensorMPSVisitor> visitor,
                 double theta) -> double {
    buffer->resetBuffer();
    // Initialize the visitor
    visitor->initialize(buffer);

    Eigen::VectorXd v;
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

  auto mstrs = buffer->getMeasurementStrings();
  for (auto s : mstrs) {
    EXPECT_TRUE(s == "11");
  }
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
