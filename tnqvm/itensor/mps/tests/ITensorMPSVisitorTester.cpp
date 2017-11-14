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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ITensorMPSVisitorTester

#include <memory>
#include <boost/test/included/unit_test.hpp>
#include "ITensorMPSVisitor.hpp"
#include "GateFunction.hpp"
#include "Hadamard.hpp"
#include "CNOT.hpp"
#include "GateQIR.hpp"
#include "X.hpp"
#include "InstructionIterator.hpp"
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "XACC.hpp"

using namespace xacc::quantum;
using namespace tnqvm;
using namespace xacc;

BOOST_AUTO_TEST_CASE(checkSimpleSimulation) {

	auto gateRegistry = GateInstructionRegistry::instance();

	auto statePrep =
			std::make_shared<GateFunction>("statePrep",
					std::vector<InstructionParameter> { InstructionParameter(
							"theta") });

	auto term0 =
			std::make_shared<GateFunction>("term0",
					std::vector<InstructionParameter> { InstructionParameter(
							"theta") });

	auto rx = gateRegistry->create("Rx", std::vector<int>{0});
	InstructionParameter p0(3.1415926);
	rx->setParameter(0, p0);

	auto ry = gateRegistry->create("Ry", std::vector<int>{1});
	InstructionParameter p1(3.1415926/2.0);
	ry->setParameter(0, p1);

	auto rx2 = gateRegistry->create("Rx", std::vector<int>{0});
	InstructionParameter p2(7.8539752);
	rx2->setParameter(0, p2);

	auto cnot1 = gateRegistry->create("CNOT", std::vector<int>{1,0});

	auto rz = gateRegistry->create("Rz", std::vector<int>{0});
	InstructionParameter p3("theta");
	rz->setParameter(0, p3);

	auto cnot2 = gateRegistry->create("CNOT", std::vector<int>{1,0});

	auto ry2 = gateRegistry->create("Ry", std::vector<int>{1});
	InstructionParameter p4(7.8539752);
	ry2->setParameter(0, p4);

	auto rx3 = gateRegistry->create("Rx", std::vector<int>{0});
	InstructionParameter p5(3.1415926/2.0);
	rx3->setParameter(0, p5);

	auto meas = gateRegistry->create("Measure", std::vector<int>{0});
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
	buffer->set_verbose(0);

	// Get the visitor backend
	auto visitor = std::make_shared<ITensorMPSVisitor>();
	auto visCast =
				std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

	auto run =
			[&](std::shared_ptr<ITensorMPSVisitor> visitor, double theta) -> double {
				buffer->resetBuffer();
				// Initialize the visitor
				visitor->initialize(buffer);

				term0->evaluateVariableParameters(std::vector<InstructionParameter> {
							InstructionParameter(theta)});

				// Walk the IR tree, and visit each node
				InstructionIterator it(term0);
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

	BOOST_VERIFY(std::fabs(-1 - run(visitor, -pi)) < 1e-8);
	BOOST_VERIFY(std::fabs(0.128844 - run(visitor, -1.44159)) < 1e-8);
	BOOST_VERIFY(std::fabs(0.307333 - run(visitor, 1.25841)) < 1e-8);
	BOOST_VERIFY(std::fabs(-.283662 - run(visitor, 1.85841)) < 1e-8);
	BOOST_VERIFY(std::fabs(-1 - run(visitor, pi)) < 1e-8);

}

BOOST_AUTO_TEST_CASE(checkOneQubitBug) {

	auto gateRegistry = GateInstructionRegistry::instance();

	auto statePrep =
			std::make_shared<GateFunction>("statePrep",
					std::vector<InstructionParameter> { InstructionParameter(
							"theta") });

	auto term0 =
			std::make_shared<GateFunction>("term0",
					std::vector<InstructionParameter> { InstructionParameter(
							"theta") });

	auto rx = gateRegistry->create("Rx", std::vector<int>{0});
	InstructionParameter p0("theta");
	rx->setParameter(0, p0);

	auto ry = gateRegistry->create("Ry", std::vector<int> { 1 });
	InstructionParameter p1(3.1415926 / 2.0);
	ry->setParameter(0, p1);
	auto rz = gateRegistry->create("Rz", std::vector<int> { 0 });
	InstructionParameter p3("theta");
	rz->setParameter(0, p3);

	auto h = gateRegistry->create("H", std::vector<int>{0});

	auto meas = gateRegistry->create("Measure", std::vector<int> { 0 });
	InstructionParameter p6(0);
	meas->setParameter(0, p6);

	statePrep->addInstruction(rx);
	statePrep->addInstruction(rz);

	term0->addInstruction(statePrep);
	term0->addInstruction(h);
	term0->addInstruction(meas);

	auto buffer = std::make_shared<TNQVMBuffer>("qreg", 1);
	buffer->set_verbose(0);

	// Get the visitor backend
	auto visitor = std::make_shared<ITensorMPSVisitor>();
	auto visCast =
				std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

	auto run =
			[&](std::shared_ptr<ITensorMPSVisitor> visitor, double theta) -> double {
				buffer->resetBuffer();
				// Initialize the visitor
				visitor->initialize(buffer);

				term0->evaluateVariableParameters(std::vector<InstructionParameter> {
							InstructionParameter(theta)});

				// Walk the IR tree, and visit each node
				InstructionIterator it(term0);
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

BOOST_AUTO_TEST_CASE(checkSampling) {

	auto gateRegistry = GateInstructionRegistry::instance();

	auto term0 =
			std::make_shared<GateFunction>("term0",
					std::vector<InstructionParameter> {});

	auto x1 = gateRegistry->create("X", std::vector<int>{0});
	auto x2 = gateRegistry->create("X", std::vector<int> { 1 });

	term0->addInstruction(x1);
	term0->addInstruction(x2);
	auto meas1 = gateRegistry->create("Measure", std::vector<int> { 0 });
	InstructionParameter p6(0);
	meas1->setParameter(0, p6);

	auto meas2 = gateRegistry->create("Measure", std::vector<int> { 1 });
	InstructionParameter p7(1);
	meas2->setParameter(0, p7);

	term0->addInstruction(meas1);
	term0->addInstruction(meas2);

	auto buffer = std::make_shared<TNQVMBuffer>("qreg", 2);
	buffer->set_verbose(0);

	// Get the visitor backend
	auto visitor = std::make_shared<ITensorMPSVisitor>();
	auto visCast =
				std::dynamic_pointer_cast<BaseInstructionVisitor>(visitor);

	auto run =
			[&](std::shared_ptr<ITensorMPSVisitor> visitor, double theta) -> double {
				buffer->resetBuffer();
				// Initialize the visitor
				visitor->initialize(buffer);

				term0->evaluateVariableParameters(std::vector<InstructionParameter> {
							InstructionParameter(theta)});

				// Walk the IR tree, and visit each node
				InstructionIterator it(term0);
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
		BOOST_VERIFY(s == "11");
	}
}
