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
 *   Initial API and implementation - Thien Nguyen
 *
 **********************************************************************************/
#include <memory>
#include <gtest/gtest.h>
#include "ExatnVisitor.hpp"
#include "xacc.hpp"
#include "utils/GateMatrixAlgebra.hpp"
#include <cmath>


using namespace tnqvm;

std::complex<double> calcMatrixTrace(const std::vector<std::complex<double>>& in_flattenedMatrix)
{
  const auto isPowerOfTwo = [](size_t in_number) -> bool {
     if(in_number == 0)
     {
        return false;
     }

     return (ceil(log2(in_number)) == floor(log2(in_number)));
  };

  assert(isPowerOfTwo(in_flattenedMatrix.size()));
  double dim;
  const auto fractionalPart = std::modf(sqrt(in_flattenedMatrix.size()), &dim);
  assert(fractionalPart == 0.0);
  const size_t nbQubit = std::lround(dim);
  std::complex<double> trace = 0.0;
  for (size_t i = 0; i < nbQubit; ++i)
  {
    const size_t pos = i*nbQubit + i;
    assert(pos < in_flattenedMatrix.size());
    trace += in_flattenedMatrix[pos];
  }

  return trace;
}

// Test ExpVal calculation:
TEST(ExatnVisitorInternalTester, testTensorExpValCalc) {
  // Test 1: deuteron_2 using *direct* observable for <X0X1>,
  // i.e. no need to change basis and measure.
  {
    auto xasmCompiler = xacc::getCompiler("xasm");
    // Just the base ansatz (no measurement)
    auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
    })");

    auto program = ir->getComposite("ansatz");
    // Expected results from deuteron_2qbit_xasm_X0X1
    const std::vector<double> expectedResults {
        0.0,
        -0.324699,
        -0.614213,
        -0.837166,
        -0.9694,
        -0.996584,
        -0.915773,
        -0.735724,
        -0.475947,
        -0.164595,
        0.164595,
        0.475947,
        0.735724,
        0.915773,
        0.996584,
        0.9694,
        0.837166,
        0.614213,
        0.324699,
        0.0
    };

    const auto angles = xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
    auto gateRegistry = xacc::getIRProvider("quantum");
    auto x0 = gateRegistry->createInstruction("X", std::vector<std::size_t>{0});
    auto x1 = gateRegistry->createInstruction("X", std::vector<std::size_t>{1});
    // Observable term: X0X1
    const ExatnVisitor<std::complex<double>>::ObservableTerm term1({x0, x1});

    for (size_t i = 0; i < angles.size(); ++i)
    {
        auto exatnVisitor = std::make_shared<DefaultExatnVisitor>();
        auto buffer = xacc::qalloc(2);
        auto evaled = program->operator()({ angles[i] });
        // Calculate the expVal
        const auto expVal = exatnVisitor->observableExpValCalc(buffer, evaled, { term1 });

        EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
        EXPECT_NEAR(expVal.real(), expectedResults[i], 1e-4);
    }
  }
  // Test 2: Full Observable expression: multiple terms
  {
    auto xasmCompiler = xacc::getCompiler("xasm");
    // Just the base ansatz (no measurement)
    auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz1(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
    })");

    auto program = ir->getComposite("ansatz1");

    auto gateRegistry = xacc::getIRProvider("quantum");
    auto x0 = gateRegistry->createInstruction("X", std::vector<std::size_t>{0});
    auto x1 = gateRegistry->createInstruction("X", std::vector<std::size_t>{1});
    auto y0 = gateRegistry->createInstruction("Y", std::vector<std::size_t>{0});
    auto y1 = gateRegistry->createInstruction("Y", std::vector<std::size_t>{1});
    auto z0 = gateRegistry->createInstruction("Z", std::vector<std::size_t>{0});
    auto z1 = gateRegistry->createInstruction("Z", std::vector<std::size_t>{1});

    // Observable: E = 5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1
    const ExatnVisitor<std::complex<double>>::ObservableTerm term0({}, 5.907);
    const ExatnVisitor<std::complex<double>>::ObservableTerm term1({x0, x1}, -2.1433);
    const ExatnVisitor<std::complex<double>>::ObservableTerm term2({y0, y1}, -2.1433);
    const ExatnVisitor<std::complex<double>>::ObservableTerm term3({z0}, 0.21829);
    const ExatnVisitor<std::complex<double>>::ObservableTerm term4({z1}, -6.125);
    // Theta -> Energy data (from VQE run using Pauli Observable)
    std::vector<std::pair<double, double>> expectedResults =
    {
      {0.0, -0.43629},
      {1.5708, 1.62039983003},
      {-1.5708, 10.19359983},
      {-0.785398, 4.4527004535},
      {0.392699, -1.59384659069},
      {0.785398, -1.60946732175},
      {0.981748, -1.18132079195},
      {0.687223, -1.71581973894},
      {0.589049, -1.74876023748},
      {0.490874, -1.70797158239},
      {0.539961, -1.73757413396},
      {0.613592, -1.7474369643},
      {0.576777, -1.74769248655},
      {0.595185, -1.74886176773},
      {0.60132, -1.74867505912},
      {0.592117, -1.74884703248},
      {0.596719, -1.74884211288},
      {0.594418, -1.74886483954},
      {0.593651, -1.7488634076},
      {0.594801, -1.7488638666}
    };

    for (const auto& expectedResult : expectedResults)
    {
      const double theta = expectedResult.first;
      const double energy = expectedResult.second;
      auto exatnVisitor = std::make_shared<DefaultExatnVisitor>();
      auto buffer = xacc::qalloc(2);
      auto evaled = program->operator()({ theta });
      // Calculate the expVal
      const auto expVal = exatnVisitor->observableExpValCalc(buffer, evaled, { term0, term1, term2, term3, term4 });
      // The expVal is real (no imaginary part) and the real part matches the expected result.
      EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
      EXPECT_NEAR(expVal.real(), energy, 1e-4);
    }
  }
}


// Test RDM calculation: verify that the expected value calculated by RDM is consistent with regular simulation (i.e. via Measure)
TEST(ExatnVisitorInternalTester, testReducedDensityMatrixCalc) {
  const auto generateRandomAngle = []() -> double {
    static std::uniform_real_distribution<double> uniformDist(-M_PI, M_PI);
    static std::default_random_engine randomEngine;
    return uniformDist(randomEngine);
  };

  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q, double theta) {
    H(q[0]);
    H(q[1]);
    H(q[2]);
    H(q[3]);
    H(q[4]);
    Rx(q[5], theta);
    H(q[6]);
    H(q[7]);
    H(q[8]);
    H(q[9]);
    Measure(q[5]);
  })");

  const auto calcExpValByRdm = [](std::shared_ptr<CompositeInstruction>& in_function) -> double {
    auto exatnVisitor = std::make_shared<DefaultExatnVisitor>();
    auto buffer = xacc::qalloc(10);
    const auto rdm = exatnVisitor->getReducedDensityMatrix(buffer, in_function, {5});
    {
      // Validate trace of the RDM
      const auto trace = calcMatrixTrace(rdm);
      EXPECT_NEAR(trace.imag(), 0.0, 1e-12);
      EXPECT_NEAR(trace.real(), 1.0, 1e-12);
    }
    // Exp-Val-Z = Rho_00 - Rho11
    const auto expVal = rdm.front() - rdm.back();
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
    return expVal.real();
  };

  // Calculate expectation value via normal execution
  const auto calcExpVal = [](std::shared_ptr<CompositeInstruction>& in_function) -> double {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn")});
    auto qubitReg = xacc::qalloc(10);
    qpu->execute(qubitReg, in_function);
    return qubitReg["exp-val-z"].as<double>();
  };

  auto program = ir->getComposite("test1");
  const size_t nbRandomTests = 1;
  for (size_t i = 0; i < nbRandomTests; ++i)
  {
    const auto angle = generateRandomAngle();
    auto evaled = program->operator()({ angle });
    const double resultViaRdm = calcExpValByRdm(evaled);
    const double expectedResult = calcExpVal(evaled);
    EXPECT_NEAR(resultViaRdm, expectedResult, 1e-6);
  }
}

// Test tensor sampling by sequential collapse -> project
TEST(ExatnVisitorInternalTester, testSequentialCollapse)
{
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test2(qbit q) {
    H(q[0]);
    CNOT(q[0], q[1]);
    CNOT(q[0], q[2]);
    CNOT(q[0], q[3]);
    CNOT(q[0], q[4]);
  })");
  auto program = ir->getComposite("test2");
  auto exatnVisitor = std::make_shared<DefaultExatnVisitor>();
  auto buffer = xacc::qalloc(5);
  // Randomly select a subset of qubits
  const auto sampleBitString = exatnVisitor->getMeasureSample(buffer, program, { 1, 3 });
  const bool areAllBitsEqual = std::adjacent_find(sampleBitString.cbegin(), sampleBitString.cend(), std::not_equal_to<>()) == sampleBitString.cend();
  EXPECT_TRUE(areAllBitsEqual);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  xacc::Initialize();
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
