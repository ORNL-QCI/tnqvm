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
#include <random>
namespace {
inline double generateRandomProbability() {
  auto randFunc =
      std::bind(std::uniform_real_distribution<double>(0, 1),
                std::mt19937(std::chrono::high_resolution_clock::now()
                                 .time_since_epoch()
                                 .count()));
  return randFunc();
}
} // namespace

TEST(ExatnExpValSumReduceTester, testDeuteron) {
  auto accelerator = xacc::getAccelerator(
      "tnqvm", {{"tnqvm-visitor", "exatn"}, {"max-qubit", 1}});
  auto xasmCompiler = xacc::getCompiler("xasm");

  auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
      H(q[0]);
      H(q[1]);
      Measure(q[0]);
      Measure(q[1]);
    })",
                                  accelerator);

  auto program = ir->getComposite("ansatz");
  // Expected results from deuteron_2qbit_xasm_X0X1
  const std::vector<double> expectedResults{
      0.0,       -0.324699, -0.614213, -0.837166, -0.9694,
      -0.996584, -0.915773, -0.735724, -0.475947, -0.164595,
      0.164595,  0.475947,  0.735724,  0.915773,  0.996584,
      0.9694,    0.837166,  0.614213,  0.324699,  0.0};

  const auto angles =
      xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  for (size_t i = 0; i < angles.size(); ++i) {
    auto buffer = xacc::qalloc(2);
    auto evaled = program->operator()({angles[i]});
    accelerator->execute(buffer, evaled);
    std::cout << "Angle = " << angles[i]
              << ": Exp-val = " << buffer->getExpectationValueZ() << "\n";
    EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResults[i], 1e-6);
  }
}

TEST(ExatnExpValSumReduceTester, testSliceOfMultipleQubits) {
  auto accelerator = xacc::getAccelerator(
      "tnqvm", {{"tnqvm-visitor", "exatn"}, {"max-qubit", 2}});
  xacc::qasm(R"(
        .compiler xasm
        .circuit test_circuit
        .qbit q
        U(q[1], 1.5708,0,3.14159);
        U(q[0], 1.5708,1.5708,4.71239); 
        CNOT(q[0], q[1]);
        U(q[2], 1.5708,-3.14159,3.14159); 
        U(q[3], 1.5708,0,3.14159); 
        CNOT(q[2], q[3]);
        Rz(q[3], 0.101476); 
        CNOT(q[2], q[3]);
        CNOT(q[1], q[2]);
        CNOT(q[0], q[1]);
        U(q[3], 1.5708,0,3.14159); 
        U(q[2], 1.5708,0,3.14159); 
        U(q[0], 1.5708,1.5708,4.71239); 
        U(q[1], 1.5708,0,3.14159); 
        H(q[0]);
        H(q[1]);
        H(q[2]);
        H(q[3]);
        Measure(q[0]);
        Measure(q[1]);
        Measure(q[2]);
        Measure(q[3]);
    )");

  auto program = xacc::getCompiled("test_circuit");
  auto buffer = xacc::qalloc(4);
  accelerator->execute(buffer, program);
  buffer->print();
}

TEST(ExatnExpValSumReduceTester, testDeuteronH3) {
  auto accelerator = xacc::getAccelerator(
      "tnqvm", {{"tnqvm-visitor", "exatn"}, {"max-qubit", 2}});
  xacc::qasm(R"(
        .compiler xasm
        .circuit ansatz
        .parameters t0, t1
        .qbit q
        X(q[0]);
        exp_i_theta(q, t0, {{"pauli", "0.5 X0 Y1 - 0.5 Y0 X1"}});
        exp_i_theta(q, t1, {{"pauli", "0.5 X0 Z1 Y2 - 0.5 X2 Z1 Y0"}});
    )");

  auto program = xacc::getCompiled("ansatz");

  const auto t0Angles =
      xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  const auto t1Angles =
      xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  auto provider = xacc::getService<xacc::IRProvider>("quantum");

  auto h0 = provider->createInstruction("H", {0});
  auto h1 = provider->createInstruction("H", {1});
  auto m0 = provider->createInstruction("Measure", {0});
  auto m1 = provider->createInstruction("Measure", {1});

  // Expected results from QPP
  const std::vector<double> expectedResults{
      -2.22045e-16, -2.59945e-11, -4.91719e-11, -6.70207e-11, -7.7607e-11,
      -7.97832e-11, -7.33136e-11, -5.88994e-11, -3.81025e-11, -1.31766e-11,
      1.3177e-11,   3.81029e-11,  5.88997e-11,  7.33138e-11,  7.97833e-11,
      7.76069e-11,  6.70206e-11,  4.91716e-11,  2.5994e-11,   -2.22045e-16,
      -0.614213,    -0.580933,    -0.4847,      -0.335943,    -0.15078,
      0.0507213,    0.246726,     0.415995,     0.540184,     0.605836,
      0.605836,     0.540184,     0.415995,     0.246726,     0.0507213,
      -0.15078,     -0.335943,    -0.4847,      -0.580933,    -0.614213,
      -0.9694,      -0.916875,    -0.764993,    -0.530212,    -0.237974,
      0.0800524,    0.389404,     0.656557,     0.852562,     0.956179,
      0.956179,     0.852562,     0.656557,     0.389404,     0.0800524,
      -0.237974,    -0.530212,    -0.764993,    -0.916875,    -0.9694,
      -0.915773,    -0.866154,    -0.722674,    -0.500881,    -0.224809,
      0.075624,     0.367862,     0.620236,     0.805399,     0.903283,
      0.903283,     0.805399,     0.620236,     0.367862,     0.075624,
      -0.224809,    -0.500881,    -0.722674,    -0.866154,    -0.915773,
      -0.475947,    -0.450159,    -0.375589,    -0.260319,    -0.116838,
      0.0393034,    0.191186,     0.32235,      0.418583,     0.469456,
      0.469456,     0.418583,     0.32235,      0.191186,     0.0393034,
      -0.116838,    -0.260319,    -0.375589,    -0.450159,    -0.475947,
      0.164595,     0.155676,     0.129888,     0.0900247,    0.0404056,
      -0.0135921,   -0.0661169,   -0.111477,    -0.144757,    -0.16235,
      -0.16235,     -0.144757,    -0.111477,    -0.0661169,   -0.0135921,
      0.0404056,    0.0900247,    0.129888,     0.155676,     0.164595,
      0.735724,     0.69586,      0.58059,      0.402403,     0.18061,
      -0.0607556,   -0.295537,    -0.498292,    -0.64705,     -0.72569,
      -0.72569,     -0.64705,     -0.498292,    -0.295537,    -0.0607556,
      0.18061,      0.402403,     0.58059,      0.69586,      0.735724,
      0.996584,     0.942587,     0.786445,     0.54508,      0.244647,
      -0.0822973,   -0.400323,    -0.674968,    -0.87647,     -0.982992,
      -0.982992,    -0.87647,     -0.674968,    -0.400323,    -0.0822973,
      0.244647,     0.54508,      0.786445,     0.942587,     0.996584,
      0.837166,     0.791806,     0.660642,     0.457887,     0.205512,
      -0.0691327,   -0.336286,    -0.566997,    -0.736266,    -0.825749,
      -0.825749,    -0.736266,    -0.566997,    -0.336286,    -0.0691327,
      0.205512,     0.457887,     0.660642,     0.791806,     0.837166,
      0.324699,     0.307106,     0.256234,     0.177594,     0.079709,
      -0.0268135,   -0.13043,     -0.219913,    -0.285565,    -0.320271,
      -0.320271,    -0.285565,    -0.219913,    -0.13043,     -0.0268135,
      0.079709,     0.177594,     0.256234,     0.307106,     0.324699,
      -0.324699,    -0.307106,    -0.256234,    -0.177594,    -0.079709,
      0.0268135,    0.13043,      0.219913,     0.285565,     0.320271,
      0.320271,     0.285565,     0.219913,     0.13043,      0.0268135,
      -0.079709,    -0.177594,    -0.256234,    -0.307106,    -0.324699,
      -0.837166,    -0.791806,    -0.660642,    -0.457887,    -0.205512,
      0.0691327,    0.336286,     0.566997,     0.736266,     0.825749,
      0.825749,     0.736266,     0.566997,     0.336286,     0.0691327,
      -0.205512,    -0.457887,    -0.660642,    -0.791806,    -0.837166,
      -0.996584,    -0.942587,    -0.786445,    -0.54508,     -0.244647,
      0.0822973,    0.400323,     0.674968,     0.87647,      0.982992,
      0.982992,     0.87647,      0.674968,     0.400323,     0.0822973,
      -0.244647,    -0.54508,     -0.786445,    -0.942587,    -0.996584,
      -0.735724,    -0.69586,     -0.58059,     -0.402403,    -0.18061,
      0.0607556,    0.295537,     0.498292,     0.64705,      0.72569,
      0.72569,      0.64705,      0.498292,     0.295537,     0.0607556,
      -0.18061,     -0.402403,    -0.58059,     -0.69586,     -0.735724,
      -0.164595,    -0.155676,    -0.129888,    -0.0900247,   -0.0404056,
      0.0135921,    0.0661169,    0.111477,     0.144757,     0.16235,
      0.16235,      0.144757,     0.111477,     0.0661169,    0.0135921,
      -0.0404056,   -0.0900247,   -0.129888,    -0.155676,    -0.164595,
      0.475947,     0.450159,     0.375589,     0.260319,     0.116838,
      -0.0393034,   -0.191186,    -0.32235,     -0.418583,    -0.469456,
      -0.469456,    -0.418583,    -0.32235,     -0.191186,    -0.0393034,
      0.116838,     0.260319,     0.375589,     0.450159,     0.475947,
      0.915773,     0.866154,     0.722674,     0.500881,     0.224809,
      -0.075624,    -0.367862,    -0.620236,    -0.805399,    -0.903283,
      -0.903283,    -0.805399,    -0.620236,    -0.367862,    -0.075624,
      0.224809,     0.500881,     0.722674,     0.866154,     0.915773,
      0.9694,       0.916875,     0.764993,     0.530212,     0.237974,
      -0.0800524,   -0.389404,    -0.656557,    -0.852562,    -0.956179,
      -0.956179,    -0.852562,    -0.656557,    -0.389404,    -0.0800524,
      0.237974,     0.530212,     0.764993,     0.916875,     0.9694,
      0.614213,     0.580933,     0.4847,       0.335943,     0.15078,
      -0.0507213,   -0.246726,    -0.415995,    -0.540184,    -0.605836,
      -0.605836,    -0.540184,    -0.415995,    -0.246726,    -0.0507213,
      0.15078,      0.335943,     0.4847,       0.580933,     0.614213,
      -6.66134e-16, -2.59949e-11, -4.91722e-11, -6.7021e-11,  -7.76071e-11,
      -7.97832e-11, -7.33135e-11, -5.88991e-11, -3.81022e-11, -1.31762e-11,
      1.31775e-11,  3.81033e-11,  5.89e-11,     7.3314e-11,   7.97833e-11,
      7.76068e-11,  6.70204e-11,  4.91712e-11,  2.59938e-11,  -6.66134e-16};
  int testCaseId = 0;
  for (const auto &t0 : t0Angles) {
    for (const auto &t1 : t1Angles) {
      // To keep test time reasonable, we only run a few test cases.
      // Only run 5% (20 cases) of all test cases.
      const double PROBABILITY_TO_RUN = 0.05;
      if (generateRandomProbability() < PROBABILITY_TO_RUN) {
        auto buffer = xacc::qalloc(3);
        auto evaled = program->operator()({t0, t1});
        evaled->addInstructions({h0, h1, m0, m1});
        accelerator->execute(buffer, evaled);
        const auto expectedResult = expectedResults[testCaseId];
        std::cout << "Result = " << buffer->getExpectationValueZ() << " vs "
                  << expectedResult << "\n";
        EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResult, 1e-6);
      }
      ++testCaseId;
    }
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
