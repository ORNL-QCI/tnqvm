#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "NoiseModel.hpp"
#include "Optimizer.hpp"
#include "xacc_observable.hpp"
#include "Algorithm.hpp"

TEST(ExaTnGenTester, checkPrecisionSpecification) {
  xacc::set_verbose(true);
  auto accelerator1 =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});
  auto accelerator2 =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen:float"}});
  auto accelerator3 =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen:double"}});
}

TEST(ExaTnGenTester, checkExpVal) {
  xacc::set_verbose(true);
  {
    // Check exp-val by tensor expansion
    auto accelerator =
        xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto program = xasmCompiler
                       ->compile(R"(__qpu__ void testRY(qbit q, double t) {
        Ry(q[0], t);
        Measure(q[0]);
      })",
                                 accelerator)
                       ->getComposite("testRY");
    const auto angles = xacc::linspace(-M_PI, M_PI, 20);
    for (const auto &angle : angles) {
      auto evaled = program->operator()({angle});
      const double expResult =
          -1.0 + 2.0 * std::cos(angle / 2.0) * std::cos(angle / 2.0);
      std::cout << "Angle = " << angle << ": " << expResult << "\n";
      auto buffer = xacc::qalloc(1);
      accelerator->execute(buffer, evaled);
      // buffer->print();
      std::cout << "Angle = " << angle << ": " << buffer->getExpectationValueZ()
                << " vs. " << expResult << "\n";
      EXPECT_NEAR(buffer->getExpectationValueZ(), expResult, 1e-3);
    }
  }
  {
    auto accelerator =
        xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t)
    {
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
                << ": Expected = " << expectedResults[i]
                << "; ExaTN = " << buffer->getExpectationValueZ() << "\n";
      EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResults[i], 1e-6);
    }
  }
}

TEST(ExaTnGenTester, checkVqeH2) {
  auto accelerator =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});
  // Create the N=2 deuteron Hamiltonian
  auto H_N_2 = xacc::quantum::getObservable(
      "pauli", std::string("5.907 - 2.1433 X0X1 "
                           "- 2.1433 Y0Y1"
                           "+ .21829 Z0 - 6.125 Z1"));

  auto optimizer = xacc::getOptimizer("nlopt");
  xacc::qasm(R"(
        .compiler xasm
        .circuit deuteron_ansatz
        .parameters theta
        .qbit q
        X(q[0]);
        Ry(q[1], theta);
        CNOT(q[1],q[0]);
    )");
  auto ansatz = xacc::getCompiled("deuteron_ansatz");

  // Get the VQE Algorithm and initialize it
  auto vqe = xacc::getAlgorithm("vqe");
  vqe->initialize({std::make_pair("ansatz", ansatz),
                   std::make_pair("observable", H_N_2),
                   std::make_pair("accelerator", accelerator),
                   std::make_pair("optimizer", optimizer)});

  // Allocate some qubits and execute
  auto buffer = xacc::qalloc(2);
  vqe->execute(buffer);
  std::cout << "Energy = " << (*buffer)["opt-val"].as<double>() << "\n";
  // Expected result: -1.74886
  EXPECT_NEAR((*buffer)["opt-val"].as<double>(), -1.74886, 1e-4);
}

TEST(ExaTnGenTester, checkVqeH3) {
  auto accelerator = xacc::getAccelerator(
      "tnqvm", {{"tnqvm-visitor", "exatn-gen"}, {"reconstruct-layers", -1}});
  // Create the N=3 deuteron Hamiltonian
  auto H_N_3 = xacc::quantum::getObservable(
      "pauli",
      std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + "
                  "9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2"));

  auto optimizer = xacc::getOptimizer("nlopt");

  xacc::qasm(R"(
        .compiler xasm
        .circuit deuteron_ansatz_h3
        .parameters t0, t1
        .qbit q
        X(q[0]);
        exp_i_theta(q, t0, {{"pauli", "X0 Y1 - Y0 X1"}});
        exp_i_theta(q, t1, {{"pauli", "X0 Z1 Y2 - X2 Z1 Y0"}});
    )");
  auto ansatz = xacc::getCompiled("deuteron_ansatz_h3");

  // Get the VQE Algorithm and initialize it
  auto vqe = xacc::getAlgorithm("vqe");
  vqe->initialize({std::make_pair("ansatz", ansatz),
                   std::make_pair("observable", H_N_3),
                   std::make_pair("accelerator", accelerator),
                   std::make_pair("optimizer", optimizer)});

  // Allocate some qubits and execute
  auto buffer = xacc::qalloc(3);
  vqe->execute(buffer);
  std::cout << "Energy = " << (*buffer)["opt-val"].as<double>() << "\n";
  auto optParams = (*buffer)["opt-params"].as<std::vector<double>>();
  for (const auto &param : optParams) {
    std::cout << "Param: " << param << "\n";
  }
  // Expected result: -2.04482
  EXPECT_NEAR((*buffer)["opt-val"].as<double>(), -2.04482, 1e-3);
}

TEST(ExaTnGenTester, checkBitstringAmpl) {
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
            H(q[0]);
            for (int i = 0; i < 7; i++) {
                CNOT(q[i], q[i + 1]);
            }
        })");
  std::vector<int> bitstring(8, -1);
  auto program = ir->getComposite("test1");
  auto accelerator =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen:float"},
                                     {"reconstruct-layers", 2},
                                     {"reconstruct-tolerance", 0.01},
                                     {"bitstring", bitstring}});
  auto qreg = xacc::qalloc(8);
  accelerator->execute(qreg, program);
  qreg->print();
  const auto realAmpl = (*qreg)["amplitude-real-vec"].as<std::vector<double>>();
  const auto imagAmpl = (*qreg)["amplitude-imag-vec"].as<std::vector<double>>();
  const int nb_elems = 256;
  EXPECT_EQ(realAmpl.size(), nb_elems);
  EXPECT_EQ(imagAmpl.size(), nb_elems);
  // GHZ: |000000> + |111111>/sqrt(2)
  for (size_t i = 0; i < nb_elems; ++i) {
    EXPECT_NEAR(imagAmpl[i], 0.0, 0.1);
    EXPECT_NEAR(realAmpl[i],
                (i == 0 || i == nb_elems - 1) ? 1.0 / std::sqrt(2.0) : 0.0,
                0.1);
  }
}

// TEST(ExaTnGenTester, checkVqeH3Approx) {
//   // Use very high tolerance to save test time
//   auto accelerator =
//       xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"},
//                                      {"reconstruct-layers", 10},
//                                      {"reconstruct-tolerance", 0.01}});
//   xacc::set_verbose(true);
//   xacc::qasm(R"(
//         .compiler xasm
//         .circuit deuteron_ansatz_h3_2
//         .parameters t0, t1
//         .qbit q
//         X(q[0]);
//         exp_i_theta(q, t0, {{"pauli", "X0 Y1 - Y0 X1"}});
//         exp_i_theta(q, t1, {{"pauli", "X0 Z1 Y2 - X2 Z1 Y0"}});
//     )");
//   auto ansatz = xacc::getCompiled("deuteron_ansatz_h3_2");
//   auto H_N_3 = xacc::quantum::getObservable(
//       "pauli",
//       std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + "
//                   "9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2"));
//   auto optimizer = xacc::getOptimizer("nlopt");
//   // Allocate some qubits and execute
//   auto buffer = xacc::qalloc(3);
//   auto vqe = xacc::getAlgorithm("vqe");
//   vqe->initialize({std::make_pair("ansatz", ansatz),
//                    std::make_pair("observable", H_N_3),
//                    std::make_pair("accelerator", accelerator),
//                    std::make_pair("optimizer", optimizer)});
//   // The reconstruction can take a long time, so we just test a single
//   // observable evaluation.
//   auto energies = vqe->execute(buffer, {0.0684968, 0.17797});
//   buffer->print();
//   std::cout << "Energy = " << energies[0] << "\n";
//   EXPECT_NEAR(energies[0], -2.04482, 0.1);
// }

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}