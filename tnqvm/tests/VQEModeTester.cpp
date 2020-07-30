#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "Optimizer.hpp"
#include "xacc_observable.hpp"
#include "Algorithm.hpp"

using namespace xacc;

TEST(VQEModeTester, checkH2) 
{
    auto accelerator = xacc::getAccelerator("tnqvm", { std::make_pair("tnqvm-visitor", "exatn") });
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

TEST(VQEModeTester, checkH3) 
{
    auto accelerator = xacc::getAccelerator("tnqvm", { std::make_pair("tnqvm-visitor", "exatn") });
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
    // Expected result: -2.04482
    EXPECT_NEAR((*buffer)["opt-val"].as<double>(), -2.04482, 1e-4);
}

TEST(VQEModeTester, checkUcc) 
{
    const std::string rucc = R"rucc(__qpu__ void f(qbit q, double t0) {
        X(q[0]);
        X(q[1]);
        Rx(q[0],1.5707);
        H(q[1]);
        H(q[2]);
        H(q[3]);
        CNOT(q[0],q[1]);
        CNOT(q[1],q[2]);
        CNOT(q[2],q[3]);
        Rz(q[3], t0);
        CNOT(q[2],q[3]);
        CNOT(q[1],q[2]);
        CNOT(q[0],q[1]);
        Rx(q[0],-1.5707);
        H(q[1]);
        H(q[2]);
        H(q[3]);
    })rucc";

    auto acc = xacc::getAccelerator("tnqvm", { std::make_pair("tnqvm-visitor", "exatn") });
    auto buffer = xacc::qalloc(4); 
    auto compiler = xacc::getCompiler("xasm");
    auto ir = compiler->compile(rucc, nullptr);
    auto ruccsd = ir->getComposite("f");
    auto optimizer = xacc::getOptimizer("nlopt");
    auto observable = xacc::quantum::getObservable("pauli", std::string(
        "(0.174073,0) Z2 Z3 + (0.1202,0) Z1 Z3 + (0.165607,0) Z1 Z2 + "
        "(0.165607,0) Z0 Z3 + (0.1202,0) Z0 Z2 + (-0.0454063,0) Y0 Y1 X2 X3 + "
        "(-0.220041,0) Z3 + (-0.106477,0) + (0.17028,0) Z0 + (-0.220041,0) Z2 "
        "+ (0.17028,0) Z1 + (-0.0454063,0) X0 X1 Y2 Y3 + (0.0454063,0) X0 Y1 "
        "Y2 X3 + (0.168336,0) Z0 Z1 + (0.0454063,0) Y0 X1 X2 Y3"));

    auto vqe = xacc::getService<Algorithm>("vqe");
    EXPECT_TRUE(vqe->initialize({{"ansatz", ruccsd},
                                {"accelerator", acc},
                                {"observable", observable},
                                {"optimizer", optimizer}}));
    vqe->execute(buffer);
    std::cout << "Energy = " << (*buffer)["opt-val"].as<double>() << "\n";
    EXPECT_NEAR(-1.13717, (*buffer)["opt-val"].as<double>(), 1e-4);
}

int main(int argc, char **argv) 
{
    xacc::set_verbose(true);   
    xacc::Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    xacc::Finalize();
    return ret;
} 
