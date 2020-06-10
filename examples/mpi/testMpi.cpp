#include "xacc.hpp"

int main (int argc, char** argv) {

    // Initialize the XACC Framework
    xacc::Initialize(argc, argv);
    auto qpu = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn-mps"),
        std::make_pair("shots", 10),
    });

    // Allocate a register of 40 qubits
    auto qubitReg = xacc::qalloc(40);

    // Create a Program
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test(qbit q) {
        H(q[0]);
        CX(q[0], q[1]);
        CX(q[1], q[2]);
        CX(q[2], q[3]);
        CX(q[3], q[4]);
        CX(q[4], q[5]);
        CX(q[5], q[6]);
        CX(q[6], q[7]);
        CX(q[7], q[8]);
        CX(q[8], q[9]);
        CX(q[9], q[10]);
        CX(q[10], q[11]);
        CX(q[11], q[12]);
        CX(q[12], q[13]);
        CX(q[13], q[14]);
        CX(q[14], q[15]);
        CX(q[15], q[16]);
        CX(q[16], q[17]);
        CX(q[17], q[18]);
        CX(q[18], q[19]);
        CX(q[19], q[20]);
        CX(q[20], q[21]);
        CX(q[21], q[22]);
        CX(q[22], q[23]);
        CX(q[23], q[24]);
        CX(q[24], q[25]);
        CX(q[25], q[26]);
        CX(q[26], q[27]);
        CX(q[27], q[28]);
        CX(q[28], q[29]);
        CX(q[29], q[30]);
        CX(q[30], q[31]);
        CX(q[31], q[32]);
        CX(q[32], q[33]);
        CX(q[33], q[34]);
        CX(q[34], q[35]);
        CX(q[35], q[36]);
        CX(q[36], q[37]);
        CX(q[37], q[38]);
        CX(q[38], q[39]);
        // Measure two random qubits
        // should only get entangled bitstrings:
        // i.e. 00 or 11
        Measure(q[2]);
        Measure(q[37]);
    })", qpu);

    // Request the quantum kernel representing
    // the above source code
    auto program = ir->getComposite("test");
    // Execute!
    qpu->execute(qubitReg, program);

    qubitReg->print();

    // Finalize the XACC Framework
    xacc::Finalize();

    return 0;
}
