#include "xacc.hpp"

int main(int argc, char **argv) {
  // Initialize the XACC Framework
  xacc::Initialize(argc, argv);
  xacc::set_verbose(true);
  auto compiler = xacc::getCompiler("staq");
  auto circ = compiler
                  ->compile(R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[16];
x q[16];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
x q[0];
x q[2];
x q[4];
x q[6];
x q[8];
x q[10];
x q[12];
x q[14];
cx q[0],q[16];
cx q[1],q[16];
cx q[2],q[16];
cx q[3],q[16];
cx q[4],q[16];
cx q[5],q[16];
cx q[6],q[16];
cx q[7],q[16];
cx q[8],q[16];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
x q[0];
x q[2];
x q[4];
x q[6];
x q[8];
x q[10];
x q[12];
x q[14];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
)")
                  ->getComposites()[0];
  std::cout << "HOWDY:\n" << circ->toString() << "\n";
  auto qpu = xacc::getAccelerator("tnqvm:exatn-mps", {{"shots", 2}});
  auto qubitReg = xacc::qalloc(17);
  qpu->execute(qubitReg, circ);
  qubitReg->print();

  // Finalize the XACC Framework
  xacc::Finalize();

  return 0;
}
