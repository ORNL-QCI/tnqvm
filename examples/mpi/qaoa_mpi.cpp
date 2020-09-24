#include "xacc.hpp"
#include "xacc_service.hpp"

int main(int argc, char **argv) {
  xacc::Initialize();
  xacc::set_verbose(true);
  // Use a small buffer size to enforce splitting the wave-function into slices
  // across MPI processes.
  auto acc = xacc::getAccelerator("tnqvm",
                                  {std::make_pair("tnqvm-visitor", "exatn"),
                                   std::make_pair("exatn-buffer-size-gb", 1)});
  const size_t nbNodes = 28;
  const int nbSteps = 1;
  auto buffer = xacc::qalloc(nbNodes);
  auto optimizer = xacc::getOptimizer("nlopt", {{"nlopt-maxeval", 1}});
  auto qaoa = xacc::getService<xacc::Algorithm>("QAOA");
  auto graph = xacc::getService<xacc::Graph>("boost-digraph")
                   ->gen_random_graph(nbNodes, 0.005);
  const bool initOk = qaoa->initialize(
      {std::make_pair("accelerator", acc),
       std::make_pair("optimizer", optimizer), std::make_pair("graph", graph),
       // number of time steps (p) param
       std::make_pair("steps", nbSteps),
       // "Standard" or "Extended"
       std::make_pair("parameter-scheme", "Standard")});
  qaoa->execute(buffer);
  std::cout << "Min Val: " << (*buffer)["opt-val"].as<double>() << "\n";

  xacc::Finalize();
  return 0;
}
