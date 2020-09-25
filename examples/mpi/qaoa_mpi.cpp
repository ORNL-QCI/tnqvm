#include "xacc.hpp"
#include "xacc_service.hpp"

int main(int argc, char **argv) {
  xacc::Initialize();
  xacc::logToFile(true);
  xacc::set_verbose(true);
  auto acc = xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn"},
                                            {"exatn-buffer-size-gb", 16},
                                            // Note: fixes a low number here for debugging memory leak,
                                            // i.e. bypassing the estimate based on memory buffer size.
                                            {"max-qubit", 20}});
  const size_t nbNodes = 28;
  const int nbSteps = 1;
  auto buffer = xacc::qalloc(nbNodes);
  auto optimizer = xacc::getOptimizer("nlopt", {{"nlopt-maxeval", 1}});
  auto qaoa = xacc::getService<xacc::Algorithm>("QAOA");
  auto graph = xacc::getService<xacc::Graph>("boost-digraph")
                   ->gen_random_graph(nbNodes, 0.01);
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
