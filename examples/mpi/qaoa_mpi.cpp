#include "xacc.hpp"
#include "xacc_service.hpp"

int main(int argc, char **argv) {
  xacc::set_verbose(true);
  xacc::Initialize();
  auto acc = xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn:float"},
                                            {"exatn-buffer-size-gb", 1}});
  const size_t nbNodes = 30;
  auto graph = xacc::getService<xacc::Graph>("boost-digraph");

  for (size_t i = 0; i < nbNodes; i++) {
    graph->addVertex();
  }
  // Ring graph
  for (size_t i = 0; i < nbNodes - 1; i++) {
    graph->addEdge(i, i + 1);
  }
  graph->addEdge(nbNodes - 1, 0);
  
  const int nbSteps = 4;
  auto buffer = xacc::qalloc(nbNodes);
  auto optimizer = xacc::getOptimizer("nlopt", {{"nlopt-maxeval", 1}});
  auto qaoa = xacc::getService<xacc::Algorithm>("QAOA");
 
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
