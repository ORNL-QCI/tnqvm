#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"


TEST(GateAggregatorTester, checkSimple) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("qft");
    auto qft = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    qft->expand({std::make_pair("nq", 20)});
    // DEBUG:
    std::cout << "QFT Circuit: \n" << qft->toString() << "\n";
    auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qreg = xacc::qalloc(20);
    accelerator->execute(qreg, qft);
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 
