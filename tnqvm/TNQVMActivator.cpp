#include "TNQVM.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

using namespace cppmicroservices;

class US_ABI_LOCAL TNQVMActivator : public BundleActivator {
public:
  TNQVMActivator() {}

  void Start(BundleContext context) {
    auto acc = std::make_shared<tnqvm::TNQVM>();
    context.RegisterService<xacc::Accelerator>(acc);
    context.RegisterService<xacc::OptionsProvider>(acc);
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(TNQVMActivator)
