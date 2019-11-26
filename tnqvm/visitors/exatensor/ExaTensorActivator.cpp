#include "ExaTensorMPSVisitor.hpp"

#include "OptionsProvider.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

using namespace cppmicroservices;

class US_ABI_LOCAL ExaTensorActivator : public BundleActivator {
public:
  ExaTensorActivator() {}

  void Start(BundleContext context) {
    auto visitor = std::make_shared<tnqvm::ExaTensorMPSVisitor>();
    context.RegisterService<tnqvm::TNQVMVisitor>(visitor);
    context.RegisterService<xacc::OptionsProvider>(visitor);
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExaTensorActivator)
