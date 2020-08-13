#include "ExatnVisitor.hpp"

#include "OptionsProvider.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "ExatnTearDown.hpp"

using namespace cppmicroservices;

class US_ABI_LOCAL ExaTNActivator : public BundleActivator {
public:
  ExaTNActivator() {}

  void Start(BundleContext context) {
    auto visitor = std::make_shared<tnqvm::DefaultExatnVisitor>();
    context.RegisterService<tnqvm::TNQVMVisitor>(visitor);
    context.RegisterService<xacc::OptionsProvider>(visitor);
    // Register the ExaTN teardown service to properly finalize ExaTN.
    context.RegisterService<xacc::TearDown>(std::make_shared<tnqvm::ExatnTearDown>());
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExaTNActivator)
