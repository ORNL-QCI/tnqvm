#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "ExaTnPmpsVisitor.hpp"

using namespace cppmicroservices;

class US_ABI_LOCAL ExaTnPmpsActivator : public BundleActivator {
public:
  ExaTnPmpsActivator() {}

  void Start(BundleContext context) 
  {
    context.RegisterService<tnqvm::TNQVMVisitor>(std::make_shared<tnqvm::ExaTnPmpsVisitor>());
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExaTnPmpsActivator)
