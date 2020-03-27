#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "ExaTnMpsVisitor.hpp"

using namespace cppmicroservices;

class US_ABI_LOCAL ExatnMpsActivator : public BundleActivator {
public:
  ExatnMpsActivator() {}

  void Start(BundleContext context) 
  {
    context.RegisterService<tnqvm::TNQVMVisitor>(std::make_shared<tnqvm::ExatnMpsVisitor>());
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExatnMpsActivator)
