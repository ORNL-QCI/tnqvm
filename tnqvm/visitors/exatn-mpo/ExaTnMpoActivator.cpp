#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "ExaTnMpoVisitor.hpp"

using namespace cppmicroservices;

class US_ABI_LOCAL ExatnMpoActivator : public BundleActivator {
public:
  ExatnMpoActivator() {}

  void Start(BundleContext context) 
  {
    context.RegisterService<tnqvm::TNQVMVisitor>(std::make_shared<tnqvm::ExatnMpoVisitor>());
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExatnMpoActivator)
