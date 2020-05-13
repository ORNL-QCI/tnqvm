#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "ExaTnMpsVisitor.hpp"
#include "NearestNeighborTransform.hpp"
#include "RandomCircuitGen.hpp"

using namespace cppmicroservices;

class US_ABI_LOCAL ExatnMpsActivator : public BundleActivator {
public:
  ExatnMpsActivator() {}

  void Start(BundleContext context) 
  {
    context.RegisterService<tnqvm::TNQVMVisitor>(std::make_shared<tnqvm::ExatnMpsVisitor>());
    context.RegisterService<xacc::IRTransformation>(std::make_shared<xacc::quantum::NearestNeighborTransform>());
    context.RegisterService<xacc::Instruction>(std::make_shared<xacc::circuits::RCS>());
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExatnMpsActivator)
