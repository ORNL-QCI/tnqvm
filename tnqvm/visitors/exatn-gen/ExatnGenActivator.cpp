#include "xacc_plugin.hpp"
#include "ExatnGenVisitor.hpp"
using namespace cppmicroservices;

class US_ABI_LOCAL ExatnGenActivator : public BundleActivator {
public:
  ExatnGenActivator() {}

  void Start(BundleContext context) {
    context.RegisterService<tnqvm::TNQVMVisitor>(
        std::make_shared<tnqvm::DefaultExatnGenVisitor>());
    context.RegisterService<tnqvm::TNQVMVisitor>(
        std::make_shared<tnqvm::SinglePrecisionExatnGenVisitor>());
    context.RegisterService<tnqvm::TNQVMVisitor>(
        std::make_shared<tnqvm::DoublePrecisionExatnGenVisitor>());
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExatnGenActivator)
