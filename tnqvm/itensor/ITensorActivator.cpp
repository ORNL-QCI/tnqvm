#include "ITensorMPSVisitor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

using namespace cppmicroservices;

class US_ABI_LOCAL ITensorActivator: public BundleActivator {
public:
	ITensorActivator(){}

    void Start(BundleContext context){
        auto vis = std::make_shared<tnqvm::ITensorMPSVisitor>();
        context.RegisterService<tnqvm::TNQVMVisitor>(vis);
    }

    void Stop(BundleContext context){}
};


CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ITensorActivator)
