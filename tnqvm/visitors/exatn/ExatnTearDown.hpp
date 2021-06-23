#pragma once

#include "TearDown.hpp"
#include "exatn.hpp"
#include "xacc.hpp"

// Impl xacc TearDown interface to finalize ExaTN when we are done (XACC::Finalize)
namespace tnqvm {
class ExatnTearDown : public xacc::TearDown
{
public:
    virtual void tearDown() override
    {
        // Note: we lazily initialize ExaTN but always register this TearDown service by default,
        // hence must check whether exaTN is initialized or not before finalizing it.
        if(exatn::isInitialized())
        {
            xacc::debug("[exatn tear down] Finalizing ExaTN service...");
            exatn::finalize();
        }
    }
};
} // namespace