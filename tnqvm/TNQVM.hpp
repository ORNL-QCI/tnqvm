/***********************************************************************************
 * Copyright (c) 2017, UT-Battelle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the xacc nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Initial API and implementation - Alex McCaskey
 *
 **********************************************************************************/
#ifndef TNQVM_TNQVM_HPP_
#define TNQVM_TNQVM_HPP_

#include "xacc.hpp"
#include "xacc_service.hpp"
#include "TNQVMVisitor.hpp"
#include <cassert>

namespace tnqvm {

class TNQVM : public Accelerator {
public:
  void initialize(const xacc::HeterogeneousMap &params = {}) override {
    if (xacc::optionExists("tnqvm-verbose")) {
      __verbose = 1;
    } else {
      __verbose = 0;
    }    

    // Force a configuration update.
    updateConfiguration(params);
  }
  void updateConfiguration(const HeterogeneousMap &config) override {
    if (config.keyExists<bool>("verbose") && config.get<bool>("verbose")) {
      __verbose = 1;
    }
    if (config.keyExists<bool>("vqe-mode")) {
      vqeMode = config.get<bool>("vqe-mode");
    }

    if (config.stringExists("tnqvm-visitor")) {
      const auto requestedBackend = config.getString("tnqvm-visitor");
      const auto& allVisitorServices = xacc::getServices<TNQVMVisitor>();
      // We must have at least one TNQVM service registered.
      assert(!allVisitorServices.empty());
      bool foundRequestedBackend = false;
      
      for (const auto& registeredService: allVisitorServices)
      {
        if (registeredService->name() == requestedBackend)
        {
          // Found it, use that service name.
          backendName = registeredService->name();
          foundRequestedBackend = true;
          break;
        }
      }
      // A visitor backend was explicitly specified but the corresponding service cannot be found,
      // e.g. the service name was misspelled.
      if (!foundRequestedBackend)
      {
        backendName = DEFAULT_VISITOR_BACKEND;
        xacc::warning("The requested TNQVM visitor backend '" + requestedBackend + "' cannot be found in the service registry. Please make sure the name is correct and the service is installed.\n"
          "The default visitor backend of type '" + DEFAULT_VISITOR_BACKEND + "' will be used.");
      }
    }
    else {
      // No visitor type was specified, use the default.
      backendName = DEFAULT_VISITOR_BACKEND;
      xacc::warning("No visitor backend was specified. The default visitor backend of type '" + DEFAULT_VISITOR_BACKEND + "' will be used.");
    }    
  }
  const std::vector<std::string> configurationKeys() override { return {}; }
//   const std::string getSignature() override {return name()+":";}

  void
  execute(std::shared_ptr<AcceleratorBuffer> buffer,
          const std::shared_ptr<xacc::CompositeInstruction> kernel) override;

  void execute(std::shared_ptr<AcceleratorBuffer> buffer,
               const std::vector<std::shared_ptr<CompositeInstruction>>
                   functions) override;

  const std::vector<std::complex<double>>
  getAcceleratorState(std::shared_ptr<CompositeInstruction> program) override;

  const std::string name() const override { return "tnqvm"; }
  
  const std::string description() const override {
    return "XACC tensor network quantum virtual machine (TNQVM) Accelerator";
  }

  const std::string& getVisitorName() const { return backendName; }
  
  virtual ~TNQVM() {}

  int verbose() const { return __verbose; }
  void verbose(int level) { __verbose = level; }
  void set_verbose(int level) { __verbose = level; }
  void mute() { __verbose = 0; }
  void unmute() { __verbose = 1; } // default to 1
  
protected:
  std::shared_ptr<TNQVMVisitor> visitor;

private:
  int __verbose = 1;
  bool executedOnce = false;
  bool vqeMode = true;
  // Default visitor backend is ITensor.
  // TODO: we may eventually use our exatn-mps as default.
  static const std::string DEFAULT_VISITOR_BACKEND;
  // The backend name that is configured.
  std::string backendName;
};
} // namespace tnqvm

#endif
