/***********************************************************************************
 * Copyright (c) 2018, UT-Battelle
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
#ifndef TNQVM_TNQVMVISITOR_HPP_
#define TNQVM_TNQVMVISITOR_HPP_

#include "Identifiable.hpp"
#include "AllGateVisitor.hpp"
#include "xacc.hpp"
#include <sstream>

using namespace xacc;
using namespace xacc::quantum;

template <typename... Ts>
std::string concat(Ts&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Ts>(args));
  return oss.str();
}

// Extra predicate to control timing log.
// i.e. conditionally disable timing log if neccessary (MPI execution)
extern bool tnqvm_timing_log_enabled;
// Macro to define a Telemetry zone whose execution time is tracked.
// Using macros so that we can opt out if not need telemetry.
#define TNQVM_TELEMETRY_ZONE(NAME, FILE, LINE) \
  xacc::ScopeTimer __telemetry__timer(concat("tnqvm::", NAME, " (", FILE, ":", LINE, ")"), xacc::verbose && tnqvm_timing_log_enabled);

namespace tnqvm {
class TNQVMVisitor : public AllGateVisitor, public OptionsProvider,
                     public xacc::Cloneable<TNQVMVisitor> {
public:
  virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots = 1) = 0;
  virtual const double
  getExpectationValueZ(std::shared_ptr<CompositeInstruction> function) = 0;

  virtual const std::vector<std::complex<double>> getState() {
    return std::vector<std::complex<double>>{};
  }
  virtual void finalize() = 0;
  void setOptions(const HeterogeneousMap& in_options) { options = in_options; }
  virtual void setKernelName(const std::string& in_kernelName) {}
  // Does this visitor implementation support VQE mode execution?
  // i.e. ability to cache the state vector after simulating the ansatz.
  virtual bool supportVqeMode() const { return false; }
  // Execution information that visitor wants to persist.
  HeterogeneousMap getExecutionInfo() const { return executionInfo; }

protected:
  std::shared_ptr<AcceleratorBuffer> buffer;
  HeterogeneousMap options;
  // Visitor impl to set if need be.
  HeterogeneousMap executionInfo;
};

} // namespace tnqvm

#endif /* TNQVM_TNQVMVISITOR_HPP_ */
