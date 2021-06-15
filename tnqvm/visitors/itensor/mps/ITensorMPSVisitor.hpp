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
 *   Initial implementation - Mengsu Chen 2017.7
 *
 **********************************************************************************/
#ifndef QUANTUM_GATE_ACCELERATORS_TNQVM_ITensorMPSVisitor_HPP_
#define QUANTUM_GATE_ACCELERATORS_TNQVM_ITensorMPSVisitor_HPP_

#include <cstdlib>
#include "TNQVMVisitor.hpp"
#include "Cloneable.hpp"
#include "itensor/all.h"

namespace tnqvm {

class ITensorMPSVisitor : public TNQVMVisitor {
  using ITensor = itensor::ITensor;
  using Index = itensor::Index;
  using IndexVal = itensor::IndexVal;

public:
  ITensorMPSVisitor();
  virtual ~ITensorMPSVisitor();
  
  virtual std::shared_ptr<TNQVMVisitor> clone() {
    return std::make_shared<ITensorMPSVisitor>();
  }

  virtual const double getExpectationValueZ(std::shared_ptr<CompositeInstruction> function) override;

  virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots = 1) override;
  virtual void finalize() override;
  // Service name as defined in manifest.json
  virtual const std::string name() const { return "itensor-mps"; }

  virtual const std::string description() const { return ""; }

  /**
   * Return all relevant TNQVM runtime options.
   */
  virtual OptionPairs getOptions() {
    OptionPairs desc{{"itensor-svd-cutoff",
                        "Provide the cutoff (default 1e-4) for the singular "
                        "value decomposition."}};
    return desc;
  }

   // one-qubit gates
  void visit(Identity &gate) {}
  void visit(Hadamard &gate);
  void visit(X &gate);
  void visit(Y &gate);
  void visit(Z &gate);
  void visit(Rx &gate);
  void visit(Ry &gate);
  void visit(Rz &gate);
  void visit(CPhase &cp);
  void visit(U& u);

  // two-qubit gates
  void visit(CNOT &gate);
  void visit(Swap &gate);
  void visit(CZ &gate);
  // others
  void visit(Measure &gate);
  virtual bool supportVqeMode() const override { return true; }

private:
  void applySingleQubitGate(xacc::Instruction &in_gate);
  void applyTwoQubitGate(itensor::ITensor &in_gateTensor, size_t in_siteId1,
                         size_t in_siteId2);
  itensor::Index getSiteIndex(size_t site_id);

  // Returns Up1, Dn1, Up2, Dn2, UpP1, DnP1, UpP2, DnP2
  // indices to construct 2-q gate tensor.
  std::tuple<itensor::IndexVal, itensor::IndexVal, itensor::IndexVal,
             itensor::IndexVal, itensor::IndexVal, itensor::IndexVal,
             itensor::IndexVal, itensor::IndexVal>
  getTwoQubitOpInds(size_t in_siteId1, size_t in_siteId2);
  itensor::ITensor createTwoQubitOpTensor(size_t in_siteId1, size_t in_siteId2);
  double compute_expectation_z(const std::vector<size_t> &in_measureBits);

private:
  itensor::MPS m_mps;
  std::vector<size_t> m_measureBits;
  std::shared_ptr<AcceleratorBuffer> m_buffer;
  // SVD options
  double m_svdCutoff;
  int m_maxDim;
  std::shared_ptr<CompositeInstruction> m_conjCircuit;
  bool m_logSvd = false;
};

} // namespace tnqvm
#endif
