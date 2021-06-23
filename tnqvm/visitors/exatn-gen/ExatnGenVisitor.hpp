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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Implementation - Thien Nguyen
 *   Implementation - Dmitry Lyakh
 *
 **********************************************************************************/
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// |  Initialization Parameter   |                  Parameter Description                                 |    type     |         default          |
// +=============================+========================================================================+=============+==========================+
// | reconstruct-layers          | Perform reconstruction after this number of consecutive 2-q gates      |    int      | -1 (no reconstruct)      |
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// | reconstruct-tolerance       | Reconstruction convergence tolerance                                   |    double   | 1e-4                     |
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// | max-bond-dim                | Reconstruction max bond dimension (inside approximating tensor network)|    int      | 512                      |
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// | reconstruct-builder         | Reconstruction network builder (builds the tensor network ansatz)      |    string   | "MPS"                    |
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
#pragma once

#ifdef TNQVM_HAS_EXATN
#include "TNQVMVisitor.hpp"
#include "exatn.hpp"
#include "base/Gates.hpp"
#include "utils/GateMatrixAlgebra.hpp"

namespace tnqvm {
enum class ObsOpType { I, X, Y, Z, NA };

// Simple struct to identify a concrete quantum gate instance,
// For example, parametric gates, e.g. Rx(theta), will have an instance for each
// value of theta that is used to instantiate the gate matrix.
struct GateInstanceIdentifier {
  GateInstanceIdentifier(const std::string &in_gateName);

  template <typename... GateParams>
  GateInstanceIdentifier(const std::string &in_gateName,
                         const GateParams &... in_params);

  template <typename GateParam> void addParam(const GateParam &in_param);

  template <typename GateParam, typename... MoreParams>
  void addParam(const GateParam &in_param, const MoreParams &... in_moreParams);

  std::string toNameString() const;

private:
  std::string m_gateName;
  std::vector<std::string> m_gateParams;
};

// XACC simulation backend (TNQVM) visitor based on ExaTN
template <typename TNQVM_COMPLEX_TYPE>
class ExatnGenVisitor : public TNQVMVisitor {
public:
  // Constructor
  ExatnGenVisitor();
  typedef typename TNQVM_COMPLEX_TYPE::value_type TNQVM_FLOAT_TYPE;
  virtual exatn::TensorElementType getExatnElementType() const = 0;
  // Virtual function impls:
  virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer,
                          int nbShots) override;
  virtual void finalize() override;

  // Service name as defined in manifest.json
  virtual const std::string name() const override { return "exatn-gen"; }

  virtual const std::string description() const override {
    return "ExaTN Visitor";
  }

  virtual OptionPairs getOptions() override { /*TODO: define options */
    return OptionPairs{};
  }

  virtual void setKernelName(const std::string &in_kernelName) override {
    m_kernelName = in_kernelName;
  }

  // one-qubit gates
  virtual void visit(Identity &in_IdentityGate) override;
  virtual void visit(Hadamard &in_HadamardGate) override;
  virtual void visit(X &in_XGate) override;
  virtual void visit(Y &in_YGate) override;
  virtual void visit(Z &in_ZGate) override;
  virtual void visit(Rx &in_RxGate) override;
  virtual void visit(Ry &in_RyGate) override;
  virtual void visit(Rz &in_RzGate) override;
  virtual void visit(T &in_TGate) override;
  virtual void visit(Tdg &in_TdgGate) override;
  virtual void visit(CPhase &in_CPhaseGate) override;
  virtual void visit(U &in_UGate) override;
  // two-qubit gates
  virtual void visit(CNOT &in_CNOTGate) override;
  virtual void visit(Swap &in_SwapGate) override;
  virtual void visit(CZ &in_CZGate) override;
  virtual void visit(iSwap &in_iSwapGate) override;
  virtual void visit(fSim &in_fsimGate) override;
  // others
  virtual void visit(Measure &in_MeasureGate) override;
  virtual bool supportVqeMode() const override { return true; }
  virtual const double getExpectationValueZ(
      std::shared_ptr<CompositeInstruction> in_function) override;

private:
  template <tnqvm::CommonGates GateType, typename... GateParams>
  void appendGateTensor(const xacc::Instruction &in_gateInstruction,
                        GateParams &&... in_params);
  std::vector<ObsOpType>
  analyzeObsSubCircuit(std::shared_ptr<CompositeInstruction> in_function) const;
  exatn::TensorOperator
  constructObsTensorOperator(const std::vector<ObsOpType> &in_obsOps) const;
  void reconstructCircuitTensor();

private:
  std::shared_ptr<exatn::TensorNetwork> m_qubitNetwork;
  exatn::TensorExpansion m_tensorExpansion;
  int m_layersReconstruct;
  double m_reconstructTol;
  int m_layerCounter;
  int m_maxBondDim;
  std::string m_reconstructBuilder;
  std::string m_kernelName;
  std::shared_ptr<AcceleratorBuffer> m_buffer;
  std::unordered_map<std::string, std::vector<TNQVM_COMPLEX_TYPE>>
      m_gateTensorBodies;
  std::set<int> m_measuredBits;
  std::shared_ptr<exatn::TensorOperator> m_obsTensorOperator;
  std::unordered_map<std::string, size_t> m_compositeNameToComponentId;
  std::shared_ptr<exatn::TensorExpansion> m_evaluatedExpansion;
};

template class ExatnGenVisitor<std::complex<double>>;
template class ExatnGenVisitor<std::complex<float>>;

class DoublePrecisionExatnGenVisitor : public ExatnGenVisitor<std::complex<double>> {
  virtual const std::string name() const override { return "exatn-gen:double"; }
  virtual exatn::TensorElementType getExatnElementType() const override {
    return exatn::TensorElementType::COMPLEX64;
  }
  virtual std::shared_ptr<TNQVMVisitor> clone() override {
    return std::make_shared<DoublePrecisionExatnGenVisitor>();
  }
};

class SinglePrecisionExatnGenVisitor : public ExatnGenVisitor<std::complex<float>> {
  virtual const std::string name() const override { return "exatn-gen:float"; }
  virtual exatn::TensorElementType getExatnElementType() const override {
    return exatn::TensorElementType::COMPLEX32;
  }
  virtual std::shared_ptr<TNQVMVisitor> clone() override {
    return std::make_shared<SinglePrecisionExatnGenVisitor>();
  }
};

class DefaultExatnGenVisitor : public DoublePrecisionExatnGenVisitor {
  virtual const std::string name() const override { return "exatn-gen"; }
};
} // end namespace tnqvm

#endif // TNQVM_HAS_EXATN
