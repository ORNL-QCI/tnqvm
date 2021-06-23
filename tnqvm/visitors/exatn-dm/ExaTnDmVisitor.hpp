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
 * Full density matrix noisy visitor:
 * Name: "exatn-dm"
 * Supported initialization keys:
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
 * |  Initialization Parameter   |                  Parameter Description                                 |    type     |         default          |
 * +=============================+========================================================================+=============+==========================+
 * | backend-json                | Backend configuration JSON to estimate the noise model from.           |    string   | None                     |
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
 * | backend                     | Name of the IBMQ backend to query the backend configuration.           |    string   | None                     |
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
 * If either `backend-json` or `backend` is provided, the `exatn-dm` simulator will simulate the backend noise associated with each quantum gate.
*/

#pragma once

#include "TNQVMVisitor.hpp"
#include "tensor_network.hpp"
#include "exatn.hpp"

namespace xacc {
// Forward declaration
class NoiseModel;
struct NoiseChannelKraus;
} // namespace xacc
namespace tnqvm {
// Single-site Kraus Op (this Accelerator can only model this)
struct KrausOp {
  size_t qubit;
  // Choi matrix
  std::vector<std::vector<std::complex<double>>> mats;
};

class ExaTnDmVisitor : public TNQVMVisitor
{
public:
    // Constructor
    ExaTnDmVisitor();

    // Virtual function impls:
    virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) override;
    virtual void finalize() override;

    // Service name as defined in manifest.json
    virtual const std::string name() const override { return "exatn-dm"; }
    virtual const std::string description() const override { return "ExaTN Density Matrix Visitor"; }
    virtual std::shared_ptr<TNQVMVisitor> clone() override { return std::make_shared<ExaTnDmVisitor>(); }

    // one-qubit gates
    virtual void visit(Identity& in_IdentityGate) override;
    virtual void visit(Hadamard& in_HadamardGate) override;
    virtual void visit(X& in_XGate) override;
    virtual void visit(Y& in_YGate) override;
    virtual void visit(Z& in_ZGate) override;
    virtual void visit(Rx& in_RxGate) override;
    virtual void visit(Ry& in_RyGate) override;
    virtual void visit(Rz& in_RzGate) override;
    virtual void visit(T& in_TGate) override;
    virtual void visit(Tdg& in_TdgGate) override;
    virtual void visit(CPhase& in_CPhaseGate) override;
    virtual void visit(U& in_UGate) override;
    // two-qubit gates
    virtual void visit(CNOT& in_CNOTGate) override;
    virtual void visit(Swap& in_SwapGate) override;
    virtual void visit(CZ& in_CZGate) override;
    virtual void visit(iSwap& in_iSwapGate) override;
    virtual void visit(fSim& in_fsimGate) override;
    // others
    virtual void visit(Measure& in_MeasureGate) override;

    virtual const double getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) override;
    class ExaTnTensorFunctor : public talsh::TensorFunctor<exatn::Identifiable>
    {
    public:
        ExaTnTensorFunctor(const std::function<int(talsh::Tensor& in_tensor)>& in_func) :
            m_func(in_func)
        {}

        const std::string name() const override { return "default"; }
        const std::string description() const override { return ""; }
        virtual void pack(BytePacket& packet) override {};
        virtual void unpack(BytePacket& packet) override {};
        virtual int apply(talsh::Tensor& local_tensor) override { return m_func(local_tensor); }
    private:
        std::function<int(talsh::Tensor& in_tensor)> m_func;
    };

    [[nodiscard]] exatn::TensorNetwork buildInitialNetwork(size_t in_nbQubits) const;
    void applySingleQubitGate(xacc::quantum::Gate& in_gateInstruction);
    void applyTwoQubitGate(xacc::quantum::Gate& in_gateInstruction);
    void applyNoise(xacc::quantum::Gate &in_gateInstruction);

  private:
    exatn::TensorNetwork m_tensorNetwork;
    std::shared_ptr<AcceleratorBuffer> m_buffer;
    std::vector<size_t> m_measuredBits;
    int m_nbShots;
    int m_tensorIdCounter;
    std::shared_ptr<xacc::NoiseModel> m_noiseConfig;
};
} // namespace tnqvm