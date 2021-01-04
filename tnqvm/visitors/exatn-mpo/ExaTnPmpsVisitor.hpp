#pragma once

#include "TNQVMVisitor.hpp"
#include "tensor_network.hpp"
#include "exatn.hpp"

// Purified-MPS visitor:
// Name: "exatn-pmps"
// Supported initialization keys:
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// |  Initialization Parameter   |                  Parameter Description                                 |    type     |         default          |
// +=============================+========================================================================+=============+==========================+
// | backend-json                | Backend configuration JSON to estimate the noise model from.           |    string   | None                     |
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// | backend                     | Name of the IBMQ backend to query the backend configuration.           |    string   | None                     |
// +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
// If either `backend-json` or `backend` is provided, the `exatn-pmps` simulator will simulate the backend noise associated with each quantum gate.

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

class ExaTnPmpsVisitor : public TNQVMVisitor
{
public:
    // Constructor
    ExaTnPmpsVisitor();

    // Virtual function impls:        
    virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) override;
    virtual void finalize() override;

    // Service name as defined in manifest.json
    virtual const std::string name() const override { return "exatn-pmps"; }
    virtual const std::string description() const override { return "ExaTN Purified MPS Visitor"; }
    virtual std::shared_ptr<TNQVMVisitor> clone() override { return std::make_shared<ExaTnPmpsVisitor>(); }

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

    [[nodiscard]] exatn::TensorNetwork buildInitialNetwork(size_t in_nbQubits, bool in_createQubitTensors) const;
    void applySingleQubitGate(xacc::quantum::Gate& in_gateInstruction);
    void applyTwoQubitGate(xacc::quantum::Gate& in_gateInstruction);
    void applyKrausOp(const KrausOp& in_op);
    // Apply a local (single-site) Kraus operator
    void applyLocalKrausOp(size_t in_siteId, const std::string& in_opTensorName);
    void truncateSvdTensors(const std::string& in_leftTensorName, const std::string& in_rightTensorName, double in_eps = 1e-9);
    std::vector<KrausOp> convertNoiseChannel(const std::vector<NoiseChannelKraus>& in_channels) const;
private:
    exatn::TensorNetwork m_pmpsTensorNetwork;
    std::shared_ptr<AcceleratorBuffer> m_buffer;
    std::shared_ptr<xacc::NoiseModel> m_noiseConfig;
    std::vector<size_t> m_measuredBits;
    int m_nbShots;
};
} // namespace tnqvm