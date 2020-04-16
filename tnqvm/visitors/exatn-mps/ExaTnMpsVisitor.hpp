#pragma once

#include "TNQVMVisitor.hpp"
#include "GateTensorAggregator.hpp"
#include "tensor_network.hpp"

namespace tnqvm {
class ExatnMpsVisitor : public TNQVMVisitor, public IAggreratorListener
{
public:
    // Constructor
    ExatnMpsVisitor();

    // Virtual function impls:        
    virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) override;
    virtual void finalize() override;

    // Service name as defined in manifest.json
    virtual const std::string name() const override { return "exatn-mps"; }
    virtual const std::string description() const override { return "ExaTN MPS Visitor"; }
    virtual std::shared_ptr<TNQVMVisitor> clone() override { return std::make_shared<ExatnMpsVisitor>(); }

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
    // others
    virtual void visit(Measure& in_MeasureGate) override;

    virtual const double getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) override;
    virtual void onFlush(const AggreratedGroup& in_group) override;

private:
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
    
    // Evaluates the whole tensor network and returns state vector
    void evaluateTensorNetwork(exatn::numerics::TensorNetwork& io_tensorNetwork, std::vector<std::complex<double>>& out_stateVec);
    void addMeasureBitStringProbability(const std::vector<size_t>& in_bits, const std::vector<std::complex<double>>& in_stateVec, int in_shotCount);
private:
    TensorAggrerator m_aggrerator;
    std::shared_ptr<AcceleratorBuffer> m_buffer; 
    std::vector<std::string> m_qubitTensorNames;
    exatn::numerics::TensorNetwork m_tensorNetwork;
    size_t m_tensorIdCounter;
    size_t m_aggregatedGroupCounter;
    std::unordered_set<std::string> m_registeredGateTensors;
    std::vector<std::complex<double>> m_stateVec;
    std::vector<size_t> m_measureQubits;
    int m_shotCount;
    bool m_aggrerateEnabled; 
};
} 