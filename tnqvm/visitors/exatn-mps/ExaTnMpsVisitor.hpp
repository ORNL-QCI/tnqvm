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
 * MPS visitor:
 * Name: "exatn-mps"
 * Supported initialization keys:
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
 * |  Initialization Parameter   |                  Parameter Description                                 |    type     |         default          |
 * +=============================+========================================================================+=============+==========================+
 * | svd-cutoff                  | SVD cut-off limit.                                                     |    double   | numeric_limits::min      |
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
 * | max-bond-dim                | Max bond dimension to keep.                                            |    int      | no limit                 |
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
 * | mpi-communicator            | The MPI communicator to initialize ExaTN runtime with.                 |    void*    | <unused>                 |
 * |                             | If not provided, by default, ExaTN will use `MPI_COMM_WORLD`.          |             |                          |
 * +-----------------------------+------------------------------------------------------------------------+-------------+--------------------------+
*/

#pragma once

#include "TNQVMVisitor.hpp"
#include "GateTensorAggregator.hpp"
#include "tensor_network.hpp"

namespace tnqvm {
class ExatnMpsVisitor : public TNQVMVisitor, public IAggregatorListener
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
    virtual void visit(iSwap& in_iSwapGate) override;
    virtual void visit(fSim& in_fsimGate) override;
    // others
    virtual void visit(Measure& in_MeasureGate) override;

    virtual const double getExpectationValueZ(std::shared_ptr<CompositeInstruction> in_function) override;
    virtual void onFlush(const AggregatedGroup& in_group) override;

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
    
private:
    // Evaluates the whole tensor network and returns state vector
    void evaluateTensorNetwork(exatn::numerics::TensorNetwork& io_tensorNetwork, std::vector<std::complex<double>>& out_stateVec);
    void addMeasureBitStringProbability(const std::vector<size_t>& in_bits, const std::vector<std::complex<double>>& in_stateVec, int in_shotCount);
    void applyGate(xacc::Instruction& in_gateInstruction);
    void applyTwoQubitGate(xacc::Instruction& in_gateInstruction);
    // Get a sample measurement bit string:
    // In this function, we get RDM by opening one qubit line at a time (same order as the provided list).
    // Then, we contract the whole tensor network to get the RDM for that qubit.
    // Randomly select a binary (1/0) result based on the RDM, then close that tensor leg by projecting it onto the selected result.
    // Continue with the next qubit line (conditioned on the previous measurement result).
    std::vector<uint8_t> getMeasureSample(const std::vector<size_t>& in_qubitIdx, exatn::ProcessGroup *in_processGroup = nullptr);
    void printStateVec();
    // Truncate the bond dimension between two tensors that are decomposed by SVD
    void truncateSvdTensors(const std::string &in_leftTensorName,
                            const std::string &in_rightTensorName,
                            double in_eps = std::numeric_limits<double>::min(),
                            exatn::ProcessGroup *in_processGroup = nullptr);
    std::vector<std::complex<double>> computeWaveFuncSlice(const exatn::numerics::TensorNetwork& in_tensorNetwork, const std::vector<int>& bitString, const exatn::ProcessGroup& in_processGroup) const;
    double computeStateVectorNorm(const exatn::numerics::TensorNetwork& in_tensorNetwork, const exatn::ProcessGroup& in_processGroup) const;

private:
    TensorAggregator m_aggregator;
    std::shared_ptr<AcceleratorBuffer> m_buffer;
    std::vector<std::string> m_qubitTensorNames;
    std::shared_ptr<exatn::numerics::Tensor> m_rootTensor;
    std::shared_ptr<exatn::numerics::TensorNetwork> m_tensorNetwork;
    size_t m_tensorIdCounter;
    size_t m_aggregatedGroupCounter;
    std::unordered_set<std::string> m_registeredGateTensors;
    std::vector<std::complex<double>> m_stateVec;
    std::vector<size_t> m_measureQubits;
    int m_shotCount;
    bool m_aggregateEnabled;
    double m_svdCutoff;
    int m_maxBondDim;
#ifdef TNQVM_MPI_ENABLED
    // Rebuild the tensor network (m_tensorNetwork) from individual MPS tensors:
    // e.g. after bond dimension changes.
    void rebuildTensorNetwork();
    // Min-max qubit range (inclusive) that this process handles
    std::pair<size_t, size_t> m_qubitRange;
    // The self process group that the current process belongs to.
    std::shared_ptr<exatn::ProcessGroup> m_selfProcessGroup;
    // Left and right shared process groups (to communicate with neighboring processes)
    // One of them can be null for the leftmost and rightmost processes.
    std::shared_ptr<exatn::ProcessGroup> m_leftSharedProcessGroup;
    std::shared_ptr<exatn::ProcessGroup> m_rightSharedProcessGroup;
    size_t m_rank;
    // Map from qubit indices to MPI rank which owns the qubit tensor.
    std::unordered_map<size_t, size_t> m_qubitIdxToRank;
#endif
};
}
