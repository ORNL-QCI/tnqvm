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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Initial sketch - Mengsu Chen 2017/07/17;
 *   Implementation - Dmitry Lyakh 2017/10/05 - active;
 *
 **********************************************************************************/
#pragma once

#ifdef TNQVM_HAS_EXATN

#include <cstdlib>
#include <complex>
#include <vector>
#include <utility>
#include "TNQVMVisitor.hpp"
#include "tensor_network.hpp"

using namespace xacc;
using namespace xacc::quantum;
using TensorFunctor = talsh::TensorFunctor<exatn::Identifiable>;

namespace tnqvm {
    // Simple struct to identify a concrete quantum gate instance,
    // For example, parametric gates, e.g. Rx(theta), will have an instance for each value of theta
    // that is used to instantiate the gate matrix.
    struct GateInstanceIdentifier
    {  
        GateInstanceIdentifier(const std::string& in_gateName);
        
        template<typename... GateParams>
        GateInstanceIdentifier(const std::string& in_gateName, const GateParams&... in_params);
        
        template<typename GateParam>
        void addParam(const GateParam& in_param);

        template<typename GateParam, typename... MoreParams>
        void addParam(const GateParam& in_param, const MoreParams&... in_moreParams);
        
        std::string toNameString() const;

        private:
            std::string m_gateName;
            std::vector<std::string> m_gateParams; 
    };

    class DefaultTNQVMTensorFunctor : public TensorFunctor 
    {
        const std::string name() const override { return "TNQVM Tensor Functor"; }
        const std::string description() const override { return "TNQVM Tensor Functor"; }
        virtual void pack(BytePacket& packet) override {};
        virtual void unpack(BytePacket& packet) override {};
    }; 

    // Functor to inspect/print out tensor components
    class TensorComponentPrintFunctor : public DefaultTNQVMTensorFunctor 
    {
    public:
        virtual int apply(talsh::Tensor& local_tensor) override;
    };

    // Functor to retrieve the state vector from the final/root tensor
    class ReconstructStateVectorFunctor : public DefaultTNQVMTensorFunctor
    {
    public:
        ReconstructStateVectorFunctor(const std::shared_ptr<AcceleratorBuffer>& buffer, std::vector<std::complex<double>>& io_stateVec);
        virtual int apply(talsh::Tensor& local_tensor) override;

    private:
        int m_qubits;
        std::vector<std::complex<double>>& m_stateVec;
    }; 

    // Functor to calculate the expected Z value across a list of qubit.
    class CalculateExpectationValueFunctor : public DefaultTNQVMTensorFunctor
    {
    public:
        CalculateExpectationValueFunctor(const std::vector<int>& qubitIndex);
        virtual int apply(talsh::Tensor& local_tensor) override;
        double getResult() const { return m_result; }
    private:
        const std::vector<int>& m_qubitIndices;
        double m_result;
    };

    class ApplyQubitMeasureFunctor : public DefaultTNQVMTensorFunctor
    {
    public:
        ApplyQubitMeasureFunctor(int qubitIndex);
        virtual int apply(talsh::Tensor& local_tensor) override;
        // Get the result: 0/1 as boolean
        bool getResult() const { return m_result; }
    private:
        int m_qubitIndex;
        bool m_result;
    };

    class ResetTensorDataFunctor : public DefaultTNQVMTensorFunctor
    {
    public:
        ResetTensorDataFunctor(const std::vector<std::complex<double>>& in_stateVec);
        virtual int apply(talsh::Tensor& local_tensor) override;
    private:
        const std::vector<std::complex<double>>& m_stateVec;
    };  

    // Forward declarations:
    enum class CommonGates: int;
   
    // ExaTN type alias:
    using TensorNetwork = exatn::numerics::TensorNetwork;
    using Tensor = exatn::numerics::Tensor;
    using TensorShape = exatn::numerics::TensorShape;
    using TensorLeg = exatn::numerics::TensorLeg;    
    
    class ExatnVisitor;

    // Special listener interface for unit testing/logging purposes
    class IExatnListener
    {
        public:
        // Called when the tensor network has been constructed, ready to be submitted.
        virtual void preEvaluate(ExatnVisitor* in_backEnd) = 0;
        // Called just before measurement is applied 
        // i.e. collapse then normalize the state vector. 
        virtual void preMeasurement(ExatnVisitor* in_backEnd, Measure& in_measureGate) = 0;
        // After measurement
        virtual void postMeasurement(ExatnVisitor* in_backEnd, Measure& in_measureGate, bool in_bitResult, double in_expectedValue) = 0;
        // Called before finalization, i.e. destroy the tensors.
        // Note: if there is no measurement in the circuit, 
        // this is where you can retrieve the final state vector.
        virtual void onEvaluateComplete(ExatnVisitor* in_backEnd) = 0;
    };

    // Debug logger
    class ExatnDebugLogger: public IExatnListener
    {
    public:
        static ExatnDebugLogger* GetInstance() {
            static ExatnDebugLogger instance;
            return &instance;
        }

        virtual void preEvaluate(ExatnVisitor* in_backEnd) override; 
        virtual void preMeasurement(ExatnVisitor* in_backEnd, Measure& in_measureGate) override;
        virtual void postMeasurement(tnqvm::ExatnVisitor* in_backEnd, xacc::quantum::Measure& in_measureGate, bool in_bitResult, double in_expectedValue) override; 
        virtual void onEvaluateComplete(tnqvm::ExatnVisitor* in_backEnd) override {}

    private:
        ExatnDebugLogger() = default;
        ~ExatnDebugLogger() = default;
    
        ExatnDebugLogger(const ExatnDebugLogger&) = delete;
        ExatnDebugLogger& operator=(const ExatnDebugLogger&) = delete;
        ExatnDebugLogger(ExatnDebugLogger&&) = delete;
        ExatnDebugLogger& operator=(ExatnDebugLogger&&) = delete;
    };

    // XACC simulation backend (TNQVM) visitor based on ExaTN
    class ExatnVisitor : public TNQVMVisitor
    {
    public:
        // Constructor
        ExatnVisitor();
        
        // Virtual function impls:        
        virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer, int nbShots) override;
        virtual void finalize() override;
        
        // Service name as defined in manifest.json
        virtual const std::string name() const override { return "exatn"; }

        virtual const std::string description() const override { return "ExaTN MPS Visitor"; }

        virtual std::shared_ptr<TNQVMVisitor> clone() override { return std::make_shared<ExatnVisitor>(); }
        
        virtual OptionPairs getOptions() override { /*TODO: define options */ return OptionPairs{}; }
        
        virtual void setKernelName(const std::string& in_kernelName) override { m_kernelName = in_kernelName; }

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

        void subscribe(IExatnListener* listener) { m_listeners.emplace_back(listener); }
        std::vector<std::complex<double>> retrieveStateVector();

        // Advance API (based on unique capabilities of ExaTN/ExaTensor):
        // (1) Calculate the expectation value for an arbitrary observable operator
        // without explicitly evaluate the state vector.
        // i.e. get the result of `<psi| Hamiltonian |psi>` for an arbitrary Hamiltonian.
        // The Hamiltonian must be expressed as a sum of products of gates (with a complex coefficient for each term):
        // e.g. a1*Z1Z2 + a2*X0X1 + .. + Y2Y3, etc.
        struct ObservableTerm
        {
            ObservableTerm(const std::vector<std::shared_ptr<Instruction>>& in_operatorsInProduct, const std::complex<double>& in_coeff = 1.0);
            std::complex<double> coefficient;
            std::vector<std::shared_ptr<Instruction>> operators;
        };
        // Note: the composite function is the *ORIGINAL* circuit (aka the ansatz), i.e. no change-of-basis or measurement is required.
        std::complex<double> observableExpValCalc(std::shared_ptr<AcceleratorBuffer>& in_buffer, std::shared_ptr<CompositeInstruction>& in_function, const std::vector<ObservableTerm>& in_observableExpression); 
        

        // (2) Get the RDM for a subset of qubit,
        // e.g. to simulate samping (measurement) of those qubits.
        //////////////////////////////////////////////////////
        // o------|---------|-------------|---------|------o
        // o------| Quantum |-------------| Inverse |------o
        // o------| Circuit |---o RDM o---| Quantum |------o
        // o------|         |---o RDM o---| Circuit |------o
        // o------|---------|-------------|---------|------o
        /////////////////////////////////////////////////////
        // Returns the *flatten* RDM (size = 2^(2N)), N = number of *open* qubit wires.
        std::vector<std::complex<double>> getReducedDensityMatrix(std::shared_ptr<AcceleratorBuffer>& in_buffer, std::shared_ptr<CompositeInstruction>& in_function, const std::vector<size_t>& in_qubitIdx);
        

        // (3) Get a sample measurement bit string:
        // In this mode, we get RDM by opening one qubit line at a time (same order as the provided list).
        // Then, we contract the whole tensor network to get the RDM for that qubit.
        // Randomly select a binary (1/0) result based on the RDM, then close that tensor leg by projecting it onto the selected result.
        // Continue with the next qubit line (conditioned on the previous measurement result).
        std::vector<uint8_t> getMeasureSample(std::shared_ptr<AcceleratorBuffer>& in_buffer, std::shared_ptr<CompositeInstruction>& in_function, const std::vector<size_t>& in_qubitIdx);
    private:
        template<tnqvm::CommonGates GateType, typename... GateParams>
        void appendGateTensor(const xacc::Instruction& in_gateInstruction, GateParams&&... in_params);
        void evaluateNetwork(); 
        void resetExaTN(); 
        void resetNetwork();
        std::complex<double> expVal(const std::vector<ObservableTerm>& in_observableExpression); 
        std::complex<double> evaluateTerm(const std::vector<std::shared_ptr<Instruction>>& in_observableTerm); 
        void applyInverse();
        std::vector<uint8_t> generateMeasureSample(const TensorNetwork& in_tensorNetwork, const std::vector<int>& in_qubitIdx);
        // Calculate the flops and memory requirements to generate a full sample (all qubits) for the input tensor network.
        // Note: this doesn't actually contract the tensor network, just getting this data from the ExaTN optimizer.
        // Output: pairs of flops and memory (in bytes); one pair for each qubit.
        std::vector<std::pair<double, double>> calcFlopsAndMemoryForSample(const TensorNetwork& in_tensorNetwork);

    private:
       TensorNetwork m_tensorNetwork;
       unsigned int m_tensorIdCounter;
       bool m_hasEvaluated;
       // The AcceleratorBuffer shared_ptr, null if not initialized.
       std::shared_ptr<AcceleratorBuffer> m_buffer; 
       
       // List of listeners
       std::vector<IExatnListener*> m_listeners;

       // Bit-string of the measurement result, the bit order of this 
       // is the order in which the measure gates are specified:
       // e.g. Measure(q[2]); Measure (q[1]); Measure (q[0]); => q2q1q0, etc.
       std::string m_resultBitString;
       // Count of the number of shots that was requested.
       int m_shots; 
       std::vector<int> m_measureQbIdx;

       // Map of gate tensors that we've initialized with ExaTN.
       // The list is indexed by Tensor Name.
       // Note: the tensor name is unique for each gate tensor, hence, for parameterized gate, 
       // the name is a full/expanded name, e.g. Rx(1.234), Ry(2.3456), etc.
       // This map is lazily constructed while we traverse the gate sequence,
       // i.e. if we encounter a new gate whose tensor has not been initialized, 
       // then we initialize before append the gate tensor (referencing the tensor by name) to the network.  
       std::unordered_map<std::string, std::vector<std::complex<double>>> m_gateTensorBodies; 
       
       // List of gate tensors (name and leg pairing) that we have appended to the network.
       // We use this list to construct the inverse tensor sequence (e.g. isometric collapse)
       std::vector<std::pair<std::string, std::vector<unsigned int>>> m_appendedGateTensors;
       bool m_isAppendingCircuitGates;
       // Tensor network of the qubit register (to close the tensor network for expectation calculation)
       TensorNetwork m_qubitRegTensor;
       std::string m_kernelName;
       // Make the debug logger friend, e.g. retrieve internal states for logging purposes.
       friend class ExatnDebugLogger;
    };
} //end namespace tnqvm

#endif //TNQVM_HAS_EXATN