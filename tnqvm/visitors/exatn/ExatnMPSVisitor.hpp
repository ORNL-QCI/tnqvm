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
#ifndef TNQVM_EXATENSORMPSVISITOR_HPP_
#define TNQVM_EXATENSORMPSVISITOR_HPP_

#ifdef TNQVM_HAS_EXATENSOR

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

    // Functor to calculate the expected Z value of a qubit.
    class CalculateExpectationValueFunctor : public DefaultTNQVMTensorFunctor
    {
    public:
        CalculateExpectationValueFunctor(int qubitIndex);
        virtual int apply(talsh::Tensor& local_tensor) override;
        double getResult() const { return m_result; }
    private:
        int m_qubitIndex;
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

// class ExaTNMPSVisitor : public TNQVMVisitor {

// public:

// //Static constants:
//  static const std::size_t INITIAL_VALENCE = 2; //default initial dimension extent for virtual MPS indices
//  static const unsigned int MAX_GATES = 8; //max number of gates in the gate sequence before evaluation

// //Life cycle:
//  ExaTNMPSVisitor();
//  virtual ~ExaTNMPSVisitor();

//  void initialize(std::shared_ptr<AcceleratorBuffer> buffer); //accelerator buffer
//  void finalize();
 
//  virtual std::shared_ptr<TNQVMVisitor> clone() override { return std::make_shared<ExaTNMPSVisitor>(); }
//  virtual const double getExpectationValueZ(std::shared_ptr<CompositeInstruction> function) override;
// //Visitor methods:
//  void visit(Identity& gate) {}
//  void visit(Hadamard & gate);
//  void visit(X & gate);
//  void visit(Y & gate);
//  void visit(Z & gate);
//  void visit(Rx & gate);
//  void visit(Ry & gate);
//  void visit(Rz & gate);
//  void visit(U & gate);
//  void visit(CPhase & gate);
//  void visit(CNOT & gate);
//  void visit(CZ & gate);
//  void visit(Swap & gate);
//  void visit(Measure & gate);
//  void visit(ConditionalFunction & condFunc);
//  void visit(GateFunction & gateFunc);

// //Numerical evaluation:
//  bool isInitialized(); //returns TRUE if the wavefunction has been initialized
//  bool isEvaluated(); //returns TRUE if the wavefunction is fully evaluated and gate sequence is empty, FALSE otherwise
//  void setEvaluationStrategy(const bool eagerEval); //sets tensor network evaluation strategy
//  void setInitialMPSValence(const std::size_t initialValence); //initial dimension extent for virtual MPS indices
//  int evaluate(); //evaluates the constructed tensor network (returns an error or 0)

// private:

// //Type aliases:
//  using TensDataType = GateFactory::TensDataType;
//  using Tensor = GateFactory::Tensor;
//  using TensorLeg = exatensor::TensorLeg;
//  using TensorNetwork = exatensor::TensorNetwork<TensDataType>;
//  using WaveFunction = std::vector<Tensor>;

// //Gate factory member:
//  GateFactory GateTensors;

// //Data members:
//  std::shared_ptr<AcceleratorBuffer> Buffer;        //accelerator buffer
//  WaveFunction StateMPS;                            //MPS wave-function of qubits (MPS tensors)
//  TensorNetwork TensNet;                            //currently constructed tensor network
//  std::vector<std::pair<Tensor, unsigned int *>> GateSequence; //sequence of visited quantum gates before evaluation
//  std::pair<int, int> QubitRange;                   //range of involved qubits in the current tensor network
//  std::vector<unsigned int> OptimizedTensors;       //IDs of the tensors to be optimized in the closed tensor network
//  std::size_t InitialValence;                       //initial dimension extent for virtual dimensions of MPS tensors
//  bool EagerEval;                                   //if TRUE each gate will be applied immediately (defaults to FALSE)

// //Private member functions:
//  void initMPSTensor(Tensor & tensor); //initializes an MPS tensor to a pure |0> state
//  void buildWaveFunctionNetwork(int firstQubit = 0, int lastQubit = -1); //builds a TensorNetwork object for the wavefunction of qubits [first:last]
//  void appendGateSequence(); //appends gate tensors from the current gate sequence to the wavefunction tensor network of qubits [first:last]
//  void closeCircuitNetwork(); //closes the circuit TensorNetwork object with output tensors (those to be optimized)
//  int appendNBodyGate(const Tensor & gate, const unsigned int qubit_id[]); //appends (applies) an N-body quantum gate to the wavefunction

// }; //end class ExaTNMPSVisitor
    
    // Forward declarations:
    enum class CommonGates: int;
   
    // ExaTN type alias:
    using TensorNetwork = exatn::numerics::TensorNetwork;
    using Tensor = exatn::numerics::Tensor;
    using TensorShape = exatn::numerics::TensorShape;
    using TensorLeg = exatn::numerics::TensorLeg;    
    
    class ExatnMPSVisitor;

    // Special listener interface for unit testing/logging purposes
    class IExatnListener
    {
        public:
        // Called when the tensor network has been constructed, ready to be submitted.
        virtual void preEvaluate(ExatnMPSVisitor* in_backEnd) = 0;
        // Called just before measurement is applied 
        // i.e. collapse then normalize the state vector. 
        virtual void preMeasurement(ExatnMPSVisitor* in_backEnd, Measure& in_measureGate) = 0;
        // After measurement
        virtual void postMeasurement(ExatnMPSVisitor* in_backEnd, Measure& in_measureGate, bool in_bitResult, double in_expectedValue) = 0;
        // Called before finalization, i.e. destroy the tensors.
        // Note: if there is no measurement in the circuit, 
        // this is where you can retrieve the final state vector.
        virtual void onEvaluateComplete(ExatnMPSVisitor* in_backEnd) = 0;
    };

    // Debug logger
    class ExatnDebugLogger: public IExatnListener
    {
    public:
        static ExatnDebugLogger* GetInstance() {
            static ExatnDebugLogger instance;
            return &instance;
        }

        virtual void preEvaluate(ExatnMPSVisitor* in_backEnd) override; 
        virtual void preMeasurement(ExatnMPSVisitor* in_backEnd, Measure& in_measureGate) override;
        virtual void postMeasurement(tnqvm::ExatnMPSVisitor* in_backEnd, xacc::quantum::Measure& in_measureGate, bool in_bitResult, double in_expectedValue) override; 
        virtual void onEvaluateComplete(tnqvm::ExatnMPSVisitor* in_backEnd) override {}

    private:
        ExatnDebugLogger() = default;
        ~ExatnDebugLogger() = default;
    
        ExatnDebugLogger(const ExatnDebugLogger&) = delete;
        ExatnDebugLogger& operator=(const ExatnDebugLogger&) = delete;
        ExatnDebugLogger(ExatnDebugLogger&&) = delete;
        ExatnDebugLogger& operator=(ExatnDebugLogger&&) = delete;
    };

    // XACC simulation backend (TNQVM) visitor based on ExaTN
    class ExatnMPSVisitor : public TNQVMVisitor
    {
    public:
        // Constructor
        ExatnMPSVisitor();
        
        // Virtual function impls:        
        virtual void initialize(std::shared_ptr<AcceleratorBuffer> buffer) override;
        virtual void finalize() override;
        
        // Service name as defined in manifest.json
        virtual const std::string name() const override { return "exatn-mps"; }

        virtual const std::string description() const override { return "ExaTN MPS Visitor"; }

        virtual std::shared_ptr<TNQVMVisitor> clone() override { return std::make_shared<ExatnMPSVisitor>(); }
        
        virtual OptionPairs getOptions() override { /*TODO: define options */ return OptionPairs{}; }
    
        // one-qubit gates
        virtual void visit(Identity& in_IdentityGate) override;
        virtual void visit(Hadamard& in_HadamardGate) override;
        virtual void visit(X& in_XGate) override;
        virtual void visit(Y& in_YGate) override;
        virtual void visit(Z& in_ZGate) override;
        virtual void visit(Rx& in_RxGate) override;
        virtual void visit(Ry& in_RyGate) override;
        virtual void visit(Rz& in_RzGate) override;
        virtual void visit(CPhase& in_CPhaseGate) override;
        virtual void visit(U& in_UGate) override;
        // two-qubit gates
        virtual void visit(CNOT& in_CNOTGate) override;
        virtual void visit(Swap& in_SwapGate) override;
        virtual void visit(CZ& in_CZGate) override;
        // others
        virtual void visit(Measure& in_MeasureGate) override;

        void subscribe(IExatnListener* listener) { m_listeners.emplace_back(listener); }
        std::vector<std::complex<double>> retrieveStateVector();
    
    private:
        template<tnqvm::CommonGates GateType, typename... GateParams>
        void appendGateTensor(const xacc::Instruction& in_gateInstruction, GateParams&&... in_params);
        void evaluateNetwork(); 
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

       // Map of gate tensors that we've initialized with ExaTN.
       // The list is indexed by Tensor Name.
       // Note: the tensor name is unique for each gate tensor, hence, for parameterized gate, 
       // the name is a full/expanded name, e.g. Rx(1.234), Ry(2.3456), etc.
       // This map is lazily constructed while we traverse the gate sequence,
       // i.e. if we encounter a new gate whose tensor has not been initialized, 
       // then we initialize before append the gate tensor (referencing the tensor by name) to the network.  
       std::unordered_map<std::string, std::vector<std::complex<double>>> m_gateTensorBodies; 
       // Tensor body data represents a single qubit in the zero state. 
       static const std::vector<std::complex<double>> Q_ZERO_TENSOR_BODY;
       static const std::vector<std::complex<double>> Q_ONE_TENSOR_BODY;
       // Make the debug logger friend, e.g. retrieve internal states for logging purposes.
       friend class ExatnDebugLogger;
    };
} //end namespace tnqvm

#endif //TNQVM_HAS_EXATENSOR

#endif //TNQVM_EXATENSORMPSVISITOR_HPP_