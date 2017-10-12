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
 *   Implementation - Dmitry Lyakh 2017/10/12;
 *
 **********************************************************************************/
#ifndef QUANTUM_GATE_ACCELERATORS_TNQVM_GATEFACTORY_HPP_
#define QUANTUM_GATE_ACCELERATORS_TNQVM_GATEFACTORY_HPP_

#include <cstdlib>
#include <complex>
#include "AllGateVisitor.hpp"

namespace xacc {
namespace quantum {

class GateBodyFactory{

private:

//Type aliases:

 using TensDataType = std::complex<double>;

//Static data (gate tensors):

 static constexpr const TensDataType HBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType XBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType YBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType ZBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType RxBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType RyBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType RzBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType CPBody[4] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType CNBody[16] = {
  TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType SWBody[16] = {
  TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0)
 };

//Data members:

 std::shared_ptr<TensDataType> HTensor;  //Hadamard
 std::shared_ptr<TensDataType> XTensor;  //Pauli X
 std::shared_ptr<TensDataType> YTensor;  //Pauli Y
 std::shared_ptr<TensDataType> ZTensor;  //Pauli Z
 std::shared_ptr<TensDataType> RxTensor; //Rx
 std::shared_ptr<TensDataType> RyTensor; //Ry
 std::shared_ptr<TensDataType> RzTensor; //Rz
 std::shared_ptr<TensDataType> CPTensor; //CPhase
 std::shared_ptr<TensDataType> CNTensor; //CNOT
 std::shared_ptr<TensDataType> SWTensor; //SWAP

public:

//Life cycle:

 GateBodyFactory():
  HTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  XTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  YTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  ZTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  RxTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  RyTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  RzTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  CPTensor(new TensDataType[4], [](TensDataType * ptr){delete[] ptr;}),
  CNTensor(new TensDataType[16], [](TensDataType * ptr){delete[] ptr;}),
  SWTensor(new TensDataType[16], [](TensDataType * ptr){delete[] ptr;})
 {
  {auto body = HTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=HBody[i];}
  {auto body = XTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=XBody[i];}
  {auto body = YTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=YBody[i];}
  {auto body = ZTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=ZBody[i];}
  {auto body = RxTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=RxBody[i];}
  {auto body = RyTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=RyBody[i];}
  {auto body = RzTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=RzBody[i];}
  {auto body = CPTensor.get(); for(unsigned int i = 0; i < 4; ++i) body[i]=CPBody[i];}
  {auto body = CNTensor.get(); for(unsigned int i = 0; i < 16; ++i) body[i]=CNBody[i];}
  {auto body = SWTensor.get(); for(unsigned int i = 0; i < 16; ++i) body[i]=SWBody[i];}
 }

//Returns the body of each concrete gate tensor:

 std::shared_ptr<TensDataType> getBody(const Hadamard & gate){return HTensor;}
 std::shared_ptr<TensDataType> getBody(const X & gate){return XTensor;}
 std::shared_ptr<TensDataType> getBody(const Y & gate){return YTensor;}
 std::shared_ptr<TensDataType> getBody(const Z & gate){return ZTensor;}
 std::shared_ptr<TensDataType> getBody(const Rx & gate){return RxTensor;}
 std::shared_ptr<TensDataType> getBody(const Ry & gate){return RyTensor;}
 std::shared_ptr<TensDataType> getBody(const Rz & gate){return RzTensor;}
 std::shared_ptr<TensDataType> getBody(const CPhase & gate){return CPTensor;}
 std::shared_ptr<TensDataType> getBody(const CNOT & gate){return CNTensor;}
 std::shared_ptr<TensDataType> getBody(const Swap & gate){return SWTensor;}

}; //end class GateBodyFactory

} //end namespace quantum
} //end namespace xacc


#ifdef TNQVM_HAS_EXATENSOR

#include "exatensor.hpp"

namespace xacc {
namespace quantum {

class GateFactory{

private:

//Type aliases:

 using TensDataType = std::complex<double>;
 using Tensor = exatensor::TensorDenseAdpt<TensDataType>;

//Constants:

 static const unsigned int OneBodyRank = 2;
 static constexpr const std::size_t OneBodyShape[OneBodyRank] = {2,2};

 static const unsigned int TwoBodyRank = 4;
 static constexpr const std::size_t TwoBodyShape[TwoBodyRank] = {2,2,2,2};

//Gate body factory member:

 GateBodyFactory GateBodies;

//Data members:

 Tensor HadamardTensor;
 Tensor XTensor;
 Tensor YTensor;
 Tensor ZTensor;
 Tensor RxTensor;
 Tensor RyTensor;
 Tensor RzTensor;
 Tensor CPhaseTensor;
 Tensor CNOTTensor;
 Tensor SwapTensor;

public:

//Life cycle:

 GateFactory():
  HadamardTensor(OneBodyRank,OneBodyShape),
  XTensor(OneBodyRank,OneBodyShape),
  YTensor(OneBodyRank,OneBodyShape),
  ZTensor(OneBodyRank,OneBodyShape),
  RxTensor(OneBodyRank,OneBodyShape),
  RyTensor(OneBodyRank,OneBodyShape),
  RzTensor(OneBodyRank,OneBodyShape),
  CPhaseTensor(OneBodyRank,OneBodyShape),
  CNOTTensor(TwoBodyRank,TwoBodyShape),
  SwapTensor(TwoBodyRank,TwoBodyShape)
 {
 }

//Returns the tensor body for a concrete quantum gate:

 const Tensor & getTensor(const Hadamard & gate){
  if(!(HadamardTensor.hasBody())) HadamardTensor.setBody(GateBodies.getBody(gate));
  return HadamardTensor;
 }

 const Tensor & getTensor(const X & gate){
  if(!(XTensor.hasBody())) XTensor.setBody(GateBodies.getBody(gate));
  return XTensor;
 }

 const Tensor & getTensor(const Y & gate){
  if(!(YTensor.hasBody())) YTensor.setBody(GateBodies.getBody(gate));
  return YTensor;
 }

 const Tensor & getTensor(const Z & gate){
  if(!(ZTensor.hasBody())) ZTensor.setBody(GateBodies.getBody(gate));
  return ZTensor;
 }

 const Tensor & getTensor(const Rx & gate){
  if(!(RxTensor.hasBody())) RxTensor.setBody(GateBodies.getBody(gate));
  return RxTensor;
 }

 const Tensor & getTensor(const Ry & gate){
  if(!(RyTensor.hasBody())) RyTensor.setBody(GateBodies.getBody(gate));
  return RyTensor;
 }

 const Tensor & getTensor(const Rz & gate){
  if(!(RzTensor.hasBody())) RzTensor.setBody(GateBodies.getBody(gate));
  return RzTensor;
 }

 const Tensor & getTensor(const CPhase & gate){
  if(!(CPhaseTensor.hasBody())) CPhaseTensor.setBody(GateBodies.getBody(gate));
  return CPhaseTensor;
 }

 const Tensor & getTensor(const CNOT & gate){
  if(!(CNOTTensor.hasBody())) CNOTTensor.setBody(GateBodies.getBody(gate));
  return CNOTTensor;
 }

 const Tensor & getTensor(const Swap & gate){
  if(!(SwapTensor.hasBody())) SwapTensor.setBody(GateBodies.getBody(gate));
  return SwapTensor;
 }

}; //end class GateFactory

} //end namespace quantum
} //end namespace xacc

#endif //TNQVM_HAS_EXATENSOR

#endif //QUANTUM_GATE_ACCELERATORS_TNQVM_GATEFACTORY_HPP_
