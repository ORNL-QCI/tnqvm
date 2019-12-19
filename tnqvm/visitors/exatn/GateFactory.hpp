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
#ifndef TNQVM_GATEFACTORY_HPP_
#define TNQVM_GATEFACTORY_HPP_

#include <cstdlib>
#include <complex>

#include "AllGateVisitor.hpp"

using namespace xacc;
using namespace xacc::quantum;

namespace tnqvm {

//Constants:

static const std::size_t BASE_SPACE_DIM = 2; //basic space dimension (2 for a qubit)
static const unsigned int ONE_BODY_RANK = 2; //rank of a one-body tensor
static const std::size_t ONE_BODY_VOL = BASE_SPACE_DIM * BASE_SPACE_DIM; //volume of a one-body tensor
static const unsigned int TWO_BODY_RANK = 4; //rank of a two-body tensor
static const std::size_t TWO_BODY_VOL = BASE_SPACE_DIM * BASE_SPACE_DIM * BASE_SPACE_DIM * BASE_SPACE_DIM; //volume of a two-body tensor


/** This class computes and/or caches gate tensor bodies. **/
class GateBodyFactory{

public:

//Type aliases:

 using TensDataType = std::complex<double>;

private:

//Static data declaration (gate tensor bodies):

 static constexpr const TensDataType HBody[ONE_BODY_VOL] = {
  TensDataType(1.0,0.0), TensDataType( 1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType XBody[ONE_BODY_VOL] = {
  TensDataType(0.0,0.0), TensDataType(1.0,0.0),
  TensDataType(1.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType YBody[ONE_BODY_VOL] = {
  TensDataType(0.0,0.0), TensDataType(0.0,-1.0),
  TensDataType(0.0,1.0), TensDataType(0.0, 0.0)
 };

 static constexpr const TensDataType ZBody[ONE_BODY_VOL] = {
  TensDataType(1.0,0.0), TensDataType( 0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(-1.0,0.0)
 };

 static constexpr const TensDataType RxBody[ONE_BODY_VOL] = {
  TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType RyBody[ONE_BODY_VOL] = {
  TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType RzBody[ONE_BODY_VOL] = {
  TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType CPBody[TWO_BODY_VOL] = {
  TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType CNBody[TWO_BODY_VOL] = {
  TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType CZBody[TWO_BODY_VOL] = {
  TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0)
 };

 static constexpr const TensDataType SWBody[TWO_BODY_VOL] = {
  TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(1.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0),
  TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(0.0,0.0), TensDataType(1.0,0.0)
 };

//Data members (gate tensor body pointers):

 std::shared_ptr<TensDataType> HTensor;  //Hadamard
 std::shared_ptr<TensDataType> XTensor;  //Pauli X
 std::shared_ptr<TensDataType> YTensor;  //Pauli Y
 std::shared_ptr<TensDataType> ZTensor;  //Pauli Z
 std::shared_ptr<TensDataType> RxTensor; //Rx
 std::shared_ptr<TensDataType> RyTensor; //Ry
 std::shared_ptr<TensDataType> RzTensor; //Rz
 std::shared_ptr<TensDataType> CPTensor; //CPhase
 std::shared_ptr<TensDataType> CNTensor; //CNOT
 std::shared_ptr<TensDataType> CZTensor; //CZ
 std::shared_ptr<TensDataType> SWTensor; //SWAP

public:

//Life cycle:

 GateBodyFactory():
  HTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  XTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  YTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  ZTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  RxTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  RyTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  RzTensor(new TensDataType[ONE_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  CPTensor(new TensDataType[TWO_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  CNTensor(new TensDataType[TWO_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  CZTensor(new TensDataType[TWO_BODY_VOL], [](TensDataType * ptr){delete[] ptr;}),
  SWTensor(new TensDataType[TWO_BODY_VOL], [](TensDataType * ptr){delete[] ptr;})
 {
  {auto body = HTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=HBody[i];}
  {auto body = XTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=XBody[i];}
  {auto body = YTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=YBody[i];}
  {auto body = ZTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=ZBody[i];}
  {auto body = RxTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=RxBody[i];}
  {auto body = RyTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=RyBody[i];}
  {auto body = RzTensor.get(); for(unsigned int i = 0; i < ONE_BODY_VOL; ++i) body[i]=RzBody[i];}
  {auto body = CPTensor.get(); for(unsigned int i = 0; i < TWO_BODY_VOL; ++i) body[i]=CPBody[i];}
  {auto body = CNTensor.get(); for(unsigned int i = 0; i < TWO_BODY_VOL; ++i) body[i]=CNBody[i];}
  {auto body = CZTensor.get(); for(unsigned int i = 0; i < TWO_BODY_VOL; ++i) body[i]=CZBody[i];}
  {auto body = SWTensor.get(); for(unsigned int i = 0; i < TWO_BODY_VOL; ++i) body[i]=SWBody[i];}
 }

//Returns the body of each concrete gate tensor:

 const std::shared_ptr<TensDataType> getBody(const Hadamard & gate){return HTensor;}
 const std::shared_ptr<TensDataType> getBody(const X & gate){return XTensor;}
 const std::shared_ptr<TensDataType> getBody(const Y & gate){return YTensor;}
 const std::shared_ptr<TensDataType> getBody(const Z & gate){return ZTensor;}
 const std::shared_ptr<TensDataType> getBody(const Rx & gate){return RxTensor;}
 const std::shared_ptr<TensDataType> getBody(const Ry & gate){return RyTensor;}
 const std::shared_ptr<TensDataType> getBody(const Rz & gate){return RzTensor;}
 const std::shared_ptr<TensDataType> getBody(const CPhase & gate){return CPTensor;}
 const std::shared_ptr<TensDataType> getBody(const CNOT & gate){return CNTensor;}
 const std::shared_ptr<TensDataType> getBody(const CZ & gate){return CZTensor;}
 const std::shared_ptr<TensDataType> getBody(const Swap & gate){return SWTensor;}

}; //end class GateBodyFactory

} //end namespace tnqvm


#ifdef TNQVM_HAS_EXATN

namespace tnqvm {

// class GateFactory{

// public:

// //Type aliases:

//  using TensDataType = GateBodyFactory::TensDataType;
//  using Tensor = exatensor::TensorDenseAdpt<TensDataType>;

// private:

// //Static constants declaration:

//  static constexpr const std::size_t OneBodyShape[ONE_BODY_RANK] = {BASE_SPACE_DIM,BASE_SPACE_DIM};
//  static constexpr const std::size_t TwoBodyShape[TWO_BODY_RANK] = {BASE_SPACE_DIM,BASE_SPACE_DIM,BASE_SPACE_DIM,BASE_SPACE_DIM};

// //Gate body factory member:

//  GateBodyFactory GateBodies;

// //Data members (gate tensors):

//  Tensor HadamardTensor;
//  Tensor XTensor;
//  Tensor YTensor;
//  Tensor ZTensor;
//  Tensor RxTensor;
//  Tensor RyTensor;
//  Tensor RzTensor;
//  Tensor CPhaseTensor;
//  Tensor CNOTTensor;
//  Tensor CZTensor;
//  Tensor SwapTensor;

// public:

// //Life cycle:

//  GateFactory():
//   HadamardTensor(ONE_BODY_RANK,OneBodyShape),
//   XTensor(ONE_BODY_RANK,OneBodyShape),
//   YTensor(ONE_BODY_RANK,OneBodyShape),
//   ZTensor(ONE_BODY_RANK,OneBodyShape),
//   RxTensor(ONE_BODY_RANK,OneBodyShape),
//   RyTensor(ONE_BODY_RANK,OneBodyShape),
//   RzTensor(ONE_BODY_RANK,OneBodyShape),
//   CPhaseTensor(TWO_BODY_RANK,TwoBodyShape),
//   CNOTTensor(TWO_BODY_RANK,TwoBodyShape),
//   CZTensor(TWO_BODY_RANK,TwoBodyShape),
//   SwapTensor(TWO_BODY_RANK,TwoBodyShape)
//  {
//  }

// //Returns the tensor for a concrete quantum gate:

//  const Tensor & getTensor(const Hadamard & gate){
//   if(!(HadamardTensor.hasBody())) HadamardTensor.setBody(GateBodies.getBody(gate));
//   return HadamardTensor;
//  }

//  const Tensor & getTensor(const X & gate){
//   if(!(XTensor.hasBody())) XTensor.setBody(GateBodies.getBody(gate));
//   return XTensor;
//  }

//  const Tensor & getTensor(const Y & gate){
//   if(!(YTensor.hasBody())) YTensor.setBody(GateBodies.getBody(gate));
//   return YTensor;
//  }

//  const Tensor & getTensor(const Z & gate){
//   if(!(ZTensor.hasBody())) ZTensor.setBody(GateBodies.getBody(gate));
//   return ZTensor;
//  }

//  const Tensor & getTensor(const Rx & gate){
//   if(!(RxTensor.hasBody())) RxTensor.setBody(GateBodies.getBody(gate));
//   return RxTensor;
//  }

//  const Tensor & getTensor(const Ry & gate){
//   if(!(RyTensor.hasBody())) RyTensor.setBody(GateBodies.getBody(gate));
//   return RyTensor;
//  }

//  const Tensor & getTensor(const Rz & gate){
//   if(!(RzTensor.hasBody())) RzTensor.setBody(GateBodies.getBody(gate));
//   return RzTensor;
//  }

//  const Tensor & getTensor(const CPhase & gate){
//   if(!(CPhaseTensor.hasBody())) CPhaseTensor.setBody(GateBodies.getBody(gate));
//   return CPhaseTensor;
//  }

//  const Tensor & getTensor(const CNOT & gate){
//   if(!(CNOTTensor.hasBody())) CNOTTensor.setBody(GateBodies.getBody(gate));
//   return CNOTTensor;
//  }

//  const Tensor & getTensor(const CZ & gate){
//   if(!(CZTensor.hasBody())) CZTensor.setBody(GateBodies.getBody(gate));
//   return CZTensor;
//  }

//  const Tensor & getTensor(const Swap & gate){
//   if(!(SwapTensor.hasBody())) SwapTensor.setBody(GateBodies.getBody(gate));
//   return SwapTensor;
//  }

// }; //end class GateFactory

} //end namespace tnqvm

#endif //TNQVM_HAS_EXATN

#endif //TNQVM_GATEFACTORY_HPP_