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
 *   Implementation - Dmitry Lyakh 2017/11/04;
 *
 **********************************************************************************/

#include "GateFactory.hpp"

namespace tnqvm {

//Static class member storage:

constexpr const GateBodyFactory::TensDataType GateBodyFactory::HBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::XBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::YBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::ZBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::RxBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::RyBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::RzBody[ONE_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::CPBody[TWO_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::CNBody[TWO_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::CZBody[TWO_BODY_VOL];
constexpr const GateBodyFactory::TensDataType GateBodyFactory::SWBody[TWO_BODY_VOL];

#ifdef TNQVM_HAS_EXATENSOR
constexpr const std::size_t GateFactory::OneBodyShape[ONE_BODY_RANK];
constexpr const std::size_t GateFactory::TwoBodyShape[TWO_BODY_RANK];
#endif //TNQVM_HAS_EXATENSOR

} //namespace tnqvm
