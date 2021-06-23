/***********************************************************************************
 * Copyright (c) 2016, UT-Battelle
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
 *   Initial API and implementation - Alex McCaskey
 *
**********************************************************************************/
#pragma once
#include <string>
#include <array>
#include <complex>
#include <cassert>

namespace tnqvm {
    // Enum of common quantum gates.
    // The enum corresponds to a gate name string.
    // See: XACC's CommonGates.hpp
    enum class CommonGates: int {
        I = 0,
        H,
        CNOT,
        Rx,
        Ry,
        Rz,
        Swap,
        U,
        X,
        Y,
        Z,
        Measure,
        CZ,
        CPhase,
        S,
        Sdg,
        T,
        Tdg,
        CY,
        CH,
        CRZ,
        iSwap,
        fSim,
        // Count of defined common gates
        GateCount
    };

    static const std::array<std::string, static_cast<size_t>(CommonGates::GateCount)> CommonGateNames = {
        "I",
        "H",
        "CNOT",
        "Rx",
        "Ry",
        "Rz",
        "Swap",
        "U",
        "X",
        "Y",
        "Z",
        "Measure",
        "CZ",
        "CPhase",
        "S",
        "Sdg",
        "T",
        "Tdg",
        "CY",
        "CH",
        "CRZ",
        "iSwap",
        "fSim"
    };

    inline std::string GetGateName(CommonGates in_gateEnum) { return CommonGateNames[static_cast<size_t>(in_gateEnum)]; }

    inline CommonGates GetGateType(const std::string& in_gateName)
    {
        for (int i = 0; i < static_cast<int>(CommonGates::GateCount); ++i)
        {
            if (CommonGateNames[i] == in_gateName)
            {
                return static_cast<CommonGates>(i);
            }
        }
        // Invalid
        return CommonGates::GateCount;
    }

    inline bool IsControlGate(CommonGates in_gateEnum) {
        switch (in_gateEnum)
        {
            case CommonGates::CNOT:
            case CommonGates::CZ:
            case CommonGates::CPhase:
            case CommonGates::CY:
            case CommonGates::CH:
            case CommonGates::CRZ: return true;
            default:
                return false;
        }
    }

    template<typename tnqvm::CommonGates, typename... Args>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix(Args... in_gateArgs) {
        // We don't expect this to be called if template specialization is not provided for a specific gate.
        assert(false);
        return {};
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::I>() {
        return
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::H>() {
        return
        {
            { M_SQRT1_2, M_SQRT1_2 },
            { M_SQRT1_2, -M_SQRT1_2 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::X>() {
        return
        {
            { 0.0, 1.0 },
            { 1.0, 0.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Y>() {
        return
        {
            { 0.0, std::complex<double>(0, -1) },
            { std::complex<double>(0, 1), 0.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Z>() {
        return
        {
            { 1.0, 0.0 },
            { 0.0, -1.0 }
        };
    }

    // Rx(theta) gate:
    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Rx>(double in_theta) {
        return
        {
            { std::cos(0.5 * in_theta), std::complex<double>(0, -1) * std::sin(0.5 * in_theta) },
            { std::complex<double>(0, -1) * std::sin(0.5 * in_theta), std::cos(0.5 * in_theta) }
        };
    }

    // Ry(theta) gate:
    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Ry>(double in_theta) {
        return
        {
            { std::cos(0.5 * in_theta), -std::sin(0.5 * in_theta) },
            { std::sin(0.5 * in_theta), std::cos(0.5 * in_theta) }
        };
    }

    // Rz(theta) gate:
    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Rz>(double in_theta) {
        return
        {
            { std::exp(std::complex<double>(0, -0.5 * in_theta)), 0.0 },
            { 0.0, std::exp(std::complex<double>(0, 0.5 * in_theta)) }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::T>() {
        return
        {
            { 1.0, 0.0 },
            { 0.0, std::exp(std::complex<double>(0, M_PI_4)) }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Tdg>() {
        return
        {
            { 1.0, 0.0 },
            { 0.0, std::exp(std::complex<double>(0, -M_PI_4)) }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>>
    GetGateMatrix<CommonGates::U>(double in_theta, double in_phi,
                                  double in_lambda) {
      return {
          {std::cos(in_theta / 2.0),
           -std::exp(std::complex<double>(0, in_lambda)) *
               std::sin(in_theta / 2.0)},
          {std::exp(std::complex<double>(0, in_phi)) * std::sin(in_theta / 2.0),
           std::exp(std::complex<double>(0, in_phi + in_lambda)) *
               std::cos(in_theta / 2.0)}};
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::CNOT>() {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, 0.0 , 1.0 },
            { 0.0, 0.0, 1.0 , 0.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::CZ>() {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, 1.0 , 0.0 },
            { 0.0, 0.0, 0.0 , -1.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::CY>() {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, 0.0, std::complex<double>(0.0, -1.0) },
            { 0.0, 0.0, std::complex<double>(0.0, 1.0), 0.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::CH>() {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, M_SQRT1_2, M_SQRT1_2 },
            { 0.0, 0.0, M_SQRT1_2, -M_SQRT1_2 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::CRZ>(double in_theta) {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, std::exp(std::complex<double>(0.0, -0.5 * in_theta)) , 0.0 },
            { 0.0, 0.0, 0.0 , std::exp(std::complex<double>(0.0, 0.5 * in_theta)) }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::CPhase>(double in_theta) {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, 1.0 , 0.0 },
            { 0.0, 0.0, 0.0 , std::exp(std::complex<double>(0.0, in_theta)) }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Swap>() {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 0.0, 1.0 , 0.0 },
            { 0.0, 1.0, 0.0 , 0.0 },
            { 0.0, 0.0, 0.0 , 1.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::iSwap>() {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, 0.0, std::complex<double>(0.0, 1.0), 0.0 },
            { 0.0, std::complex<double>(0.0, 1.0), 0.0 , 0.0 },
            { 0.0, 0.0, 0.0 , 1.0 }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::fSim>(double in_theta, double in_phi) {
        return
        {
            { 1.0, 0.0, 0.0 , 0.0 },
            { 0.0, std::cos(in_theta), std::complex<double>(0.0, -std::sin(in_theta)), 0.0 },
            { 0.0, std::complex<double>(0.0, -std::sin(in_theta)), std::cos(in_theta), 0.0 },
            { 0.0, 0.0, 0.0, std::exp(std::complex<double>(0.0, -in_phi)) }
        };
    }

    template <>
    std::vector<std::vector<std::complex<double>>> GetGateMatrix<CommonGates::Measure>() {
        return {};
    }
}
