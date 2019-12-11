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
 *   Initial API and implementation - Thien Nguyen
 * 
**********************************************************************************/

// Simple linear algebra quantum gate simulator
#pragma once
#include<complex>
#include<vector>
#include <assert.h>

typedef std::vector<std::complex<double>> StateVectorType;
typedef std::vector<std::vector<std::complex<double>>> GateMatrixType;

void ApplySingleQubitGate(StateVectorType& io_psi, size_t in_index, const GateMatrixType& in_gateMatrix)
{
	assert(in_gateMatrix.size() == 2 && in_gateMatrix[0].size() == 2 &&  in_gateMatrix[1].size() == 2);
    const uint64_t N = io_psi.size();
	const uint64_t k_range = 1ULL << in_index;

    auto stateVectorCopy = io_psi;    
    for (uint64_t g = 0; g < N; g += (k_range * 2))
    {
        for (uint64_t i = g; i < g + k_range; ++i)
        {
            // See https://arxiv.org/pdf/1601.07195.pdf Figure 2 for pseudo-code
            stateVectorCopy[i] = in_gateMatrix[0][0] * io_psi[i] + in_gateMatrix[0][1] * io_psi[i + k_range];
            stateVectorCopy[i + k_range] = in_gateMatrix[1][0] * io_psi[i] + in_gateMatrix[1][1] * io_psi[i + k_range];
        }
    }

    // Assign the result back to the state vector.
    io_psi = stateVectorCopy;
}

void ApplyCNOTGate(StateVectorType& io_psi, size_t in_controlIndex, size_t in_targetIndex)
{
    // Must have at least 2 qubits
    assert(io_psi.size() >= 4);
    // Qubit index
    assert(io_psi.size() >= (1ULL << in_targetIndex));
    assert(io_psi.size() >= (1ULL << in_targetIndex));

    const uint64_t N = io_psi.size();
	const uint64_t k_range = 1ULL << in_targetIndex;
	const uint64_t ctrlmask = 1ULL << in_controlIndex;
	// Note: the effect of a CNOT gate is a remapping of state vector:
    // e.g. |1>|0> ==> |1>|1> (first qubit is the control), etc.    
    if (in_controlIndex > in_targetIndex)
	{
		for (uint64_t g = ctrlmask; g < N; g += (ctrlmask * 2))
		{
			for (uint64_t i = 0; i < ctrlmask; i += (k_range * 2))
			{
				for (uint64_t ii = 0; ii < k_range; ++ii)
                {
                    std::swap(io_psi[i + g + ii], io_psi[i + g + ii + k_range]);
                }
			}
		}
	}
	else
	{
    	for (uint64_t g = 0; g < N; g += (k_range * 2))
		{
			for (uint64_t i = ctrlmask; i < k_range; i += (ctrlmask * 2))
			{
				for (uint64_t ii = 0; ii < ctrlmask; ++ii)
                {
                    std::swap(io_psi[i + g + ii], io_psi[i + g + ii + k_range]);
                }					
			}
		}
	}
}

StateVectorType AllocateStateVector(size_t in_nbQubits)
{
    StateVectorType stateVector(1ULL << in_nbQubits);
    // Default is the |0> state
    stateVector[0] = 1.0;
    return stateVector;
}