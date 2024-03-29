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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Initial implementation - Mengsu Chen 2017.7
 *
 **********************************************************************************/
#include "ITensorMPSVisitor.hpp"
#include "AllGateVisitor.hpp"
#include "itensor/all.h"
#include <complex>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include "Eigen/Dense"
#include "utils/GateMatrixAlgebra.hpp"
#include "base/Gates.hpp"
namespace {
using namespace tnqvm;

itensor::ITensor singleQubitTensor(const itensor::Index &s,
                                   const xacc::Instruction &in_gate) {
  const auto gate_mat = [&]() {
    const auto gateEnum = GetGateType(in_gate.name());
    switch (gateEnum) {
    case CommonGates::Rx:
      return GetGateMatrix<CommonGates::Rx>(
          in_gate.getParameter(0).as<double>());
    case CommonGates::Ry:
      return GetGateMatrix<CommonGates::Ry>(
          in_gate.getParameter(0).as<double>());
    case CommonGates::Rz:
      return GetGateMatrix<CommonGates::Rz>(
          in_gate.getParameter(0).as<double>());
    case CommonGates::U:
      return GetGateMatrix<CommonGates::U>(
          in_gate.getParameter(0).as<double>(),
          in_gate.getParameter(1).as<double>(),
          in_gate.getParameter(2).as<double>());
    case CommonGates::I:
      return GetGateMatrix<CommonGates::I>();
    case CommonGates::H:
      return GetGateMatrix<CommonGates::H>();
    case CommonGates::X:
      return GetGateMatrix<CommonGates::X>();
    case CommonGates::Y:
      return GetGateMatrix<CommonGates::Y>();
    case CommonGates::Z:
      return GetGateMatrix<CommonGates::Z>();
    case CommonGates::T:
      return GetGateMatrix<CommonGates::T>();
    case CommonGates::Tdg:
      return GetGateMatrix<CommonGates::Tdg>();
    default:
      xacc::error("Invalid single qubit gates!");
      return GetGateMatrix<CommonGates::I>();
    }
  }();
  assert(gate_mat.size() == 2 && gate_mat[0].size() == 2 && gate_mat[1].size());
  auto sP = itensor::prime(s);
  auto Up = s(1);
  auto UpP = sP(1);
  auto Dn = s(2);
  auto DnP = sP(2);
  auto Op = itensor::ITensor(itensor::dag(s), sP);
  Op.set(Up, UpP, gate_mat[0][0]);
  Op.set(Up, DnP, gate_mat[0][1]);
  Op.set(Dn, UpP, gate_mat[1][0]);
  Op.set(Dn, DnP, gate_mat[1][1]);

  // std::cout << "Op:\n";
  // itensor::PrintData(Op);
  return Op;
}

} // namespace

namespace tnqvm {
/// Constructor
ITensorMPSVisitor::ITensorMPSVisitor() {}

void ITensorMPSVisitor::initialize(
    std::shared_ptr<AcceleratorBuffer> accbuffer_in, int nbShots) {
  // std::cout << "Initialize " << accbuffer_in->size() << " qubits.\n";
  itensor::SpinHalf sites(accbuffer_in->size(), {"ConserveQNs=", false});
  // Set all spins to be Up
  itensor::InitState state(sites, "Up");
  m_mps = itensor::MPS(state);
  m_measureBits.clear();
  m_buffer = accbuffer_in;

  // Parse SVD options:
  // Default:
  m_svdCutoff = 1e-6;
  m_maxDim = 1024;
  if (xacc::optionExists("itensor-svd-cutoff")) {
    std::string cutoffStr = "";
    try {
      cutoffStr = xacc::getOption("itensor-svd-cutoff");
      m_svdCutoff = std::stod(cutoffStr);
      if (verbose)
        xacc::info("ITensorMPSVisitor setting SVD cutoff to " + cutoffStr);
    } catch (std::exception &e) {
      xacc::error("ITensorMPSVisitor: invalid svd cutoff value " + cutoffStr);
    }
  }
  // Also support setting 'svd-cutoff' from the initialize HetMap.
  if (options.keyExists<double>("svd-cutoff")) {
    m_svdCutoff = options.get<double>("svd-cutoff");
    xacc::info("ITensorMPSVisitor setting SVD cut-off to " +
               std::to_string(m_svdCutoff));
  }
  if (options.keyExists<int>("svd-max-dim")) {
    m_maxDim = options.get<int>("svd-max-dim");
    xacc::info("ITensorMPSVisitor setting SVD max bond dimension to " +
               std::to_string(m_maxDim));
  }
  // Debug
  // itensor::PrintData(m_mps);
  // Option to provide a circuit to construct the conjugate state for inner
  // product calculation.
  m_conjCircuit.reset();
  if (options.pointerLikeExists<xacc::CompositeInstruction>(
          "conjugate-circuit-inner-product")) {
    m_conjCircuit =
        xacc::as_shared_ptr(options.getPointerLike<xacc::CompositeInstruction>(
            "conjugate-circuit-inner-product"));
  }

  m_logSvd = false;
  if (options.keyExists<bool>("log-svd")) {
    m_logSvd = options.get<bool>("log-svd");
  }
}

itensor::Index ITensorMPSVisitor::getSiteIndex(size_t site_id) {
  if (m_buffer->size() > 1) {
    return itensor::siteIndex(m_mps, site_id);
  }

  auto idxs = itensor::siteInds(m_mps, site_id);
  for (auto iter = idxs.begin(); iter != idxs.end(); ++iter) {
    if (iter->dim() == 2) {
      return *iter;
    }
  }
  __builtin_unreachable();
}

double ITensorMPSVisitor::compute_expectation_z(
    const std::vector<size_t> &in_measureBits) {
  if (in_measureBits.size() == 1) {
    const auto site_idx = in_measureBits[0] + 1;
    // IMPORTTANT: shift the gauge position
    m_mps.position(site_idx);
    auto ket = m_mps(site_idx);
    auto bra = itensor::dag(itensor::prime(ket, "Site"));
    auto Z_op = singleQubitTensor(getSiteIndex(site_idx), Z(in_measureBits[0]));
    return itensor::eltC(bra * Z_op * ket).real();
  } else {
    // Follow the recipe here:
    // https://www.itensor.org/docs.cgi?vers=cppv3&page=tutorials/correlations
    //'gauge' the MPS to site i
    // any 'position' between i and j, inclusive, would work here
    auto sortedMeasureBits = in_measureBits;
    std::sort(sortedMeasureBits.begin(), sortedMeasureBits.end());
    const auto site_idx = sortedMeasureBits[0] + 1;
    m_mps.position(site_idx);

    auto &psi = m_mps;
    // Create the bra/dual version of the MPS psi
    auto psidag = itensor::dag(m_mps);

    // Prime the link indices to make them distinct from
    // the original ket links
    psidag.prime("Link");
    auto i = site_idx;
    // Handle first qubit: either has measure or not
    auto C = [&]() {
      Z obsGate(site_idx - 1);
      auto op_i = singleQubitTensor(getSiteIndex(site_idx), obsGate);
      if (site_idx != 1) {
        // No measure at q[0]: need to get left link:
        auto li_1 = itensor::leftLinkIndex(psi, site_idx);
        auto Cval = itensor::prime(psi(i), li_1) * op_i;
        Cval *= itensor::prime(psidag(i), "Site");
        return Cval;
      } else {
        auto Cval = psi(i) * op_i;
        Cval *= itensor::prime(psidag(i), "Site");
        return Cval;
      }
    }();
    for (int obs_id = 1; obs_id < sortedMeasureBits.size() - 1; ++obs_id) {
      auto j = sortedMeasureBits[obs_id] + 1;
      for (int k = i + 1; k < j; ++k) {
        C *= psi(k);
        C *= psidag(k);
      }
      Z obsGateJ(sortedMeasureBits[obs_id]);
      auto op_j = singleQubitTensor(getSiteIndex(j), obsGateJ);
      C *= psi(j) * op_j;
      C *= prime(psidag(j), "Site");
      i = j;
    }
    // Last measure bit:
    i = sortedMeasureBits[sortedMeasureBits.size() - 2] + 1;
    auto j = sortedMeasureBits[sortedMeasureBits.size() - 1] + 1;
    for (int k = i + 1; k < j; ++k) {
      C *= psi(k);
      C *= psidag(k);
    }
    Z obsGateJ(j - 1);
    auto op_j = singleQubitTensor(getSiteIndex(j), obsGateJ);
    if (j < m_buffer->size()) {
      auto lj = itensor::rightLinkIndex(psi, j);
      C *= prime(psi(j), lj) * op_j;
      C *= prime(psidag(j), "Site");
    } else {
      C *= psi(j) * op_j;
      C *= prime(psidag(j), "Site");
    }
    const auto result = itensor::eltC(C);
    // std::cout << "Result = " << result << "\n";
    return result.real();
  }
}

void ITensorMPSVisitor::finalize() {
  if (m_conjCircuit) {
    itensor::SpinHalf sites(m_buffer->size(), {"ConserveQNs=", false});
    // Set all spins to be Up
    itensor::InitState state(sites, "Up");
    auto bra_mps = itensor::MPS(state);
    // Cache the ket MPS
    auto ket_mps = m_mps;

    // Set the running MPS to bra and visit the conj. circuit:
    m_mps = bra_mps;
    InstructionIterator it(m_conjCircuit);
    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled() && !nextInst->isComposite()) {
        if (nextInst->name() != "Measure") {
          nextInst->accept(this);
        } else {
          xacc::error("Illegal Measure instructions in conjugate circuit.");
        }
      }
    }

    const std::complex<double> inner_product = itensor::innerC(ket_mps, m_mps);
    // std::cout << "Inner product = " << inner_product << "\n";
    m_buffer->addExtraInfo("amplitude-real", inner_product.real());
    m_buffer->addExtraInfo("amplitude-imag", inner_product.imag());
    return;
  }

  // Assign the exp-val-z (single Composite mode)
  if (!m_measureBits.empty()) {
    m_buffer->addExtraInfo("exp-val-z", compute_expectation_z(m_measureBits));
  }
}

void ITensorMPSVisitor::applySingleQubitGate(xacc::Instruction &in_gate) {
  assert(in_gate.bits().size() == 1);
  // ITensor use 1-base indexing (weird)
  auto bit_loc = in_gate.bits()[0] + 1;
  // itensor::PrintData(itensor::siteInds(m_mps, 1));
  // IMPORTTANT: shift the gauge position
  m_mps.position(bit_loc);
  auto Op = singleQubitTensor(getSiteIndex(bit_loc), in_gate);
  auto newA = Op * m_mps(bit_loc);
  newA.noPrime();
  m_mps.set(bit_loc, newA);
}

// Single-qubit gates:
void ITensorMPSVisitor::visit(Hadamard &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(X &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(Y &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(Z &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(Rx &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(Ry &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(Rz &gate) { applySingleQubitGate(gate); }
void ITensorMPSVisitor::visit(U &gate) { applySingleQubitGate(gate); }

// two-qubit gates
void ITensorMPSVisitor::applyTwoQubitGate(itensor::ITensor &in_gateTensor,
                                          size_t in_siteId1,
                                          size_t in_siteId2) {
  assert(std::abs(in_siteId1 - in_siteId2) == 1);
  // IMPORTTANT: shift the gauge position
  m_mps.position(in_siteId1);
  auto wf = m_mps(in_siteId1) * m_mps(in_siteId2);
  wf *= in_gateTensor;
  wf.noPrime();
  // itensor::PrintData(wf);
  auto [U, S, V] = itensor::svd(wf, itensor::inds(m_mps(in_siteId1)),
                                {"Cutoff=", m_svdCutoff, "MaxDim=", m_maxDim});
  if (m_logSvd) {
    std::cout << "Singular values ";
    itensor::PrintData(S);
  }

  m_mps.set(in_siteId1, U);
  m_mps.set(in_siteId2, S * V);
}

std::tuple<itensor::IndexVal, itensor::IndexVal, itensor::IndexVal,
           itensor::IndexVal, itensor::IndexVal, itensor::IndexVal,
           itensor::IndexVal, itensor::IndexVal>
ITensorMPSVisitor::getTwoQubitOpInds(size_t in_siteId1, size_t in_siteId2) {

  auto s1 = getSiteIndex(in_siteId1);
  auto s2 = getSiteIndex(in_siteId2);
  auto sP1 = itensor::prime(s1);
  auto sP2 = itensor::prime(s2);
  auto Up1 = s1(1);
  auto UpP1 = sP1(1);
  auto Dn1 = s1(2);
  auto DnP1 = sP1(2);
  auto Up2 = s2(1);
  auto UpP2 = sP2(1);
  auto Dn2 = s2(2);
  auto DnP2 = sP2(2);

  return std::make_tuple(Up1, Dn1, Up2, Dn2, UpP1, DnP1, UpP2, DnP2);
}

itensor::ITensor ITensorMPSVisitor::createTwoQubitOpTensor(size_t in_siteId1,
                                                           size_t in_siteId2) {
  auto s1 = getSiteIndex(in_siteId1);
  auto s2 = getSiteIndex(in_siteId2);
  auto sP1 = itensor::prime(s1);
  auto sP2 = itensor::prime(s2);
  return itensor::ITensor(itensor::dag(s1), itensor::dag(s2), sP1, sP2);
}

void ITensorMPSVisitor::visit(CNOT &gate) {
  auto bit_loc1 = gate.bits()[0] + 1;
  auto bit_loc2 = gate.bits()[1] + 1;
  auto [Up1, Dn1, Up2, Dn2, UpP1, DnP1, UpP2, DnP2] =
      getTwoQubitOpInds(bit_loc1, bit_loc2);
  auto Op = createTwoQubitOpTensor(bit_loc1, bit_loc2);
  Op.set(Up1, Up2, UpP1, UpP2, 1.0);
  Op.set(Up1, Dn2, UpP1, DnP2, 1.0);
  Op.set(Dn1, Up2, DnP1, DnP2, 1.0);
  Op.set(Dn1, Dn2, DnP1, UpP2, 1.0);
  if (m_logSvd) {
    std::cout << "Gate: " << gate.toString() << "\n";
  }
  applyTwoQubitGate(Op, bit_loc1, bit_loc2);
}

void ITensorMPSVisitor::visit(Swap &gate) {
  auto bit_loc1 = gate.bits()[0] + 1;
  auto bit_loc2 = gate.bits()[1] + 1;
  auto [Up1, Dn1, Up2, Dn2, UpP1, DnP1, UpP2, DnP2] =
      getTwoQubitOpInds(bit_loc1, bit_loc2);
  auto Op = createTwoQubitOpTensor(bit_loc1, bit_loc2);
  // up-up and down-down intact
  Op.set(Up1, Up2, UpP1, UpP2, 1.0);
  Op.set(Dn1, Dn2, DnP1, DnP2, 1.0);

  // Swap: up-down => down-up and vice-versa
  Op.set(Up1, Dn2, DnP1, UpP2, 1.0);
  Op.set(Dn1, Up2, UpP1, DnP2, 1.0);
  if (m_logSvd) {
    std::cout << "Gate: " << gate.toString() << "\n";
  }
  applyTwoQubitGate(Op, bit_loc1, bit_loc2);
}

void ITensorMPSVisitor::visit(CZ &gate) {
  auto bit_loc1 = gate.bits()[0] + 1;
  auto bit_loc2 = gate.bits()[1] + 1;
  auto [Up1, Dn1, Up2, Dn2, UpP1, DnP1, UpP2, DnP2] =
      getTwoQubitOpInds(bit_loc1, bit_loc2);
  auto Op = createTwoQubitOpTensor(bit_loc1, bit_loc2);

  Op.set(Up1, Up2, UpP1, UpP2, 1.0);
  Op.set(Up1, Dn2, UpP1, DnP2, 1.0);
  Op.set(Dn1, Up2, DnP1, UpP2, 1.0);
  // -1 the last one
  Op.set(Dn1, Dn2, DnP1, DnP2, -1.0);
  if (m_logSvd) {
    std::cout << "Gate: " << gate.toString() << "\n";
  }
  applyTwoQubitGate(Op, bit_loc1, bit_loc2);
}

void ITensorMPSVisitor::visit(CPhase &cp) {
  const double theta = cp.getParameter(0).as<double>();
  auto bit_loc1 = cp.bits()[0] + 1;
  auto bit_loc2 = cp.bits()[1] + 1;
  auto [Up1, Dn1, Up2, Dn2, UpP1, DnP1, UpP2, DnP2] =
      getTwoQubitOpInds(bit_loc1, bit_loc2);
  auto Op = createTwoQubitOpTensor(bit_loc1, bit_loc2);

  Op.set(Up1, Up2, UpP1, UpP2, 1.0);
  Op.set(Up1, Dn2, UpP1, DnP2, 1.0);
  Op.set(Dn1, Up2, DnP1, UpP2, 1.0);
  // exp(itheta) the last one
  Op.set(Dn1, Dn2, DnP1, DnP2, std::exp(std::complex<double>(0.0, theta)));
  if (m_logSvd) {
    std::cout << "Gate: " << cp.toString() << "\n";
  }
  applyTwoQubitGate(Op, bit_loc1, bit_loc2);
}

// others
void ITensorMPSVisitor::visit(Measure &gate) {
  m_measureBits.emplace_back(gate.bits()[0]);
}

const double ITensorMPSVisitor::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> in_function) {
  auto cached_mps = m_mps;
  // Walk the remaining circuit and visit all gates
  InstructionIterator it(in_function);
  assert(m_measureBits.empty());
  std::vector<size_t> measureBits;
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled() && !nextInst->isComposite()) {
      if (nextInst->name() != "Measure") {
        nextInst->accept(this);
      } else {
        measureBits.emplace_back(nextInst->bits()[0]);
      }
    }
  }
  assert(!measureBits.empty());

  const double exp_val = compute_expectation_z(measureBits);
  // Restore the mps:
  m_mps = cached_mps;

  return exp_val;
}

ITensorMPSVisitor::~ITensorMPSVisitor() {}
} // namespace tnqvm
