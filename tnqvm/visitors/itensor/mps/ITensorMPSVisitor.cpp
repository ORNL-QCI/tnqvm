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

  std::cout << "Op:\n";
  itensor::PrintData(Op);
  return Op;
}

} // namespace

namespace tnqvm {
/// Constructor
ITensorMPSVisitor::ITensorMPSVisitor() {}

void ITensorMPSVisitor::initialize(
    std::shared_ptr<AcceleratorBuffer> accbuffer_in, int nbShots) {
  std::cout << "Initialize " << accbuffer_in->size() << " qubits.\n";
  itensor::SpinHalf sites(accbuffer_in->size(), {"ConserveQNs=", false});
  // Set all spins to be Up
  itensor::InitState state(sites, "Up");
  m_mps = itensor::MPS(state);
  m_measureBits.clear();
  m_buffer = accbuffer_in;

  // Debug
  itensor::PrintData(m_mps);
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

void ITensorMPSVisitor::finalize() {
  std::cout << "Finalize\n";
  std::cout << "Measure: ";
  for (const auto& bit: m_measureBits) {
    std::cout << bit << " ";
  }
  std::cout << "\n";
  std::cout << "Site index:\n";
  itensor::PrintData(itensor::siteInds(m_mps, 1));
  auto hamOp =
      xacc::container::contains(m_measureBits, 0)
          ? singleQubitTensor(getSiteIndex(1), Z(0))
          : singleQubitTensor(getSiteIndex(1), Identity(0));

  for (size_t i = 2; i <= m_buffer->size(); ++i) {
    if (xacc::container::contains(m_measureBits, i - 1)) {
      Z obsGate(i - 1);
      auto Op = singleQubitTensor(getSiteIndex(i), obsGate);
      hamOp *= Op;
    } else {
      Identity obsGate(i - 1);
      auto Op = singleQubitTensor(getSiteIndex(i), obsGate);
      hamOp *= Op;
    }
  }

  std::cout << "Observable:\n";
  // Debug
  itensor::PrintData(hamOp);
  auto bond_ket = m_mps(1);
  for (size_t i = 2; i <= m_buffer->size(); ++i) {
    bond_ket *= m_mps(i);
  }
  auto bond_bra = itensor::dag(itensor::prime(bond_ket, "Site"));
  const double exp_val_z = itensor::eltC(bond_bra * hamOp * bond_ket).real();
  std::cout << "Exp-val = " << exp_val_z << "\n";
  m_buffer->addExtraInfo("exp-val-z", exp_val_z);
}

void ITensorMPSVisitor::applySingleQubitGate(xacc::Instruction &in_gate) {
  assert(in_gate.bits().size() == 1);
  // ITensor use 1-base indexing (weird)
  auto bit_loc = in_gate.bits()[0] + 1;
  itensor::PrintData(itensor::siteInds(m_mps, 1));
  // IMPORTTANT: shift the gauge position 
  m_mps.position(bit_loc);
  auto Op = singleQubitTensor(getSiteIndex(bit_loc), in_gate);
  auto newA = Op * m_mps(bit_loc);
  newA.noPrime();
  m_mps.set(bit_loc, newA);
  std::cout << "Apply: " << in_gate.toString() << "\n";
  itensor::PrintData(m_mps);
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
void ITensorMPSVisitor::visit(CNOT &gate) {
  auto bit_loc1 = gate.bits()[0] + 1;
  auto bit_loc2 = gate.bits()[1] + 1;
  // IMPORTTANT: shift the gauge position 
  m_mps.position(bit_loc1);
  auto s1 = getSiteIndex(bit_loc1);
  auto s2 = getSiteIndex(bit_loc2);
  auto sP1 = itensor::prime(s1);
  auto sP2 = itensor::prime(s2);
  auto Up1 = s1(1);
  auto UpP1 = sP1(1);
  auto Dn1 = s1(2);
  auto DnP1 = sP1(2);

  auto Up2= s2(1);
  auto UpP2 = sP2(1);
  auto Dn2 = s2(2);
  auto DnP2 = sP2(2);

  auto Op = itensor::ITensor(itensor::dag(s1), itensor::dag(s2), sP1, sP2);
  Op.set(Up1, Up2, UpP1, UpP2, 1.0);
  Op.set(Up1, Dn2, UpP1, DnP2, 1.0);
  Op.set(Dn1, Up2, DnP1, DnP2, 1.0);
  Op.set(Dn1, Dn2, DnP1, UpP2, 1.0);

  auto wf = m_mps(bit_loc1) * m_mps(bit_loc2);
  wf *= Op;
  wf.noPrime();
  itensor::PrintData(wf);
  auto [U, S, V] = itensor::svd(wf, itensor::inds(m_mps(bit_loc1)), {"Cutoff=", 1E-8});
  m_mps.set(bit_loc1, U);
  m_mps.set(bit_loc2, S * V);
  std::cout << "After CNOT:\n";
  itensor::PrintData(m_mps);
}

void ITensorMPSVisitor::visit(Swap &gate) {}
void ITensorMPSVisitor::visit(CZ &gate) {}
void ITensorMPSVisitor::visit(CPhase &cp) {}

// others
void ITensorMPSVisitor::visit(Measure &gate) {
  m_measureBits.emplace_back(gate.bits()[0]);
}

// void ITensorMPSVisitor::visit(Hadamard &gate) {
//   auto iqbit_in = gate.bits()[0];
//   if (verbose) {
//     std::cout << "applying " << gate.name() << " @ " << iqbit_in <<
//     std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   // 0 -> 0+1 where 0 is at position 1 of input axis(space)
//   const double half_sqrt2 = .5 * std::sqrt(2);
//   tGate.set(ind_in(1), ind_out(1), half_sqrt2);
//   tGate.set(ind_in(1), ind_out(2), half_sqrt2);
//   // 1 -> 0-1
//   tGate.set(ind_in(2), ind_out(1), half_sqrt2);
//   tGate.set(ind_in(2), ind_out(2), -half_sqrt2);
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(CZ &gate) {
//   xacc::error("CZ not supported yet.");
// }

// void ITensorMPSVisitor::visit(CNOT &gate) {
//   auto iqbit_in0_ori = (int) gate.bits()[0];
//   auto iqbit_in1_ori = (int) gate.bits()[1];
//   int iqbit_in0, iqbit_in1;
//   if (iqbit_in0_ori < iqbit_in1_ori - 1) {
//     permute_to(iqbit_in0_ori, iqbit_in1_ori - 1);
//     iqbit_in0 = iqbit_in1_ori - 1;
//     iqbit_in1 = iqbit_in1_ori;
//   } else if (iqbit_in1_ori < iqbit_in0_ori - 1) {
//     permute_to(iqbit_in1_ori, iqbit_in0_ori - 1);
//     iqbit_in0 = iqbit_in0_ori;
//     iqbit_in1 = iqbit_in0_ori - 1;
//   } else {
//     iqbit_in0 = iqbit_in0_ori;
//     iqbit_in1 = iqbit_in1_ori;
//   }
//   if (verbose) {
//     std::cout << "applying " << gate.name() << " @ " << iqbit_in0 << " , "
//               << iqbit_in1 << std::endl;
//   }
//   auto ind_in0 = ind_for_qbit(iqbit_in0); // control
//   auto ind_in1 = ind_for_qbit(iqbit_in1);
//   auto ind_out0 = itensor::Index(gate.name(), 2);
//   auto ind_out1 = itensor::Index(gate.name(), 2);
//   Index ind_lower;
//   if (iqbit_in0 < iqbit_in1) {
//     ind_lower = ind_out0;
//   } else {
//     ind_lower = ind_out1;
//   }
//   auto tGate = itensor::ITensor(ind_in0, ind_in1, ind_out0, ind_out1);
//   tGate.set(ind_out0(1), ind_out1(1), ind_in0(1), ind_in1(1), 1.);
//   tGate.set(ind_out0(1), ind_out1(2), ind_in0(1), ind_in1(2), 1.);
//   tGate.set(ind_out0(2), ind_out1(1), ind_in0(2), ind_in1(2), 1.);
//   tGate.set(ind_out0(2), ind_out1(2), ind_in0(2), ind_in1(1), 1.);
//   int min_iqbit = std::min(iqbit_in0, iqbit_in1);
//   int max_iqbit = std::max(iqbit_in0, iqbit_in1);
//   // itensor::PrintData(tGate);
//   // itensor::PrintData(legMats[iqbit_in0]);
//   // itensor::PrintData(legMats[iqbit_in1]);
//   // itensor::PrintData(bondMats[min_iqbit]);
//   auto tobe_svd =
//       tGate * legMats[iqbit_in0] * bondMats[min_iqbit] * legMats[iqbit_in1];
//   // itensor::PrintData(tobe_svd);
//   ITensor legMat(legMats[min_iqbit].inds()[1], ind_lower), bondMat,
//   restTensor; itensor::svd(tobe_svd, legMat, bondMat, restTensor, {"Cutoff",
//   svdCutoff});
//   // itensor::PrintData(legMat);
//   // std::cout<<"svd done"<<std::endl;
//   legMats[min_iqbit] = legMat;
//   bondMats[min_iqbit] = bondMat;
//   kickback_ind(restTensor, restTensor.inds()[1]);
//   assert(restTensor.r() == 3);
//   legMats[max_iqbit] = restTensor;
//   if (iqbit_in0_ori < iqbit_in1_ori - 1) {
//     permute_to(iqbit_in1_ori - 1, iqbit_in0_ori);
//   } else if (iqbit_in1_ori < iqbit_in0_ori - 1) {
//     permute_to(iqbit_in0_ori - 1, iqbit_in1_ori);
//   }
//   printWavefunc();
//   execTime += twoQubitTime;
// }

// void ITensorMPSVisitor::visit(X &gate) {
//   auto iqbit_in = gate.bits()[0];
//   if (verbose) {
//     std::cout << "applying " << gate.name() << " @ " << iqbit_in <<
//     std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(2), 1.);
//   tGate.set(ind_out(2), ind_in(1), 1.);
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(Y &gate) {
//   auto iqbit_in = gate.bits()[0];
//   if (verbose) {
//     std::cout << "applying " << gate.name() << " @ " << iqbit_in <<
//     std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(2), std::complex<double>(0, -1.));
//   tGate.set(ind_out(2), ind_in(1), std::complex<double>(0, 1.));
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(Z &gate) {
//   auto iqbit_in = gate.bits()[0];
//   if (verbose) {
//     std::cout << "applying " << gate.name() << " @ " << iqbit_in <<
//     std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(1), 1.);
//   tGate.set(ind_out(2), ind_in(2), -1.);
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// /** The inner product is carried out in the following way
//  *   so that:
//  *   1. no need to prime legs or bonds
//  *   2. no high rank tensor (eg. a rank-nqbits tensor) pops out during the
//  * computation
//  *
//  *    /\              /\
//  *   L--L            L--L
//  *   |  |            |  |
//  *   b      ---->    b  b
//  *   |               |  |
//  *                   L--L
//  *                   |  |
//  *                   b
//  *                   |
//  *   where L is a legMat, b is a bondMat
//  */
// double ITensorMPSVisitor::wavefunc_inner() {
//   ITensor inner = itensor::conj(legMats[0] * bondMats[0]) * legMats[0];
//   for (int i = 1; i < n_qbits - 1; ++i) {
//     inner = inner * itensor::conj(legMats[i] * bondMats[i]) * bondMats[i - 1] *
//             legMats[i];
//   }
//   inner = inner * itensor::conj(legMats[n_qbits - 1]) * bondMats[n_qbits - 2] *
//           legMats[n_qbits - 1];

//   double val = 0.0;
//   try {
//     val = inner.real();
//   } catch (std::exception &e) {
//     xacc::warning(
//         "Warning, possible error in ITensorMPSVisitor.wavefunc_inner():\n" +
//         std::string(e.what()));
//     val = std::real(inner.cplx());
//   }
//   return val;
// }

// double ITensorMPSVisitor::average(int iqbit, const ITensor &op_tensor) {
//   ITensor inner;
//   if (iqbit == 0) {
//     auto bra = itensor::conj(legMats[0] * bondMats[0]) * op_tensor;
//     bra.noprime();
//     inner = bra * legMats[0];
//   } else {
//     inner = itensor::conj(legMats[0] * bondMats[0]) * legMats[0];
//   }
//   for (int i = 1; i < n_qbits - 1; ++i) {
//     if (i == iqbit) {
//       auto bra = inner * itensor::conj(legMats[i] * bondMats[i]) * op_tensor;
//       bra.noprime();
//       inner = bra * bondMats[i - 1] * legMats[i];
//     } else {
//       inner = inner * itensor::conj(legMats[i] * bondMats[i]) *
//               bondMats[i - 1] * legMats[i];
//     }
//   }
//   if (iqbit == n_qbits - 1) {
//     auto bra = inner * itensor::conj(legMats[n_qbits - 1]) * op_tensor;
//     bra.noprime();
//     inner = bra * bondMats[n_qbits - 2] * legMats[n_qbits - 1];
//   } else {
//     inner = inner * itensor::conj(legMats[n_qbits - 1]) *
//             bondMats[n_qbits - 2] * legMats[n_qbits - 1];
//   }
//   // itensor::PrintData(inner);
//   return inner.cplx().real();
// }

const double ITensorMPSVisitor::getExpectationValueZ(
    std::shared_ptr<CompositeInstruction> function) {
  return 0.0;
}
//   // std::cout << "F:\n" << function->toString("q") << "\n";
//   std::map<std::string, std::pair<int, double>> reverseGates;
//   std::set<int> bitsToMeasure;

//   // Snapshot of tensor network before
//   // change of basis and measurement
//   auto copyLegMats = legMats;
//   auto copyBondMats = bondMats;

//   // Walk the tree and execute the instructions
//   // This will be hadamards, rx, and measure
//   InstructionIterator it(function);
//   while (it.hasNext()) {
//     auto nextInst = it.next();
//     if (nextInst->isEnabled()) {
//       nextInst->accept(this);
//     }
//   }

//   auto exp = mpark::get<double>(
//       buffer->getInformation("exp-val-z")); // getExpectationValueZ();

//   snapped = false;
//   legMats_m.clear();
//   bondMats_m.clear();
//   legMats = copyLegMats;
//   bondMats = copyBondMats;
//   cbits.clear();
//   cbits.resize(buffer->size());
//   iqbits_m.clear();

//   return exp;
// }

// /// iqbits: the indecies of qits to measure
// double ITensorMPSVisitor::averZs(std::set<int> iqbits) {
//   ITensor inner;
//   if (iqbits.find(0) != iqbits.end()) {
//     auto bra = itensor::conj(legMats_m[0] * bondMats_m[0]) * tZ_measure_on(0);
//     bra.noprime();
//     inner = bra * legMats_m[0];
//   } else {
//     inner = itensor::conj(legMats_m[0] * bondMats_m[0]) * legMats_m[0];
//   }
//   for (int i = 1; i < n_qbits - 1; ++i) {
//     if (iqbits.find(i) != iqbits.end()) {
//       auto bra = inner * itensor::conj(legMats_m[i] * bondMats_m[i]) *
//                  tZ_measure_on(i);
//       bra.noprime();
//       inner = bra * bondMats_m[i - 1] * legMats_m[i];
//     } else {
//       inner = inner * itensor::conj(legMats_m[i] * bondMats_m[i]) *
//               bondMats_m[i - 1] * legMats_m[i];
//     }
//   }

//   if (iqbits.find(n_qbits - 1) != iqbits.end()) {
//     auto bra = inner * itensor::conj(legMats_m[n_qbits - 1]) *
//                tZ_measure_on(n_qbits - 1);
//     bra.noprime();
//     inner = bra * bondMats_m[n_qbits - 2] * legMats_m[n_qbits - 1];
//   } else {
//     inner = inner * itensor::conj(legMats_m[n_qbits - 1]) *
//             bondMats_m[n_qbits - 2] * legMats_m[n_qbits - 1];
//   }
//   // itensor::PrintData(inner);
//   std::complex<double> aver = inner.cplx();
//   // std::cout << "AVER: " << aver << "\n";
//   assert(aver.imag() < 1e-10);
//   return aver.real();
// }

// /// tensor of Z gate on qbit i
// itensor::ITensor ITensorMPSVisitor::tZ_measure_on(int iqbit_measured) {
//   auto ind_measured = ind_for_qbit(iqbit_measured);
//   auto ind_measured_p = ind_for_qbit(iqbit_measured);
//   ind_measured_p.prime();
//   auto tZ = itensor::ITensor(ind_measured, ind_measured_p);
//   tZ.set(ind_measured_p(1), ind_measured(1), 1.);
//   tZ.set(ind_measured_p(2), ind_measured(2), -1.);
//   return tZ;
// }

// void ITensorMPSVisitor::snap_wavefunc() {
//   if (!snapped) {
//     legMats_m = legMats;
//     bondMats_m = bondMats;
//     snapped = true;
//     // std::cout<<"wave function inner = "<<wavefunc_inner()<<std::endl;
//   }
// }

// void ITensorMPSVisitor::visit(Measure &gate) {
//   snap_wavefunc();
//   auto iqbit_measured = gate.bits()[0];
//   iqbits_m.insert(iqbit_measured);
//   auto expVal = averZs(iqbits_m);
//   buffer->addExtraInfo("exp-val-z", expVal);//setExpectationValueZ(expVal);
//   auto ind_measured = ind_for_qbit(iqbit_measured);
//   if (verbose) {
//     std::cout << "applying " << gate.name() << " @ " << iqbit_measured << ",
//     "
//               << expVal << std::endl;
//   }
//   auto ind_measured_p = ind_for_qbit(iqbit_measured);
//   ind_measured_p.prime();

//   auto tMeasure0 = itensor::ITensor(ind_measured, ind_measured_p);
//   tMeasure0.set(ind_measured_p(1), ind_measured(1), 1.);
//   double p0 = average(iqbit_measured, tMeasure0) / wavefunc_inner();
//   // accbuffer->aver_from_wavefunc *= (2*p0-1);

//   double rv = (std::rand() % 1000000) / 1000000.;
//   // std::cout<<"rv= "<<rv<<"   p0= "<<p0<<std::endl;

//   if (rv < p0) {
//     cbits[iqbit_measured] = 0;
//     legMats[iqbit_measured] =
//         tMeasure0 * legMats[iqbit_measured]; // collapse wavefunction
//     legMats[iqbit_measured].prime(ind_measured_p, -1);
//   } else {
//     cbits[iqbit_measured] = 1;
//     auto tMeasure1 = itensor::ITensor(ind_measured, ind_measured_p);
//     tMeasure1.set(ind_measured_p(2), ind_measured(2), 1.);
//     legMats[iqbit_measured] =
//         tMeasure1 * legMats[iqbit_measured]; // collapse wavefunction
//     legMats[iqbit_measured].prime(ind_measured_p, -1);
//   }
//   printWavefunc();
//   execTime += twoQubitTime;
// }

// void ITensorMPSVisitor::visit(ConditionalFunction &c) {
//   auto classicalBitIdx = c.getConditionalQubit();
//   if (verbose) {
//     std::cout << "applying " << c.name() << " @ " << classicalBitIdx
//               << std::endl;
//   }
//   if (cbits[classicalBitIdx] == 1) { // TODO: add else
//     for (auto inst : c.getInstructions()) {
//       inst->accept(this);
//     }
//   }
// }

// void ITensorMPSVisitor::visit(Rx &gate) {
//   auto iqbit_in = gate.bits()[0];
//   double theta = ipToDouble(gate.getParameter(0));
//   if (verbose) {
//     std::cout << "applying " << gate.name() << "(" << theta << ") @ "
//               << iqbit_in << std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(1), std::cos(.5 * theta));
//   tGate.set(ind_out(1), ind_in(2),
//             std::complex<double>(0, -1) * std::sin(.5 * theta));
//   tGate.set(ind_out(2), ind_in(1),
//             std::complex<double>(0, -1) * std::sin(.5 * theta));
//   tGate.set(ind_out(2), ind_in(2), std::cos(.5 * theta));
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(Ry &gate) {
//   auto iqbit_in = gate.bits()[0];
//   double theta = ipToDouble(gate.getParameter(0));
//   if (verbose) {
//     std::cout << "applying " << gate.name() << "(" << theta << ") @ "
//               << iqbit_in << std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(1), std::cos(.5 * theta));
//   tGate.set(ind_out(1), ind_in(2), -std::sin(.5 * theta));
//   tGate.set(ind_out(2), ind_in(1), std::sin(.5 * theta));
//   tGate.set(ind_out(2), ind_in(2), std::cos(.5 * theta));
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(Rz &gate) {
//   auto iqbit_in = gate.bits()[0];
//   double theta = ipToDouble(gate.getParameter(0));
//   if (verbose) {
//     std::cout << "applying " << gate.name() << "(" << theta << ") @ "
//               << iqbit_in << std::endl;
//   }
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(gate.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(1),
//             std::exp(std::complex<double>(0, -.5 * theta)));
//   tGate.set(ind_out(2), ind_in(2),
//             std::exp(std::complex<double>(0, .5 * theta)));
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(U &u) {
//   auto iqbit_in = u.bits()[0];
//   if (verbose) {
//     std::cout << "applying " << u.name() << " @ " << iqbit_in << std::endl;
//   }
//   const double theta = ipToDouble(u.getParameter(0));
//   const double phi = ipToDouble(u.getParameter(1));
//   const double lambda = ipToDouble(u.getParameter(2));
//   auto ind_in = ind_for_qbit(iqbit_in);
//   auto ind_out = itensor::Index(u.name(), 2);
//   auto tGate = itensor::ITensor(ind_in, ind_out);
//   tGate.set(ind_out(1), ind_in(1), std::cos(theta / 2.0));
//   tGate.set(ind_out(1), ind_in(2), -std::exp(std::complex<double>(0, lambda))
//   *
//                std::sin(theta / 2.0));
//   tGate.set(ind_out(2), ind_in(1), std::exp(std::complex<double>(0, phi)) *
//   std::sin(theta / 2.0)); tGate.set(ind_out(2), ind_in(2),
//   std::exp(std::complex<double>(0, phi + lambda)) *
//                std::cos(theta / 2.0));
//   legMats[iqbit_in] = tGate * legMats[iqbit_in];
//   printWavefunc();
//   execTime += singleQubitTime;
// }

// void ITensorMPSVisitor::visit(CPhase &cp) {
//   xacc::error("ITensorMPS Visitor CPhase visit unimplemented.");
// }

// void ITensorMPSVisitor::permute_to(int iqbit, int iqbit_to) {
//   if (verbose) {
//     std::cout << "permute " << iqbit << " to " << iqbit_to << std::endl;
//   }
//   int delta = iqbit < iqbit_to ? 1 : -1;
//   while (iqbit != iqbit_to) {
//     Swap gate(iqbit, iqbit + delta);
//     visit(gate);
//     iqbit = iqbit + delta;
//   }
// }

// void ITensorMPSVisitor::visit(Swap &gate) {
//   auto iqbit_in0_ori = (int)gate.bits()[0];
//   auto iqbit_in1_ori = (int)gate.bits()[1];
//   // std::cout<<"applying "<<gate.name()<<" @ "<<iqbit_in0_ori<<" ,
//   // "<<iqbit_in1_ori<<std::endl;
//   int iqbit_in0, iqbit_in1;
//   if (iqbit_in0_ori < iqbit_in1_ori - 1) {
//     permute_to(iqbit_in0_ori, iqbit_in1_ori - 1);
//     iqbit_in0 = iqbit_in1_ori - 1;
//     iqbit_in1 = iqbit_in1_ori;
//   } else if (iqbit_in1_ori < iqbit_in0_ori - 1) {
//     permute_to(iqbit_in1_ori, iqbit_in0_ori - 1);
//     iqbit_in0 = iqbit_in0_ori;
//     iqbit_in1 = iqbit_in0_ori - 1;
//   } else {
//     iqbit_in0 = iqbit_in0_ori;
//     iqbit_in1 = iqbit_in1_ori;
//   }

//   auto ind_in0 = ind_for_qbit(iqbit_in0); // control
//   auto ind_in1 = ind_for_qbit(iqbit_in1);
//   auto ind_out0 = itensor::Index(gate.name(), 2);
//   auto ind_out1 = itensor::Index(gate.name(), 2);
//   Index ind_lower;
//   if (iqbit_in0 < iqbit_in1) {
//     ind_lower = ind_out0;
//   } else {
//     ind_lower = ind_out1;
//   }
//   auto tGate = itensor::ITensor(ind_in0, ind_in1, ind_out0, ind_out1);
//   tGate.set(ind_out0(1), ind_out1(1), ind_in0(1), ind_in1(1), 1.);
//   tGate.set(ind_out0(1), ind_out1(2), ind_in0(2), ind_in1(1), 1.);
//   tGate.set(ind_out0(2), ind_out1(1), ind_in0(1), ind_in1(2), 1.);
//   tGate.set(ind_out0(2), ind_out1(2), ind_in0(2), ind_in1(2), 1.);
//   int min_iqbit = std::min(iqbit_in0, iqbit_in1);
//   int max_iqbit = std::max(iqbit_in0, iqbit_in1);
//   // itensor::PrintData(tGate);
//   // itensor::PrintData(legMats[iqbit_in0]);
//   // itensor::PrintData(legMats[iqbit_in1]);
//   // itensor::PrintData(bondMats[min_iqbit]);
//   auto tobe_svd =
//       tGate * legMats[iqbit_in0] * bondMats[min_iqbit] * legMats[iqbit_in1];
//   // itensor::PrintData(tobe_svd);
//   ITensor legMat(legMats[min_iqbit].inds()[1], ind_lower), bondMat,
//   restTensor; itensor::svd(tobe_svd, legMat, bondMat, restTensor, {"Cutoff",
//   svdCutoff}); legMats[min_iqbit] = legMat; bondMats[min_iqbit] = bondMat;
//   kickback_ind(restTensor, restTensor.inds()[1]);
//   assert(restTensor.r() == 3);
//   legMats[max_iqbit] = restTensor;
//   if (iqbit_in0_ori < iqbit_in1_ori - 1) {
//     permute_to(iqbit_in1_ori - 1, iqbit_in0_ori);
//   } else if (iqbit_in1_ori < iqbit_in0_ori - 1) {
//     permute_to(iqbit_in0_ori - 1, iqbit_in1_ori);
//   }
//   printWavefunc();
// }

// void ITensorMPSVisitor::kickback_ind(ITensor &tensor, const Index &ind) {
//   // auto ind_p = itensor::prime(ind);
//   // ITensor identity(ind, ind_p);
//   // for (int i = 1; i <= ind.m(); ++i) {
//   //   identity.set(ind(i), ind_p(i), 1.);
//   // }
//   // tensor *= identity;
//   // tensor.prime(ind_p, -1);
// }

// void ITensorMPSVisitor::visit(Circuit &f) { return; }

ITensorMPSVisitor::~ITensorMPSVisitor() {}

// /// init the wave function tensor
// void ITensorMPSVisitor::initWavefunc(int n_qbits) {
//   // Index head("head", 1);
//   // Index prev_rbond = head;
//   // for (int i = 0; i < n_qbits - 1; ++i) {
//   //   Index qbit("qbit", 2);
//   //   Index lbond("lbond", 1);
//   //   itensor::ITensor legMat(qbit, prev_rbond, lbond);
//   //   legMat.set(qbit(1), prev_rbond(1), lbond(1), 1.);
//   //   legMats.push_back(legMat);

//   //   Index rbond("rbond", 1);
//   //   itensor::ITensor bondMat(lbond, rbond);
//   //   bondMat.set(lbond(1), rbond(1), 1.);
//   //   bondMats.push_back(bondMat);
//   //   prev_rbond = rbond;
//   // }
//   // Index qbit("qbit", 2);
//   // Index tail("tail", 1);
//   // itensor::ITensor legMat(qbit, prev_rbond, tail);
//   // legMat.set(qbit(1), prev_rbond(1), tail(1), 1.);
//   // legMats.push_back(legMat);

//   // for (int i = 0; i < n_qbits - 1; ++i) {
//   //   // itensor::PrintData(legMats[i]);
//   //   // itensor::PrintData(bondMats[i]);
//   // }
//   // // itensor::PrintData(legMats[n_qbits-1]);
// }

// void ITensorMPSVisitor::initWavefunc_bysvd(int n_qbits) {
//   // std::vector<ITensor> tInitQbits;
//   // for (int i = 0; i < n_qbits; ++i) {
//   //   Index ind_qbit("qbit", 2);
//   //   ITensor tInitQbit(ind_qbit);
//   //   tInitQbit.set(ind_qbit(1), 1.);
//   //   tInitQbits.push_back(tInitQbit);
//   //   iqbit2iind.push_back(i);
//   // }
//   // Index ind_head("head", 1);
//   // ITensor head(ind_head);
//   // head.set(ind_head(1), 1.);
//   // tInitQbits.push_back(head);
//   // wavefunc = tInitQbits[0];
//   // for (int i = 1; i < n_qbits + 1; ++i) {
//   //   wavefunc = wavefunc / tInitQbits[i];
//   // }
//   // reduce_to_MPS();
//   // for (int i = 0; i < n_qbits - 1; ++i) {
//   //   // itensor::PrintData(legMats[i]);
//   //   // itensor::PrintData(bondMats[i]);
//   // }
//   // // itensor::PrintData(legMats[n_qbits-1]);
// }

// itensor::Index ITensorMPSVisitor::ind_for_qbit(int iqbit) const {
//   // if (legMats.size() <= iqbit) {
//   //   return wavefunc.inds()[iqbit];
//   // } else {
//   //   return legMats[iqbit].inds()[0];
//   // }
// }

// void ITensorMPSVisitor::printWavefunc() const {
//   //	  std::cout<<">>>>>>>>----------wf--------->>>>>>>>>>\n";
//   //	 auto mps = legMats[0];
//   //	 for(int i=1; i<n_qbits;++i){
//   //	     mps *= bondMats[i-1];
//   //	     mps *= legMats[i];
//   //	 }
//   //
//   //	 unsigned long giind = 0;
//   //	 const int n_qbits = iqbit2iind.size();
//   //	 auto print_nz = [&giind, n_qbits, this](itensor::Cplx c){
//   //	     if(std::norm(c)>0){
//   //	         for(int iind=0; iind<n_qbits; ++iind){
//   //	             auto spin = (giind>>iind) & 1UL;
//   //	              std::cout<<spin;
//   //	         }
//   //	          std::cout<<"    "<<c<<std::endl;
//   //	     }
//   //	     ++giind;
//   //	 };
//   //	 auto normed_wf = mps / itensor::norm(mps);
//   //	 normed_wf.visit(print_nz);
//   //	   itensor::PrintData(mps);
//   //	  std::cout<<"<<<<<<<<---------------------<<<<<<<<<<\n"<<std::endl;
// }

// const std::vector<std::complex<double>> ITensorMPSVisitor::getState() {
//   // auto mps = legMats[0];
//   // for (int i = 1; i < n_qbits; ++i) {
//   //   mps *= bondMats[i - 1];
//   //   mps *= legMats[i];
//   // }
//   // std::vector<std::complex<double>> wf;
//   // auto store_wf = [&](itensor::Cplx c) {
//   //   auto real = c.real();
//   //   auto imag = c.imag();

//   //   if (std::fabs(real) < 1e-12)
//   //     real = 0.0;
//   //   if (std::fabs(imag) < 1e-12)
//   //     imag = 0.0;

//   //   wf.push_back(std::complex<double>(real, imag));
//   // };
//   // auto normed_wf = mps / itensor::norm(mps);
//   // normed_wf.visit(store_wf);

//   // auto vec = Eigen::Map<Eigen::VectorXcd>(wf.data(), wf.size());
//   // vec.reverseInPlace();
//   // return std::vector<std::complex<double>>(vec.data(), vec.data() + vec.size());
// }

// /** The process of SVD is to decompose a tensor,
//  *  for example, a rank 3 tensor T
//  *  |                    |
//  *  |                    |
//  *  T====  becomes    legMat---bondMat---restTensor===

//  *                       |                  |
//  *                       |                  |
//  *         becomes    legMat---bondMat---legMat---bondMat---restTensor---

//  */
// void ITensorMPSVisitor::reduce_to_MPS() {
//   // ITensor tobe_svd = wavefunc;
//   // ITensor bondMat, restTensor;
//   // Index last_rbond = wavefunc.inds()[n_qbits];
//   // for (int i = 0; i < n_qbits - 1; ++i) {
//   //   // std::cout<<"i= "<<i<<std::endl;
//   //   ITensor legMat(last_rbond, ind_for_qbit(i));
//   //   itensor::svd(tobe_svd, legMat, bondMat, restTensor, {"Cutoff", 1E-4});
//   //   legMats.push_back(
//   //       legMat); // the indeces of legMat in order: leg, last_rbond, lbond
//   //   bondMats.push_back(bondMat);
//   //   tobe_svd = restTensor;
//   //   last_rbond = bondMat.inds()[1];
//   // }
//   // Index ind_tail("tail", 1);
//   // ITensor tail(ind_tail);
//   // tail.set(ind_tail(1), 1.);
//   // legMats.push_back(restTensor / tail);
//   // printWavefunc();
// }

} // namespace tnqvm
