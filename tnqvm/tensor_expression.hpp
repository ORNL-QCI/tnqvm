/** C++ adapters for ExaTENSOR: Header

!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISION: 2017/07/06

!Copyright (C) 2014-2017 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2014-2017 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of ExaTensor.

!ExaTensor is free software: you can redistribute it and/or modify
!it under the terms of the GNU Lesser General Public License as published
!by the Free Software Foundation, either version 3 of the License, or
!(at your option) any later version.

!ExaTensor is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
!GNU Lesser General Public License for more details.

!You should have received a copy of the GNU Lesser General Public License
!along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef _TENSOR_EXPRESSION_H
#define _TENSOR_EXPRESSION_H

#include <assert.h>
#include <iostream>
#include <memory>
#include <vector>

#define _DEBUG_DIL

namespace exatensor {

/** Simple dense tensor wrapper with imported body **/
template <typename T>
class TensorDenseAdpt {
   private:
    unsigned int Rank;  // VAL: tensor rank (number of dimensions)
    std::unique_ptr<std::size_t[]> DimExtent;  // VAL: tensor dimension extents
    std::shared_ptr<T>
        Body;  // REF: pointer to the imported tensor body (tensor elements)

   public:
    // Life cycle:
    TensorDenseAdpt(unsigned int rank, std::size_t dimExtent[],
                    std::shared_ptr<T> data)
        : Rank(rank), DimExtent(new std::size_t[rank]), Body(data) {
        for (unsigned int i = 0; i < rank; ++i) DimExtent[i] = dimExtent[i];
    }

    TensorDenseAdpt(const TensorDenseAdpt& tensor)
        : Rank(tensor.Rank),
          DimExtent(new std::size_t[tensor.Rank]),
          Body(tensor.Body) {
        for (unsigned int i = 0; i < tensor.Rank; ++i)
            DimExtent[i] = tensor.DimExtent[i];
    }

    TensorDenseAdpt& operator=(const TensorDenseAdpt& tensor) {
        if (&tensor == this) return *this;
        if (tensor.Rank != Rank) {
            DimExtent.reset(new std::size_t[tensor.Rank]);
            Rank = tensor.Rank;
        }
        std::copy(&tensor.DimExtent[0], &tensor.DimExtent[0] + tensor.Rank,
                  &DimExtent[0]);
        Body = tensor.Body;
        return *this;
    }

    virtual ~TensorDenseAdpt() {}

    // Accessors:
    unsigned int getRank() const { return Rank; }

    std::size_t getDimExtent(unsigned int dimension) const {
#ifdef _DEBUG_DIL
        assert(dimension < Rank);
#endif
        return DimExtent[dimension];
    }

    const std::size_t* getDimExtents() const { return DimExtent.get(); }

    std::shared_ptr<T>& getBodyAccess() const { return Body; }

    std::size_t getVolume() const {
        std::size_t vol = 1;
        for (unsigned int i = 0; i < Rank; ++i) vol *= DimExtent[i];
        return vol;
    }

    std::size_t getSize() const { return (this->getVolume()) * sizeof(T); }

    // Print:
    void printIt() const {
        // std::cout << std::endl;
        std::cout << "TensorDenseAdpt{" << std::endl;
        std::cout << " Rank = " << Rank << std::endl;
        std::cout << " Dim extents:";
        for (unsigned int i = 0; i < Rank; ++i)
            std::cout << " " << DimExtent[i];
        std::cout << std::endl;
        std::cout << " Data pointer: " << Body.get() << std::endl;
        std::cout << "}" << std::endl;
        return;
    }
};

/** Tensor leg **/
class TensorLeg {
   private:
    unsigned int TensorId;  // tensor id: 0 is output tensor, >0 is input tensor
    unsigned int DimesnId;  // tensor dimension id: [0..rank-1]

   public:
    // Life cycle:
    TensorLeg() : TensorId(0), DimesnId(0) {}

    TensorLeg(unsigned int tensorId, unsigned int dimesnId)
        : TensorId(tensorId), DimesnId(dimesnId) {}

    // Accesors:
    unsigned int getTensorId() const { return TensorId; }
    unsigned int getDimensionId() const { return DimesnId; }

    // Print:
    void printIt() const {
        std::cout << "{" << TensorId << ":" << DimesnId << "}";
        return;
    }
};

/** Tensor connected to other tensors via tensor legs **/
template <typename T>
class TensorConn {
   private:
    TensorDenseAdpt<T> Tensor;  // tensor
    std::vector<TensorLeg>
        Leg;  // tensor legs (connections to other tensors): [1..rank]

   public:
    // Life cycle:
    TensorConn(const TensorDenseAdpt<T>& tensor,
               const std::vector<TensorLeg>& connections)
        : Tensor(tensor), Leg(connections) {
#ifdef _DEBUG_DIL
        assert(tensor.getRank() == connections.size());
#endif
    }

    virtual ~TensorConn() {}

    // Accessors:
    std::size_t getDimExtent(unsigned int dimension) const {
        return Tensor.getDimExtent(dimension);
    }

    const TensorLeg& getTensorLeg(unsigned int leg) const {
#ifdef _DEBUG_DIL
        assert(leg < Tensor.getRank());
#endif
        return Leg.at(leg);
    }

    unsigned int getNumLegs() const { return Tensor.getRank(); }

    // Print:
    void printIt() const {
        // std::cout << std::endl;
        std::cout << "TensorConn{" << std::endl;
        Tensor.printIt();
        std::cout << "Legs:";
        for (unsigned int i = 0; i < Tensor.getRank(); ++i) {
            std::cout << " ";
            Leg.at(i).printIt();
        }
        std::cout << std::endl << "}" << std::endl;
        return;
    }
};

/** Tensor network (contraction of multiple tensors) **/
template <typename T>
class TensorNetwork {
   private:
    unsigned int NumInputTensors;  // number of input tensors:
                                   // [1..NumInputTensors], tensor 0 is always
                                   // output
    std::vector<TensorConn<T>> Tensors;  // tensors: [0;1..NumInpTensors]

   public:
    // Life cycle:
    TensorNetwork() : NumInputTensors(0) {}

    TensorNetwork(unsigned int numInputTensors)
        : NumInputTensors(numInputTensors) {}

    virtual ~TensorNetwork() {}

    // Mutators:
    void appendTensor(const TensorDenseAdpt<T>& tensor,
                      const std::vector<TensorLeg>& connections) {
        auto num_tens = Tensors.size();  // current total number of tensors set
                                         // in the tensor network
// Check the consistency of the new tensor candidate:
#ifdef _DEBUG_DIL
        assert(num_tens < (1 + NumInputTensors));
        assert(tensor.getRank() == connections.size());
        unsigned int i = 0;
        for (auto it = connections.cbegin(); it != connections.cend(); ++it) {
            const TensorLeg& leg = *it;  // new tensor leg
            auto tens_id =
                leg.getTensorId();  // tensor to which the new leg is connected
            assert(tens_id <= NumInputTensors);  // that tensor id is within
                                                 // constructed bounds
            std::cout<<"tens_id = "<<tens_id<<std::endl;
            if (tens_id < num_tens) {  // that tensor has already been appended
                                       // into the tensor network
                TensorConn<T>& tensconn =
                    Tensors[tens_id];  // reference to that tensor
                auto dimsn =
                    leg.getDimensionId();  // specific dimension of that tensor
                const TensorLeg& other_leg =
                    tensconn.getTensorLeg(dimsn);  // leg on the other side
                assert(other_leg.getTensorId() == num_tens &&
                       other_leg.getDimensionId() ==
                           i);  // legs connectivity must match
                assert(tensor.getDimExtent(i) ==
                       tensconn.getDimExtent(
                           dimsn));  // dimension extents must match as well
            } else if (tens_id == num_tens) {       // self-contraction
                auto dimsn = leg.getDimensionId();  // specific dimension of the
                                                    // same tensor
                assert(dimsn != i);  // dimension of a tensor cannot be
                                     // contracted with itself
                const TensorLeg& other_leg = connections.at(
                    dimsn);  // other leg of the same tensor (loop)
                assert(other_leg.getTensorId() == num_tens &&
                       other_leg.getDimensionId() ==
                           i);  // legs connectivity must match
                assert(tensor.getDimExtent(i) ==
                       tensor.getDimExtent(
                           dimsn));  // dimension extents must match as well
            }
            ++i;
        }
#endif
        // append the new tensor into the tensor network:
        Tensors.push_back(TensorConn<T>(tensor, connections));
        return;
    }

    // Transforms:
    int contractTensors(unsigned int tensor1, unsigned int tensor2,
                        TensorNetwork<T>* tensornet) {
#ifdef _DEBUG_DIL
        assert(NumInputTensors > 0 && tensor1 > 0 && tensor2 > 0);
#endif
        if (tensor1 > tensor2) std::swap(tensor1, tensor2);
        tensornet = new TensorNetwork<T>(NumInputTensors - 1);
        return 0;
    }

    // Print:
    void printIt() const {
        std::cout << "TensorNetwork{" << std::endl;
        std::cout << "Number of input tensors = " << NumInputTensors
                  << std::endl;
        for (unsigned int i = 0; i <= NumInputTensors; ++i)
            Tensors[i].printIt();
        std::cout << "}" << std::endl;
        return;
    }
};

}  // end namespace exatensor

#endif  //_TENSOR_EXPRESSION_H
