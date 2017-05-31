/**----------------------------------------------------------------------------
 Copyright (c) 2016-, UT-Battelle LLC
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 * Neither the name of fern nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 Author(s): Alex McCaskey (mccaskeyaj <at> ornl <dot> gov)
 -----------------------------------------------------------------------------*/
#ifndef TENSOR_TALSHTENSORPROVIDER_HPP_
#define TENSOR_TALSHTENSORPROVIDER_HPP_

#if defined __GNUC__ && __GNUC__>=6
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif

#include "TensorProvider.hpp"
#include "TensorUtils.hpp"
#include "talsh.h"
#include <array>
#include <hyper_array.hpp>

template<typename T>
void null_deleter(T *) {
}

using namespace fire;

namespace talsh {

/**
 * The TalsTensorProvider is a TensorProvider that provides
 * tensor data and operations using the Talsh C tensor module.
 */
template<const int Rank, typename Scalar = double>
class TalshTensorProvider: public fire::TensorProvider<
		TalshTensorProvider<Rank, Scalar>> {

private:

	using MultiDimArray = hyper_array::array<Scalar, Rank>;

	/**
	 * Keep track of the number of available OMP threads
	 */
	int nThreads = 0;

	talsh_tens_t * tensor;

	std::shared_ptr<MultiDimArray> dataView;

	int dataType = R8;

	void getTalshTensorByReference(TensorReference<Scalar>& reference, talsh_tens_t * ten) {
		// Get the data and shape
		auto shape = reference.second;
		auto tensorData = reference.first.data();

		size_t host_buf_size;
		void * data;
		int host_arg_max, err;
		constexpr int nDims = shape.dimensions().size();
		const int * dims = shape.dimensions().data();
		static_assert(Rank == nDims, "Must provide correct number of dimensions.");

		// By default we keep the type as R8, check
		// if the user wants floats though.
		if (std::is_same<Scalar, float>::value) {
			dataType = R4;
		}
		// Initialize Talsh
		err = talshInit(&host_buf_size, &host_arg_max, 0, 0, 0, 0, 0, 0);
		err = talshTensorCreate(&ten);
		err = talshTensorConstruct(ten, dataType, Rank, dims, 0);

		std::cout << "Talsh Tensor By Reference Rank: " << talshTensorRank(tensor) << "\n";
		std::cout << "Talsh Tensor By Reference Volume: " << talshTensorVolume(tensor)
				<< "\n";

		std::array<std::size_t, Rank> dimsList;
		for (int i = 0; i < Rank; i++) {
			dimsList[i] = shape.dimensions()[i];
		}
		err = talshTensorGetBodyAccess(ten, &data, dataType, 0);
		double * ddata = static_cast<double*>(data);
		ddata = tensorData;

		return;
	}

public:

	/**
	 * Static reference to the rank of the tensor
	 * wrapped by this provider
	 */
	static const int rank = Rank;

	/**
	 * The Constructor
	 */
	TalshTensorProvider() :
			tensor(0) {
		int n = 0;
#pragma omp parallel reduction(+:n)
		n += 1;
		nThreads = n;
	}

	/*
	 * Initialize the Talsh Tensor with all zeros.
	 * @param firstDim
	 * @param otherDims
	 */
	template<typename ... Dimensions>
	void initializeTensorBackend(int firstDim, Dimensions ... otherDims) {
		size_t host_buf_size;
		void * data;
		int host_arg_max, err;
		constexpr int nDims = sizeof...(otherDims) + 1;
		const int dims[nDims] = { firstDim, otherDims... };
		static_assert(Rank == nDims, "Must provide correct number of dimensions.");

		// By default we keep the type as R8, check
		// if the user wants floats though.
		if (std::is_same<Scalar, float>::value) {
			dataType = R4;
		}
		// Initialize Talsh
		err = talshInit(&host_buf_size, &host_arg_max, 0, 0, 0, 0, 0, 0);
		err = talshTensorCreate(&tensor);
		err = talshTensorConstruct(tensor, dataType, Rank, dims, 0);

		std::cout << "Talsh Tensor Rank: " << talshTensorRank(tensor) << "\n";
		std::cout << "Talsh Tensor Volume: " << talshTensorVolume(tensor)
				<< "\n";

		std::array<std::size_t, Rank> dimsList { firstDim, otherDims... };
		err = talshTensorGetBodyAccess(tensor, &data, dataType, 0);
		double * ddata = static_cast<double*>(data);

		dataView = std::shared_ptr < MultiDimArray
				> (new MultiDimArray(dimsList, ddata), &null_deleter<
						MultiDimArray> );
	}

	/**
	 * Initialize the Talsh Tensor from an existing TensorReference
	 * @param reference
	 */
	void initializeTensorBackendWithReference(TensorReference<Scalar>& reference) {

		getTalshTensorByReference(reference, tensor);

		// Get the data and shape
		auto shape = reference.second;
		auto tensorData = reference.first.data();
		void * data;

		std::array<std::size_t, Rank> dimsList;
		for (int i = 0; i < Rank; i++) {
			dimsList[i] = shape.dimensions()[i];
		}
		int err = talshTensorGetBodyAccess(tensor, &data, dataType, 0);
		double * ddata = static_cast<double*>(data);
		ddata = tensorData;

		dataView = std::shared_ptr < MultiDimArray
				> (new MultiDimArray(dimsList, ddata), &null_deleter<
						MultiDimArray> );
	}

	/**
	 * Return the coefficient at the given tensor indices.
	 *
	 * @param indices The indices for the desired value
	 * @return val The value at the indices.
	 */
	template<typename ... Indices>
	double& tensorCoefficient(Indices ... indices) {
		return dataView->operator()(indices...);
	}

	/**
	 * Return true if the provided TensorReference as a tensor is
	 * equal to this tensor.
	 *
	 * @param other TensorReference view of the other Tensor
	 * @return equal A boolean indicating if these Tensors are equal
	 */
	bool checkEquality(TensorReference<Scalar>& other) {
		return false;
	}

	/**
	 * Compute the tensor contraction of this Talsh Tensor with the provided
	 * Other Tensor.
	 *
	 * @param t2 The other Tensor
	 * @param indices The contraction indices.
	 * @return result The contraction result as a TensorReference
	 */
	template<typename OtherDerived, typename ContractionDims>
	TensorReference<Scalar> executeContraction(OtherDerived& t2,
			ContractionDims& cIndices) {

		// Compute new Tensor rank
		static constexpr int newRank = Rank + OtherDerived::getRank()
				- 2 * fire::array_size<ContractionDims>::value;

	}

	/**
	 * Return the 1-D array wrapped by this Talsh Tensor
	 *
	 * @return data 1-D array of data representing the tensor in this TensorProvider
	 */
	Scalar * data() {
		void * data;
		int err = talshTensorGetBodyAccess(tensor, &data, dataType, 0);
		return static_cast<Scalar*>(data);
	}

	/**
	 * Return the rank of this Talsh Tensor
	 *
	 * @return rank The rank of this Tensor
	 */
	static constexpr int getRank() {
		return Rank;
	}

	/**
	 * Return a TensorReference representing the sum of this Talsh Tensor
	 * and an Talsh Tensor represented by the other TensorReference.
	 *
	 * @param other TensorReference view of the other Tensor
	 * @return result A new TensorReference representing the sum of this and other.
	 */
	TensorReference<Scalar> add(TensorReference<Scalar>& other) {

//		talsh_tens_t * otherTensor, * resultTensor;
//		talsh_task_t * task;
//
//		int err = talshTaskCreate(&task);
//
//		getTalshTensorByReference(other, otherTensor);
//
//		err = talshTensorCopy(resultTensor, tensor, NULL, 1.0, 0.0, 0, DEV_NULL, COPY_MT, task);
//
//		err = talshTensorAdd(resultTensor, otherTensor, 1., 0.0, 0, DEV_NULL, COPY_MT, task);
//
//		void * data;
//		err = talshTensorGetBodyAccess(resultTensor, &data, dataType, 0);
//
//		TensorReference newReference = fire::make_tensor_reference(
//						static_cast<Scalar*>(data), other.second);
//
//		err = talshTaskDestruct(task);
//
//		return newReference;
	}

	/**
	 * Set the Talsh Tensor values using nested initializer_list
	 *
	 * @param vals The values as a nest std::initializer_lists
	 */
	template<typename InitList>
	void setTensorValues(InitList& vals) {
	}

	/**
	 * Output this Talsh Tensor to the provided output stream.
	 *
	 * @param outputStream The output stream to write the tensor to.
	 */
	void printTensor(std::ostream& stream) {
	}

	/**
	 * Set the Talsh Tensor values to random values.
	 */
	void fillWithRandomValues() {
	}

	/**
	 * Multiply all elements of this Talsh Tensor by the provided Scalar.
	 *
	 * @param val Scalar to multiply this tensor by.
	 * @return result A TensorReference representing the result
	 */
	TensorReference<Scalar> scalarProduct(Scalar& val) {
	}

	/**
	 * Reshape the Talsh Tensor with a new array of dimensions
	 *
	 * @param array Array of new dimensions for each rank index
	 * @return reshapedTensor A TensorReference representing new reshaped tensor.
	 */
	template<typename DimArray>
	TensorReference<Scalar> reshapeTensor(DimArray& array) {

	}

	template<typename DimArray>
	TensorReference<Scalar> shuffleTensor(DimArray& array) {
	}

	std::pair<TensorReference<Scalar>, TensorReference<Scalar>> computeSvd(TensorReference<Scalar>& ref,
			double cutoff) {

	}

	virtual ~TalshTensorProvider() {
		talshTensorDestroy(tensor);
	}
};

/**
 * This class provides a mechanism for building
 * TalshTensorProviders. It is used by the Tensor
 * class to appropriately construct a TensorProvider
 * backed by Talsh Tensors.
 */
class TalshProvider: fire::ProviderBuilder {
public:
	template<const int Rank, typename Scalar>
	TalshTensorProvider<Rank, Scalar> build() {
		TalshTensorProvider<Rank, Scalar> prov;
		return prov;
	}
};

}

#endif
