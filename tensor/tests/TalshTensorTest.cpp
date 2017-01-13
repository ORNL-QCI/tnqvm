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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Tensors

#include <boost/test/included/unit_test.hpp>
#include "Tensor.hpp"
#include "TalshTensorProvider.hpp"

using namespace boost;

BOOST_AUTO_TEST_CASE(checkConstruction) {

	fire::Tensor<5, talsh::TalshProvider> a(1, 2, 3, 4, 5);

	double d = a(0,0,0,0,0);
	BOOST_VERIFY(a.dimension(0) == 1);
	BOOST_VERIFY(a.dimension(1) == 2);
	BOOST_VERIFY(a.dimension(2) == 3);
	BOOST_VERIFY(a.dimension(3) == 4);
	BOOST_VERIFY(a.dimension(4) == 5);

	int counter = 0;
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 4; l++) {
					for (int m = 0; m < 5; m++) {
						counter++;
						BOOST_VERIFY(a(i, j, k, l, m) == 0.0);
					}
				}
			}
		}
	}
	BOOST_VERIFY(counter == 120);

	fire::Tensor<3, talsh::TalshProvider> epsilon(3, 3, 3);
	epsilon(0, 1, 2) = 1;
	BOOST_VERIFY(epsilon(0, 1, 2) == 1);
	epsilon(1, 2, 0) = 1;
	BOOST_VERIFY(epsilon(1, 2, 0) == 1);
	epsilon(2, 0, 1) = 1;
	BOOST_VERIFY(epsilon(2, 0, 1) == 1);
	epsilon(1, 0, 2) = -1;
	BOOST_VERIFY(epsilon(1, 0, 2) == -1);
	epsilon(2, 1, 0) = -1;
	BOOST_VERIFY(epsilon(2, 1, 0) == -1);
	epsilon(0, 2, 1) = -1;
	BOOST_VERIFY(epsilon(0, 2, 1) == -1);

	fire::Tensor<4, talsh::TalshProvider> grassmannIdentity(3, 3, 3, 3);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					for (int m = 0; m < 3; m++) {
						grassmannIdentity(i, j, l, m) += epsilon(i, j, k)
								* epsilon(k, l, m);
					}
				}
			}
		}
	}

	// verify
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int l = 0; l < 3; l++) {
				for (int m = 0; m < 3; m++) {
					BOOST_VERIFY(
							grassmannIdentity(i, j, l, m)
									== (int(i == l) * int(j == m)
											- int(i == m) * int(j == l)));
				}
			}
		}
	}

	// dimensionalities
	BOOST_VERIFY(epsilon.dimension(0) == 3);
	BOOST_VERIFY(epsilon.dimension(1) == 3);
	BOOST_VERIFY(epsilon.dimension(2) == 3);
}

BOOST_AUTO_TEST_CASE(checkAddition) {
	fire::Tensor<2, talsh::TalshProvider> a(2, 2);
	fire::Tensor<2, talsh::TalshProvider> b(2, 2);

	a(0, 0) = 1;
	b(0, 0) = 1;

//	fire::Tensor<2, talsh::TalshProvider> result = a + b;
//
//	BOOST_VERIFY(result.getRank() == 2);
//	BOOST_VERIFY(result.dimension(0) == 2);
//	BOOST_VERIFY(result.dimension(1) == 2);
//	BOOST_VERIFY(result(0, 0) == 2);
//	BOOST_VERIFY(result(0, 1) == 0);
//	BOOST_VERIFY(result(1, 0) == 0);
//	BOOST_VERIFY(result(1, 1) == 0);
//
}
//
//BOOST_AUTO_TEST_CASE(checkEquality) {
//	fire::Tensor<2, talsh::TalshProvider> a(2, 2);
//	fire::Tensor<2, talsh::TalshProvider> b(2, 2);
//
//	a(0, 0) = 1;
//	b(0, 0) = 1;
//
//	BOOST_VERIFY(a == b);
//
//	a(0, 0) = 2;
//
//	BOOST_VERIFY(!(a == b));
//	BOOST_VERIFY(a != b);
//}
//
//BOOST_AUTO_TEST_CASE(checkContraction) {
//
//	using namespace fire;
//	using IndexPair = std::pair<int,int>;
//
//	Tensor<2, talsh::TalshProvider> mat1(2, 3);
//	Tensor<2, talsh::TalshProvider> mat2(2, 3);
//
//	mat1.setRandom();
//	mat2.setRandom();
//
//	Tensor<2, talsh::TalshProvider> mat4(3, 3);
//
//	std::array<IndexPair, 1> contractionIndices;
//	contractionIndices[0] = std::make_pair(0, 0);
//	mat4 = mat1.contract(mat2, contractionIndices);
//
//	BOOST_VERIFY(
//			mat4(0, 0) == (mat1(0, 0) * mat2(0, 0) + mat1(1, 0) * mat2(1, 0)));
//	BOOST_VERIFY(
//			mat4(0, 1) == (mat1(0, 0) * mat2(0, 1) + mat1(1, 0) * mat2(1, 1)));
//	BOOST_VERIFY(
//			mat4(0, 2) == (mat1(0, 0) * mat2(0, 2) + mat1(1, 0) * mat2(1, 2)));
//	BOOST_VERIFY(
//			mat4(1, 0) == (mat1(0, 1) * mat2(0, 0) + mat1(1, 1) * mat2(1, 0)));
//	BOOST_VERIFY(
//			mat4(1, 1) == (mat1(0, 1) * mat2(0, 1) + mat1(1, 1) * mat2(1, 1)));
//	BOOST_VERIFY(
//			mat4(1, 2) == (mat1(0, 1) * mat2(0, 2) + mat1(1, 1) * mat2(1, 2)));
//	BOOST_VERIFY(
//			mat4(2, 0) == (mat1(0, 2) * mat2(0, 0) + mat1(1, 2) * mat2(1, 0)));
//	BOOST_VERIFY(
//			mat4(2, 1) == (mat1(0, 2) * mat2(0, 1) + mat1(1, 2) * mat2(1, 1)));
//	BOOST_VERIFY(
//			mat4(2, 2) == (mat1(0, 2) * mat2(0, 2) + mat1(1, 2) * mat2(1, 2)));
//
//}
//
//BOOST_AUTO_TEST_CASE(checkTensorProduct) {
//	using namespace fire;
//	using IndexPair = std::pair<int,int>;
//
//	Tensor<2, talsh::TalshProvider> mat1(2, 3);
//	Tensor<2, talsh::TalshProvider> mat2(4, 1);
//	mat1.setRandom();
//	mat2.setRandom();
//
//	// Note user must know tensor product produces rank n*m
//	Tensor<4, talsh::TalshProvider> result = mat1 * mat2;
//
//	BOOST_VERIFY(result.dimension(0) == 2);
//	BOOST_VERIFY(result.dimension(1) == 3);
//	BOOST_VERIFY(result.dimension(2) == 4);
//	BOOST_VERIFY(result.dimension(3) == 1);
//	for (int i = 0; i < result.dimension(0); ++i) {
//		for (int j = 0; j < result.dimension(1); ++j) {
//			for (int k = 0; k < result.dimension(2); ++k) {
//				for (int l = 0; l < result.dimension(3); ++l) {
//					BOOST_VERIFY(result(i, j, k, l) == mat1(i, j) * mat2(k, l));
//				}
//			}
//		}
//	}
//}
//
//BOOST_AUTO_TEST_CASE(checkSetValuesInitializerList) {
//	using namespace fire;
//
//	Tensor<2, talsh::TalshProvider> t(2, 3);
//
//	t.setValues( { { 0, 1, 2 }, { 3, 4, 5 } });
//
//	BOOST_VERIFY(t(0, 0) == 0);
//	BOOST_VERIFY(t(0, 1) == 1);
//	BOOST_VERIFY(t(0, 2) == 2);
//	BOOST_VERIFY(t(1, 0) == 3);
//	BOOST_VERIFY(t(1, 1) == 4);
//	BOOST_VERIFY(t(1, 2) == 5);
//
////	std::cout
////			<< "Checking setValues with initializer_list.\nTensor<2>(2,3) = \n";
////	t.print(std::cout);
////	std::cout << std::endl;
//}
//
//BOOST_AUTO_TEST_CASE(checkScalarMultiply) {
//	using namespace fire;
//
//	Tensor<2, talsh::TalshProvider> s(2, 3);
//
//	s.setValues( { { 0, 1, 2 }, { 3, 4, 5 } });
//
//	Tensor<2, talsh::TalshProvider> t = s * 2.0;
//
//	BOOST_VERIFY(t(0, 0) == 0);
//	BOOST_VERIFY(t(0, 1) == 2);
//	BOOST_VERIFY(t(0, 2) == 4);
//	BOOST_VERIFY(t(1, 0) == 6);
//	BOOST_VERIFY(t(1, 1) == 8);
//	BOOST_VERIFY(t(1, 2) == 10);
//
////	std::cout
////			<< "Checking setValues with initializer_list.\nTensor<2>(2,3) = \n";
////	t.print(std::cout);
//
//}
//
//BOOST_AUTO_TEST_CASE(checkTensorReshapeAndShuffle) {
//	using namespace fire;
//
//	Tensor<2, talsh::TalshProvider> tensor(7, 11);
//
//	std::array<int, 3> newShape { { 7, 11, 1 } };
//
//	Tensor<3, talsh::TalshProvider> reShapedTensor = tensor.reshape(newShape);
//
//	BOOST_VERIFY(reShapedTensor.getRank() == 3);
//	BOOST_VERIFY(reShapedTensor.dimension(0) == 7);
//	BOOST_VERIFY(reShapedTensor.dimension(1) == 11);
//	BOOST_VERIFY(reShapedTensor.dimension(2) == 1);
//
//	Tensor<3, talsh::TalshProvider> input(20, 30, 50);
//	input.setRandom();
//
//	std::array<int, 3> permutation { { 1, 2, 0 } };
//	Tensor<3, talsh::TalshProvider> output = input.shuffle(permutation);
//	BOOST_VERIFY(output.dimension(0) == 30);
//	BOOST_VERIFY(output.dimension(1) == 50);
//	BOOST_VERIFY(output.dimension(2) == 20);
//
//	BOOST_VERIFY(output(3, 7, 11) == input(11, 3, 7));
//}
//
//BOOST_AUTO_TEST_CASE(checkSVD) {
//	using namespace fire;
//
//	Tensor<4, talsh::TalshProvider> tensor(2, 2, 2, 2);
//	tensor.setRandom();
//
//	std::cout << "\n-----TensorTest.cpp------\nRandom Rank 4 Tensor = \n"; tensor.print(std::cout); std::cout << "\n----------" << std::endl << std::endl;
//	std::array<int,2> leftCut {0, 1}, rightCut {2, 3};
//
//	auto result = tensor.svd(leftCut, rightCut);
//
//	std::cout << "\n----TensorTest.cpp------\n\nTENSOR U: \n"; result.first.print(std::cout); std::cout << std::endl;
//	std::cout << "TENSOR V: \n"; result.second.print(std::cout); std::cout << "\n--------" << std::endl;
//
//}
