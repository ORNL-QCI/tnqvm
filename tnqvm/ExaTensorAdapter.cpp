#include "ExaTensorAdapter.hpp"
namespace tnqvm {
void ExaTensorAdapter::execute(const std::string& qasmString) {

	__exatensor_MOD_exa_tensor();

}

std::shared_ptr<ComplexRankTwoTensor> ExaTensorAdapter::getRankTwoTensor(const TensorType& type) {

	// DO STUFF WITH EXATENSOR, GET TENSOR DATA as 1D Array

	// To create a fire::Tensor, do this
	//	complex<double>* data = ... get 1D tensor data array
	//	std::vector dimensions(tensorRank);
	// set dimensions for each index
	//	TensorShape shape(dimensions);
	//	auto ref = fire::make_tensor_reference(data, shape);
	//	auto tensor = std::make_shared<ComplexRankTwoTensor>(ref);

}

int ExaTensorAdapter::getMeasurementResult(const int qubitIndex) {

}
}
