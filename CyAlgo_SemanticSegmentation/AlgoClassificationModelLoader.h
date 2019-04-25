#pragma once

#include "AlgoModelLoaderBase.h"

namespace tf_model
{
	//Model loader for feed forward neural network
	class ClassificationFeatureAdapter :public FeatureAdapterBase
	{
	public:
		ClassificationFeatureAdapter();
		~ClassificationFeatureAdapter();

		//////////////////////////////////////////////////////////////////////////
		//Print out the top five highest-scoring values
		//@param: const std::vector<tensorflow::Tensor>& outputs, the output of the graph run
		//@param: const std::string& labelsFileName, a file contains all labels
		//////////////////////////////////////////////////////////////////////////
		int printTopLabels(const std::vector<tensorflow::Tensor>& outputs, const std::string& labelsFileName);

		//////////////////////////////////////////////////////////////////////////
		//Test whether the top label index is the one that's expected
		//@param: const std::vector<tensorflow::Tensor>& outputs, the graph run output
		//@param: int expected, top label index
		//@param: bool* isExpected, whether expected or not
		int checkTopLabel(const std::vector<tensorflow::Tensor>& outputs, int expected, bool* isExpected);

	};

	class ClassificationModelLoader :public ModelLoaderBase
	{
	public:
		ClassificationModelLoader();
		~ClassificationModelLoader();


		//////////////////////////////////////////////////////////////////////////
		// Make new prediction
		//@param std::unique_ptr<tensorflow::Session>* session, work session
		//@param std::string inputNode, tensor name of input node
		//@param tensorflow::Tensor& input, input tensor
		//@param std::string outputNode, tensor name of output node
		//@param std::vector<tensorflow::Tensor>& outputs, output tensor vector
		//////////////////////////////////////////////////////////////////////////
		int predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
			const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs);

	public:
		ClassificationFeatureAdapter inputFeat;
	};
}
