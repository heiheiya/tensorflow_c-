#pragma once

#include "AlgoModelLoaderBase.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

namespace tf_model
{
	//Model loader for feed forward neural network
	class ClassificationFeatureAdapter :public FeatureAdapterBase
	{
	public:
		ClassificationFeatureAdapter();
		~ClassificationFeatureAdapter();

	};

	class ClassificationModelLoader :public ModelLoaderBase
	{
	public:
		ClassificationModelLoader();
		~ClassificationModelLoader();


		//////////////////////////////////////////////////////////////////////////
		// Make new prediction
		//@param std::unique_ptr<tensorflow::Session>* session, work session
		//@param FeatureAdapterBase& inputFeature, common interface of input feature
		//@param std::string outputNode, tensor name of output node
		//@param double* prediction, prediction values
		//////////////////////////////////////////////////////////////////////////
		int predict(std::unique_ptr<tensorflow::Session>* session, const FeatureAdapterBase& inputFeature,
			const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs) override;

		//////////////////////////////////////////////////////////////////////////
		//Analyzes the output of the graph to retrieve the highest scores and their positions
		//@param: const std::vector<tensorflow::Tensor>& outputs, the graph output
		//@param: int howManyLabels, how many labels want to parse
		//@param: tensorflow::Tensor* indices, indices
		//@param: tensorflow::Tensor* scores, scores
		//////////////////////////////////////////////////////////////////////////
		int getTopLabels(const std::vector<tensorflow::Tensor>& outputs, int howManyLabels,
			tensorflow::Tensor* indices, tensorflow::Tensor* scores);

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

	public:
		ClassificationFeatureAdapter inputFeat;
	};
}