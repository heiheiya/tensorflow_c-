#pragma once

#include "AlgoModelLoaderBase.h"
#include "opencv2/opencv.hpp"

namespace tf_model
{
	//Model loader for feed forward neural network
	class SegmentationFeatureAdapter :public FeatureAdapterBase
	{
	public:
		SegmentationFeatureAdapter();
		~SegmentationFeatureAdapter();
	};

	class SegmentationModelLoader :public ModelLoaderBase
	{
	public:
		SegmentationModelLoader();
		~SegmentationModelLoader();


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
		SegmentationFeatureAdapter inputFeat;
	};
}