#pragma once

#include "AlgoModelLoaderBase.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
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
		//@param FeatureAdapterBase& inputFeature, common interface of input feature
		//@param std::string outputNode, tensor name of output node
		//@param double* prediction, prediction values
		//////////////////////////////////////////////////////////////////////////
		int predict(std::unique_ptr<tensorflow::Session>* session, const FeatureAdapterBase& inputFeature,
			const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs) override;


	public:
		SegmentationFeatureAdapter inputFeat;
	};
}