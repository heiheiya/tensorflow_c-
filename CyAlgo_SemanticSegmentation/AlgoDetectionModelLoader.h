#pragma once

#include "AlgoModelLoaderBase.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "opencv2/opencv.hpp"

namespace tf_model
{
	//Model loader for feed forward neural network
	class DetectionFeatureAdapter :public FeatureAdapterBase
	{
	public:
		DetectionFeatureAdapter();
		~DetectionFeatureAdapter();

		//////////////////////////////////////////////////////////////////////////
		//Convert a opencv Mat to tensorflow tensor
		//@param: cv::Mat input, a input opencv Mat
		//@param: tensorflow::Tensor& outputTensor, an output tensorflow tensor
		//////////////////////////////////////////////////////////////////////////
		void cvMat2tfTensor(cv::Mat input, tensorflow::Tensor& outputTensor);

		//////////////////////////////////////////////////////////////////////////
		//Convert a tensorflow tensor to opencv Mat
		//@param: tensorflow::Tensor& inputTensor, an input tensorflow tensor
		//@param: cv::Mat output, a output opencv Mat
		//////////////////////////////////////////////////////////////////////////
		int tfTensor2cvMat(const tensorflow::Tensor& inputTensor, cv::Mat& output);

	private:
		tensorflow::int32 inputWidth;
		tensorflow::int32 inputHeight;
		float inputMean;
		float inputSTD;

	};

	class DetectionModelLoader :public ModelLoaderBase
	{
	public:
		DetectionModelLoader();
		~DetectionModelLoader();


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
		DetectionFeatureAdapter inputFeat;
	};
}