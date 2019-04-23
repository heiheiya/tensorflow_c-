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
		//@param: float normal = 1 / 255.0, normalize OpenCV mat
		//////////////////////////////////////////////////////////////////////////
		tensorflow::Tensor cvMat2tfTensor(cv::Mat input, float normal = 1 / 255.0);

		//////////////////////////////////////////////////////////////////////////
		//Convert a tensorflow tensor to opencv Mat
		//@param: tensorflow::Tensor& inputTensor, an input tensorflow tensor
		//@param: cv::Mat output, a output opencv Mat
		//////////////////////////////////////////////////////////////////////////
		int tfTensor2cvMat(tensorflow::Tensor& inputTensor, cv::Mat& output);

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
		//@param std::string inputNode, tensor name of input node
		//@param tensorflow::Tensor& input, input tensor
		//@param std::string outputNode, tensor name of output node
		//@param std::vector<tensorflow::Tensor>& outputs, output tensor vector
		//////////////////////////////////////////////////////////////////////////
		int predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
			const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs) override;


	public:
		DetectionFeatureAdapter inputFeat;
	};
}