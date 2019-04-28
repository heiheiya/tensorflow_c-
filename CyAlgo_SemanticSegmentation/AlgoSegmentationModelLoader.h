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

		//////////////////////////////////////////////////////////////////////////
		//Convert a opencv Mat to tensorflow tensor
		//@param: cv::Mat input, a input opencv Mat
		//@param: float normal = 1 / 255.0, normalize OpenCV mat
		//////////////////////////////////////////////////////////////////////////
		tensorflow::Tensor cvMat2tfTensor(cv::Mat input, float normal = 1 / 255.0) override;

		//////////////////////////////////////////////////////////////////////////
		//Convert a tensorflow tensor to opencv Mat
		//@param: tensorflow::Tensor& inputTensor, an input tensorflow tensor
		//@param: cv::Mat output, a output opencv Mat
		//////////////////////////////////////////////////////////////////////////
		int tfTensor2cvMat(tensorflow::Tensor* inputTensor, cv::Mat& output) override;

		//////////////////////////////////////////////////////////////////////////
		//Read data from image file and convert to Tensor
		//@param: std::string fileName, image file path
		//@param: const int inputHeight, input image height
		//@param: const int inputWidth, input image width
		//@param: const int channels, input image channels
		//@param: const float intputSTD, input image standard deviation
		//@param: const float inputMean, input image mean value
		//@param: std::vector<Tensor*> outTensors, image tensor
		//////////////////////////////////////////////////////////////////////////
		virtual int readTensorFromImageFile(const std::string& fileName, const int inputHeight,
			const int inputWidth, const int channels,
			const float intputSTD, const float inputMean,
			std::vector<tensorflow::Tensor>* outTensors);

		void colourSegmentation(cv::Mat& input, cv::Mat& output);
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