#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "AlgoDetectionModelLoader.h"
#include "Common.h"

namespace tf_model
{

	DetectionFeatureAdapter::DetectionFeatureAdapter()
	{

	}

	DetectionFeatureAdapter::~DetectionFeatureAdapter()
	{

	}

	tensorflow::Tensor DetectionFeatureAdapter::cvMat2tfTensor(cv::Mat input, float normal/* = 1/255.0*/)
	{
		//auto outputTensorMapped = outputTensor.tensor<float, 4>();
		int height = input.size().height;
		int width = input.size().width;
		int depth = input.channels();
		tensorflow::Tensor outputTensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, height, width, depth }));

		float* tensorDataPtr = outputTensor.flat<float>().data();
		cv::Mat tempMat(height, width, CV_32FC(depth), tensorDataPtr);
		input.convertTo(tempMat, CV_32FC(depth));

		tempMat *= normal;

		return outputTensor;

		//input.convertTo(input, CV_32FC3);
		//cv::resize(input, input, cv::Size(inputWidth, inputHeight));
		//input = 1 - input / 255.0;

		//float *p = outputTensor.flat<float>().data();
		//cv::Mat tempMat(height, width, CV_32FC(depth), p);
		//input.convertTo(tempMat, CV_32FC(depth));

		//input = input - inputMean;
		//input = input / inputSTD;



		//const float* data = (float*)input.data;
		//for (int y = 0; y < height; ++y)
		//{
		//	const float* dataRow = data + (y * width * depth);
		//	for (int x = 0; x < width; ++x)
		//	{
		//		const float* dataPixel = dataRow + (x * depth);
		//		for (int c = 0; c < depth; ++c)
		//		{
		//			const float* dataValue = dataPixel + c;
		//			outputTensorMapped(0, y, x, c) = *dataValue;
		//		}
		//	}
		//}
	}

	int DetectionFeatureAdapter::tfTensor2cvMat(tensorflow::Tensor& inputTensor, cv::Mat& output)
	{
		tensorflow::TensorShape inputTensorShape = inputTensor.shape();
		if (inputTensorShape.dims() != 4)
		{
			std::cout << "ERROR: The input tensor shape is not 4 dimension..." << "(code:" << CYAL_TF_TENSOR_DIMENSION_ERROR << ")" << std::endl;
			return CYAL_TF_TENSOR_DIMENSION_ERROR;
		}

		int height = inputTensorShape.dim_size(1);
		int width = inputTensorShape.dim_size(2);
		int depth = inputTensorShape.dim_size(3);

		float* tensorDataPtr = inputTensor.flat<float>().data();

		cv::Mat tempMat(height, width, CV_32FC(depth), tensorDataPtr);
		tempMat *= 255.0;
		tempMat.convertTo(output, CV_8UC(depth));

		//output = cv::Mat(height, width, CV_32FC(depth), tensorDataPtr);
		//auto inputTensorMapped = inputTensor.tensor<float, 4>();
		//float* data = (float*)output.data;
		//for (int y = 0; y < height; ++y)
		//{
		//	float* dataRow = data + (y * width * depth);
		//	for (int x = 0; x < width; ++x)
		//	{
		//		float* dataPixel = dataRow + (x * depth);
		//		for (int c = 0; c < depth; ++c)
		//		{
		//			float* dataValue = dataPixel + c;
		//			*dataValue = inputTensorMapped(0, y, x, c);
		//		}
		//	}
		//}
		return CYAL_SUCCESS;
	}

	DetectionModelLoader::DetectionModelLoader()
	{

	}

	DetectionModelLoader::~DetectionModelLoader()
	{

	}

	int DetectionModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
		const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs)
	{
		return CYAL_SUCCESS;
	}

}

