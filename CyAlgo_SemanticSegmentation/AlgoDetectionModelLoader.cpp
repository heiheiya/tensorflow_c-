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

	void DetectionFeatureAdapter::cvMat2tfTensor(cv::Mat input, tensorflow::Tensor& outputTensor)
	{
		auto outputTensorMapped = outputTensor.tensor<float, 4>();

		input.convertTo(input, CV_32FC3);
		cv::resize(input, input, cv::Size(inputWidth, inputHeight));

		input = input - inputMean;
		input = input / inputSTD;

		int height = input.size().height;
		int width = input.size().width;
		int depth = input.channels();

		const float* data = (float*)input.data;
		for (int y = 0; y < height; ++y)
		{
			const float* dataRow = data + (y * width * depth);
			for (int x = 0; x < width; ++x)
			{
				const float* dataPixel = dataRow + (x * depth);
				for (int c = 0; c < depth; ++c)
				{
					const float* dataValue = dataPixel + c;
					outputTensorMapped(0, y, x, c) = *dataValue;
				}
			}
		}
	}

	int DetectionFeatureAdapter::tfTensor2cvMat(const tensorflow::Tensor& inputTensor, cv::Mat& output)
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

		output = cv::Mat(height, width, CV_32FC(depth));
		auto inputTensorMapped = inputTensor.tensor<float, 4>();
		float* data = (float*)output.data;
		for (int y = 0; y < height; ++y)
		{
			float* dataRow = data + (y * width * depth);
			for (int x = 0; x < width; ++x)
			{
				float* dataPixel = dataRow + (x * depth);
				for (int c = 0; c < depth; ++c)
				{
					float* dataValue = dataPixel + c;
					*dataValue = inputTensorMapped(0, y, x, c);
				}
			}
		}
		return CYAL_SUCCESS;
	}

	DetectionModelLoader::DetectionModelLoader()
	{

	}

	DetectionModelLoader::~DetectionModelLoader()
	{

	}

	int DetectionModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, const FeatureAdapterBase& inputFeature, const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs)
	{
		return CYAL_SUCCESS;
	}

}

