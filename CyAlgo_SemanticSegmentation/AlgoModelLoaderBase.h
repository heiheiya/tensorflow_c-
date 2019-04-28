#pragma once

#include <vector>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "opencv2/opencv.hpp"

namespace tf_model
{
	class FeatureAdapterBase
	{
	public:
		FeatureAdapterBase() {};
		virtual ~FeatureAdapterBase() {};

		//////////////////////////////////////////////////////////////////////////
		//Feature adapter: convert 1-D double vector to Tensor, shape [1, ndim]
		//@param: std::string tname, tensor name
		//@param: std::vector<double>* vec, input vector
		//////////////////////////////////////////////////////////////////////////
		virtual int vec2Tensor(const std::string& tname, std::vector<double>* vec);//tensor_name, tensor_double_vector

		virtual tensorflow::Tensor cvMat2tfTensor(cv::Mat input, float normal = 1 / 255.0) = 0;

		virtual int tfTensor2cvMat(tensorflow::Tensor* inputTensor, cv::Mat& output) = 0;

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

		//////////////////////////////////////////////////////////////////////////
		//Takes a file name, and loads it as a Tensor
		//@param: tensorflow::Env* env, tensorflow environment
		//@param: const std::string& fileName, file name
		//@param: ensorflow::Tensor* output, a Tensor
		//////////////////////////////////////////////////////////////////////////
		virtual int readEntireFile(tensorflow::Env* env, const std::string& fileName, tensorflow::Tensor* output);

		//////////////////////////////////////////////////////////////////////////
		//Takes a file name, and load a list of labels from it
		//@param: const std::string& fileName, a file name
		//@param: std::vector<std::string>* result, the labels stored in a vector
		//it pads with empty strings so the length is a multiple of 16
		//@param: std::size_t* foundLabelCount, the found label count
		//////////////////////////////////////////////////////////////////////////
		virtual int readLabelsFile(const std::string& fileName, std::vector<std::string>* result, std::size_t* foundLabelCount);

		//////////////////////////////////////////////////////////////////////////
		//Get the highest scores and their positions in the tensor
		//@param: const std::vector<tensorflow::Tensor>& outputs, a vector of tensor
		//@param: int howManyLabels, how many labels want to parse
		//@param: tensorflow::Tensor* indices, indices
		//@param: tensorflow::Tensor* scores, scores
		//////////////////////////////////////////////////////////////////////////
		virtual int getTopDetections(const std::vector<tensorflow::Tensor>& outputs, int howManyLabels,
			tensorflow::Tensor* indices, tensorflow::Tensor* scores);

	public:
		std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
	 };

	class ModelLoaderBase
	{
	public:
		ModelLoaderBase() {};
		virtual ~ModelLoaderBase() {};

		//////////////////////////////////////////////////////////////////////////
		//Load graph file and new session
		//@param std::unique_ptr<tensorflow::Session>* session, add the graph to the session
		//@param std::string modelPath, absolute path to exported protobuf file *.pb
		//////////////////////////////////////////////////////////////////////////
		virtual int loadGraph(std::unique_ptr<tensorflow::Session>* session, const std::string& modelPath);

	public:
		tensorflow::GraphDef graphdef;
	};
}