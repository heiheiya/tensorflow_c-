#include <iostream>
#include <map>
#include <fstream>
#include <algorithm>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "AlgoSegmentationModelLoader.h"
#include "Common.h"

namespace tf_model
{
	SegmentationFeatureAdapter::SegmentationFeatureAdapter()
	{
	
	}

	SegmentationFeatureAdapter::~SegmentationFeatureAdapter()
	{

	}


	tensorflow::Tensor SegmentationFeatureAdapter::cvMat2tfTensor(cv::Mat input, float normal /*= 1 / 255.0*/)
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

	int SegmentationFeatureAdapter::tfTensor2cvMat(tensorflow::Tensor* inputTensor, cv::Mat& output)
	{
		tensorflow::TensorShape inputTensorShape = (*inputTensor).shape();
		if (inputTensorShape.dims() != 3)
		{
			std::cout << "ERROR: The input tensor shape is not 3 dimension...but " << inputTensorShape.dims() << "(code:" << CYAL_TF_TENSOR_DIMENSION_ERROR << ")" << std::endl;
			return CYAL_TF_TENSOR_DIMENSION_ERROR;
		}

		//std::cout << "[ " << inputTensorShape.dim_size(0) << " "
		//	<< inputTensorShape.dim_size(1) << " "
		//	<< inputTensorShape.dim_size(2) << " ]" << std::endl;
		int height = inputTensorShape.dim_size(1);
		int width = inputTensorShape.dim_size(2);
		int depth = 1;

		//std::cout << inputTensor->dtype() << std::endl;
		tensorflow::int32* tensorDataPtr = inputTensor->flat<tensorflow::int32>().data();

		output = cv::Mat(height, width, CV_32SC1, tensorDataPtr);
		output.convertTo(output, CV_8UC1);
		//cv::imwrite("temp3.jpg", output);
		//cv::cvtColor(output, output, cv::COLOR_BGR2RGB);

		return CYAL_SUCCESS;
	}

	int SegmentationFeatureAdapter::readTensorFromImageFile(const std::string& fileName, const int inputHeight, const int inputWidth, const int channels, const float intputSTD, const float inputMean, std::vector<tensorflow::Tensor>* outTensors)
	{
		auto root = tensorflow::Scope::NewRootScope();
		std::string inputName = "file_reader";
		std::string outputName = "normalized";
		std::string originalName = "identity";

		auto fileReader = tensorflow::ops::ReadFile(root.WithOpName(inputName), fileName);
		tensorflow::Output imageReader;
		if (tensorflow::str_util::EndsWith(fileName, ".png"))
		{
			imageReader = tensorflow::ops::DecodePng(root.WithOpName("png_reader"), fileReader, tensorflow::ops::DecodePng::Channels(channels));
		}
		else if (tensorflow::str_util::EndsWith(fileName, ".jpg"))
		{
			imageReader = tensorflow::ops::DecodeJpeg(root.WithOpName("jpeg_reader"), fileReader, tensorflow::ops::DecodeJpeg::Channels(channels));
		}
		else
		{
			std::cout << "ERROR: Only saving of png or jpeg files is supported..." << "(code:" << CYAL_TF_SAVE_FORMAT_NOT_SUPPORT_ERROR << ")" << std::endl;
			return CYAL_TF_SAVE_FORMAT_NOT_SUPPORT_ERROR;
		}

		auto originalImage = tensorflow::ops::Identity(root.WithOpName(originalName), imageReader);
		auto floatCaster = tensorflow::ops::Cast(root.WithOpName("float_caster"), originalImage, tensorflow::DT_FLOAT);
		//add a batch dimension of 1 to the start with ExpandDims().
		auto dimsExpander = tensorflow::ops::ExpandDims(root, floatCaster, 0);
		auto resized = tensorflow::ops::ResizeBilinear(root.WithOpName(outputName), dimsExpander, tensorflow::ops::Const(root.WithOpName("resize"), { inputHeight, inputWidth }));
		//tensorflow::ops::Transpose(root.WithOpName("transpose"), resized, { 0, 2 ,1, 3 });
		//tensorflow::ops::Div(root.WithOpName(outputName), tensorflow::ops::Sub(root, resized, { inputMean }), { intputSTD });

		tensorflow::GraphDef graph;
		tensorflow::Status status = root.ToGraphDef(&graph);
		if (!status.ok())
		{
			std::cout << "ERROR: Run graphdef failed..." << "(code:" << CYAL_TF_CONVERT_GRAPHDEF_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_CONVERT_GRAPHDEF_ERROR;
		}

		std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
		status = session->Create(graph);
		if (!status.ok())
		{
			std::cout << "ERROR: Create graph failed..." << "(code:" << CYAL_TF_CREATE_GRAPH_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_CREATE_GRAPH_ERROR;
		}
		status = session->Run({}, { outputName, originalName }, {}, outTensors);
		if (!status.ok())
		{
			std::cout << "ERROR: Run session failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}

		return CYAL_SUCCESS;
	}

	void SegmentationFeatureAdapter::colourSegmentation(cv::Mat& input, cv::Mat& output)
	{
		std::vector<cv::Vec3b> colorTab = {
			cv::Vec3b(64, 128, 64),
			cv::Vec3b(192, 0, 128), cv::Vec3b(0, 128, 192),
			cv::Vec3b(0, 128, 64), cv::Vec3b(128, 0, 0),
			cv::Vec3b(64, 0, 128), cv::Vec3b(64, 0, 192),
			cv::Vec3b(192, 128, 64), cv::Vec3b(192, 192, 128),
			cv::Vec3b(64, 64, 128), cv::Vec3b(128, 0, 192),
			cv::Vec3b(192, 0, 64), cv::Vec3b(128, 128, 64),
			cv::Vec3b(192, 0, 192), cv::Vec3b(128, 64, 64),
			cv::Vec3b(64, 192, 128), cv::Vec3b(64, 64, 0),
			cv::Vec3b(128, 64, 128), cv::Vec3b(128, 128, 192),
			cv::Vec3b(0, 0, 192), cv::Vec3b(192, 128, 128),
			cv::Vec3b(128, 128, 128), cv::Vec3b(64, 128, 192),
			cv::Vec3b(0, 0, 64), cv::Vec3b(0, 64, 64),
			cv::Vec3b(192, 64, 128), cv::Vec3b(128, 128, 0),
			cv::Vec3b(192, 128, 192), cv::Vec3b(64, 0, 64),
			cv::Vec3b(192, 192, 0), cv::Vec3b(0, 0, 0),
			cv::Vec3b(64, 192, 0)
		};

		//std::cout << colorTab.size() << std::endl;

		//output = cv::Mat(input.size(), CV_8UC3);
		int index;
		//std::cout << "=========================" << std::endl;
		//std::cout << colorTab[1].val[0] << std::endl;
		//std::cout << colorTab[1].val[1] << std::endl;
		//std::cout << colorTab[1].val[2] << std::endl;
		for (int i = 0; i < input.rows; ++i)
		{
			uchar* data = output.ptr<uchar>(i);
			for ( int j = 0; j < input.cols; ++j)
			{
				index = input.at<char>(i, j);
				//std::cout << index << ": "<< colorTab[index].val[0] << " "<< colorTab[index].val[1] << " " << colorTab[index].val[1] << std::endl;
				//output.at<cv::Vec3b>(i, j) = colorTab[index];
				//output.at<cv::Vec3b>(i, j)[1] = colorTab[index][1];
				//output.at<cv::Vec3b>(i, j)[2] = colorTab[index][2];
				//data[3 * j] = colorTab[index].val[0];
				//data[3 * j + 1] = colorTab[index].val[1];
				//data[3 * j + 2] = colorTab[index].val[2];
				output.at<cv::Vec3b>(i, j) = colorTab[index];
			}
		}
		//std::cout << output.at<cv::Vec3b>(511, 511) << std::endl;
		cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
		//cv::imwrite("color.png", output);
	}

	SegmentationModelLoader::SegmentationModelLoader()
	{

	}

	SegmentationModelLoader::~SegmentationModelLoader()
	{

	}

	int SegmentationModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
		const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs)
	{
		std::vector<tensorflow::Tensor> temps;
		tensorflow::Status status = (*session)->Run({ { inputNode, input } }, { outputNode }, {}, &temps);
		if (!status.ok())
		{
			std::cout << "ERROR: Session run failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}

		//std::cout << "Output tensor size: " << temps.size() << std::endl;
		//for (std::size_t i = 0; i < temps.size(); i++)
		//{
		//	std::cout << "=========================" << std::endl;
		//	std::cout << temps[i].DebugString() << std::endl;
		//	std::cout << "=========================" << std::endl;
		//}
		
		//tensorflow::Tensor t = temps[0];
		//auto tmap = t.tensor<float, 4>();
		//int outputHeight = t.shape().dim_size(1);
		//int outputWidth = t.shape().dim_size(2);
		//int channels = t.shape().dim_size(3);
		//int output_class_id = -1;
		//double output_prob = 0.0;
		//cv::Mat output(cv::Size(outputHeight, outputWidth), CV_8UC1);
		//for (int row = 0; row < outputHeight; ++row)
		//{
		//	for (int col = 0; col < outputWidth; ++col)
		//	{
		//		output_class_id = -1;
		//		output_prob = 0.0;
		//		for (int c = 0; c < channels; ++c)
		//		{
		//			if (tmap(0, row, col, c) >= output_prob)
		//			{
		//				output_class_id = c;
		//				output_prob = tmap(0, row, col, c);
		//			}
		//			output.at<uchar>(row, col) = output_class_id;
		//		}
		//	}
		//}
		//cv::imwrite("temp.jpg", output);

		auto root = tensorflow::Scope::NewRootScope();
		auto dim = tensorflow::ops::Const(root, -1);
		//auto originalImage = tensorflow::ops::Identity(root.WithOpName("identity"), outputs[0]);
		auto attrs = tensorflow::ops::ArgMax::OutputType(tensorflow::DataType::DT_INT32);
		auto maxOutputs = tensorflow::ops::ArgMax(root.WithOpName("argmax"), temps[0], dim, attrs);

		tensorflow::GraphDef graph;
		status = root.ToGraphDef(&graph);
		if (!status.ok())
		{
			std::cout << "ERROR: Run graphdef failed..." << "(code:" << CYAL_TF_CONVERT_GRAPHDEF_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_CONVERT_GRAPHDEF_ERROR;
		}

		std::unique_ptr<tensorflow::Session> sess(tensorflow::NewSession(tensorflow::SessionOptions()));
		status = sess->Create(graph);
		if (!status.ok())
		{
			std::cout << "ERROR: Create graph failed..." << "(code:" << CYAL_TF_CREATE_GRAPH_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_CREATE_GRAPH_ERROR;
		}

		status = sess->Run({}, {"argmax" }, {}, &outputs);
		if (!status.ok())
		{
			std::cout << "ERROR: Run session failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}
		//for (std::size_t i = 0; i < outputs.size(); i++)
		//{
		//	std::cout << "=========================" << std::endl;
		//	std::cout << outputs[i].DebugString() << std::endl;
		//	std::cout << "=========================" << std::endl;
		//}
		//std::cout << std::endl;

		//tensorflow::Tensor t = outputs[0];
		//auto tmap = t.tensor<tensorflow::int32, 3>();
		//int outputHeight = t.shape().dim_size(1);
		//int outputWidth = t.shape().dim_size(2);
		//cv::Mat output(cv::Size(outputHeight, outputWidth), CV_8UC1);
		//for (int row = 0; row < outputHeight; ++row)
		//{
		//	for (int col = 0; col < outputWidth; ++col)
		//	{
		//		output.at<uchar>(row, col) = tmap(0, row, col);
		//	}
		//}
		//cv::imwrite("temp2.jpg", output);

		return CYAL_SUCCESS;
	}
}
