#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/errors.h"

#include "Common.h"
#include "AlgoModelLoaderBase.h"


namespace tf_model
{
	int FeatureAdapterBase::vec2Tensor(const std::string& tname, std::vector<double>* vec)
	{
		int ndim = vec->size();
		if (ndim == 0)
		{
			std::cout << "ERROR: Input vec size is 0 ..." << "(code:" << CYAL_INPUT_SIZE_ZERO_ERROR << ")" << std::endl;
			return  CYAL_INPUT_SIZE_ZERO_ERROR;
		}
	
		//new Tensor shape [1, ndim]
		tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, ndim }));
		auto x_map = x.tensor<float, 2>();
		for (int j = 0; j < ndim; j++)
		{
			x_map(0, j) = (*vec)[j];
		}
		inputs.push_back(std::pair<std::string, tensorflow::Tensor>(tname, x));
	
		return CYAL_SUCCESS;
	}

	int FeatureAdapterBase::readTensorFromImageFile(const std::string& fileName, const int inputHeight, const int inputWidth, const int channels, const float intputSTD, const float inputMean, std::vector<tensorflow::Tensor>* outTensors)
	{
		auto root = tensorflow::Scope::NewRootScope();
		std::string inputName = "file_reader";
		std::string outputName = "normalized";
		std::string originalName = "identity";
	
		//tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
		//int err = readEntireFile(tensorflow::Env::Default(), fileName, &input);
		//if (err != CYAL_SUCCESS)
		//{
		//	std::cout << "ERROR: Read entire file failed..." << "(code:" << CYAL_READ_FILE_ERROR << ")" << std::endl;
		//	return CYAL_READ_FILE_ERROR;
		//}
	
		//auto fileReader = tensorflow::ops::Placeholder(root.WithOpName(inputName), tensorflow::DataType::DT_STRING);
		//inputs.push_back(std::pair<std::string, tensorflow::Tensor>("input", input));
	
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
		auto resized = tensorflow::ops::ResizeBilinear(root, dimsExpander, tensorflow::ops::Const(root.WithOpName("resize"), { inputHeight, inputWidth }));
		//tensorflow::ops::Transpose(root.WithOpName("transpose"), resized, { 0, 2 ,1, 3 });
		tensorflow::ops::Div(root.WithOpName(outputName), tensorflow::ops::Sub(root, resized, { inputMean }), { intputSTD });
	
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
		status = session->Run({ }, { outputName, originalName }, {}, outTensors);
		if (!status.ok())
		{
			std::cout << "ERROR: Run session failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}
	
		return CYAL_SUCCESS;
	}
	
	int FeatureAdapterBase::readEntireFile(tensorflow::Env* env, const std::string& fileName, tensorflow::Tensor* output)
	{
		tensorflow::uint64 fileSize = 0;
		tensorflow::Status status = env->GetFileSize(fileName, &fileSize);
		if (!status.ok())
		{
			std::cout << "ERROR: Get file size failed..." << "(code:" << CYAL_TF_GET_FILE_SIZE_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_GET_FILE_SIZE_ERROR;
		}
	
		tensorflow::string contents;
		contents.resize(fileSize);
	
		std::unique_ptr<tensorflow::RandomAccessFile> file;
		status = env->NewRandomAccessFile(fileName, &file);
		if (!status.ok())
		{
			std::cout << "ERROR: Get new random access file failed..." << "(code:" << CYAL_TF_RANDOM_ACCESS_FILE_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_RANDOM_ACCESS_FILE_ERROR;
		}
	
		tensorflow::StringPiece data;
		status = file->Read(0, fileSize, &data, &(contents)[0]);
		if (!status.ok())
		{
			std::cout << "ERROR: Read file failed..." << "(code:" << CYAL_TF_READ_FILE_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_READ_FILE_ERROR;
		}
		if (data.size() != fileSize)
		{
			std::cout << "ERROR: Read data size is not equal to file size..." << "(code:" << CYAL_READ_FILE_SIZE_ERROR << ")" << std::endl;
			return CYAL_READ_FILE_SIZE_ERROR;
		}
	
		output->scalar<tensorflow::string>()() = data.ToString();
	
		return CYAL_SUCCESS;
	}
	
	int FeatureAdapterBase::getTopDetections(const std::vector<tensorflow::Tensor>& outputs, int howManyLabels, tensorflow::Tensor* indices, tensorflow::Tensor* scores)
	{
		auto root = tensorflow::Scope::NewRootScope();

		std::string outputName = "top_k";
		tensorflow::ops::TopK(root.WithOpName(outputName), outputs[0], howManyLabels);

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
			std::cout << "ERROR: Creating graph in session failed..." << "(code:" << CYAL_TF_CREATE_GRAPH_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return  CYAL_TF_CREATE_GRAPH_ERROR;
		}

		std::vector<tensorflow::Tensor> outTensors;
		status = session->Run({}, { outputName + ":0", outputName + ":1" }, {}, &outTensors);
		if (!status.ok())
		{
			std::cout << "ERROR: Session run failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}

		*scores = outTensors[0];
		*indices = outTensors[1];
		//std::cout << indices[0].matrix<tensorflow::int32>() << std::endl;

		return CYAL_SUCCESS;
	}

	int FeatureAdapterBase::readLabelsFile(const std::string& fileName, std::vector<std::string>* result, std::size_t* foundLabelCount)
	{
		std::ifstream file(fileName);
		if (!file)
		{
			std::cout << "ERROR: File " << fileName << " not found..." << "(code:" << CYAL_READ_FILE_ERROR << ")" << std::endl;
			return CYAL_READ_FILE_ERROR;
		}
	
		result->clear();
		std::string line;
		while (std::getline(file, line))
		{
			result->push_back(line);
		}
	
		*foundLabelCount = result->size();
	
		const int padding = 16;
		while (result->size() % padding)
		{
			result->emplace_back();
		}
	
		return CYAL_SUCCESS;
	}
	
	int ModelLoaderBase::loadGraph(std::unique_ptr<tensorflow::Session>* session, const std::string& modelPath)
	{
		//read the pb file into the graphdef member
		tensorflow::Status statusLoad = ReadBinaryProto(tensorflow::Env::Default(), modelPath, &graphdef);
		if (!statusLoad.ok())
		{
			std::cout << "ERROR: Loading model failed..." << modelPath << "(code:" << CYAL_TF_READ_MODEL_ERROR << ")" << std::endl;
			std::cout << statusLoad.ToString() << std::endl;
			return  CYAL_TF_READ_MODEL_ERROR;
		}
	
		//add the graph to the session
		auto options = tensorflow::SessionOptions();
		options.config.set_allow_soft_placement(true);
		session->reset(tensorflow::NewSession(options));
		tensorflow::Status statusCreate = (*session)->Create(graphdef);
		if (!statusCreate.ok())
		{
			std::cout << "ERROR: Creating graph in session failed..." << "(code:" << CYAL_TF_CREATE_GRAPH_ERROR << ")" << std::endl;
			std::cout << statusCreate.ToString() << std::endl;
			return  CYAL_TF_CREATE_GRAPH_ERROR;
		}
	
		return CYAL_SUCCESS;
	}
}
