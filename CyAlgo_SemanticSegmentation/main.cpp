#include <iostream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "AlgoClassificationModelLoader.h"
#include "Common.h"

using namespace tensorflow;

//int main()
//{
//	std::string modelPath = "pb/frozen_graph.pb";
//	std::string imagePath = "image/0001TP_007170.png";
//
//	std::string inputTensorName = "Placeholder";
//	std::string outputTensorName = "logits/BiasAdd";
//
//	tensorflow::SessionOptions sessOpt;
//	sessOpt.config.mutable_gpu_options()->set_allow_growth(true);
//	//(&session)->reset(NewSession(sessOpt));
//	std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(sessOpt));
//	//tensorflow::Status status = session->Create(NewSession(sessOpt, &session));
//	//if (!status.ok())
//	//{
//	//	std::cout << status.ToString() << std::endl;
//	//	return CYAL_TF_CREATE_SESSION_ERROR;
//	//}
//
//	tf_model::SegmentationModelLoader model;
//	int err = model.load(&session, modelPath);
//	if (CYAL_SUCCESS != err)
//	{
//		std::cout << "ERROR: Model loading failed..." << std::endl;
//		return CYAL_LOAD_MODEL_ERROR;
//	}
//
//
//
//	//int ndim = 5;
//	//std::vector<double> input;
//	//for (int i = 0; i < ndim; i++)
//	//{
//	//	input.push_back(1.0);
//	//}
//
//	tf_model::SegmentationFeatureAdapter inputFeat;
//	const int inputHeight = 512;
//	const int inputWidth = 512;
//	const int channels = 3;
//	const float inputSTD = 128;
//	const float inputMean = 128;
//	std::vector<tensorflow::Tensor> inputs;
//	err = inputFeat.readTensorFromImageFile(imagePath, inputHeight, inputWidth, channels, inputSTD, inputMean, &inputs);
//	if (CYAL_SUCCESS != err)
//	{
//		std::cout << "ERROR: Read Tensor from image file failed..." << std::endl;
//		return CYAL_READ_TENSOR_ERROR;
//	}
//
//	std::vector<tensorflow::Tensor> outputs;
//	std::pair<std::string, tensorflow::Tensor> img(inputTensorName, inputs[0]);
//	tensorflow::Status status = session->Run({ img }, { outputTensorName }, {}, &outputs);
//	if (!status.ok())
//	{
//		std::cout << "ERROR: Running model failed..." << std::endl;
//		std::cout << status.ToString() << std::endl;
//		return CYAL_TF_SESSION_RUN_ERROR;
//	}
//	//inputFeat.assign(inputTensorName, &input);
//
//	//double prediction = 0.0;
//	//err = model.predict(session, inputFeat, outputTensorName, outputs);
//	//if (CYAL_SUCCESS != err)
//	//{
//	//	std::cout << "ERROR: Prediction failed..." << std::endl;
//	//	return CYAL_PREDICTION_ERROR;
//	//}
//
//	tensorflow::Tensor scores = outputs[0];
//	//tensorflow::Tensor indices = outputs[1];
//	tensorflow::TTypes<float>::Flat scoresFlat = scores.flat<float>();
//	//tensorflow::TTypes<int32>::Flat indicesFlat = indices.flat<int32>();
//	std::cout << scoresFlat.size() << std::endl;
//	//for (int i = 0; i < scoresFlat.size(); i++)
//	//{
//	//	std::cout  << scoresFlat(i) << std::endl;
//	//}
//	//auto tmap = t.tensor<float, 2>();
//	//int outputDim = t.shape().dim_size(1);
//	//for (int j = 0; j < outputDim; j++)
//	//{
//	//	std::cout << "Class " << j << " prob: " << tmap(0, j) << ", " << std::endl;
//	//}
//
//	//std::cout << "output prediction value: " << prediction << std::endl;
//
//	return CYAL_SUCCESS;
//}

int main()
{
	std::string image = "image/grace_hopper.jpg";
	std::string graph = "model/inception_v3_2016_08_28_frozen.pb";
	std::string labels = "model/imagenet_slim_labels.txt";
	tensorflow::int32 inputWidth = 299;
	tensorflow::int32 intputHeight = 299;
	tensorflow::int32 channels = 3;
	float inputMean = 0;
	float inputSTD = 255;
	std::string inputLayer = "input";
	std::string outputLayer = "InceptionV3/Predictions/Reshape_1";
	bool selfTest = false;
	std::string rootDir = "";

	std::unique_ptr<tensorflow::Session> session;
	std::string graphPath = tensorflow::io::JoinPath(rootDir, graph);
	tf_model::ClassificationModelLoader model;
	int err = model.loadGraph(&session, graphPath);
	if (CYAL_SUCCESS != err)
	{
		std::cout << "ERROR: Load model failed..." << "(code:" << err << ")" << std::endl;
		return err;
	}

	std::vector<Tensor> resizedTensors;
	std::string imagePath = tensorflow::io::JoinPath(rootDir, image);
	//tf_model::SegmentationFeatureAdapter inputFeat;
	err = model.inputFeat.readTensorFromImageFile(imagePath, intputHeight, inputWidth, channels, inputSTD, inputMean, &resizedTensors);
	if (CYAL_SUCCESS != err)
	{
		std::cout << "ERROR: Read tensor from image file failed..." << "(code:" << err << ")" << std::endl;
		return err;
	}

	const tensorflow::Tensor& resizedTensor = resizedTensors[0];

	std::vector<tensorflow::Tensor> outputs;
	//model.inputFeat.inputs.push_back(std::pair<std::string, tensorflow::Tensor>(inputLayer, resizedTensor));
	err = model.predict(&session, inputLayer, resizedTensor, outputLayer, outputs);
	if (CYAL_SUCCESS != err)
	{
		std::cout << "ERROR: Predict failed..." << "(code:" << err << ")" << std::endl;
		return err;
	}
	//std::cout << outputs[0].matrix<float>() << std::endl;

	if (selfTest)
	{
		bool expectedMatches;
		err = model.inputFeat.checkTopLabel(outputs, 653, &expectedMatches);
		if (CYAL_GET_TOP_LABELS_ERROR == err)
		{
			std::cout << "ERROR: Get top labels failed..." << "(code:" << err << ")" << std::endl;
			return err;
		}
		else if (CYAL_CHECK_TOP_LABELS_ERROR == err)
		{
			std::cout << "ERROR: Check top labels failed..." << "(code:" << err << ")" << std::endl;
			std::cout << "...Self-test failed..." << std::endl;
			return err;
		}
		else{}
	}

	err = model.inputFeat.printTopLabels(outputs, labels);
	if (CYAL_SUCCESS != err)
	{
		std::cout << "ERROR: Print top labels failed..." << "(code:" << err << ")" << std::endl;
		return err;
	}

	return CYAL_SUCCESS;
}