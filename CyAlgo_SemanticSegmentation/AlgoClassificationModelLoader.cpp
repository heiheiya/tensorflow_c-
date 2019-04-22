#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/errors.h"

#include "Common.h"
#include "AlgoClassificationModelLoader.h"


namespace tf_model
{
	ClassificationFeatureAdapter::ClassificationFeatureAdapter()
	{
	
	}
	
	ClassificationFeatureAdapter::~ClassificationFeatureAdapter()
	{
	
	}
	
	
	ClassificationModelLoader::ClassificationModelLoader()
	{
	
	}
	
	ClassificationModelLoader::~ClassificationModelLoader()
	{
	
	}
	
	int ClassificationModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, const FeatureAdapterBase& inputFeature, const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs)
	{
		tensorflow::Status status = (*session)->Run(inputFeature.inputs, { outputNode }, {}, &outputs);
		if (!status.ok())
		{
			std::cout << "ERROR: Session run failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}
	
		std::cout << "Output tensor size: " << outputs.size() << std::endl;
		for (std::size_t i = 0; i < outputs.size(); i++)
		{
			std::cout << outputs[i].DebugString();
		}
		std::cout << std::endl;
	
		return CYAL_SUCCESS;
	}
	
	int ClassificationModelLoader::getTopLabels(const std::vector<tensorflow::Tensor>& outputs, int howManyLabels, tensorflow::Tensor* indices, tensorflow::Tensor* scores)
	{
		auto root = tensorflow::Scope::NewRootScope();
	
		std::string outputName = "top_k";
		tensorflow::ops::TopK(root.WithOpName(outputName), outputs[0], howManyLabels);
	
		tensorflow::GraphDef graph;
		tensorflow::Status status = root.ToGraphDef(&graph);
		if (!status.ok())
		{
			std::cout << "ERROR: Run graphdef failed..." << "(code:" << CYAL_TF_RUN_GRAPHDEF_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_RUN_GRAPHDEF_ERROR;
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
	
		return CYAL_SUCCESS;
	}
	
	int ClassificationModelLoader::printTopLabels(const std::vector<tensorflow::Tensor>& outputs, const std::string& labelsFileName)
	{
		std::vector<std::string> labels;
		std::size_t labelCount;
	
		int err = inputFeat.readLabelsFile(labelsFileName, &labels, &labelCount);
		if (CYAL_SUCCESS != err)
		{
			std::cout << "ERROR: Read Labels from file failed..." << "(code:" << err << ")" << std::endl;
			return err;
		}
	
		const int howManyLabels = std::min(5, static_cast<int>(labelCount));
		tensorflow::Tensor indices;
		tensorflow::Tensor scores;
		err = getTopLabels(outputs, howManyLabels, &indices, &scores);
		if (CYAL_SUCCESS != err)
		{
			std::cout << "ERROR: Get top labels failed..." << "(code:" << err << ")" << std::endl;
			return err;
		}
	
		tensorflow::TTypes<float>::Flat scoresFlat = scores.flat<float>();
		tensorflow::TTypes<tensorflow::int32>::Flat indicesFlat = indices.flat<tensorflow::int32>();
		for (int pos = 0; pos < howManyLabels; ++pos)
		{
			const int labelIndex = indicesFlat(pos);
			const float score = scoresFlat(pos);
			std::cout << labels[labelIndex] << "( " << labelIndex << " ): " << score << std::endl;
		}
	
		return CYAL_SUCCESS;
	}
	
	int ClassificationModelLoader::checkTopLabel(const std::vector<tensorflow::Tensor>& outputs, int expected, bool* isExpected)
	{
		*isExpected = false;
		tensorflow::Tensor indices;
		tensorflow::Tensor scores;
		const int howManyLabels = 1;
		int err = getTopLabels(outputs, howManyLabels, &indices, &scores);
		if (CYAL_SUCCESS != err)
		{
			std::cout << "ERROR: Get top labels failed..." << "(code:" << CYAL_GET_TOP_LABELS_ERROR << ")" << std::endl;
			return CYAL_GET_TOP_LABELS_ERROR;
		}
	
		tensorflow::TTypes<tensorflow::int32>::Flat indicesFlat = indices.flat<tensorflow::int32>();
		if (indicesFlat(0) != expected)
		{
			std::cout << "ERROR: Expected label#" << expected << " but got #" << indicesFlat(0) << "(code:" << CYAL_CHECK_TOP_LABELS_ERROR << ")" << std::endl;
			*isExpected = false;
			return CYAL_CHECK_TOP_LABELS_ERROR;
		}
		else
		{
			*isExpected = true;
		}
	
		return CYAL_SUCCESS;
	}
}

