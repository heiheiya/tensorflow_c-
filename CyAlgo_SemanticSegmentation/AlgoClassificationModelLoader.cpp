#include <iostream>
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

	int ClassificationFeatureAdapter::printTopLabels(const std::vector<tensorflow::Tensor>& outputs, const std::string& labelsFileName)
	{
		std::vector<std::string> labels;
		std::size_t labelCount;

		int err = readLabelsFile(labelsFileName, &labels, &labelCount);
		if (CYAL_SUCCESS != err)
		{
			std::cout << "ERROR: Read Labels from file failed..." << "(code:" << err << ")" << std::endl;
			return err;
		}

		const int howManyLabels = std::min(5, static_cast<int>(labelCount));
		tensorflow::Tensor indices;
		tensorflow::Tensor scores;
		err = getTopDetections(outputs, howManyLabels, &indices, &scores);
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

	int ClassificationFeatureAdapter::checkTopLabel(const std::vector<tensorflow::Tensor>& outputs, int expected, bool* isExpected)
	{
		*isExpected = false;
		tensorflow::Tensor indices;
		tensorflow::Tensor scores;
		const int howManyLabels = 1;
		int err = getTopDetections(outputs, howManyLabels, &indices, &scores);
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
	
	
	ClassificationModelLoader::ClassificationModelLoader()
	{
	
	}
	
	ClassificationModelLoader::~ClassificationModelLoader()
	{
	
	}
	
	int ClassificationModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
		const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs)
	{
		tensorflow::Status status = (*session)->Run({ {inputNode, input} }, { outputNode }, {}, &outputs);
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
}

