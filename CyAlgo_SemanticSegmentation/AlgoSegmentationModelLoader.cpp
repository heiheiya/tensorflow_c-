#include <iostream>
#include <vector>
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


	SegmentationModelLoader::SegmentationModelLoader()
	{

	}

	SegmentationModelLoader::~SegmentationModelLoader()
	{

	}

	int SegmentationModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, const FeatureAdapterBase& inputFeature, const std::string& outputNode, std::vector<tensorflow::Tensor>& outputs)
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

		//tensorflow::Tensor t = outputs[0];
		//int ndim = t.shape().dims();
		//Tensor shape: [batch_size, target_class_num]
		//auto tmap = t.tensor<float, 2>();
		//get the target_class_num from 1st dimension
		//int outputDim = t.shape().dim_size(1);
		//std::vector<double> tout;

		//int outputClassID = -1;
		//double outputProb = 0.0;
		//for (int j = 0; j < outputDim; j++)
		//{
		//	std::cout << "Class " << j << " prob: " << tmap(0, j) << ", " << std::endl;
		//	if (tmap(0,j) >= outputProb)
		//	{
		//		outputClassID = j;
		//		outputProb = tmap(0, j);
		//	}
		//}

		//std::cout << "Final class id: " << outputClassID << std::endl;
		//std::cout << "Final class prob: " << outputProb << std::endl;

		return CYAL_SUCCESS;
	}
}
