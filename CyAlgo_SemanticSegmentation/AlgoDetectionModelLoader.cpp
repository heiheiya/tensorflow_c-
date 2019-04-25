#include <algorithm>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/default/logging.h"

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

	int DetectionFeatureAdapter::tfTensor2cvMat(tensorflow::Tensor* inputTensor, cv::Mat& output)
	{
		tensorflow::TensorShape inputTensorShape = (*inputTensor).shape();
		if (inputTensorShape.dims() != 3)
		{
			std::cout << "ERROR: The input tensor shape is not 3 dimension...but " << inputTensorShape.dims() << "(code:" << CYAL_TF_TENSOR_DIMENSION_ERROR << ")" << std::endl;
			return CYAL_TF_TENSOR_DIMENSION_ERROR;
		}

		int height = inputTensorShape.dim_size(0);
		int width = inputTensorShape.dim_size(1);
		int depth = inputTensorShape.dim_size(2);

		std::cout << inputTensor->dtype() << std::endl;
		uint8_t* tensorDataPtr = inputTensor->flat<tensorflow::uint8>().data();

		output = cv::Mat(height, width, CV_8UC3, tensorDataPtr);
		cv::cvtColor(output, output, cv::COLOR_BGR2RGB);

		return CYAL_SUCCESS;
	}

	int DetectionFeatureAdapter::readLocationsFile(const std::string& fileName, std::vector<float>* result, std::size_t* foundLabelCount)
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
			std::vector<float> tokens;
			bool err = tensorflow::str_util::SplitAndParseAsFloats(line, ',', &tokens);
			if (!err)
			{
				std::cout << "ERROR: Parse file failed..." << "(code:" << CYAL_TF_SPLIT_PARSE_FILE_ERROR << ")" << std::endl;
				return CYAL_TF_SPLIT_PARSE_FILE_ERROR;
			}
			for (auto number : tokens)
			{
				result->push_back(number);
			}
		}
		*foundLabelCount = result->size();
		return CYAL_SUCCESS;
	}

	int DetectionFeatureAdapter::saveImage(const tensorflow::Tensor& tensor, const std::string& filePath)
	{
		auto root = tensorflow::Scope::NewRootScope();

		std::string outputName = "file_writer";

		tensorflow::Output imageEncoder;

		if (tensorflow::str_util::EndsWith(filePath, ".png"))
		{
			imageEncoder = tensorflow::ops::EncodePng(root.WithOpName("png_reader"), tensor);
		}
		else if (tensorflow::str_util::EndsWith(filePath, ".jpg"))
		{
			imageEncoder = tensorflow::ops::EncodeJpeg(root.WithOpName("jpeg_reader"), tensor);
		}
		else
		{
			std::cout << "ERROR: Only saving of png or jpeg files is supported..." << "(code:" << CYAL_TF_SAVE_FORMAT_NOT_SUPPORT_ERROR << ")" << std::endl;
			return CYAL_TF_SAVE_FORMAT_NOT_SUPPORT_ERROR;
		}

		tensorflow::ops::WriteFile fileSaver = tensorflow::ops::WriteFile(root.WithOpName(outputName), filePath, imageEncoder);

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

		std::vector<tensorflow::Tensor> outputs;
		status = session->Run({}, {}, { outputName }, &outputs);
		if (!status.ok())
		{
			std::cout << "ERROR: Run session failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}

		return CYAL_SUCCESS;
	}

	void DetectionFeatureAdapter::decodeLocation(const float* encodedLocation, const float* boxPriors, float* decodedLocation)
	{
		bool nonZero = false;
		for (int i = 0; i < 4; ++i)
		{
			const float currEncoding = encodedLocation[i];
			nonZero = nonZero || currEncoding != 0.0f;

			const float mean = boxPriors[i * 2];
			const float stdDev = boxPriors[i * 2 + 1];

			float currentLocation = currEncoding * stdDev + mean;

			currentLocation = std::max(currentLocation, 0.0f);
			currentLocation = std::min(currentLocation, 1.0f);
			decodedLocation[i] = currentLocation;
		}

		if (!nonZero)
		{
			std::cout << "WARNING: No non-zero encodings..." << std::endl;
		}
	}

	void DetectionFeatureAdapter::drawBox(const int imageWidth, const int imageHeight, int left, int top, int right, int bottom, tensorflow::TTypes<tensorflow::uint8>::Flat* image)
	{
		tensorflow::TTypes<tensorflow::uint8>::Flat imageRef = *image;

		top = std::max(0, std::min(imageHeight - 1, top));
		bottom = std::max(0, std::min(imageHeight - 1, bottom));

		left = std::max(0, std::min(imageWidth - 1, left));
		right = std::max(0, std::min(imageWidth - 1, right));

		for (int i = 0; i < 3; ++i)
		{
			tensorflow::uint8 val = i == 2 ? 255 : 0;
			for (int x = left; x <= right; ++x)
			{
				imageRef((top * imageWidth + x) * 3 + i) = val;
				imageRef((bottom * imageWidth + x) * 3 + i) = val;
			}
			for (int y = top; y <= bottom; ++y)
			{
				imageRef((y * imageWidth + left) * 3 + i) = val;
				imageRef((y * imageWidth + right) * 3 + i) = val;
			}
		}
	}

	int DetectionFeatureAdapter::printTopDetections(const std::vector<tensorflow::Tensor>& outputs, const std::string& labelsFileName, const int numBoxes, const int numDetections, const std::string& imageFileName, tensorflow::Tensor* originalTensor)
	{
		std::vector<float> locations;
		std::size_t labelCount;

		int err = readLocationsFile(labelsFileName, &locations, &labelCount);
		if (CYAL_SUCCESS != err)
		{
			std::cout << "ERROR: Read locations file failed..." << "(code:" << err << ")" << std::endl;
			return err;
		}
		CHECK_EQ(labelCount, numBoxes * 8);

		const int howManyLabels = std::min(numDetections, static_cast<int>(labelCount));
		tensorflow::Tensor indices;
		tensorflow::Tensor scores;
		err = getTopDetections(outputs, howManyLabels, &indices, &scores);
		if (CYAL_SUCCESS != err)
		{
			std::cout << "ERROR: Get top detections failed..." << "(code:" << err << ")" << std::endl;
			return err;
		}

		tensorflow::TTypes<float>::Flat scoresFlat = scores.flat<float>();
		tensorflow::TTypes<tensorflow::int32>::Flat indicesFlat = indices.flat<tensorflow::int32>();

		const tensorflow::Tensor& encodedLocations = outputs[1];
		auto locationsEncoded = encodedLocations.flat<float>();

		std::cout << originalTensor->DebugString() << std::endl;
		const int imageWidth = originalTensor->shape().dim_size(1);
		const int imageHeight = originalTensor->shape().dim_size(0);

		tensorflow::TTypes<tensorflow::uint8>::Flat imageFlat = originalTensor->flat<tensorflow::uint8>();

		std::cout << "===== Top " << howManyLabels << " Detections =====" << std::endl;
		for (int pos = 0; pos < howManyLabels; ++pos)
		{
			const int labelIndex = indicesFlat(pos);
			const float score = scoresFlat(pos);

			float decodedLocation[4];
			decodeLocation(&locationsEncoded(labelIndex * 4), &locations[labelIndex * 8], decodedLocation);

			float left = decodedLocation[0] * imageWidth;
			float top = decodedLocation[1] * imageHeight;
			float right = decodedLocation[2] * imageWidth;
			float bottom = decodedLocation[3] * imageHeight;

			std::cout << "Detection " << pos << ": "
				<< "L: " << left << " "
				<< "T: " << top << " "
				<< "R: " << right << " "
				<< "B: " << bottom << " "
				<< "(" << labelIndex << ") score: " << decodeScore(score) << std::endl;

			drawBox(imageWidth, imageHeight, left, top, right, bottom, &imageFlat);

			if (!imageFileName.empty())
			{
				err = saveImage(*originalTensor, imageFileName);
				if (CYAL_SUCCESS != err)
				{
					std::cout << "ERROR: Save image failed..." << "(code:" << err << ")" << std::endl;
					return err;
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

	int DetectionModelLoader::predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
		const std::string& outputScoreNode, const std::string& outputLocNode, std::vector<tensorflow::Tensor>& outputs)
	{
		tensorflow::Status status = (*session)->Run({ { inputNode, input } }, { outputScoreNode, outputLocNode }, {}, &outputs);
		if (!status.ok())
		{
			std::cout << "ERROR: Session run failed..." << "(code:" << CYAL_TF_SESSION_RUN_ERROR << ")" << std::endl;
			std::cout << status.ToString() << std::endl;
			return CYAL_TF_SESSION_RUN_ERROR;
		}

		std::cout << "Output tensor size: " << outputs.size() << std::endl;
		for (std::size_t i = 0; i < outputs.size(); i++)
		{
			std::cout << outputs[i].DebugString() << std::endl;
		}
		std::cout << std::endl;

		return CYAL_SUCCESS;
	}
}

