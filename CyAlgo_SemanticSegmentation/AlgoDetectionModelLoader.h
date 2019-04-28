#pragma once

#include "AlgoModelLoaderBase.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

namespace tf_model
{
	//Model loader for feed forward neural network
	class DetectionFeatureAdapter :public FeatureAdapterBase
	{
	public:
		DetectionFeatureAdapter();
		~DetectionFeatureAdapter();

		//////////////////////////////////////////////////////////////////////////
		//Convert a opencv Mat to tensorflow tensor
		//@param: cv::Mat input, a input opencv Mat
		//@param: float normal = 1 / 255.0, normalize OpenCV mat
		//////////////////////////////////////////////////////////////////////////
		tensorflow::Tensor cvMat2tfTensor(cv::Mat input, float normal = 1 / 255.0) override;

		//////////////////////////////////////////////////////////////////////////
		//Convert a tensorflow tensor to opencv Mat
		//@param: tensorflow::Tensor& inputTensor, an input tensorflow tensor
		//@param: cv::Mat output, a output opencv Mat
		//////////////////////////////////////////////////////////////////////////
		int tfTensor2cvMat(tensorflow::Tensor* inputTensor, cv::Mat& output) override;

		//////////////////////////////////////////////////////////////////////////
		//Takes a file name, and load a list of labels from it
		//@param: const std::string& fileName, a file name
		//@param: std::vector<float>* result, the box priors stored in a vector
		//separated by comma
		//@param: std::size_t* foundLabelCount, the found label count
		//////////////////////////////////////////////////////////////////////////
		int readLocationsFile(const std::string& fileName, std::vector<float>* result, std::size_t* foundLabelCount);

		//////////////////////////////////////////////////////////////////////////
		//Takes a tensor, and save it as image at the given filePath
		//@param: const tensorflow::Tensor& tensor, a tensor
		//@param:const std::string& filePath, the path you want to save tensor as image
		//////////////////////////////////////////////////////////////////////////
		int saveImage(const tensorflow::Tensor& tensor, const std::string& filePath);

		//////////////////////////////////////////////////////////////////////////
		//Converts an encoded location to an box placement with box priors
		//@param: const float* encodedLocation, the location you want to convert
		//@param: const float* boxPriors, box priors to box placement
		//@param: float* decodedLocation, box placement
		//////////////////////////////////////////////////////////////////////////
		void decodeLocation(const float* encodedLocation, const float* boxPriors, float* decodedLocation);

		float decodeScore(float encodedScore) { return 1 / (1 + exp(-encodedScore)); }

		void drawBox(const int imageWidth, const int imageHeight, int left, int top,
			int right, int bottom, tensorflow::TTypes<tensorflow::uint8>::Flat* image);

		//////////////////////////////////////////////////////////////////////////
		//Prints out the top five highest-scoring values
		//@param: const std::vector<tensorflow::Tensor>& outputs, the model run output
		//@param: const std::string& labelsFileName, a file contains the labels
		//@param: const int numBoxes, the number of boxes
		//@param: const int numDetections, the number of detections
		//@param: const std::string& imageFileName, which image to draw box on
		//@param: tensorflow::Tensor* originalTensor, tensor contains the image data
		//////////////////////////////////////////////////////////////////////////
		int printTopDetections(const std::vector<tensorflow::Tensor>& outputs, const std::string& labelsFileName,
			const int numBoxes, const int numDetections, const std::string& imageFileName,
			tensorflow::Tensor* originalTensor);

	private:
		tensorflow::int32 inputWidth;
		tensorflow::int32 inputHeight;
		float inputMean;
		float inputSTD;

	};

	class DetectionModelLoader :public ModelLoaderBase
	{
	public:
		DetectionModelLoader();
		~DetectionModelLoader();


		//////////////////////////////////////////////////////////////////////////
		// Make new prediction
		//@param std::unique_ptr<tensorflow::Session>* session, work session
		//@param std::string inputNode, tensor name of input node
		//@param tensorflow::Tensor& input, input tensor
		//@param std::string outputScoreNode, tensor name of output score node
		//@param std::string outputLocNode, tensor name of output location node
		//@param std::vector<tensorflow::Tensor>& outputs, output tensor vector
		//////////////////////////////////////////////////////////////////////////
		int predict(std::unique_ptr<tensorflow::Session>* session, std::string inputNode, const tensorflow::Tensor& input,
			const std::string& outputScoreNode, const std::string& outputLocNode, std::vector<tensorflow::Tensor>& outputs);


	public:
		DetectionFeatureAdapter inputFeat;
	};
}