
#include "sam.h"
#include "export.h"
//#include "baseModel.h"
///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

std::shared_ptr<SamEmbedding> eng_0;
std::shared_ptr<SamPromptEncoderAndMaskDecoder> eng_1;
at::Tensor image_embeddings;

#define EMBEDDING
#define SAMPROMPTENCODERANDMASKDECODER

//#include <vector>
//#include <numeric>
//#include <algorithm>
//#include <cmath>

std::vector<std::vector<double>> build_point_grid(int n_per_side) {
	// Generate a 1D array of points for one side
	double offset = 1.0 / (2 * n_per_side);
	std::vector<double> points_one_side;
	for (int i = 0; i < n_per_side; ++i) {
		points_one_side.push_back(offset + i * (1.0 - 2 * offset) / (n_per_side - 1));
	}

	// Create a 2D grid of points using the generated array
	std::vector<std::vector<double>> points(n_per_side, std::vector<double>(n_per_side));

	for (int i = 0; i < n_per_side; ++i) {
		for (int j = 0; j < n_per_side; ++j) {
			points[i][j] = points_one_side[j];
		}
	}

	return points;
}

 //Function to build all layer point grids
std::vector<std::vector<std::vector<double>>> build_all_layer_point_grids(
	int n_per_side, int n_layers, int scale_per_layer
) {
	// Generates point grids for all crop layers
	std::vector<std::vector<std::vector<double>>> points_by_layer;

	for (int i = 0; i <= n_layers; ++i) {
		int n_points = static_cast<int>(n_per_side / std::pow(scale_per_layer, i));
		points_by_layer.push_back(build_point_grid(n_points));
	}

	return points_by_layer;
}

std::vector<std::vector<int>> m_batch_iterator(int batch_size, const std::vector<std::vector<double>>& args) {
	assert(!args.empty() && std::all_of(args.begin(), args.end(),
		[&](const std::vector<double>& a) { return a.size() == args[0].size(); }),
		"Batched iteration must have inputs of all the same size.");

	int n_batches = args[0].size() / batch_size + static_cast<int>(args[0].size() % batch_size != 0);
	std::cout << "Number of Batches: " << n_batches << std::endl;

	std::vector<std::vector<int>> batches;

	for (int b = 0; b < n_batches; ++b) {
		std::vector<int> batch_args;

		// Extract batch for each argument
		for (const auto& arg : args) {
			auto start = arg.begin() + b * batch_size;
			auto end = arg.begin() + std::min((b + 1) * batch_size, static_cast<int>(arg.size()));

			// Convert each double element to int and append to batch_args
			batch_args.insert(batch_args.end(), start, end);
		}

		batches.push_back(batch_args);
	}

	return batches;
}

int main(int argc, char const* argv[])
{
	std::cout << "into main" << std::endl;

	ifstream f1("weights/vit_l_embedding.engine");
	if (!f1.good())
		export_engine_image_encoder("weights/vit_l_embedding.onnx", "weights/vit_l_embedding.engine");

	ifstream f2("weights/samlorg64pts.engine");
	if (!f2.good())
		export_engine_prompt_encoder_and_mask_decoder("weights/samlorg.onnx", "weights/samlorg22.engine");

#ifdef EMBEDDING
	{
		const std::string modelFile = "weights/vit_l_embedding.engine";
		std::cout << "into embeddings" << std::endl;
		std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
		assert(engineFile);

		int fsize;
		engineFile.seekg(0, engineFile.end);
		fsize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);
		std::vector<char> engineData(fsize);
		engineFile.read(engineData.data(), fsize);

		if (engineFile)
			std::cout << "all characters read successfully." << std::endl;
		else
			std::cout << "error: only " << engineFile.gcount() << " could be read" << std::endl;
		engineFile.close();

		std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
		std::shared_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
		cv::Mat frame = cv::imread("data/truck.jpg");
		std::cout << frame.size << std::endl;
		eng_0 = std::shared_ptr<SamEmbedding>(new SamEmbedding(std::to_string(1), mEngine, frame));
		auto res = eng_0->prepareInput();
		std::cout << "------------------prepareInput: " << res << std::endl;
		res = eng_0->infer();
		std::cout << "------------------infer: " << res << std::endl;
		image_embeddings = eng_0->verifyOutput();
		std::cout << "------------------verifyOutput: " << std::endl;
	}

#endif

#ifdef SAMPROMPTENCODERANDMASKDECODER
	{
		const std::string modelFile = "weights/samlorg64pts.engine";
		std::cout << "into prompt encoder" << std::endl;
		std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
		assert(engineFile);

		int fsize;
		engineFile.seekg(0, engineFile.end);
		fsize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);
		std::vector<char> engineData(fsize);
		engineFile.read(engineData.data(), fsize);

		if (engineFile)
			std::cout << "all characters read successfully." << std::endl;
		else
			std::cout << "error: only " << engineFile.gcount() << " could be read" << std::endl;
		engineFile.close();

		std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
		std::shared_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
		cv::Mat frame = cv::imread("data/truck.jpg");
		eng_1 = std::shared_ptr<SamPromptEncoderAndMaskDecoder>(new SamPromptEncoderAndMaskDecoder(std::to_string(1), mEngine, frame));

		int points_per_batch = 64;
		int points_per_side = 32;
		int crop_n_layers = 0;
		int crop_overlap_ratio = 512 / 1500;

		std::vector<std::vector<std::vector<double>>>point_grids = build_all_layer_point_grids(points_per_side, crop_n_layers, crop_overlap_ratio);

		int scale_x = 1800;
		int scale_y = 1200;

		std::vector<std::vector<double>> points_for_image = point_grids[0];

		//// Scale each point in the grid and store in points_for_image
		for (auto& row : points_for_image) {
			for (size_t j = 0; j < row.size(); ++j) {
				row[j] *= (j == 0) ? scale_x : scale_y;
			}

			std::vector<std::vector<int>> batches = m_batch_iterator(points_per_batch, points_for_image);

			for (const auto& batch : batches) {
				auto res = eng_1->prepareInput(batch, image_embeddings);
				std::cout << "------------------prepareInput: " << res << std::endl;

				res = eng_1->infer();
				std::cout << "------------------infer: " << res << std::endl;

				eng_1->verifyOutput();
				std::cout << "-----------------done" << std::endl;
			}

			/*auto res = eng_1->prepareInput(batches, image_embeddings);
			std::cout << "------------------prepareInput: " << res << std::endl;
			res = eng_1->infer();
			std::cout << "------------------infer: " << res << std::endl;*/
			//eng_1->verifyOutput();
			//std::cout << "-----------------done" << std::endl;
		}
	}
#endif
}