
#include "sam.h"
#include "export.h"
///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

std::shared_ptr<SamEmbedding> eng_0;
std::shared_ptr<SamPromptEncoderAndMaskDecoder> eng_1;
at::Tensor image_embeddings;

#define EMBEDDING
#define SAMPROMPTENCODERANDMASKDECODER

std::vector<std::vector<double>> build_point_grid(int n_per_side)
{
	// Calculate Offset
	double offset = 1.0 / (2 * n_per_side);
	// Generate Points Along One Side
	std::vector<double> points_one_side;
	for (int i = 0; i < n_per_side; ++i)
	{
		points_one_side.push_back(offset + i * (1.0 - 2 * offset) / (n_per_side - 1));
	}
	// Tile Points for X and Y Axes
	std::vector<std::vector<double>> points_x(n_per_side, std::vector<double>(n_per_side, 0.0));
	std::vector<std::vector<double>> points_y(n_per_side, std::vector<double>(n_per_side, 0.0));
	for (int i = 0; i < n_per_side; ++i)
	{
		for (int j = 0; j < n_per_side; ++j)
		{
			points_x[i][j] = points_one_side[j];
			points_y[i][j] = points_one_side[i];
		}
	}
	// Combine X and Y Coordinates
	std::vector<std::vector<double>> points;
	for (int i = 0; i < n_per_side; ++i)
	{
		for (int j = 0; j < n_per_side; ++j)
		{
			std::vector<double> point = {points_x[i][j], points_y[i][j]};
			points.push_back(point);
		}
	}
	return points;
}

std::vector<std::vector<std::vector<double>>> build_all_layer_point_grids(
	int n_per_side, int n_layers, double scale_per_layer)
{
	std::vector<std::vector<std::vector<double>>> points_by_layer;
	for (int i = 0; i <= n_layers; ++i)
	{
		// Calculate number of points for each layer
		int n_points = static_cast<int>(n_per_side / std::pow(scale_per_layer, i));
		// Build point grid for the current layer
		points_by_layer.push_back(build_point_grid(n_points));
	}
	return points_by_layer;
}

std::vector<std::vector<int>> batch_iterator(int batch_size, const std::vector<std::vector<double>> &points)
{
	std::vector<std::vector<int>> batches;
	int n_batches = points.size() / batch_size + (points.size() % batch_size != 0);
	std::cout << "Number of Batches: " << n_batches << "\n\n";
	for (int b = 0; b < n_batches; ++b)
	{
		int start_idx = b * batch_size;
		int end_idx = std::min((b + 1) * batch_size, static_cast<int>(points.size()));
		// Create a batch by emplacing each point
		std::vector<int> batch;
		for (int i = start_idx; i < end_idx; ++i)
		{
			batch.emplace_back(static_cast<int>(points[i][0]));
			batch.emplace_back(static_cast<int>(points[i][1]));
		}
		// Add the batch to the result without scaling
		// Print the points in the current batch
		std::cout << "Batch " << b << ":\n";
		for (size_t i = 0; i < batch.size(); i += 2)
		{
			std::cout << "(" << batch[i] << ", " << batch[i + 1] << ") ";
		}
		std::cout << "\n\n";
		batches.emplace_back(std::move(batch));
	}
	return batches;
}

int main(int argc, char const *argv[])
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
		double scale_per_layer = 512 / 1500.0;

		auto all_layer_point_grids = build_all_layer_point_grids(points_per_side, crop_n_layers, scale_per_layer);

		// Print the generated point grids for all layers
		// for (int layer = 0; layer <= n_layers; ++layer) {
		// std::cout << "Layer " << layer << ":\n";
		// for (const auto& point : all_layer_point_grids[layer]) {
		// std::cout << "(" << point[0] << ", " << point[1] << ") ";
		// }
		// }

		double scale_x = 1800.0;
		double scale_y = 1200.0;

		std::vector<std::vector<double>> points_for_image = all_layer_point_grids[0];

		// Scale each point in the grid for the first layer
		for (auto &row : points_for_image)
		{
			for (size_t j = 0; j < row.size(); ++j)
			{
				row[j] *= (j == 0) ? scale_x : scale_y;
			}
		}

		// Print the scaled points for the first layer
		// std::cout << "Scaled Points for the First Layer:\n";
		// for (const auto& point : points_for_image) {
		// std::cout << "(" << point[0] << ", " << point[1] << ") ";
		//}

		auto batches = batch_iterator(points_per_batch, points_for_image);

		for (const auto &batch : batches)
		{
			auto res = eng_1->prepareInput(batch, image_embeddings);
			std::cout << "------------------prepareInput: " << res << std::endl;

			res = eng_1->infer();
			std::cout << "------------------infer: " << res << std::endl;

			eng_1->verifyOutput();
			std::cout << "-----------------done" << std::endl;
		}
	}
#endif
}