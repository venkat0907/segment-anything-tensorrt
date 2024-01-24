
#include "sam.h"
#include "export.h"
#include "baseModel.h"
///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

std::shared_ptr<SamEmbedding> eng_0;
std::shared_ptr<SamPromptEncoderAndMaskDecoder> eng_1;
at::Tensor image_embeddings;

#define EMBEDDING
#define SAMPROMPTENCODERANDMASKDECODER

//std::vector<float> build_point_grid(int n_points) {
//	std::vector<float> point_grid;
//	for (int i = 0; i < n_points; ++i) {
//		float x = static_cast<float>(i) / n_points;
//		for (int j = 0; j < n_points; ++j) {
//			float y = static_cast<float>(j) / n_points;
//			point_grid.push_back(x);
//			point_grid.push_back(y);
//		}
//	}
//	return point_grid;
//}
//
//std::vector<std::vector<float>> build_all_layer_point_grids(int n_per_side, int n_layers, int scale_per_layer) {
//	std::vector<std::vector<float>> points_by_layer;
//	for (int i = 0; i <= n_layers; ++i) {
//		int n_points = n_per_side / (scale_per_layer * i + 1);
//		auto point_grid = build_point_grid(n_points);
//		points_by_layer.push_back(point_grid);
//	}
//	return points_by_layer;
//}

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using std::vector;
using std::iota;

vector<double> build_point_grid(int n_per_side) {
	// Calculate offset and spacing for evenly spaced points
	double offset = 1.0 / (2 * n_per_side);
	double spacing = 1.0 / n_per_side;

	// Generate points efficiently using iota and transform
	vector<double> points_x(n_per_side * n_per_side);
	iota(points_x.begin(), points_x.end(), offset);
	transform(points_x.begin(), points_x.end(), points_x.begin(),
		[spacing](double x) { return x * spacing; });

	// Reshape into 2D grid
	vector<double> points(2 * n_per_side * n_per_side);
	std::copy(points_x.begin(), points_x.end(), points.begin());
	std::copy(points_x.begin(), points_x.end(), points.begin() + n_per_side * n_per_side);

	return points;
}

vector<vector<double>> build_all_layer_point_grids(int n_per_side, int n_layers, int scale_per_layer) {
	vector<vector<double>> points_by_layer;

	for (int i = 0; i <= n_layers; ++i) {
		int n_points = int(ceil(n_per_side / pow(scale_per_layer, i)));  // Ensure integer division rounds up
		points_by_layer.push_back(build_point_grid(n_points));
	}

	return points_by_layer;
}

int main(int argc, char const* argv[])
{
	std::cout << "into main" << std::endl;

	ifstream f1("weights/vit_l_embedding.engine");
	if (!f1.good())
		export_engine_image_encoder("weights/vit_l_embedding.onnx", "weights/vit_l_embedding.engine");

	ifstream f2("weights/samlorg5pts.engine");
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
		const std::string modelFile = "weights/samlorg5pts.engine";
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
		float crop_overlap_ratio = 512.0 / 1500.0;

		vector<vector<double>> point_grids = build_all_layer_point_grids(points_per_side, crop_n_layers, crop_overlap_ratio);

		for (int i = 0; i < point_grids.size(); ++i) {
			cout << "Point grid for layer " << i << ":" << endl;
			for (const double& point : point_grids[i]) {
				cout << point << " ";
			}
			cout << endl;
		}

		vector<int> points_scale = { frame.rows, frame.cols };  // Reverse order for ::-1
		cout << "points_scale: ";
		for (int value : points_scale) {
			cout << value << " ";
		}
		cout << endl;

		vector<double>& points_for_image = point_grids[0];
		for (int i = 0; i < points_for_image.size(); ++i) {
			points_for_image[i] *= points_scale[i % 2]; // Use modulo to alternate between rows and cols
		}

		cout << "points_for_image after scaling:" << endl;
		for (const double& point : points_for_image) {
			cout << point << " ";
		}
		cout << endl;

		/*std::vector<int> mult_pts = {x,y,x-5,y-5,x+5,y+5};
		auto res = eng_1->prepareInput(mult_pts, image_embeddings);
		std::cout << "------------------prepareInput: " << res << std::endl;
		res = eng_1->infer();
		std::cout << "------------------infer: " << res << std::endl;
		eng_1->verifyOutput();
		std::cout << "-----------------done" << std::endl;*/
	}
#endif
}