
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
		
		std::vector<int> mult_pts = {
		320, 350,
		400, 420,
		450, 480,
		370, 410,
		490, 330
		};

		//std::vector<int> mult_pts = {x,y,x-5,y-5,x+5,y+5};
		auto res = eng_1->prepareInput(mult_pts, image_embeddings);
		std::cout << "------------------prepareInput: " << res << std::endl;
		res = eng_1->infer();
		std::cout << "------------------infer: " << res << std::endl;
		eng_1->verifyOutput();
		std::cout << "-----------------done" << std::endl;
	}
#endif
}