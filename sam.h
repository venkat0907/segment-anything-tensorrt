#ifndef SAM_H
#define SAM_H

#include "buffers.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
// #include <torchvision/vision.h>
#include <torch/script.h>
#include "sam_utils.h"
#include <iostream>
#include <fstream>


using namespace torch::indexing;

class ResizeLongestSide
{
public:
    ResizeLongestSide(int target_length);
    ~ResizeLongestSide();

    std::vector<int> get_preprocess_shape(int oldh, int oldw);
    at::Tensor apply_coords(at::Tensor boxes, at::IntArrayRef sz);

public:
    int m_target_length;
};

ResizeLongestSide::ResizeLongestSide(int target_length) : m_target_length(target_length)
{
}

ResizeLongestSide::~ResizeLongestSide()
{
}

std::vector<int> ResizeLongestSide::get_preprocess_shape(int oldh, int oldw)
{
    float scale = m_target_length * 1.0 / std::max(oldh, oldw);
    int newh = static_cast<int>(oldh * scale + 0.5);
    int neww = static_cast<int>(oldw * scale + 0.5);
    std::cout << " newh " << newh << "  neww " << neww << std::endl;
    std::cout << "at::IntArrayRef{newh, neww}" << at::IntArrayRef{newh, neww} << std::endl;
    return  std::vector<int>{newh, neww};
}

at::Tensor ResizeLongestSide::apply_coords(at::Tensor coords, at::IntArrayRef sz)
{
    int old_h = sz[0], old_w = sz[1];
    auto new_sz = get_preprocess_shape(old_h, old_w);
    int new_h = new_sz[0], new_w = new_sz[1];
    coords.index_put_({"...", 0}, coords.index({"...", 0}) * (1.0 * new_w / old_w));
    coords.index_put_({"...", 1}, coords.index({"...", 1}) * (1.0 * new_h / old_h));
    return coords;
}

////////////////////////////////////////////////////////////////////////////////////

class SamEmbedding
{
public:
    SamEmbedding(std::string bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width = 640, int height = 640);
    ~SamEmbedding();

    int prepareInput();
    bool infer();
    at::Tensor verifyOutput();
    at::Tensor verifyOutput(std::string output_name);

public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    cudaEvent_t start, end;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<float> pad_info;
    std::vector<std::string> names;
    cv::Mat frame;
    cv::Mat img;
    int inp_width = 640;
    int inp_height = 640;
    std::string mBufferName;
};

SamEmbedding::SamEmbedding(std::string bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width, int height) : mBufferName(bufferName), mEngine(engine), frame(im), inp_width(width), inp_height(height)
{
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }

    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        // dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        // index2srt(type);
        int vecDim = mEngine->getBindingVectorizedDim(i);
        // std::cout << "vecDim:" << vecDim << std::endl;
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            std::cout << "scalarsPerVec" << scalarsPerVec << std::endl;
        }
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

SamEmbedding::~SamEmbedding()
{
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaStreamDestroy(stream));
}

int SamEmbedding::prepareInput()
{
    int prompt_embed_dim = 256;
    int image_size = 1024;
    int vit_patch_size = 16;
    int target_length = image_size;
    auto pixel_mean = at::tensor({123.675, 116.28, 103.53}, torch::kFloat).view({-1, 1, 1});
    auto pixel_std = at::tensor({58.395, 57.12, 57.375}, torch::kFloat).view({-1, 1, 1});
    ResizeLongestSide transf(image_size);
    int newh, neww;
    auto target_size = transf.get_preprocess_shape(frame.rows, frame.cols);
    // std::cout << "    " << torch::IntArrayRef{newh,neww} << std::endl;
    std::cout << "target_size = " << target_size << std::endl;
    cv::Mat im_sz;
    std::cout << frame.size << std::endl;
    cv::resize(frame, im_sz, cv::Size(target_size[1], target_size[0]));
    im_sz.convertTo(im_sz, CV_32F, 1.0);
    at::Tensor input_image_torch =
        at::from_blob(im_sz.data, {im_sz.rows, im_sz.cols, im_sz.channels()})
            .permute({2, 0, 1})
            .contiguous()
            .unsqueeze(0);
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std;
    int h = input_image_torch.size(2);
    int w = input_image_torch.size(3);
    int padh = image_size - h;
    int padw = image_size - w;
    input_image_torch = at::pad(input_image_torch, {0, padw, 0, padh});
    auto ret = mInOut["images"]->host2device((void *)(input_image_torch.data_ptr<float>()), true, stream);
    return ret;
}

bool SamEmbedding::infer()
{
    CHECK(cudaEventRecord(start, stream));
    auto ret = context->enqueueV2(mDeviceBindings.data(), stream, nullptr);
    return ret;
}

at::Tensor SamEmbedding::verifyOutput()
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape("image_embeddings");

    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor preds;
    preds = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["image_embeddings"]->device2host((void *)(preds.data_ptr<float>()), stream);

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));
    // torch::save({preds}, "preds.pt");
    // cv::FileStorage storage("1.yaml", cv::FileStorage::WRITE);
    // storage << "image_embeddings" << points3dmatrix;
    return preds;
}

at::Tensor SamEmbedding::verifyOutput(std::string output_name)
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape(output_name.c_str());

    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor preds;
    preds = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut[output_name]->device2host((void *)(preds.data_ptr<float>()), stream);

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));
    // torch::save({preds}, "preds.pt");
    // cv::FileStorage storage("1.yaml", cv::FileStorage::WRITE);
    // storage << "image_embeddings" << points3dmatrix;
    return preds;
}

///////////////////////////////////////////////////

class SamEmbedding2
{
public:
    SamEmbedding2(std::string bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine);
    ~SamEmbedding2();

    int prepareInput(at::Tensor input_image_torch);
    bool infer();
    at::Tensor verifyOutput();

public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    cudaEvent_t start, end;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<float> pad_info;
    std::vector<std::string> names;
    std::string mBufferName;
};

SamEmbedding2::SamEmbedding2(std::string bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine) : 
mBufferName(bufferName), mEngine(engine)
{
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }

    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        // dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        // index2srt(type);
        int vecDim = mEngine->getBindingVectorizedDim(i);
        // std::cout << "vecDim:" << vecDim << std::endl;
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            std::cout << "scalarsPerVec" << scalarsPerVec << std::endl;
        }
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

SamEmbedding2::~SamEmbedding2()
{
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaStreamDestroy(stream));
}

int SamEmbedding2::prepareInput(at::Tensor input_image_torch)
{
    auto ret = mInOut["image_embeddings_part_1"]->host2device((void *)(input_image_torch.data_ptr<float>()), false, stream);
    return ret;
}

bool SamEmbedding2::infer()
{
    CHECK(cudaEventRecord(start, stream));
    auto ret = context->enqueueV2(mDeviceBindings.data(), stream, nullptr);
    return ret;
}

at::Tensor SamEmbedding2::verifyOutput()
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape("image_embeddings_part_2");

    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor preds;
    preds = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["image_embeddings_part_2"]->device2host((void *)(preds.data_ptr<float>()), stream);

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));
    // torch::save({preds}, "preds.pt");
    // cv::FileStorage storage("1.yaml", cv::FileStorage::WRITE);
    // storage << "image_embeddings" << points3dmatrix;
    return preds;
}

///////////////////////////////////////////////////



class SamPromptEncoderAndMaskDecoder
{
public:
    SamPromptEncoderAndMaskDecoder(std::string bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width = 640, int height = 640);
    ~SamPromptEncoderAndMaskDecoder();

    int prepareInput(int x, int y, at::Tensor image_embeddings);
    int prepareInput(int x, int y, int x1, int y1, int x2, int y2, at::Tensor image_embeddings);
    int prepareInput(std::vector<int> mult_pts, at::Tensor image_embeddings);
    bool infer();
    int verifyOutput();
    int verifyOutput(cv::Mat& roi);
    at::Tensor generator_colors(int num);

    template <class Type>
    Type string2Num(const std::string &str);

    at::Tensor plot_masks(at::Tensor masks, at::Tensor im_gpu, float alpha);

public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    cudaEvent_t start, end;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<float> pad_info;
    std::vector<std::string> names;
    cv::Mat frame;
    cv::Mat img;
    int inp_width = 640;
    int inp_height = 640;
    std::string mBufferName;
};

SamPromptEncoderAndMaskDecoder::SamPromptEncoderAndMaskDecoder(std::string bufferName, std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, int width, int height) : mBufferName(bufferName), mEngine(engine), frame(im), inp_width(width), inp_height(height)
{
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }
    // set input dims whichs name "point_coords "
    context->setBindingDimensions(1, nvinfer1::Dims3(1, 2, 2));
    // set input dims whichs name "point_label "
    context->setBindingDimensions(2, nvinfer1::Dims2(1, 2));
    // set input dims whichs name "point_label "
    // context->setBindingDimensions(5, nvinfer1::Dims2(frame.rows,frame.cols));
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    int nbopts = mEngine->getNbOptimizationProfiles();
    // std::cout << "nboopts: " << nbopts << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        // auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        // std::cout << "tensor_name: " << tensor_name << std::endl;
        auto dims = context->getBindingDimensions(i);
        // dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        // index2srt(type);
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

SamPromptEncoderAndMaskDecoder::~SamPromptEncoderAndMaskDecoder()
{
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaStreamDestroy(stream));
}

int SamPromptEncoderAndMaskDecoder::prepareInput(int x, int y, at::Tensor image_embeddings)
{
    // at::Tensor image_embeddings;

    // torch::load(image_embeddings, "preds.pt");
    // std::cout << image_embeddings.sizes() << std::endl;
    int image_size = 1024;
    ResizeLongestSide transf(image_size);

    auto input_point = at::tensor({x, y}, at::kFloat).reshape({-1,2});
    auto input_label = at::tensor({1}, at::kFloat);

    std::cout << "input_point: " << input_point << std::endl;
    std::cout << "input_label: " << input_label << std::endl;

    auto trt_coord = at::concatenate({input_point, at::tensor({0, 0}, at::kFloat).unsqueeze(0)}, 0).unsqueeze(0);
    auto trt_label = at::concatenate({input_label, at::tensor({-1}, at::kFloat)}, 0).unsqueeze(0);
    // auto trt_coord = at::concatenate({input_point, at::tensor({x-100, y-100, x+100, y+100}, at::kFloat).reshape({-1,2})}, 0).unsqueeze(0);
    // auto trt_label = at::concatenate({input_label, at::tensor({2,3}, at::kFloat)}, 0).unsqueeze(0);

    std::cout << "trt_coord: " << trt_coord << std::endl;
    std::cout << "trt_label: " << trt_label << std::endl;

    trt_coord = transf.apply_coords(trt_coord, {frame.rows, frame.cols});
    // std::cout << "trt_coord " << trt_coord.sizes() << std::endl;
    auto trt_mask_input = at::zeros({1, 1, 256, 256}, at::kFloat);
    auto trt_has_mask_input = at::zeros(1, at::kFloat);

    context->setBindingDimensions(1, nvinfer1::Dims3(trt_coord.size(0), trt_coord.size(1), trt_coord.size(2)));
    // set input dims whichs name "point_label "
    context->setBindingDimensions(2, nvinfer1::Dims2(trt_coord.size(0), trt_coord.size(1)));
    int nbopts = mEngine->getNbOptimizationProfiles();
    std::cout << "nboopts: " << nbopts << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        // auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        auto dims = context->getBindingDimensions(i);
        dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        index2srt(type);
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});

        mInOut[tensor_name]->resize(dims);
    }

    CHECK(mInOut["image_embeddings"]->host2device((void *)(image_embeddings.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["point_coords"]->host2device((void *)(trt_coord.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["point_labels"]->host2device((void *)(trt_label.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["mask_input"]->host2device((void *)(trt_mask_input.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["has_mask_input"]->host2device((void *)(trt_has_mask_input.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    return 0;
}

int SamPromptEncoderAndMaskDecoder::prepareInput(std::vector<int> mult_pts, at::Tensor image_embeddings)
{
    int image_size = 1024;
    ResizeLongestSide transf(image_size);

    auto input_point = at::tensor(mult_pts, at::kFloat).reshape({ -1,2 });
    std::cout << input_point << std::endl;
    auto input_label = at::ones({ int(mult_pts.size() / 2) }, at::kFloat);
    std::cout << input_label << std::endl;

    auto trt_coord = at::concatenate({ input_point, at::tensor({0, 0}, at::kFloat).unsqueeze(0) }, 0).unsqueeze(0);
    auto trt_label = at::concatenate({ input_label, at::tensor({-1}, at::kFloat) }, 0).unsqueeze(0);
    trt_coord = transf.apply_coords(trt_coord, { frame.rows, frame.cols });
    std::cout << "trt_coord " << trt_coord.sizes() << std::endl;
    auto trt_mask_input = at::zeros({ 1, 1, 256, 256 }, at::kFloat);
    auto trt_has_mask_input = at::zeros(1, at::kFloat);

    context->setBindingDimensions(1, nvinfer1::Dims3(trt_coord.size(0), trt_coord.size(1), trt_coord.size(2)));
    // set input dims whichs name "point_label "
    context->setBindingDimensions(2, nvinfer1::Dims2(trt_coord.size(0), trt_coord.size(1)));

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        // auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        auto dims = context->getBindingDimensions(i);
        dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        index2srt(type);
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{ 1 }, std::multiplies<int64_t>{});

        mInOut[tensor_name]->resize(dims);
    }

    CHECK(mInOut["image_embeddings"]->host2device((void*)(image_embeddings.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["point_coords"]->host2device((void*)(trt_coord.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["point_labels"]->host2device((void*)(trt_label.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["mask_input"]->host2device((void*)(trt_mask_input.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(mInOut["has_mask_input"]->host2device((void*)(trt_has_mask_input.data_ptr<float>()), true, stream));
    CHECK(cudaStreamSynchronize(stream));
    return 0;
}

bool SamPromptEncoderAndMaskDecoder::infer()
{
    CHECK(cudaEventRecord(start, stream));
    auto ret = context->enqueueV2(mDeviceBindings.data(), stream, nullptr);
    return ret;
}

void writeTensorToFile(const at::Tensor& tensor, const std::string& filename) {
    // Convert tensor to contiguous form
    at::Tensor contiguousTensor = tensor.contiguous();

    // Open the file in binary mode
    std::ofstream outFile(filename, std::ios::binary);

    // Check if the file is open
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the tensor data to the file
    outFile.write(reinterpret_cast<const char*>(contiguousTensor.data_ptr<float>()), contiguousTensor.numel() * sizeof(float));

    // Close the file
    outFile.close();
}

int SamPromptEncoderAndMaskDecoder::verifyOutput()
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape("masks");
    auto dim1 = mEngine->getTensorShape("scores");
    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor masks;
    masks = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["masks"]->device2host((void *)(masks.data_ptr<float>()), stream);

    std::string filePath = "weights/masks_data.bin";
    //at::Tensor contiguousTensor = masks.contiguous();
    //torch::save(masks, filePath2);
    writeTensorToFile(masks, filePath);

    //std::cout << masks << std::endl;
    // Wait for the work in the stream to complete

    CHECK(cudaStreamSynchronize(stream));

    int longest_side = 1024;
  
    namespace F = torch::nn::functional;
    //    auto temp = F::InterpolateFuncOptions().size(std::vector<int64_t>({ longest_side , longest_side })).mode(torch::kBilinear).align_corners(false);
    //auto temp = F::InterpolateFuncOptions().scale_factor(std::vector<double_t>({ 2.0 })).mode(torch::kBilinear).align_corners(false);
    //masks = F::interpolate(masks, temp);
    //at::IntArrayRef input_image_size{frame.rows, frame.cols};
    masks = F::interpolate(masks,F::InterpolateFuncOptions().size(std::vector<int64_t>({ 512,512 })).mode(torch::kNearest));
    ResizeLongestSide transf(longest_side);
    auto target_size = transf.get_preprocess_shape(frame.rows, frame.cols);
    masks = masks.index({"...", Slice(None, target_size[0]), Slice(None, target_size[1])});

   /* masks = F::interpolate(masks, F::InterpolateFuncOptions().size(std::vector<int64_t>({ frame.rows, frame.cols })).mode(torch::kNearest).align_corners(false));
    std::cout << "masks: " << masks.sizes() << std::endl;*/



    at::Tensor iou_predictions;
    iou_predictions = at::zeros({dim0.d[0], dim0.d[1]}, at::kFloat);
    mInOut["scores"]->device2host((void *)(iou_predictions.data_ptr<float>()), stream);
    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);
    masks = masks.gt(0.) * 1.0;
    std::cout << "max " << masks.max() << std::endl;
    // masks = masks.sigmoid();
    std::cout << "masks: " << masks.sizes() << std::endl;
    masks = masks.to(device);
    std::cout << "iou_predictions: " << iou_predictions << std::endl;
    cv::Mat img;
    // cv::Mat frame = cv::imread("D:/projects/detections/data/truck.jpg");
    frame.convertTo(img, CV_32F, 1.0 / 255);
    int r = img.channels();
    at::Tensor im_gpu =
        at::from_blob(img.data, {img.rows, img.cols, img.channels()})
            .permute({2, 0, 1})
            .contiguous()
            .unsqueeze(0)
            .to(device);
    float x = 0.5;

    auto results = plot_masks(masks, im_gpu, x);
    auto t_img = results.to(torch::kCPU).clamp(0, 255).to(torch::kU8);

    auto img_ = cv::Mat(t_img.size(0), t_img.size(1), CV_8UC3, t_img.data_ptr<uchar>());
    std::cout << "1111111111111111" << std::endl;
    //cv::cvtColor(img_, img_, cv::COLOR_RGB2BGR);
    cv::imwrite("img1111.jpg",img_);
    // cv::imshow("img_", img_);
    return 0;
}

int SamPromptEncoderAndMaskDecoder::verifyOutput(cv::Mat& roi)
{
    float ms{0.0f};
    CHECK(cudaEventRecord(end, stream));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&ms, start, end));

    auto dim0 = mEngine->getTensorShape("masks");
    auto dim1 = mEngine->getTensorShape("scores");
    // dims2str(dim0);
    // dims2str(dim1);
    at::Tensor masks;
    masks = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["masks"]->device2host((void *)(masks.data_ptr<float>()), stream);
    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    int longest_side = 1024;

    namespace F = torch::nn::functional;
    masks = F::interpolate(masks, F::InterpolateFuncOptions().size(std::vector<int64_t>({longest_side, longest_side})).mode(torch::kBilinear).align_corners(false));
    // at::IntArrayRef input_image_size{frame.rows, frame.cols};
    ResizeLongestSide transf(longest_side);
    auto target_size = transf.get_preprocess_shape(frame.rows, frame.cols);
    masks = masks.index({"...", Slice(None, target_size[0]), Slice(None, target_size[1])});

    masks = F::interpolate(masks, F::InterpolateFuncOptions().size(std::vector<int64_t>({frame.rows, frame.cols})).mode(torch::kBilinear).align_corners(false));
    // std::cout << "masks: " << masks.sizes() << std::endl;

    at::Tensor iou_predictions;
    iou_predictions = at::zeros({dim0.d[0], dim0.d[1]}, at::kFloat);
    mInOut["scores"]->device2host((void *)(iou_predictions.data_ptr<float>()), stream);
    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    masks = masks.gt(0.) * 1.0;
    masks = masks.squeeze(0).squeeze(0);
    masks = masks.to(torch::kCPU).to(torch::kU8);
    std::cout << "masks: " << masks.sizes() << std::endl;
    auto roi_ = cv::Mat(masks.size(0), masks.size(1), CV_8U, masks.data_ptr<uchar>());
    roi_.copyTo(roi);

    return 0;
}

/*
    return [r g b] * n
*/
at::Tensor SamPromptEncoderAndMaskDecoder::generator_colors(int num)
{

    std::vector<std::string> hexs = {"FF37C7", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                                     "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "", "520085", "CB38FF", "FF95C8", "FF3838"};

    std::vector<int> tmp;
    for (int i = 0; i < num; ++i)
    {
        int r = string2Num<int>(hexs[i].substr(0, 2));
        // std::cout << r << std::endl;
        int g = string2Num<int>(hexs[i].substr(2, 2));
        // std::cout << g << std::endl;
        int b = string2Num<int>(hexs[i].substr(4, 2));
        // std::cout << b << std::endl;
        tmp.emplace_back(r);
        tmp.emplace_back(g);
        tmp.emplace_back(b);
    }
    return at::from_blob(tmp.data(), {(int)tmp.size()}, at::TensorOptions(at::kInt));
}

template <class Type>
Type SamPromptEncoderAndMaskDecoder::string2Num(const std::string &str)
{
    std::istringstream iss(str);
    Type num;
    iss >> std::hex >> num;
    return num;
}

/*
        Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
*/

at::Tensor SamPromptEncoderAndMaskDecoder::plot_masks(at::Tensor masks, at::Tensor im_gpu, float alpha)
{
    int n = masks.size(0);
    auto colors = generator_colors(n);
    colors = colors.to(masks.device()).to(at::kFloat).div(255).reshape({-1, 3}).unsqueeze(1).unsqueeze(2);
    // std::cout << "colors: " << colors.sizes() << std::endl;
    masks = masks.permute({0, 2, 3, 1}).contiguous();
    // std::cout << "masks: " << masks.sizes() << std::endl;
    auto masks_color = masks * (colors * alpha);
    // std::cout << "masks_color: " << masks_color.sizes() << std::endl;
    auto inv_alph_masks = (1 - masks * alpha);
    inv_alph_masks = inv_alph_masks.cumprod(0);
    // std::cout << "inv_alph_masks: " << inv_alph_masks.sizes() << std::endl;

    //image*alpha + mask(1-alpha)
    auto mcs = masks_color * inv_alph_masks;
    mcs = mcs.sum(0) * 2;
    std::cout << "mcs: " << mcs.sizes() << std::endl;
    im_gpu = im_gpu.flip({0});
    std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
    im_gpu = im_gpu.permute({1, 2, 0}).contiguous();
    std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
    //im_gpu = im_gpu * inv_alph_masks[-1] + mcs;
    //im_gpu = im_gpu * inv_alph_masks.index({ "...", Slice(None, inv_alph_masks.size(2)), Slice(None, inv_alph_masks.size(3)) }) + mcs;
    im_gpu = im_gpu * inv_alph_masks.index({ "..." }).slice(2, 0, inv_alph_masks.size(2)).slice(3, 0, inv_alph_masks.size(3)) + mcs;


    // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
    auto im_mask = (im_gpu * 255);
    return im_mask;
}

#endif