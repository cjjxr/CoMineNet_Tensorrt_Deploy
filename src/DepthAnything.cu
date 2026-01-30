#include "DepthAnything.hpp"
#include <fstream>
#include <iostream>
#include <device_launch_parameters.h>

#define CHECK_DA(status) do { auto ret = (status); if (ret != 0) { std::cerr << "CUDA Error at line " << __LINE__ << ": " << ret << std::endl; abort(); } } while (0)

// CUDA 预处理：双线性插值 + 归一化
__global__ void preprocess_kernel_da(unsigned char* src, float* dst, int src_w, int src_h, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_w && y < dst_h) {
        float scale_x = (float)src_w / dst_w;
        float scale_y = (float)src_h / dst_h;
        float src_xf = (x + 0.5f) * scale_x - 0.5f;
        float src_yf = (y + 0.5f) * scale_y - 0.5f;
        int x0 = max(0, min((int)floorf(src_xf), src_w - 2));
        int y0 = max(0, min((int)floorf(src_yf), src_h - 2));
        float u = src_xf - x0; float v = src_yf - y0;

        float mean[] = {0.485f, 0.456f, 0.406f};
        float std[] = {0.229f, 0.224f, 0.225f};

        for (int c = 0; c < 3; ++c) {
            float p00 = src[(y0 * src_w + x0) * 3 + (2 - c)];
            float p10 = src[(y0 * src_w + x0 + 1) * 3 + (2 - c)];
            float p01 = src[((y0 + 1) * src_w + x0) * 3 + (2 - c)];
            float p11 = src[((y0 + 1) * src_w + x0 + 1) * 3 + (2 - c)];

            float val = (1.0f - u) * (1.0f - v) * p00 + u * (1.0f - v) * p10 + (1.0f - u) * v * p01 + u * v * p11;
            dst[c * dst_w * dst_h + y * dst_w + x] = (val / 255.0f - mean[c]) / std[c];
        }
    }
}

__global__ void postprocess_kernel_da(float* src, unsigned char* dst, int size, float max_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = fminf(src[idx], max_dist);
        dst[idx] = (unsigned char)((val / max_dist) * 255.0f);
    }
}

void DALogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) std::cout << "[DepthAnything] " << msg << std::endl;
}

DepthAnything::DepthAnything(const Config& config) {
    CHECK_DA(cudaStreamCreate(&stream_));
    load_engine(config.model_path);
    allocate_buffers();
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), [](nvinfer1::IExecutionContext* e){ delete e; });
}

void DepthAnything::load_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) { std::cerr << "FATAL: Could not open engine: " << path << std::endl; abort(); }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size), [](nvinfer1::ICudaEngine* e){ delete e; });

    // 自动获取维度
    input_tensor_name_ = engine_->getIOTensorName(0);
    output_tensor_name_ = engine_->getIOTensorName(1);
    nvinfer1::Dims in_dims = engine_->getTensorShape(input_tensor_name_.c_str());
    nvinfer1::Dims out_dims = engine_->getTensorShape(output_tensor_name_.c_str());
    
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];
    std::cout << "[DepthAnything] Engine Size Detected: " << input_w_ << "x" << input_h_ << std::endl;
}

void DepthAnything::allocate_buffers() {
    size_t model_in_sz = 3 * input_h_ * input_w_ * sizeof(float);
    size_t model_out_sz = input_h_ * input_w_ * sizeof(float);
    src_capacity_ = 3840 * 2160 * 3; // 预留 4K 空间

    CHECK_DA(cudaMalloc(&device_input_ptr_, model_in_sz));
    CHECK_DA(cudaMalloc(&device_output_ptr_, model_out_sz));
    CHECK_DA(cudaMalloc(&device_src_ptr_, src_capacity_));
    CHECK_DA(cudaMalloc(&device_post_ptr_, input_h_ * input_w_));
    CHECK_DA(cudaHostAlloc(&host_output_ptr_, input_h_ * input_w_, cudaHostAllocDefault));
    CHECK_DA(cudaHostAlloc(&host_output_float_ptr_, input_h_ * input_w_ * sizeof(float), cudaHostAllocDefault));
}

DepthAnything::~DepthAnything() {
    if (device_input_ptr_) cudaFree(device_input_ptr_);
    if (device_output_ptr_) cudaFree(device_output_ptr_);
    if (device_src_ptr_) cudaFree(device_src_ptr_);
    if (device_post_ptr_) cudaFree(device_post_ptr_);
    if (host_output_ptr_) cudaFreeHost(host_output_ptr_);
    if (host_output_float_ptr_) cudaFreeHost(host_output_float_ptr_);
    if (stream_) cudaStreamDestroy(stream_);
}

cv::Mat DepthAnything::predict_float(const cv::Mat& image) {
    if (image.empty()) return cv::Mat();
    
    size_t raw_size = image.total() * 3;
    if (raw_size > src_capacity_) { // 自动扩容防止溢出
        cudaFree(device_src_ptr_);
        src_capacity_ = raw_size;
        cudaMalloc(&device_src_ptr_, src_capacity_);
    }

    CHECK_DA(cudaMemcpyAsync(device_src_ptr_, image.data, raw_size, cudaMemcpyHostToDevice, stream_));

    dim3 block(16, 16);
    dim3 grid((input_w_ + block.x - 1) / block.x, (input_h_ + block.y - 1) / block.y);
    preprocess_kernel_da<<<grid, block, 0, stream_>>>((unsigned char*)device_src_ptr_, (float*)device_input_ptr_, 
                                                  image.cols, image.rows, input_w_, input_h_);

    context_->setTensorAddress(input_tensor_name_.c_str(), device_input_ptr_);
    context_->setTensorAddress(output_tensor_name_.c_str(), device_output_ptr_);
    context_->enqueueV3(stream_);

    int total_pixels = input_w_ * input_h_;
    CHECK_DA(cudaMemcpyAsync(host_output_float_ptr_, device_output_ptr_, total_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // 返回 Resize 回原图尺寸的深度图，确保对齐
    cv::Mat depth_map(input_h_, input_w_, CV_32F, host_output_float_ptr_);
    cv::Mat resized_depth;
    cv::resize(depth_map, resized_depth, image.size());
    return resized_depth.clone();
}

cv::Mat DepthAnything::predict(const cv::Mat& image) {
    cv::Mat float_depth = predict_float(image);
    cv::Mat u8_depth;
    double min_val, max_val;
    cv::minMaxLoc(float_depth, &min_val, &max_val);
    float_depth.convertTo(u8_depth, CV_8U, 255.0 / (max_val + 1e-6));
    return u8_depth;
}