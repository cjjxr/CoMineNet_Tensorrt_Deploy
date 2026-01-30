#ifndef DEPTH_ANYTHING_HPP
#define DEPTH_ANYTHING_HPP

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

// 显式定义虚析构函数，防止 vtable 错误
class DALogger : public nvinfer1::ILogger {
public:
    DALogger() = default;
    virtual ~DALogger() = default; 
    void log(Severity severity, const char* msg) noexcept override;
};

class DepthAnything {
public:
    struct Config {
        std::string model_path;
        // 宽高现在由引擎自动识别，不再需要手动指定
    };

    DepthAnything(const Config& config);
    ~DepthAnything();
    
    // 返回 U8 类型的可视化深度图
    cv::Mat predict(const cv::Mat& image);
    // 返回原始 Float 类型的深度图（用于距离计算）
    cv::Mat predict_float(const cv::Mat& image);

private:
    void load_engine(const std::string& engine_path);
    void allocate_buffers();

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;
    DALogger logger_;

    int input_w_, input_h_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;

    // 显存指针
    void* device_input_ptr_ = nullptr;
    void* device_output_ptr_ = nullptr;
    void* device_src_ptr_ = nullptr;
    void* device_post_ptr_ = nullptr;
    unsigned char* host_output_ptr_ = nullptr;
    float* host_output_float_ptr_ = nullptr;
    
    size_t src_capacity_ = 0; // 用于动态管理输入图显存
};

#endif