#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include "DepthAnything.hpp"

class JetsonDetector {
public:
    JetsonDetector(const std::string& engine_path);
    ~JetsonDetector();

    void process_frame(cv::Mat& img, DepthAnything& depth_engine);

private:
    struct InferDeleter { template <typename T> void operator()(T* obj) const { delete obj; } };
    template <typename T> using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

    TrtUniquePtr<nvinfer1::IRuntime> m_runtime{nullptr};
    TrtUniquePtr<nvinfer1::ICudaEngine> m_engine{nullptr};
    TrtUniquePtr<nvinfer1::IExecutionContext> m_context{nullptr};
    cudaStream_t m_stream;
    std::map<std::string, void*> m_buffers;
    std::string m_in_name, m_det_name, m_seg_name;
    bool m_has_seg = false;
    size_t m_det_size = 0;
    uint8_t* m_gpu_img_buffer = nullptr;
    size_t m_gpu_img_capacity = 0;
    float* m_cpu_det_buffer = nullptr;
    nvinfer1::Dims m_seg_dims;
    nvinfer1::Dims m_det_dims;
    int m_input_w;
    int m_input_h;
    
    void load_engine(const std::string& path);
    void prepare_buffers(int w, int h);
    void draw_results(cv::Mat& img, const cv::Mat& depth_map, float* det_data);
};
