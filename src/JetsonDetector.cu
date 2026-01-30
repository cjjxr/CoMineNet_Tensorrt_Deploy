#include "JetsonDetector.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <device_launch_parameters.h>

#define CHECK(status) { if (status != 0) { std::cerr << "CUDA Error: " << status << std::endl; abort(); } }

namespace {
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
        }
    } gLogger;

    // 距离映射函数
    float get_distance_meters(float raw_val) {
        const float scale = 1.0f; 
        return std::clamp(raw_val * scale, 0.1f, 50.0f);
    }

    // 鲁棒深度计算逻辑
    float calculate_robust_depth(const cv::Mat& depth_roi) {
        if (depth_roi.empty()) return 0.0f;
        std::vector<float> data;
        for (int i = 0; i < depth_roi.rows; ++i) {
            for (int j = 0; j < depth_roi.cols; ++j) {
                float val = depth_roi.at<float>(i, j);
                if (val > 0.1f) data.push_back(val);
            }
        }
        if (data.size() < 10) return 0.0f;
        try {
            cv::Mat samples(data.size(), 1, CV_32F);
            for (size_t i = 0; i < data.size(); ++i) samples.at<float>(i) = data[i];
            cv::Mat labels, centers;
            cv::kmeans(samples, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
            return std::min(centers.at<float>(0), centers.at<float>(1)); 
        } catch (...) {
            std::nth_element(data.begin(), data.begin() + (int)(data.size() * 0.25), data.end());
            return data[(int)(data.size() * 0.25)];
        }
    }

    // CUDA 预处理：自适应 Letterbox (BGR2RGB + Resize + Pad + Normalize)
    __global__ void preprocess_kernel_adaptive(uint8_t* src, float* dst, int src_w, int src_h, int dst_w, int dst_h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= dst_w || y >= dst_h) return;

        float scale = min((float)dst_w / src_w, (float)dst_h / src_h);
        int nw = (int)(src_w * scale); 
        int nh = (int)(src_h * scale);
        int dx = (dst_w - nw) / 2; 
        int dy = (dst_h - nh) / 2;

        int idx_r = 0 * dst_h * dst_w + y * dst_w + x;
        int idx_g = 1 * dst_h * dst_w + y * dst_w + x;
        int idx_b = 2 * dst_h * dst_w + y * dst_w + x;

        if (x >= dx && x < dx + nw && y >= dy && y < dy + nh) {
            int src_x = (int)((x - dx) / scale); 
            int src_y = (int)((y - dy) / scale);
            src_x = min(src_x, src_w - 1);
            src_y = min(src_y, src_h - 1);
            int src_idx = (src_y * src_w + src_x) * 3;
            // BGR to RGB 归一化
            dst[idx_r] = src[src_idx + 2] / 255.0f;
            dst[idx_g] = src[src_idx + 1] / 255.0f;
            dst[idx_b] = src[src_idx + 0] / 255.0f;
        } else {
            // 背景填充 (Gray)
            dst[idx_r] = 0.447f; dst[idx_g] = 0.447f; dst[idx_b] = 0.447f;
        }
    }

    // 自适应分割掩码 Kernel：解决补边导致的偏移 
    __global__ void overlay_mask_kernel_adaptive(uint8_t* img, const float* mask, int img_w, int img_h, 
                                                int mask_w, int mask_h, int net_w, int net_h,
                                                int dw, int dh, float scale) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= img_w || y >= img_h) return;

        // 1. 映射到网络输入的坐标系
        float net_x = (float)x * scale + (float)dw;
        float net_y = (float)y * scale + (float)dh;

        // 2. 边界判定准则：仅在有效图像视野内采样
        if (net_x >= (float)dw && net_x < (float)(net_w - dw) && 
            net_y >= (float)dh && net_y < (float)(net_h - dh)) {
            
            // 3. 映射到分割张量索引
            int mx = (int)(net_x * (float)mask_w / (float)net_w);
            int my = (int)(net_y * (float)mask_h / (float)net_h);

            if (mx >= 0 && mx < mask_w && my >= 0 && my < mask_h) {
                int mask_idx = my * mask_w + mx;
                if (mask[mask_idx] > 0.65f) { // 阈值过滤环境噪声
                    int img_idx = (y * img_w + x) * 3;
                    img[img_idx + 1] = (uint8_t)(img[img_idx + 1] * 0.5f + 127.5f); // 渲染绿色路面
                }
            }
        }
    }
}

JetsonDetector::JetsonDetector(const std::string& engine_path) {
    CHECK(cudaStreamCreate(&m_stream));
    load_engine(engine_path);
}

JetsonDetector::~JetsonDetector() {
    cudaStreamDestroy(m_stream);
    for (auto& p : m_buffers) cudaFree(p.second);
    if (m_gpu_img_buffer) cudaFree(m_gpu_img_buffer);
    if (m_cpu_det_buffer) cudaFreeHost(m_cpu_det_buffer);
}

void JetsonDetector::load_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) exit(1);
    
    file.seekg(0, file.end); size_t size = file.tellg(); file.seekg(0, file.beg);
    std::vector<char> dat(size); file.read(dat.data(), size);
    
    m_runtime.reset(nvinfer1::createInferRuntime(gLogger));
    m_engine.reset(m_runtime->deserializeCudaEngine(dat.data(), size));
    m_context.reset(m_engine->createExecutionContext());

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_engine->getTensorShape(name);
        size_t sz = 1;
        for(int j=0; j<dims.nbDims; ++j) {
            if(dims.d[j] < 0) dims.d[j] = 1; 
            sz *= dims.d[j];
        }
        void* ptr; CHECK(cudaMalloc(&ptr, sz * sizeof(float)));
        m_buffers[name] = ptr;

        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            m_in_name = name;
            m_input_h = dims.d[2]; m_input_w = dims.d[3];
            std::cout << "[Adaptive Perception] Engine size: " << m_input_w << "x" << m_input_h << std::endl;
        } else if (dims.nbDims == 3) { // Detection
            m_det_name = name; m_det_size = sz * sizeof(float); m_det_dims = dims;
            CHECK(cudaMallocHost(&m_cpu_det_buffer, m_det_size));
        } else if (dims.nbDims == 4) { // Segmentation
            m_seg_name = name; m_seg_dims = dims; m_has_seg = true; 
        }
    }
}

void JetsonDetector::process_frame(cv::Mat& img, DepthAnything& depth_engine) {
    if (img.empty()) return;
    prepare_buffers(img.cols, img.rows);
    CHECK(cudaMemcpyAsync(m_gpu_img_buffer, img.data, img.total() * 3, cudaMemcpyHostToDevice, m_stream));
    
    // 1. 动态计算缩放参数
    float scale = std::min((float)m_input_w / img.cols, (float)m_input_h / img.rows);
    int dw = (m_input_w - (int)(img.cols * scale)) / 2;
    int dh = (m_input_h - (int)(img.rows * scale)) / 2;

    // 2. 自适应预处理
    dim3 blockSize(16, 16);
    dim3 gridSize((m_input_w + 15) / 16, (m_input_h + 15) / 16);
    preprocess_kernel_adaptive<<<gridSize, blockSize, 0, m_stream>>>(m_gpu_img_buffer, (float*)m_buffers[m_in_name], img.cols, img.rows, m_input_w, m_input_h);
    
    // 3. 推理
    for (auto& p : m_buffers) m_context->setTensorAddress(p.first.c_str(), p.second);
    m_context->enqueueV3(m_stream);
    
    // 4. 同步执行深度估计
    cv::Mat depth_map = depth_engine.predict_float(img);

    // 5. 分割掩码叠加 (自适应映射)
    if (m_has_seg) {
        int mask_h = m_seg_dims.d[2]; int mask_w = m_seg_dims.d[3];
        dim3 maskGrid((img.cols + 15) / 16, (img.rows + 15) / 16);
        overlay_mask_kernel_adaptive<<<maskGrid, blockSize, 0, m_stream>>>(
            m_gpu_img_buffer, (float*)m_buffers[m_seg_name], 
            img.cols, img.rows, mask_w, mask_h, m_input_w, m_input_h,
            dw, dh, scale
        );
    }

    CHECK(cudaMemcpyAsync(m_cpu_det_buffer, m_buffers[m_det_name], m_det_size, cudaMemcpyDeviceToHost, m_stream));
    CHECK(cudaMemcpyAsync(img.data, m_gpu_img_buffer, img.total() * 3, cudaMemcpyDeviceToHost, m_stream));
    CHECK(cudaStreamSynchronize(m_stream));

    draw_results(img, depth_map, m_cpu_det_buffer);
}

void JetsonDetector::draw_results(cv::Mat& img, const cv::Mat& depth_map, float* det_data) {
    float scale = std::min((float)m_input_w / img.cols, (float)m_input_h / img.rows);
    float dx = (m_input_w - img.cols * scale) / 2.0f;
    float dy = (m_input_h - img.rows * scale) / 2.0f;
    float dsw = (float)depth_map.cols / img.cols, dsh = (float)depth_map.rows / img.rows;

    for (int i = 0; i < m_det_dims.d[1]; ++i) {
        float* feat = det_data + i * m_det_dims.d[2];
        if (feat[4] > 0.25f) {
            float cx = (feat[0] * m_input_w - dx) / scale;
            float cy = (feat[1] * m_input_h - dy) / scale;
            float w = (feat[2] * m_input_w) / scale, h = (feat[3] * m_input_h) / scale;

            cv::Rect box((int)(cx - w*0.5f), (int)(cy - h*0.5f), (int)w, (int)h);
            box &= cv::Rect(0, 0, img.cols, img.rows);
            
            if (box.area() > 0) {
                cv::Mat roi = depth_map(cv::Rect((int)(box.x*dsw), (int)(box.y*dsh), (int)(box.width*dsw), (int)(box.height*dsh)));
                float dist = get_distance_meters(calculate_robust_depth(roi));
                cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);
                cv::putText(img, std::to_string(dist).substr(0,4)+"m", cv::Point(box.x, box.y-10), 0, 0.6, cv::Scalar(0,255,0), 2);
            }
        }
    }
}

void JetsonDetector::prepare_buffers(int w, int h) {
    size_t size = w * h * 3;
    if (size > m_gpu_img_capacity) {
        if (m_gpu_img_buffer) cudaFree(m_gpu_img_buffer);
        CHECK(cudaMalloc(&m_gpu_img_buffer, size));
        m_gpu_img_capacity = size;
    }
}