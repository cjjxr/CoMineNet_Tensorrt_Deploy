#include <iostream>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "DepthAnything.hpp"
#include "JetsonDetector.hpp"

namespace fs = std::filesystem;

int main() {
    std::string config_path = "../config/config.yaml";
    if (!fs::exists(config_path)) {
        std::cerr << "Config file not found!" << std::endl;
        return 1;
    }

    YAML::Node config = YAML::LoadFile(config_path);

    std::string input_source = config["common"]["input_dir"].as<std::string>();
    std::string save_dir = config["common"]["output_dir"].as<std::string>();
    std::string det_model = config["CoMineNet"]["model_path"].as<std::string>();
    std::string depth_model = config["depth_anything"]["model_path"].as<std::string>();

    // 初始化：现在两者都只需要路径即可
    JetsonDetector detector(det_model);
    
    DepthAnything::Config da_cfg;
    da_cfg.model_path = depth_model;
    DepthAnything depth_engine(da_cfg);

    fs::create_directories(save_dir);

    if (fs::is_directory(input_source)) {
        std::cout << ">>> Processing Folder: " << input_source << std::endl;
        for (const auto& e : fs::directory_iterator(input_source)) {
            std::string ext = e.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                cv::Mat img = cv::imread(e.path().string());
                if (img.empty()) continue;

                // 推理并绘制
                detector.process_frame(img, depth_engine);
                
                cv::imwrite(save_dir + "/" + e.path().filename().string(), img);
            }
        }
    } else {
        cv::VideoCapture cap(input_source);
        if (!cap.isOpened()) return 1;

        cv::VideoWriter writer(save_dir + "/output.mp4", 
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                               cap.get(cv::CAP_PROP_FPS), 
                               cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat frame;
        while (cap.read(frame)) {
            detector.process_frame(frame, depth_engine);
            writer.write(frame);
            std::cout << "." << std::flush;
        }
        cap.release();
        writer.release();
    }

    std::cout << "\nFinished. Results saved to: " << save_dir << std::endl;
    return 0;
}