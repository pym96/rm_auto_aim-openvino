#ifndef _MODULE_DETECT_HPP_
#define _MODULE_DETECT_HPP_

#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <set>
#include <unordered_set>

#include <inference_engine.hpp>

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

/***
 * @description: 网络推理用到的参数
 */

static constexpr int DEVICE                      = 0;  // GPU id
static constexpr int BATCH_SIZE                  = 1;

static constexpr int APEX_NUM = 4;

static constexpr int INPUT_W     = 416;  // 网络推理输入图像的宽
static constexpr int INPUT_H     = 416;  // 网络推理输入图像的高
static constexpr int NUM_CLASSES = 8;    // 分类数
static constexpr int NUM_COLORS  = 3;    // 颜色
static constexpr int TOPK        = 20;   // TOP K

static constexpr float NMS_THRESH       = 0.1;  // NMS 阈值
static constexpr float BBOX_CONF_THRESH = 0.6;  // bbox 阈值
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU    = 0.2;

static const char * INPUT_BLOB_NAME  = "input_0";
static const char * OUTPUT_BLOB_NAME = "output_0";

struct ArmorObject
{
    // 从左上 逆时针旋转
    cv::Point2f apex[APEX_NUM];
    cv::Rect_<float> rect;
    int cls;
    int color;
    float prob;
    std::vector<cv::Point2f> pts;
};

const std::unordered_map<int, std::string> BIG_ARMOR_CLASSES{{1, "1"}, {7, "Base"}};
const std::unordered_map<int, std::string> SMALL_ARMOR_CLASSES{{0, "G"}, {2, "2"}, {3, "3"},
                                                               {4, "4"}, {5, "5"}, {6, "O"}};
const std::unordered_map<int, std::string> ARMOR_CLASSES{{1, "1"}, {7, "Base"}, {0, "G"}, {2, "2"},
                                                         {3, "3"}, {4, "4"},    {5, "5"}, {6, "O"}};
const std::unordered_map<int, std::string> ARMOR_COLORS{{0, "Blue"}, {1, "Red"}, {2, "None"}};

namespace Modules
{
struct Detection_pack  //每帧的打包数据结构
{
    cv::Mat img;                      //图像
    double timestamp;                 //时间戳
    std::vector<ArmorObject> armors;  //装甲板

    Detection_pack() = default;
    Detection_pack(cv::Mat & img, double timestamp) : img(img), timestamp(timestamp)
    {
        // armours.reserve(30000);
    }
};

class Detector
{
public:
    Detector(const std::string & model_path);
    ~Detector();

    bool detect(Detection_pack & detection_pack);

    InferenceEngine::Core ie;                               // 网络
    InferenceEngine::CNNNetwork network;                    // 可执行网络
    InferenceEngine::ExecutableNetwork executable_network;  // 推理请求
    InferenceEngine::InferRequest infer_request;
    InferenceEngine::MemoryBlob::CPtr moutput;

    std::string input_name;
    std::string output_name;
    int inputIndex, outputIndex;
    std::string engineName;
};
}  // namespace Modules

#endif