#include "detect_armour/detect.hpp"

#include <fstream>

std::vector<GridAndStride> grid_strides;

/***
 * @description: argmax, 获取数组中最大值的索引
 * @param ptr: 数组
 * @param len: 数组长度t
 * @return *: 最大值索引
 */
static int argmax(const float * ptr, int len)
{
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }

    return max_arg;
}

/***
 * @description: 对即将送入神经网络的图像进行缩放
 * @param img: 原图
 * @return *:
 */
inline cv::Mat scaleResize(cv::Mat & img)
{
    // 最小缩放比例
    float r     = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    int unpad_w = int(r * img.cols);
    int unpad_h = int(r * img.rows);

    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;

    dw /= 2;
    dh /= 2;

    //  transform_matrix << 1.0 / r, 0, -dw / r, //
    //      0, 1.0 / r, -dh / r,                 //
    //      0, 0, 1;                             //

    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    //  cv::copyMakeBorder(re, out, dh, dh, dw, dw, cv::BORDER_CONSTANT);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    return out;
}

/***
 * @description
 * @param target_w: width
 * @param target_h: height
 * @param strides: 步长数组
 * @param grid_strides: grid_strides数组, 返回值
 * @return *:
 */
static void generate_grids_and_stride(
    const int target_w, const int target_h, std::vector<int> & strides,
    std::vector<GridAndStride> & grid_strides)
{
    for (int stride : strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;

        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.emplace_back((GridAndStride){g0, g1, stride});
            }
        }
    }

    return;
}

/***
 * @description:
 * @param grid_stride: grid stride
 * @param feat_ptr: 图像数组
 * @param transfrom_matrix: 变换矩阵
 * @param prob_threshold: 置信度阈值
 * @param objects: 返回值
 * @return *:
 */
static void generateYoloxProposals(
    const std::vector<GridAndStride> & grid_stride, const float * feat_ptr, float prob_threshold,
    std::vector<ArmorObject> & objects)
{
    const int num_anchors = grid_stride.size();
    // 遍历所有anchor
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int grid0  = grid_stride[anchor_idx].grid0;
        const int grid1  = grid_stride[anchor_idx].grid1;
        const int stride = grid_stride[anchor_idx].stride;

        // points(2 * APEX_NUM) + conf(1) + colors + classes
        const int basic_pos = anchor_idx * (2 * APEX_NUM + 1 + NUM_COLORS + NUM_CLASSES);

        Eigen::Matrix<float, 2, APEX_NUM> apex_norm;
        for (int i = 0; i < APEX_NUM; i++) {
            apex_norm(0, i) = (feat_ptr[basic_pos + i * 2] + grid0) * stride;
            apex_norm(1, i) = (feat_ptr[basic_pos + i * 2 + 1] + grid1) * stride;
        }

        int box_color = argmax(feat_ptr + basic_pos + 2 * APEX_NUM + 1, NUM_COLORS);
        int box_class = argmax(feat_ptr + basic_pos + 2 * APEX_NUM + 1 + NUM_COLORS, NUM_CLASSES);

        float box_objectness = (feat_ptr[basic_pos + 2 * APEX_NUM]);

        float color_conf = (feat_ptr[basic_pos + 2 * APEX_NUM + 1 + box_color]);
        float cls_conf   = (feat_ptr[basic_pos + 2 * APEX_NUM + 1 + NUM_COLORS + box_class]);

        float box_prob = box_objectness;
        if (box_prob > 1) {
            int i = 0;
        }
        if (box_prob >= prob_threshold) {
            ArmorObject obj;

            // for (int i = 0; i < 5; i++)
            //   obj.apex[i] = cv::Point2f(apex_dst(0, i), apex_dst(1, i));
            for (int i = 0; i < APEX_NUM; i++) {
                //        obj.apex[i] = cv::Point2f(apex_dst(0, i), apex_dst(1, i));
                obj.apex[i] = cv::Point2f(apex_norm(0, i), apex_norm(1, i));
                obj.pts.push_back(obj.apex[i]);
            }
            std::vector<cv::Point2f> tmp(obj.apex, obj.apex + APEX_NUM);
            obj.rect = cv::boundingRect(tmp);

            obj.cls   = box_class;
            obj.color = box_color;
            obj.prob  = box_prob;

            objects.push_back(obj);
        }
    }
}

/***
 * @description: 计算两个buffObject的交集
 * @param a:
 * @param b:
 * @return *:
 */
static inline float intersection_area(const ArmorObject & a, const ArmorObject & b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

/***
 * @description: 对目标数组按置信度从高到底快速排序, 并行、递归
 * @param left:
 * @param right:
 * @return *:
 */
static void qsort_descent_inplace(std::vector<ArmorObject> & objects, int left, int right)
{
    int i   = left;
    int j   = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p) i++;

        while (objects[j].prob < p) j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<ArmorObject> & objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

/***
 * @description: NMS
 * @param objects:
 * @param picked:
 * @param nms_threshold: NMS 阈值
 * @return *:
 */
static void nms_sorted_bboxes(
    std::vector<ArmorObject> & objects, std::vector<int> & picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        std::vector<cv::Point2f> object_apex_temp(objects[i].apex, objects[i].apex + APEX_NUM);

        //    areas[i] = cv::contourArea(object_apex_temp);

        cv::Rect_<float> r = cv::boundingRect(object_apex_temp);
        areas[i]           = r.area();
    }

    for (int i = 0; i < n; i++) {
        ArmorObject & a = objects[i];
        std::vector<cv::Point2f> apex_a(a.apex, a.apex + APEX_NUM);
        int keep = 1;

        for (int j = 0; j < (int)picked.size(); j++) {
            ArmorObject & b = objects[picked[j]];
            std::vector<cv::Point2f> apex_b(b.apex, b.apex + APEX_NUM);
            std::vector<cv::Point2f> apex_inter;
            // intersection over union
            float inter_area = intersection_area(a, b);
            // float union_area = areas[i] + areas[picked[j]] - inter_area;
            // TODO:此处耗时较长，大约1ms，可以尝试使用其他方法计算IOU与多边形面积
            // float inter_area = cv::intersectConvexConvex(apex_a, apex_b,
            // apex_inter);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou        = inter_area / union_area;

            if (iou > nms_threshold || isnan(iou)) {
                keep = 0;
                if (iou > MERGE_MIN_IOU && abs(a.prob - b.prob) < MERGE_CONF_ERROR &&
                    a.cls == b.cls && a.color == b.color) {
                    for (int p = 0; p < APEX_NUM; p++) b.pts.push_back(a.apex[p]);
                }
                break;
            }
        }

        if (keep) picked.push_back(i);
    }
}

static void decodeOutputs(
    const float * prob, std::vector<ArmorObject> & object, float scale, const int img_w,
    const int img_h)
{
    std::vector<ArmorObject> proposals;
    std::vector<int> strides = {8, 16, 32};

    if (grid_strides.size() == 0) {
        generate_grids_and_stride(img_w, img_h, strides, grid_strides);
    }

    generateYoloxProposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);
    qsort_descent_inplace(proposals);

    if (proposals.size() > TOPK) proposals.resize(TOPK);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    int count = picked.size();
    object.resize(count);

    for (int i = 0; i < count; i++) {
        object[i] = proposals[picked[i]];

        for (int p = 0; p < APEX_NUM; p++) {
            object[i].apex[p] /= scale;
        }
        for (auto p : object[i].pts) {
            p /= scale;
        }
    }
}

namespace Modules
{
Detector::Detector(const std::string & model_path)
{
    network = ie.ReadNetwork(model_path);

    // Step 1. Read a model in OpenVINO Intermediate Representation
    // (.xml and .bin files) or ONNX (.onnx file) format
    if (network.getOutputsInfo().size() != 1)
        throw std::logic_error("Sample supports topologies with 1 output only");

    // Step 2. Configure input & output
    //  Prepare input blobs
    InferenceEngine::InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_name                                 = network.getInputsInfo().begin()->first;

    std::cout << "input name = " << input_name << std::endl;

    //  Prepare output blobs
    if (network.getOutputsInfo().empty()) {
        std::cerr << "Network outputs info is empty" << std::endl;
        return;
    }
    InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
    output_name                          = network.getOutputsInfo().begin()->first;
    std::cout << "output name = " << output_name << std::endl;

    // output_info->setPrecision(Precision::FP16);
    // Step 3. Loading a model to the device
    // executable_network = ie.LoadNetwork(network, "MULTI:GPU");
    // executable_network = ie.LoadNetwork(network, "GPU");
    executable_network = ie.LoadNetwork(network, "CPU");

    // Step 4. Create an infer request
    infer_request                                = executable_network.CreateInferRequest();
    const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
    // Blob::Ptr input = infer_request.GetBlob(input_name);     // just wrap Mat
    // data by Blob::Ptr
    if (!moutput) {
        throw std::logic_error(
            "We expect output to be inherited from MemoryBlob, "
            "but by fact we were not able to cast output to MemoryBlob");
    }
}

bool Detector::detect(Detection_pack & detection_pack)
{
    if (detection_pack.img.empty()) {
        return false;
    }

    cv::Mat pr_img = scaleResize(detection_pack.img);
    cv::imshow("pr_img", pr_img);

    // U8 转 FP32
    cv::Mat pre;
    cv::Mat pre_split[3];
    pr_img.convertTo(pre, CV_32FC3);
    // 分离颜色通道, 便于拷贝内存
    cv::split(pre, pre_split);

    InferenceEngine::Blob::Ptr imgBlob = infer_request.GetBlob(input_name);

    InferenceEngine::MemoryBlob::Ptr mblob =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(imgBlob);

    auto mblobHolder  = mblob->wmap();
    float * blob_data = mblobHolder.as<float *>();

    auto img_offset = INPUT_W * INPUT_H;
    // Copy img to blob
    for (int c = 0; c < 3; c++) {
        memcpy(blob_data + c * img_offset, pre_split[c].data, INPUT_W * INPUT_H * sizeof(float));
    }

    infer_request.Infer();

    auto moutputHolder     = moutput->rmap();
    const float * net_pred = moutputHolder.as<
        const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    float scale = std::min(
        INPUT_W / (detection_pack.img.cols * 1.0), INPUT_H / (detection_pack.img.rows * 1.0));

    decodeOutputs(net_pred, detection_pack.armors, scale, INPUT_W, INPUT_H);

    if (!detection_pack.armors.empty())
        return true;
    else
        return false;
}

Detector::~Detector()
{
}

}  // namespace Modules