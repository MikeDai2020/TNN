//
// Created by tencent on 2020-12-18.
//
#include "blazepose_detector.h"
#include "blazepose_landmark.h"
#include "pose_detect_landmark.h"
#include "tnn/utils/mat_utils.h"
#include "utils/utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

#include <opencv2/opencv.hpp>


static std::shared_ptr<TNN_NS::PoseDetectLandmark> gDetector;
static std::shared_ptr<TNN_NS::PoseDetectLandmark> gFullBodyDetector;
static std::shared_ptr<TNN_NS::BlazePoseDetector> gBlazePoseDetector;
static std::shared_ptr<TNN_NS::BlazePoseLandmark> gBlazePoseLandmark;
static std::shared_ptr<TNN_NS::BlazePoseLandmark> gBlazePoseFullBodyLandmark;

using namespace cv;

int init(std::string modelPath) 
{
    //setBenchResult("");
    gDetector                  = std::make_shared<TNN_NS::PoseDetectLandmark>();
    gFullBodyDetector          = std::make_shared<TNN_NS::PoseDetectLandmark>();
    gBlazePoseDetector         = std::make_shared<TNN_NS::BlazePoseDetector>();
    gBlazePoseLandmark         = std::make_shared<TNN_NS::BlazePoseLandmark>();
    gBlazePoseFullBodyLandmark = std::make_shared<TNN_NS::BlazePoseLandmark>();
    std::string protoContent, modelContent;
    protoContent = fdLoadFile(modelPath + "/pose_detection.tnnproto");
    modelContent = fdLoadFile(modelPath + "/pose_detection.tnnmodel");
    printf("pose detection proto content size %lld model content size %lld\n", protoContent.length(), modelContent.length());
    TNN_NS::Status status = TNN_NS::TNN_OK;
    {
        auto option                       = std::make_shared<TNN_NS::BlazePoseDetectorOption>();
        option->compute_units             = TNN_NS::TNNComputeUnitsCPU;
        option->library_path              = "";
        option->proto_content             = protoContent;
        option->model_content             = modelContent;
        option->min_score_threshold       = 0.5;
        option->min_suppression_threshold = 0.3;
        status                = gBlazePoseDetector->Init(option);
        if (status != TNN_NS::TNN_OK) {
            printf("blaze pose detector init failed %d\n", (int)status);
            return -1;
        }
    }
    protoContent = fdLoadFile(modelPath + "/pose_landmark_upper_body.tnnproto");
    modelContent = fdLoadFile(modelPath + "/pose_landmark_upper_body.tnnmodel");
    printf("pose landmark proto content size %lld model content size %lld\n", protoContent.length(), modelContent.length());

    {
        auto option                           = std::make_shared<TNN_NS::BlazePoseLandmarkOption>();
        option->compute_units                 = TNN_NS::TNNComputeUnitsCPU;
        option->library_path                  = "";
        option->proto_content                 = protoContent;
        option->model_content                 = modelContent;
        option->pose_presence_threshold       = 0.5;
        option->landmark_visibility_threshold = 0.1;

        status                = gBlazePoseLandmark->Init(option);
        if (status != TNN_NS::TNN_OK) {
            printf("blaze pose landmark init failed %d\n", (int)status);
            return -1;
        }
    }

    protoContent = fdLoadFile(modelPath + "/pose_landmark_full_body.tnnproto");
    modelContent = fdLoadFile(modelPath + "/pose_landmark_full_body.tnnmodel");
    printf("pose landmark full body proto content size %lld model content size %lld\n", protoContent.length(),
         modelContent.length());
    {
        auto option                           = std::make_shared<TNN_NS::BlazePoseLandmarkOption>();
        option->compute_units                 = TNN_NS::TNNComputeUnitsCPU;
        option->library_path                  = "";
        option->proto_content                 = protoContent;
        option->model_content                 = modelContent;
        option->pose_presence_threshold       = 0.5;
        option->landmark_visibility_threshold = 0.1;
        option->full_body                     = true;

        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        status                = gBlazePoseFullBodyLandmark->Init(option);
       
        if (status != TNN_NS::TNN_OK) {
            printf("blaze pose landmark init failed %d\n", (int)status);
            return -1;
        }
    }

    status = gDetector->Init({gBlazePoseDetector, gBlazePoseLandmark});
    if (status != TNN_NS::TNN_OK) {
        printf("pose detector init failed %d\n", (int)status);
        return -1;
    }

    status = gFullBodyDetector->Init({gBlazePoseDetector, gBlazePoseFullBodyLandmark});
    if (status != TNN_NS::TNN_OK) {
        printf("pose full body detector init failed %d\n", (int)status);
        return -1;
    }
    return 0;
}

int deinit() {
    gDetector                  = nullptr;
    gFullBodyDetector          = nullptr;
    gBlazePoseDetector         = nullptr;
    gBlazePoseLandmark         = nullptr;
    gBlazePoseFullBodyLandmark = nullptr;
    return 0;
}

void detectPose() 
{
    const char *name = "yoga.jpg";
    cv::Mat mat          = imread(name);
    //cvtColor(mat, mat, COLOR_BGR2BGRA);
    if (mat.empty())
        return;
    std::shared_ptr<TNN_NS::PoseDetectLandmark> asyncRefDetector;
    int detect_type = 1;
    if (detect_type == 0) {
        asyncRefDetector = gDetector;
    } else {
        asyncRefDetector = gFullBodyDetector;
    }
    std::vector<TNN_NS::BlazePoseInfo> objectInfoList;
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_X86;

    TNN_NS::DimsVector input_dims = {1, mat.channels(), mat.rows, mat.cols};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC3, input_dims, mat.data);

    std::shared_ptr<TNN_NS::TNNSDKInput> input   = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    if (status != TNN_NS::TNN_OK) {
        printf("failed to detect %d\n", (int)status);
        return;
    }

    if (!output) {
        printf("Get empty output\n");
        return;
    } else {
        TNN_NS::BlazePoseLandmarkOutput *ptr = dynamic_cast<TNN_NS::BlazePoseLandmarkOutput *>(output.get());
        if (!ptr) {
            printf("BlazePose Landmark output empty\n");
            return;
        }
    }

    objectInfoList = dynamic_cast<TNN_NS::BlazePoseLandmarkOutput *>(output.get())->body_list;
    if (status != TNN_NS::TNN_OK) {
        printf("failed to detect %d\n", (int)status);
        return;
    }

    printf("object info list size %lld\n", objectInfoList.size());
    if (objectInfoList.size() > 0) 
    {
        for (int i = 0; i < objectInfoList.size(); i++) 
        {
            auto &objectInfo      = objectInfoList[i];
            int keypointsNum      = objectInfo.key_points_3d.size();
            int linesNum          = objectInfo.lines.size();
            printf("object %f %f %f %f score %f key points size %d, label_id: %d, line num: %d\n", objectInfo.x1,
                 objectInfo.y1, objectInfo.x2, objectInfo.y2, objectInfo.score, keypointsNum, objectInfo.class_id,
                 linesNum);
            //auto object_orig = objectInfo.AdjustToViewSize(800, 600, 2);
            // from here start to create point
            // Create the returnable jobjectArray with an initial value
            for (int j = 0; j < keypointsNum; j++) {
                float temp[] = {std::get<0>(objectInfo.key_points_3d[j]), std::get<1>(objectInfo.key_points_3d[j])};
                std::cout << temp[0] << "  " << temp[1] << std::endl;
                cv::circle(mat, cv::Point2f(temp[0], temp[1]), 3, cv::Scalar(0, 0, 255), -1);
            }

            // from here start to create line
            // Create the returnable jobjectArray with an initial value
            for (int j = 0; j < linesNum; j++) {
                int temp[] = {objectInfo.lines[j].first, objectInfo.lines[j].second};
            }
        }
    }
    cv::imshow("res.jpg", mat);
    cv::waitKey(0);
    return; 
}


using namespace TNN_NS;

int main(int argc, char** argv) 
{
    init("./");
    detectPose();
    deinit();
    return 0;
}