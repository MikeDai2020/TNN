//
// Created by tencent on 2020-12-11.
//
#include <stdio.h>
#include "skeleton_detector.h"
#include "tnn/utils/mat_utils.h"
#include "utils/utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"


static std::shared_ptr<TNN_NS::SkeletonDetector> gDetector;
static std::shared_ptr<TNN_NS::SkeletonDetector> gSmallDetector;
static int gComputeUnitType = 0; //cpu


void init() 
{
    // Reset bench description
    //setBenchResult("");
    //std::vector<int> nchw = {1, 3, height, width};
    gDetector             = std::make_shared<TNN_NS::SkeletonDetector>();
    gSmallDetector        = std::make_shared<TNN_NS::SkeletonDetector>();
    std::string protoContent, middleProtoContent, smallProtoContent, modelContent;
    protoContent      = fdLoadFile("skeleton_big.tnnproto");
    smallProtoContent = fdLoadFile("skeleton_small.tnnproto");
    modelContent      = fdLoadFile("skeleton.tnnmodel");
    printf(
        "big proto content size: %d, "
        "small proto content size: %d, "
        "model content size %d",
        protoContent.length(), smallProtoContent.length(), modelContent.length());

    TNN_NS::Status status = TNN_NS::TNN_OK, status1 = TNN_NS::TNN_OK;
    auto option           = std::make_shared<TNN_NS::SkeletonDetectorOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->library_path  = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->min_threshold = 0.15f;

    auto smallDetectorOption           = std::make_shared<TNN_NS::SkeletonDetectorOption>(*option);
    smallDetectorOption->proto_content = smallProtoContent;
    LOGI("device type: %d", gComputeUnitType);
 
    status  = gDetector->Init(option);
    status1 = gSmallDetector->Init(smallDetectorOption);
   
    if (status != TNN_NS::TNN_OK || status1 != TNN_NS::TNN_OK) {
        printf("detector init failed high precision mode status: %d, fast mode status: %d", (int)status, (int)status1);
        return;
    }
    return;
}


int deinit() 
{
    gDetector      = nullptr;
    gSmallDetector = nullptr;
    return 0;
}

void Test() 
{
    std::shared_ptr<TNN_NS::SkeletonDetector> asyncRefDetector = gDetector;
    //asyncRefDetector = gSmallDetector;
    TNN_NS::SkeletonInfo objectInfo;
    // Convert yuv to rgb
    int width;
    int height;
    const char* name        = "yoga.jpg";
    int image_channel;
    unsigned char* data     = stbi_load(name, &width, &height, &image_channel, 3);

    TNN_NS::DeviceType dt = TNN_NS::DEVICE_X86;

    TNN_NS::DimsVector input_dims = {1, 3, height, width};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC3, input_dims, data);

    std::shared_ptr<TNN_NS::TNNSDKInput> input   = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);
    free(data);
    objectInfo = dynamic_cast<TNN_NS::SkeletonDetectorOutput *>(output.get())->keypoints;
    if (status != TNN_NS::TNN_OK) {
        printf("failed to detect %d", (int)status);
        return;
    }
    int keypointsNum      = objectInfo.key_points.size();
    int linesNum          = objectInfo.lines.size();
    printf("object %f %f %f %f score %f key points size %d, label_id: %d, line num: %d", objectInfo.x1, objectInfo.y1,
         objectInfo.x2, objectInfo.y2, objectInfo.score, keypointsNum, objectInfo.class_id, linesNum);
    return;
}

int main(int argc, char** argv) {
    init();
    Test();
    return 0;
}