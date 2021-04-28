#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>

#include "object_detector_yolo.h"
#include "tnn_sdk_sample.h"
#include "macro.h"
#include "utils/utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

using namespace TNN_NS;

int main(int argc, char** argv) {
    // ´´½¨tnnÊµÀý
    auto proto_content = fdLoadFile("yolov5s.tnnproto");
    auto model_content = fdLoadFile("yolov5s.tnnmodel");

    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path  = "";
        TNNComputeUnits units = TNNComputeUnits::TNNComputeUnitsCPU;
        option->compute_units = units;
    }

    auto predictor = std::make_shared<ObjectDetectorYolo>();
    auto status    = predictor->Init(option);
    if (status != TNN_OK){
        std::cout << "Error: " << status.description() << std::endl;
        return -1;
    }
    auto input_dims   = predictor->GetInputShape(kTNNSDKDefaultName);
    char* input_imgfn = "bus.jpg";
    int image_width, image_height, image_channel;
    unsigned char* data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    DimsVector img_dims  = {1, 3, image_height, image_width};

    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::N8UC3, img_dims, data);
    std::shared_ptr<TNNSDKOutput> output = predictor->CreateSDKOutput();

    status = predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), output);
    if (status != TNN_OK) {
        std::cout << "Error: " << status.description() << std::endl;
        return -1;
    }
    //status = predictor->ProcessSDKOutput(output);
    //if (status != TNN_OK) {
    //    std::cout << "Error: " << status.description() << std::endl;
    //    return -1;
    //}

    std::vector<ObjectInfo> object_list;
    if (output && dynamic_cast<ObjectDetectorYoloOutput*>(output.get())) {
        auto obj_output = dynamic_cast<ObjectDetectorYoloOutput*>(output.get());
        object_list     = obj_output->object_list;
    }

    auto bench_result = predictor->GetBenchResult();
    std::cout << "Get objects:" << (int)object_list.size() << "  "  << bench_result.Description() << std::endl;

    int64_t size    = (int64_t)image_height * (int64_t)image_width * 4;
    uint8_t* buffer = new uint8_t[size];
    for (size_t i = 0; i < image_height * image_width; ++i) {
        buffer[i * 4]     = data[i * 3];
        buffer[i * 4 + 1] = data[i * 3 + 1];
        buffer[i * 4 + 2] = data[i * 3 + 2];
        buffer[i * 4 + 3] = 255;
    }

    float scale_x = (float)image_width / (float)input_dims[3];
    float scale_y = (float)image_height / (float)input_dims[2];
    float scale  = std::max(scale_x, scale_y);
    for (int i = 0; i < object_list.size(); i++) {
        std::string des;
        auto& obj = object_list[i];
        des.append(coco_classes[obj.class_id]);
        des = des + "," + std::to_string(obj.score);
        std::cout << des << std::endl;
        int x1    = obj.x1;
        int y1    = obj.y1;
        int x2    = obj.x2;
        int y2    = obj.y2;
        int w     = x2 - x1;
        int h     = y2 - y1;
        printf("x = %d, y = %d, w = %d, h = %d\n", x1, y1, w, h);
        TNN_NS::Rectangle((void*)buffer, image_height, image_width, x1, y1, x2, y2, scale, scale);
    }

    int success = stbi_write_bmp("res.bmp", image_width, image_height, 4, buffer);
    delete[] buffer;
    if (!success)
        return -1;
    return 0;
}