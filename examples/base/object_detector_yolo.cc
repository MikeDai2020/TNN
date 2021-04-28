// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "object_detector_yolo.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>


namespace TNN_NS {
ObjectDetectorYoloOutput::~ObjectDetectorYoloOutput() {}

MatConvertParam ObjectDetectorYolo::GetConvertParamForInput(std::string name) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 255, 1.0 / 255, 1.0 / 255, 0.0};
    input_convert_param.bias  = {0.0, 0.0, 0.0, 0.0};
    return input_convert_param;
}

std::shared_ptr<Mat> ObjectDetectorYolo::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
   auto target_dims  = GetInputShape(name);
   auto input_height = input_mat->GetHeight();
   auto input_width  = input_mat->GetWidth();
   copy_border_para  = {0, 0, 0, 0, BORDER_TYPE_CONSTANT, 0};
   if (target_dims.size() >= 4 && (input_height != target_dims[2] || input_width != target_dims[3])) {
       float scalex =   (float)target_dims[3] / (float)input_width;
       float scaley = (float)target_dims[2] / (float)input_height;
       float scale  = std::min(scalex, scaley);
       int new_w    = scale * input_width + 0.5;
       int new_h    = scale * input_height + 0.5;
       //target_dims[3]        = (new_w + 31) & ~31;
       //target_dims[2]        = (new_h + 31) & ~31;
       int top      = (target_dims[2] - new_h)/2;
       int bottom   = target_dims[2] - top - new_h;
       int left     = (target_dims[3] - new_w) / 2;
       int right    = target_dims[3] - new_w - left;

       copy_border_para.left = left;
       copy_border_para.right = right;
       copy_border_para.bottom = bottom;
       copy_border_para.top    = top;
       printf("left = %d, top = %d, right = %d, bottom = %d\n", left, top, right, bottom);
       auto dst_dims = {target_dims[0],target_dims[1],new_h, new_w};
       auto dst_mat  = std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), dst_dims);
       auto status   = Resize(input_mat, dst_mat, TNNInterpLinear);
       auto target_mat =
           std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), target_dims);
       //input_shape = target_dims;
       if (status == TNN_OK) {
           status = CopyMakeBorder(dst_mat, target_mat, top, bottom, left, right, TNNBorderConstant, 0);
           if (status == TNN_OK) {
               return target_mat;
           } else {
               LOGE("%s\n", status.description().c_str());
               return nullptr;
           }
       } else {
           LOGE("%s\n", status.description().c_str());
           return nullptr;
       }
   }
   return input_mat;
   //return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

std::shared_ptr<TNNSDKOutput> ObjectDetectorYolo::CreateSDKOutput() {
    return std::make_shared<ObjectDetectorYoloOutput>();
}

bool CompareOutput(std::shared_ptr<Mat>& mat0, std::shared_ptr<Mat>& mat1) {
    auto dims0 = mat0->GetDims();
    auto dims1 = mat1->GetDims();
    return dims0[1] < dims1[1];
}

Status ObjectDetectorYolo::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    
    auto output = dynamic_cast<ObjectDetectorYoloOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    auto out_names = GetOutputNames();  
    std::vector<std::shared_ptr<Mat>> outputs;
    for (auto name : out_names) {
        auto output_mat = output->GetMat(name); 
        RETURN_VALUE_ON_NEQ(!output_mat, false, Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
        outputs.push_back(output_mat);
    }
    std::sort(outputs.begin(), outputs.end(), CompareOutput);

    std::vector<ObjectInfo> object_list;
    auto input_shape = GetInputShape();
    GenerateDetectResult(outputs, object_list, input_shape[3], input_shape[2]);
    output->object_list = object_list;
    return status; 
}

void ObjectDetectorYolo::NMS(std::vector<ObjectInfo>& objs, std::vector<ObjectInfo>& results) {
    ::TNN_NS::NMS(objs, results, iou_threshold_, TNNHardNMS);
}

ObjectDetectorYolo::~ObjectDetectorYolo() {}

void ObjectDetectorYolo::GenerateDetectResult(std::vector<std::shared_ptr<Mat> >outputs,
                                              std::vector<ObjectInfo>& detecs, int image_width, int image_height) {
    std::vector<ObjectInfo> extracted_objs;
    int blob_index = 0;
    
    for(auto& output:outputs){
        auto dim = output->GetDims();
  
        if(dim[3] != num_anchor_ * detect_dim_) {
            LOGE("Invalid detect output, the size of last dimension is: %d\n", dim[3]);
            return;
        }
        float* data = static_cast<float*>(output->GetData());
        
        int num_potential_detecs = dim[1] * dim[2] * num_anchor_;
        for(int i=0; i<num_potential_detecs; ++i){
            float x = data[i * detect_dim_ + 0];
            float y = data[i * detect_dim_ + 1];
            float width = data[i * detect_dim_ + 2];
            float height = data[i * detect_dim_ + 3];
            
            float objectness = data[i * detect_dim_ + 4];
            if(objectness < conf_thres)
                continue;
            //center point coord
            x      = (x * 2 - 0.5 + ((i / num_anchor_) % dim[2])) * strides_[blob_index];
            y      = (y * 2 - 0.5 + ((i / num_anchor_) / dim[2]) % dim[1]) * strides_[blob_index];
            width  = pow((width  * 2), 2) * anchor_grids_[blob_index * grid_per_input_ + (i%num_anchor_) * 2 + 0];
            height = pow((height * 2), 2) * anchor_grids_[blob_index * grid_per_input_ + (i%num_anchor_) * 2 + 1];
            // compute coords
            float x1 = x - width  / 2;
            float y1 = y - height / 2;
            float x2 = x + width  / 2;
            float y2 = y + height / 2;
            // compute confidence
            auto conf_start = data + i * detect_dim_ + 5;
            auto conf_end   = data + (i+1) * detect_dim_;
            auto max_conf_iter = std::max_element(conf_start, conf_end);
            int conf_idx = static_cast<int>(std::distance(conf_start, max_conf_iter));
            float score = (*max_conf_iter) * objectness;
            
            ObjectInfo obj_info;
            obj_info.image_width = image_width;
            obj_info.image_height = image_height;
            obj_info.x1 = x1 - copy_border_para.left;
            obj_info.y1 = y1 - copy_border_para.top;
            obj_info.x2 = x2 - copy_border_para.left;
            obj_info.y2 = y2 - copy_border_para.top;
            obj_info.score = score;
            obj_info.class_id = conf_idx;
            extracted_objs.push_back(obj_info);
        }
        blob_index += 1;
    }
    NMS(extracted_objs, detecs);
}

}
