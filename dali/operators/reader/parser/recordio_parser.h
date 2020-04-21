// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
#define DALI_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_

#include <string>
#include <vector>

#include "dali/operators/reader/parser/parser.h"

namespace dali {

struct ImageRecordIOHeader {
  uint32_t flag;
  float label;
  uint64_t image_id[2];
};

class RecordIOParser : public Parser<Tensor<CPUBackend>> {
 public:
  explicit RecordIOParser(const OpSpec& spec) :
    Parser<Tensor<CPUBackend>>(spec),
    save_img_ids_(spec.GetArgument<bool>("save_img_ids")),
    read_detection_data_(spec.GetArgument<bool>("read_detection_data")),
    ltrb_(spec.GetArgument<bool>("ltrb")),
    min_size_threshold_(spec.GetArgument<float>("size_threshold")),
    ratio_(spec.GetArgument<bool>("ratio")) {
    if (!read_detection_data_) {
      DALI_ENFORCE(save_img_ids_ == false,
                  "save_img_ids option can be used only with detection data.");
    }
  }

  void ParseDetectionData(SampleWorkspace* ws, const uint8_t *input, int num_of_obj) {
    size_t dim_size = 2 * sizeof(float);
    size_t id_size = sizeof(float);
    size_t labels_size = num_of_obj*sizeof(float);
    size_t boxes_size = num_of_obj*4*sizeof(float);
    auto& boxes_output = ws->Output<CPUBackend>(1);
    auto& labels_output = ws->Output<CPUBackend>(2);

    if (save_img_ids_) {
      auto& id_output = ws->Output<CPUBackend>(3);
      id_output.Resize({1});
      auto id = id_output.mutable_data<int>();
      float id_float;
      memcpy(&id_float, input, id_size);
      // convert id to int
      *id = id_float;
    }
    std::array<float, 2> size_float;
    memcpy(&size_float[0], input + id_size, dim_size);
    int width = size_float[0];
    int height = size_float[1];

    std::vector<float> labels_float(num_of_obj);
    memcpy(&labels_float[0], input + id_size + dim_size, labels_size);
    std::vector<float> boxes_tmp(4*num_of_obj);
    memcpy(&boxes_tmp[0], input + id_size + dim_size + labels_size, boxes_size);

    int valid_obj_num = 0;
    for (int i = 0; i < num_of_obj; ++i) {
      if (boxes_tmp[4 * i + 2] >= min_size_threshold_ &&
          boxes_tmp[4 * i + 3] >= min_size_threshold_) {
          if (valid_obj_num != i) {
            boxes_tmp[4 * valid_obj_num    ] = boxes_tmp[4 * i    ];
            boxes_tmp[4 * valid_obj_num + 1] = boxes_tmp[4 * i + 1];
            boxes_tmp[4 * valid_obj_num + 2] = boxes_tmp[4 * i + 2];
            boxes_tmp[4 * valid_obj_num + 3] = boxes_tmp[4 * i + 3];
            labels_float[valid_obj_num] = labels_float[i];
          }
          ++valid_obj_num;
      }
    }

    labels_output.Resize({valid_obj_num, 1});
    boxes_output.Resize({valid_obj_num, 4});
    auto labels = labels_output.mutable_data<int>();
    auto boxes = boxes_output.mutable_data<float>();

    // convert labels to int, copy bboxes and scale them if needed
    if (!ratio_) {
      width = 1;
      height = 1;
    }
    for (int i = 0; i < valid_obj_num; ++i) {
      labels[i] = labels_float[i];
      if (ltrb_) {
        boxes_tmp[4 * i + 2] += boxes_tmp[4 * i    ];
        boxes_tmp[4 * i + 3] += boxes_tmp[4 * i + 1];
      }
      boxes[4 * i    ] = boxes_tmp[4 * i    ] / width;
      boxes[4 * i + 1] = boxes_tmp[4 * i + 1] / height;
      boxes[4 * i + 2] = boxes_tmp[4 * i + 2] / width;
      boxes[4 * i + 3] = boxes_tmp[4 * i + 3] / height;
    }
  }

  void Parse(const Tensor<CPUBackend>& data, SampleWorkspace* ws) override {
    int index = 0;
    auto input = data.data<uint8_t>();
    auto& o_image = ws->Output<CPUBackend>(0);

    uint32_t magic;
    const uint32_t kMagic = 0xced7230a;
    ReadSingle<uint32_t>(&input, &magic);
    DALI_ENFORCE(magic == kMagic, "Invalid RecordIO: wrong magic number");

    uint32_t length_flag;
    ReadSingle(&input, &length_flag);
    uint32_t cflag = DecodeFlag(length_flag);
    uint32_t clength = DecodeLength(length_flag);
    ImageRecordIOHeader hdr;
    ReadSingle(&input, &hdr);

    int num_of_obj = 0;
    if (hdr.flag == 0) {
      DALI_ENFORCE(read_detection_data_ == false,
                   "Not enough data for bboxes and lables the label field");
      auto& o_label = ws->Output<CPUBackend>(1);
      o_label.Resize({1});
      o_label.mutable_data<float>()[0] = hdr.label;
    } else {
      if (read_detection_data_) {
        DALI_ENFORCE((hdr.flag - 3) % 5 == 0,
                     "Not enough data for bboxes, ids and lables the label field");
        num_of_obj = (hdr.flag - 3) / 5;
      } else {
        auto& o_label = ws->Output<CPUBackend>(1);
        o_label.Resize({hdr.flag});
        o_label.mutable_data<float>();
      }
    }

    int64_t data_size = clength - sizeof(ImageRecordIOHeader);
    int64_t label_size = 0;
    if (read_detection_data_) {
      label_size = (hdr.flag - 1) * sizeof(float) + sizeof(int);
    } else {
      label_size = hdr.flag * sizeof(float);
    }
    int64_t image_size = data_size - label_size;
    if (cflag == 0) {
      o_image.Resize({image_size});
      uint8_t* data = o_image.mutable_data<uint8_t>();
      memcpy(data, input + label_size, image_size);
      if (hdr.flag > 0) {
        if (read_detection_data_) {
          ParseDetectionData(ws, input, num_of_obj);
        } else {
          auto& o_label = ws->Output<CPUBackend>(1);
          auto label = o_label.mutable_data<float>();
          memcpy(label, input, label_size);
        }
      }
    } else {
      std::vector<uint8_t> temp_vec(data_size);
      memcpy(&temp_vec[0], input, data_size);
      input += data_size;

      while (true) {
        size_t pad = clength - (((clength + 3U) >> 2U) << 2U);
        input += pad;

        if (cflag != 3) {
          size_t s = temp_vec.size();
          temp_vec.resize(static_cast<int64_t>(s + sizeof(kMagic)));
          memcpy(&temp_vec[s], &kMagic, sizeof(kMagic));
        } else {
          break;
        }
        ReadSingle(&input, &magic);
        ReadSingle(&input, &length_flag);
        cflag = DecodeFlag(length_flag);
        clength = DecodeLength(length_flag);
        size_t s = temp_vec.size();
        temp_vec.resize(static_cast<int64_t>(s + clength));
        memcpy(&temp_vec[s], input, clength);
        input += clength;
      }
      o_image.Resize({static_cast<Index>(temp_vec.size() - label_size)});
      uint8_t* data = o_image.mutable_data<uint8_t>();
      memcpy(data, (&temp_vec[0]) + label_size, temp_vec.size() - label_size);
      if (hdr.flag > 0) {
        if (read_detection_data_) {
          ParseDetectionData(ws, &temp_vec[0], num_of_obj);
        } else {
          auto& o_label = ws->Output<CPUBackend>(1);
          auto label = o_label.mutable_data<float>();
          memcpy(label, &temp_vec[0], label_size);
        }
      }
    }
    o_image.SetSourceInfo(data.GetSourceInfo());
  }

 private:
  inline uint32_t DecodeFlag(uint32_t rec) {
    return (rec >> 29U) & 7U;
  }

  inline uint32_t DecodeLength(uint32_t rec) {
    return rec & ((1U << 29U) - 1U);
  }

  template <typename T>
  void ReadSingle(const uint8_t** in, T* out) {
    memcpy(out, *in, sizeof(T));
    *in += sizeof(T);
  }

  bool save_img_ids_;
  bool read_detection_data_;
  bool ltrb_;
  float min_size_threshold_;
  bool ratio_;
};

};  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_RECORDIO_PARSER_H_
