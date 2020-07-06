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

#include "dali/pipeline/operator/builtin/make_contiguous.h"

namespace dali {

void MakeContiguousCPU::RunImpl(HostWorkspace &ws) {
  auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  output.SetLayout(input.GetLayout());
  // if (input.IsContiguous()) {
  //   std::swap(input, output);
  //   input.UpdateViews();
  // }
  // else {
  //   for (int i = 0; i < batch_size_; ++i) {
  //     output[i].Copy(input[i], 0);
  //   }
  // }

  // sort by the work size
  sample_ids_.clear();
  sample_ids_.reserve(batch_size_);
  auto shapes = input.shape();
  for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
    sample_ids_.emplace_back(volume(shapes[sample_id]), sample_id);
  }
  std::sort(sample_ids_.begin(), sample_ids_.end(), std::greater<VolumeSampleIdPair>());

  auto &thread_pool = ws.GetThreadPool();
  for (const auto &sample : sample_ids_) {
    auto data_idx = sample.second;
    thread_pool.DoWorkWithID([&ws, data_idx, &output, &input]
                             (int tid) {
      // HostWorkspace doesn't have any stream
      cudaStream_t stream = 0;
      output[data_idx].Copy(input[data_idx], stream);
    });
  }
  thread_pool.WaitForWork();
}

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousCPU, CPU);

DALI_SCHEMA(MakeContiguous)
  .DocStr(R"code(Move input batch to a contiguous representation, more suitable for execution on the GPU)code")
  .NumInput(1)
  .NumOutput(1)
  .MakeInternal();

}  // namespace dali
