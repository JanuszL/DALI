// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/color/brightness_contrast.h"
#include <vector>
#include "dali/kernels/imgproc/pointwise/multiply_add_gpu.h"

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::MultiplyAddGpu<Out, In, 3>;

}  // namespace

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrastGpu, GPU)
DALI_REGISTER_OPERATOR(Brightness, BrightnessContrastGpu, GPU);
DALI_REGISTER_OPERATOR(Contrast, BrightnessContrastGpu, GPU);


bool BrightnessContrastGpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<GPUBackend> &ws) {
    KMgrResize(num_threads_, max_batch_size_);
    const auto &input = ws.template Input<GPUBackend>(0);
    const auto &output = ws.template Output<GPUBackend>(0);
    output_desc.resize(1);
    AcquireArguments(ws);
    int N = input.num_samples();
    auto sh = input.shape();
    auto num_dims = sh.sample_dim();
    auto layout = input.GetLayout();
    int c_dim = layout.find('C');
    DALI_ENFORCE(c_dim == num_dims - 1 || layout.empty(), make_string("Only channel last or empty "
                "layouts are supported, received ", layout, " instead"));
    DALI_ENFORCE(num_dims >= 3 && num_dims <= 4, make_string("Only 3 and 4 dimensions are "
                "supported received ", num_dims));
    addends_.resize(N);
    multipliers_.resize(N);
    TYPE_SWITCH(input.type(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
            {
                VALUE_SWITCH(num_dims, static_dims, (3, 4),
                (
                    using Kernel = TheKernel<OutputType, InputType>;
                    kernel_manager_.Initialize<Kernel>();
                    CallSetup<Kernel, InputType, static_dims>(ws, input);
                    output_desc[0] = {input.shape(), output_type_};
                ),  // NOLINT
                (
                    DALI_FAIL("Not supported number of dims");
                ));  // NOLINT
            }
        ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
    return true;
}

void BrightnessContrastGpu::RunImpl(workspace_t<GPUBackend> &ws) {
    const auto &input = ws.template Input<GPUBackend>(0);
    auto &output = ws.template Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());
    auto sh = input.shape();
    auto num_dims = sh.sample_dim();
    TYPE_SWITCH(input.type(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
            {
                kernels::KernelContext ctx;
                ctx.gpu.stream = ws.stream();
                for (unsigned i = 0; i < input.num_samples(); i++) {
                    OpArgsToKernelArgs<OutputType, InputType>(addends_[i], multipliers_[i],
                        brightness_[i], brightness_shift_[i], contrast_[i]);
                }
                VALUE_SWITCH(num_dims, static_dims, (3, 4),
                (
                    if constexpr (static_dims == 3) {
                        using Kernel = TheKernel<OutputType, InputType>;
                        auto tvin = view<const InputType, 3>(input);
                        auto tvout = view<OutputType, 3>(output);
                        kernel_manager_.Run<Kernel>(0, 0, ctx, tvout, tvin, addends_, multipliers_);
                    } else if constexpr (static_dims == 4) {  // NOLINT
                        using Kernel = TheKernel<OutputType, InputType>;
                        auto tvin = view<const InputType, 4>(input);
                        auto tvin_reint = reinterpret<const InputType, 3>(tvin,
                                                                collapse_dim(tvin.shape, 0), true);
                        auto tvout = view<OutputType, 4>(output);
                        auto tvout_reint = reinterpret<OutputType, 3>(tvout,
                                                                collapse_dim(tvout.shape, 0), true);
                        kernel_manager_.Run<Kernel>(0, 0, ctx, tvout_reint, tvin_reint,
                                                    addends_, multipliers_);
                    }
                ),  // NOLINT
                (
                    DALI_FAIL("Not supported number of dims");
                ));  // NOLINT
            }
        ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}

}  // namespace dali
