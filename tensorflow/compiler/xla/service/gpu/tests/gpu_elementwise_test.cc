/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

// Check that some elementwise operation aren't unrolled as LLVM
// doesn't vectorize them.

namespace xla {
namespace gpu {
namespace {

class GpuElementwiseTest : public GpuCodegenTest {};

TEST_F(GpuElementwiseTest, ExpElementwise) {
    const char* hlo_string = R"(
HloModule vec

ENTRY %computation (arg0: f32[16000000]) -> f32[16000000] {
  %arg0 = f32[16000000] parameter(0), parameter_replication={false}
  ROOT %out = f32[16000000] sine(arg0)
}
)";

    CompileAndVerifyPtx(hlo_string, R"(
    CHECK: ld.global.nc.f32
    CHECK-NOT: ld.global.nc.f32
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
