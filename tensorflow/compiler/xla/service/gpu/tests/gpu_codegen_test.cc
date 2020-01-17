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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

std::unique_ptr<VerifiedHloModule>
GpuCodegenTest::CreateNewVerifiedModuleWithFTZ(bool ftz) {
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_ftz(ftz);
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  config.set_debug_options(debug_options);

  return absl::make_unique<VerifiedHloModule>(
      TestName(), config, /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      ShapeUtil::ByteSizeOfElements);
}

void GpuCodegenTest::CompileAndVerifyPtx(
    std::unique_ptr<VerifiedHloModule> hlo_module, absl::string_view pattern) {
  std::unique_ptr<Executable> executable =
      std::move(CompileToExecutable(std::move(hlo_module)).ValueOrDie());
  string ptx_str(static_cast<GpuExecutable*>(executable.get())->text());
  StatusOr<bool> filecheck_result = RunFileCheck(ptx_str, pattern);
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

void GpuCodegenTest::CompileAndVerifyPtx(const string& hlo_text,
					 absl::string_view pattern) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  CompileAndVerifyPtx(std::move(module), pattern);
}
}  // namespace gpu
}  // namespace xla
