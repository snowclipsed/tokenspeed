/*
 * Copyright (c) 2026 LightSeek Foundation
 *
 * BF16-only TokenSpeed binding around FlashInfer/TensorRT-LLM's Cutlass MoE
 * runner. The underlying runner and heuristic profiler are vendored from
 * FlashInfer v0.6.13; this file intentionally exposes only the unquantized
 * BF16 path used by AFMoE models.
 */

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/extra/module.h>

#include <memory>
#include <mutex>
#include <vector>

#include "cutlass_kernel_selector.h"
#include "moe_gemm_kernels.h"
#include "moe_kernels.h"
#include "tensorrt_llm/common/workspace.h"
#include "tvm_ffi_utils.h"

namespace common = tensorrt_llm::common;
namespace kernels = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using ActivationParams = CUTLASS_MOE_GEMM_NAMESPACE::ActivationParams;
using ActivationType = CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

namespace {

class Bf16FusedMoeRunner : public tvm::ffi::ModuleObj {
 public:
  Bf16FusedMoeRunner(DLDataType activation_dtype, DLDataType weight_dtype,
                     DLDataType output_dtype) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(activation_dtype), bfloat16_code)
        << "AFMoE local Cutlass MoE only supports BF16 activations";
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(weight_dtype), bfloat16_code)
        << "AFMoE local Cutlass MoE only supports BF16 weights";
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output_dtype), bfloat16_code)
        << "AFMoE local Cutlass MoE only supports BF16 output";

    runner_ =
        std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    profiler_ = std::make_shared<kernels::GemmProfilerBackend>();

    auto gemm1_tactics = runner_->getTactics(kernels::MoeGemmId::GEMM_1);
    auto gemm2_tactics = runner_->getTactics(kernels::MoeGemmId::GEMM_2);
    gemm1_tactic_count_ = static_cast<int64_t>(gemm1_tactics.size());
    gemm2_tactic_count_ = static_cast<int64_t>(gemm2_tactics.size());
    profiles_ = std::move(gemm1_tactics);
    profiles_.insert(profiles_.end(), gemm2_tactics.begin(), gemm2_tactics.end());
    TVM_FFI_ICHECK(!profiles_.empty()) << "No valid BF16 Cutlass MoE tactics found";
  }

  const char* kind() const final { return "afmoe_cutlass_bf16_moe_runner"; }

  tvm::ffi::Optional<tvm::ffi::Function> GetFunction(
      const tvm::ffi::String& name) final {
    if (name == "get_tactic_num") {
      return tvm::ffi::Function::FromTyped([this]() -> int64_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int64_t>(profiles_.size());
      });
    }
    if (name == "get_gemm1_tactic_count") {
      return tvm::ffi::Function::FromTyped([this]() -> int64_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return gemm1_tactic_count_;
      });
    }
    if (name == "get_gemm2_tactic_count") {
      return tvm::ffi::Function::FromTyped([this]() -> int64_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return gemm2_tactic_count_;
      });
    }
    if (name == "get_tactic_occupancy") {
      return tvm::ffi::Function::FromTyped([this](int64_t tactic_id) -> int64_t {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tactic_id < 0 || tactic_id >= static_cast<int64_t>(profiles_.size())) {
          return 0;
        }
        return static_cast<int64_t>(
            runner_->queryOccupancyForConfig(profiles_[tactic_id]));
      });
    }
    if (name == "run_gemm_profile") {
      return tvm::ffi::Function::FromTyped(
          [this](TensorView input, TensorView fc1_expert_weights,
                 tvm::ffi::Optional<TensorView> fc1_expert_biases,
                 TensorView fc2_expert_weights,
                 tvm::ffi::Optional<TensorView> fc2_expert_biases, int64_t top_k,
                 int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                 int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                 bool min_latency_mode, int64_t gemm_idx, int64_t profile_id,
                 bool do_preparation, bool enable_pdl, int64_t activation_type) {
            runGemmProfile(input, fc1_expert_weights, fc1_expert_biases,
                           fc2_expert_weights, fc2_expert_biases, top_k, tp_size,
                           tp_rank, ep_size, ep_rank, cluster_size, cluster_rank,
                           enable_alltoall, min_latency_mode, gemm_idx, profile_id,
                           do_preparation, enable_pdl,
                           static_cast<ActivationType>(activation_type));
          });
    }
    if (name == "run_moe") {
      return tvm::ffi::Function::FromTyped(
          [this](TensorView output, TensorView input, TensorView token_selected_experts,
                 tvm::ffi::Optional<TensorView> token_final_scales,
                 TensorView fc1_expert_weights,
                 tvm::ffi::Optional<TensorView> fc1_expert_biases,
                 TensorView fc2_expert_weights,
                 tvm::ffi::Optional<TensorView> fc2_expert_biases,
                 tvm::ffi::Optional<tvm::ffi::Array<Tensor>> quant_scales,
                 tvm::ffi::Optional<TensorView> input_sf,
                 tvm::ffi::Optional<TensorView> swiglu_alpha,
                 tvm::ffi::Optional<TensorView> swiglu_beta,
                 tvm::ffi::Optional<TensorView> swiglu_limit, bool swizzled_input_sf,
                 int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                 int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                 bool min_latency_mode, tvm::ffi::Optional<tvm::ffi::Array<int64_t>> profile_ids,
                 bool enable_pdl, int64_t activation_type) {
            runMoe(output, input, token_selected_experts, token_final_scales,
                   fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
                   fc2_expert_biases, quant_scales, input_sf, swiglu_alpha, swiglu_beta,
                   swiglu_limit, swizzled_input_sf, tp_size, tp_rank, ep_size, ep_rank,
                   cluster_size, cluster_rank, enable_alltoall, min_latency_mode, profile_ids,
                   enable_pdl, static_cast<ActivationType>(activation_type));
          });
    }
    return tvm::ffi::Function(nullptr);
  }

 private:
  struct WorkspaceInfo {
    Tensor workspace{};
    void* src_to_dest_map{};
  };

  std::mutex mutex_;
  std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> runner_;
  std::shared_ptr<kernels::GemmProfilerBackend> profiler_;
  Tensor profile_workspace_;
  std::vector<Profile> profiles_;
  int64_t gemm1_tactic_count_{0};
  int64_t gemm2_tactic_count_{0};

  static nvinfer1::DataType dataType(DLDataType dtype) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(dtype), bfloat16_code)
        << "AFMoE local Cutlass MoE only supports BF16";
    return nvinfer1::DataType::kBF16;
  }

  void setRunnerProfiles(tvm::ffi::Optional<tvm::ffi::Array<int64_t>> profile_ids) {
    auto best_gemm1 = profiles_.front();
    auto best_gemm2 =
        (gemm2_tactic_count_ > 0 &&
         profiles_.size() > static_cast<size_t>(gemm1_tactic_count_))
            ? profiles_.at(gemm1_tactic_count_)
            : profiles_.front();

    if (profile_ids.has_value()) {
      TVM_FFI_ICHECK_EQ(profile_ids.value().size(), 2) << "Expecting 2 profile ids";
      int64_t id1 = profile_ids.value()[0];
      if (id1 != -1) {
        TVM_FFI_ICHECK(id1 >= 0 && id1 < gemm1_tactic_count_)
            << "Invalid GEMM1 profile id: " << id1;
        best_gemm1 = profiles_.at(id1);
      }

      int64_t id2 = profile_ids.value()[1];
      if (id2 != -1) {
        int64_t absolute_id2 = id2;
        if (id2 >= 0 && id2 < gemm2_tactic_count_) {
          absolute_id2 = gemm1_tactic_count_ + id2;
        }
        TVM_FFI_ICHECK(absolute_id2 >= 0 &&
                       absolute_id2 < static_cast<int64_t>(profiles_.size()))
            << "Invalid GEMM2 profile id: " << id2;
        best_gemm2 = profiles_.at(absolute_id2);
      }
    }

    runner_->setTactic(best_gemm1, best_gemm2);
  }

  WorkspaceInfo getWorkspaceInfo(int64_t num_rows, int64_t hidden_size, int64_t inter_size,
                                 int num_experts, int experts_per_token,
                                 ActivationType activation_type,
                                 kernels::MOEParallelismConfig parallelism_config) {
    size_t moe_workspace_size = runner_->getWorkspaceSize(
        num_rows, hidden_size, inter_size, num_experts, experts_per_token, activation_type,
        parallelism_config, /*use_lora=*/false, /*use_deepseek_fp8_block_scale=*/false,
        /*use_mxfp8_act_scaling=*/false, /*min_latency_mode=*/false, /*use_awq=*/false);
    size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);
    std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};
    size_t total_workspace_size =
        common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    WorkspaceInfo info{};
    int device_id = 0;
    cudaGetDevice(&device_id);
    info.workspace = alloc_tensor({static_cast<int64_t>(total_workspace_size)}, dl_int8,
                                  DLDevice{kDLCUDA, device_id});
    info.src_to_dest_map =
        common::nextWorkspacePtr(static_cast<int8_t*>(info.workspace.data_ptr()),
                                 moe_workspace_size);
    return info;
  }

  static void validateCommon(TensorView input, TensorView fc1_expert_weights,
                             TensorView fc2_expert_weights, ActivationType activation_type) {
    CHECK_INPUT_TYPE(input, dl_bfloat16);
    CHECK_INPUT_TYPE(fc1_expert_weights, dl_bfloat16);
    CHECK_INPUT_TYPE(fc2_expert_weights, dl_bfloat16);
    CHECK_DIM(2, input);
    CHECK_DIM(3, fc1_expert_weights);
    CHECK_DIM(3, fc2_expert_weights);
    TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(0), fc2_expert_weights.size(0))
        << "fc1_expert_weights and fc2_expert_weights must have the same number of experts";
    if (isGatedActivation(activation_type)) {
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(1), fc2_expert_weights.size(2) * 2)
          << "gated fc1 inter size must be 2x fc2 inter size";
    } else {
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(1), fc2_expert_weights.size(2))
          << "non-gated fc1 inter size must equal fc2 inter size";
    }
  }

  static ActivationParams activationParams(
      ActivationType activation_type, int64_t num_experts_on_rank,
      tvm::ffi::Optional<TensorView> swiglu_alpha,
      tvm::ffi::Optional<TensorView> swiglu_beta,
      tvm::ffi::Optional<TensorView> swiglu_limit) {
    if (swiglu_alpha.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_alpha.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_alpha.value().size(0), num_experts_on_rank);
    }
    if (swiglu_beta.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_beta.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_beta.value().size(0), num_experts_on_rank);
    }
    if (swiglu_limit.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_limit.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_limit.value().size(0), num_experts_on_rank);
    }
    if (activation_type == ActivationType::Swiglu &&
        (swiglu_alpha.has_value() || swiglu_beta.has_value() ||
         swiglu_limit.has_value())) {
      activation_type = ActivationType::SwigluBias;
    }
    return ActivationParams(
        activation_type,
        reinterpret_cast<float const*>(
            swiglu_alpha.has_value() ? swiglu_alpha.value().data_ptr() : nullptr),
        reinterpret_cast<float const*>(
            swiglu_beta.has_value() ? swiglu_beta.value().data_ptr() : nullptr),
        reinterpret_cast<float const*>(
            swiglu_limit.has_value() ? swiglu_limit.value().data_ptr() : nullptr));
  }

  void runMoe(TensorView output, TensorView input, TensorView token_selected_experts,
              tvm::ffi::Optional<TensorView> token_final_scales,
              TensorView fc1_expert_weights,
              tvm::ffi::Optional<TensorView> fc1_expert_biases,
              TensorView fc2_expert_weights,
              tvm::ffi::Optional<TensorView> fc2_expert_biases,
              tvm::ffi::Optional<tvm::ffi::Array<Tensor>> quant_scales,
              tvm::ffi::Optional<TensorView> input_sf,
              tvm::ffi::Optional<TensorView> swiglu_alpha,
              tvm::ffi::Optional<TensorView> swiglu_beta,
              tvm::ffi::Optional<TensorView> swiglu_limit, bool swizzled_input_sf,
              int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
              int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
              bool min_latency_mode, tvm::ffi::Optional<tvm::ffi::Array<int64_t>> profile_ids,
              bool enable_pdl, ActivationType activation_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    TVM_FFI_ICHECK(!min_latency_mode)
        << "AFMoE local BF16 Cutlass MoE does not expose min-latency mode yet";
    TVM_FFI_ICHECK(cluster_size == 1 && cluster_rank == 0)
        << "cluster routing is only supported by FlashInfer's min-latency path";
    TVM_FFI_ICHECK(!quant_scales.has_value() || quant_scales.value().empty())
        << "quant_scales are not used by BF16/BF16 MoE";
    TVM_FFI_ICHECK(!input_sf.has_value()) << "input_sf is only valid for quantized MoE";

    CHECK_INPUT_TYPE(output, dl_bfloat16);
    CHECK_INPUT_TYPE(token_selected_experts, dl_int32);
    CHECK_DIM(2, token_selected_experts);
    CHECK_DIM(2, output);
    validateCommon(input, fc1_expert_weights, fc2_expert_weights, activation_type);
    TVM_FFI_ICHECK_EQ(input.size(0), token_selected_experts.size(0));
    TVM_FFI_ICHECK_EQ(output.size(0), input.size(0));
    TVM_FFI_ICHECK_EQ(output.size(1), fc2_expert_weights.size(1));

    if (token_final_scales.has_value()) {
      CHECK_INPUT_TYPE(token_final_scales.value(), dl_float32);
      CHECK_DIM(2, token_final_scales.value());
      TVM_FFI_ICHECK_EQ(token_final_scales.value().size(0), input.size(0));
      TVM_FFI_ICHECK_EQ(token_final_scales.value().size(1), token_selected_experts.size(1));
    }

    if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value()) {
      TVM_FFI_ICHECK(fc1_expert_biases.has_value() && fc2_expert_biases.has_value())
          << "fc1/fc2 biases must be provided together";
      CHECK_INPUT_TYPE(fc1_expert_biases.value(), dl_bfloat16);
      CHECK_INPUT_TYPE(fc2_expert_biases.value(), dl_bfloat16);
      CHECK_DIM(2, fc1_expert_biases.value());
      CHECK_DIM(2, fc2_expert_biases.value());
    }

    int const experts_per_token = token_selected_experts.size(1);
    int64_t const num_rows = input.size(0);
    int64_t const hidden_size = fc2_expert_weights.size(1);
    int64_t const inter_size = fc2_expert_weights.size(2);
    int const num_experts_on_rank = static_cast<int>(fc2_expert_weights.size(0));
    int const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(
        static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size),
        static_cast<int>(ep_rank), static_cast<int>(cluster_size),
        static_cast<int>(cluster_rank));
    auto act = activationParams(activation_type, num_experts_on_rank, swiglu_alpha, swiglu_beta,
                                swiglu_limit);

    setRunnerProfiles(profile_ids);
    auto workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
                                           experts_per_token, activation_type,
                                           parallelism_config);
    auto stream = get_stream(input.device());
    ::tensorrt_llm::kernels::LoraParams lora_params{};
    kernels::MoeMinLatencyParams min_latency_params{};

    runner_->runMoe(
        input.data_ptr(), nullptr, swizzled_input_sf,
        reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value().data_ptr())
            : nullptr,
        fc1_expert_weights.data_ptr(),
        fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr() : nullptr, act,
        fc2_expert_weights.data_ptr(),
        fc2_expert_biases.has_value() ? fc2_expert_biases.value().data_ptr() : nullptr,
        kernels::QuantParams{}, num_rows, hidden_size, hidden_size, inter_size,
        num_experts_total, experts_per_token,
        static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
        static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, enable_alltoall,
        /*use_lora=*/false, lora_params, /*use_deepseek_fp8_block_scale=*/false,
        /*use_mxfp8_act_scaling=*/false, /*min_latency_mode=*/false, min_latency_params,
        enable_pdl, stream);
  }

  void runGemmProfile(TensorView input, TensorView fc1_expert_weights,
                      tvm::ffi::Optional<TensorView> fc1_expert_biases,
                      TensorView fc2_expert_weights,
                      tvm::ffi::Optional<TensorView> fc2_expert_biases, int64_t top_k,
                      int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                      int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                      bool min_latency_mode, int64_t gemm_idx, int64_t profile_id,
                      bool do_preparation, bool enable_pdl, ActivationType activation_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    TVM_FFI_ICHECK(!min_latency_mode)
        << "AFMoE local BF16 Cutlass MoE does not expose min-latency mode yet";
    validateCommon(input, fc1_expert_weights, fc2_expert_weights, activation_type);
    TVM_FFI_ICHECK(gemm_idx == 1 || gemm_idx == 2) << "gemm_idx must be 1 or 2";
    TVM_FFI_ICHECK(top_k > 0) << "top_k must be positive";
    TVM_FFI_ICHECK(profile_id == -1 ||
                   (profile_id >= 0 && profile_id < static_cast<int64_t>(profiles_.size())))
        << "Invalid profile id: " << profile_id;

    auto profile = profile_id == -1 ? profiles_.front() : profiles_.at(profile_id);
    auto stream = get_stream(input.device());
    int64_t const num_rows = input.size(0);
    int64_t const hidden_size = fc2_expert_weights.size(1);
    int64_t const inter_size = fc2_expert_weights.size(2);
    int const num_experts = static_cast<int>(fc2_expert_weights.size(0) * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(
        static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size),
        static_cast<int>(ep_rank), static_cast<int>(cluster_size),
        static_cast<int>(cluster_rank));

    void const* expert_weights =
        gemm_idx == 1 ? fc1_expert_weights.data_ptr() : fc2_expert_weights.data_ptr();

    if (do_preparation) {
      profiler_->mGemmToProfile = gemm_idx == 1
                                      ? kernels::GemmProfilerBackend::GemmToProfile::GEMM_1
                                      : kernels::GemmProfilerBackend::GemmToProfile::GEMM_2;
      bool use_bias = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
      profiler_->init(*runner_, profiler_->mGemmToProfile, dataType(dl_bfloat16),
                      dataType(dl_bfloat16), dataType(dl_bfloat16), num_experts,
                      static_cast<int>(top_k), hidden_size, hidden_size, inter_size,
                      /*group_size=*/-1, activation_type, use_bias, /*use_lora=*/false,
                      /*min_latency_mode=*/false, /*need_weights=*/false,
                      parallelism_config, enable_alltoall);
      size_t profile_workspace_size = profiler_->getWorkspaceSize(num_rows);
      int device_id = 0;
      cudaGetDevice(&device_id);
      profile_workspace_ = alloc_tensor({static_cast<int64_t>(profile_workspace_size)}, dl_int8,
                                        DLDevice{kDLCUDA, device_id});
      profiler_->prepare(num_rows, static_cast<char*>(profile_workspace_.data_ptr()),
                         expert_weights, enable_pdl, stream);
    }

    profiler_->runProfiler(num_rows, profile,
                           static_cast<char*>(profile_workspace_.data_ptr()), expert_weights,
                           enable_pdl, stream);
  }
};

}  // namespace

tvm::ffi::Module init(DLDataType activation_dtype, DLDataType weight_dtype,
                      DLDataType output_dtype) {
  return tvm::ffi::Module(
      tvm::ffi::make_object<Bf16FusedMoeRunner>(activation_dtype, weight_dtype, output_dtype));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
