# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Local BF16/BF16 Cutlass MoE wrapper for AFMoE models."""

from __future__ import annotations

import contextlib
import functools
import os
import threading
from enum import IntEnum
from pathlib import Path
from typing import Optional

import torch


class ActivationType(IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    SwigluStep = 7
    Identity = 8
    InvalidType = 9


_PHASE1_END = 256
_PHASE2_STEP = 256
_PHASE2_END = 2048
_PHASE3_STEP = 512
_PHASE3_END = 4096

_runner_cache: dict[tuple[torch.dtype, torch.dtype, torch.dtype], object] = {}
_tactic_cache: dict[tuple[object, ...], tuple[int, int]] = {}
_cache_lock = threading.Lock()


@functools.cache
def _load_module():
    import tvm_ffi

    so_path = (
        Path(__file__).resolve().parent
        / "objs"
        / "afmoe_cutlass_bf16_moe"
        / "afmoe_cutlass_bf16_moe.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel afmoe_cutlass_bf16_moe library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def _runner(
    activation_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    key = (activation_dtype, weight_dtype, output_dtype)
    with _cache_lock:
        runner = _runner_cache.get(key)
        if runner is None:
            runner = _load_module().init(activation_dtype, weight_dtype, output_dtype)
            _runner_cache[key] = runner
        return runner


def _next_positive_power_of_2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _ceil_to_step(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def _map_to_hybrid_bucket(value: int, max_num_tokens: int) -> int:
    if value <= 0:
        return 1
    if value >= max_num_tokens:
        return max_num_tokens
    if value <= _PHASE1_END:
        return _next_positive_power_of_2(value)
    if value <= _PHASE2_END:
        return min(_ceil_to_step(value, _PHASE2_STEP), max_num_tokens)
    if value <= _PHASE3_END:
        return min(_ceil_to_step(value, _PHASE3_STEP), max_num_tokens)
    return min(_next_positive_power_of_2(value), max_num_tokens)


def _device_support_pdl(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9


def _valid_tactics(runner, stage: int) -> list[int]:
    gemm1_count = int(runner.get_gemm1_tactic_count())
    gemm2_count = int(runner.get_gemm2_tactic_count())
    if stage == 1:
        tactics = range(gemm1_count)
    elif stage == 2:
        tactics = range(gemm1_count, gemm1_count + gemm2_count)
    else:
        raise ValueError(f"invalid GEMM stage: {stage}")

    valid: list[int] = []
    for tactic in tactics:
        try:
            if int(runner.get_tactic_occupancy(tactic)) > 0:
                valid.append(tactic)
        except Exception:
            valid.append(tactic)
    return valid


def _profile_stage(
    runner,
    profile_input: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,
    fc2_expert_biases: Optional[torch.Tensor],
    top_k: int,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
    cluster_size: int,
    cluster_rank: int,
    enable_alltoall: bool,
    enable_pdl: bool,
    activation_type: ActivationType,
    stage: int,
) -> int:
    tactics = _valid_tactics(runner, stage)
    if not tactics:
        return -1

    runner.run_gemm_profile(
        profile_input,
        fc1_expert_weights,
        fc1_expert_biases,
        fc2_expert_weights,
        fc2_expert_biases,
        int(top_k),
        int(tp_size),
        int(tp_rank),
        int(ep_size),
        int(ep_rank),
        int(cluster_size),
        int(cluster_rank),
        bool(enable_alltoall),
        False,
        int(stage),
        -1,
        True,
        bool(enable_pdl),
        int(activation_type),
    )

    warmup = int(os.environ.get("TOKENSPEED_AFMOE_BF16_MOE_TUNE_WARMUP", "2"))
    repeat = int(os.environ.get("TOKENSPEED_AFMOE_BF16_MOE_TUNE_REPEAT", "5"))
    best_tactic = -1
    best_ms = float("inf")

    def run_once(tactic: int) -> None:
        runner.run_gemm_profile(
            profile_input,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            int(top_k),
            int(tp_size),
            int(tp_rank),
            int(ep_size),
            int(ep_rank),
            int(cluster_size),
            int(cluster_rank),
            bool(enable_alltoall),
            False,
            int(stage),
            int(tactic),
            False,
            bool(enable_pdl),
            int(activation_type),
        )

    for tactic in tactics:
        try:
            for _ in range(warmup):
                run_once(tactic)

            stream = torch.cuda.current_stream(profile_input.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(repeat):
                run_once(tactic)
            end.record(stream)
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / max(repeat, 1)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(profile_input.device)
                torch.cuda.cudart().cudaGetLastError()
            elapsed_ms = float("inf")

        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_tactic = tactic

    return best_tactic


def _tactics_for_shape(
    runner,
    input: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,
    fc2_expert_biases: Optional[torch.Tensor],
    top_k: int,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
    cluster_size: int,
    cluster_rank: int,
    enable_alltoall: bool,
    tune_max_num_tokens: int,
    enable_pdl: bool,
    activation_type: ActivationType,
) -> tuple[int, int]:
    if os.environ.get("TOKENSPEED_AFMOE_BF16_MOE_DISABLE_TUNING"):
        return -1, -1

    num_tokens_bucket = _map_to_hybrid_bucket(input.shape[0], tune_max_num_tokens)
    capability = torch.cuda.get_device_capability(input.device)
    key = (
        capability,
        num_tokens_bucket,
        input.shape[1],
        fc1_expert_weights.shape,
        fc1_expert_weights.stride(),
        fc2_expert_weights.shape,
        fc2_expert_weights.stride(),
        int(top_k),
        int(tp_size),
        int(tp_rank),
        int(ep_size),
        int(ep_rank),
        int(cluster_size),
        int(cluster_rank),
        bool(enable_alltoall),
        int(activation_type),
    )

    with _cache_lock:
        cached = _tactic_cache.get(key)
    if cached is not None:
        return cached

    profile_input = input
    if profile_input.shape[0] != num_tokens_bucket:
        profile_input = torch.empty(
            (num_tokens_bucket, input.shape[1]),
            device=input.device,
            dtype=input.dtype,
        )

    with _cache_lock:
        cached = _tactic_cache.get(key)
        if cached is not None:
            return cached
        gemm1 = _profile_stage(
            runner,
            profile_input,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            top_k,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            enable_alltoall,
            enable_pdl,
            activation_type,
            stage=1,
        )
        gemm2 = _profile_stage(
            runner,
            profile_input,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            top_k,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            enable_alltoall,
            enable_pdl,
            activation_type,
            stage=2,
        )
        _tactic_cache[key] = (gemm1, gemm2)
        return gemm1, gemm2


def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: Optional[list[torch.Tensor]] = None,
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_packed_weights: bool = False,
    tune_max_num_tokens: int = 8192,
    enable_pdl: Optional[bool] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    swizzled_input_sf: bool = True,
) -> list[torch.Tensor]:
    if input.dtype != torch.bfloat16:
        raise TypeError("AFMoE local Cutlass MoE only supports BF16 activations")
    if fc1_expert_weights.dtype != torch.bfloat16 or fc2_expert_weights.dtype != torch.bfloat16:
        raise TypeError("AFMoE local Cutlass MoE only supports BF16 weights")
    if output_dtype != torch.bfloat16:
        raise TypeError("AFMoE local Cutlass MoE only supports BF16 output")
    if quant_scales:
        raise NotImplementedError("quant_scales are not used by BF16/BF16 MoE")
    if input_sf is not None:
        raise NotImplementedError("input_sf is only valid for quantized MoE")
    if min_latency_mode:
        raise NotImplementedError("AFMoE local BF16 Cutlass MoE does not support min-latency mode")
    if use_deepseek_fp8_block_scale or use_w4_group_scaling or use_mxfp8_act_scaling:
        raise NotImplementedError("AFMoE local BF16 Cutlass MoE does not support quantized modes")
    if use_packed_weights:
        raise NotImplementedError("AFMoE local BF16 Cutlass MoE does not support packed weights")
    if cluster_size != 1 or cluster_rank != 0:
        raise NotImplementedError("cluster routing is only supported by min-latency mode")

    if enable_pdl is None:
        enable_pdl = _device_support_pdl(input.device)
    activation_type = ActivationType(int(activation_type))
    token_selected_experts = token_selected_experts.to(torch.int32).contiguous()
    token_final_scales = token_final_scales.to(torch.float32).contiguous()

    if output is None:
        output = torch.empty(
            (input.shape[0], fc2_expert_weights.shape[1]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    if input.shape[0] == 0:
        return [
            output,
            torch.empty((1,), dtype=torch.int32, device=input.device),
            torch.empty(
                (fc2_expert_weights.shape[0], 0), dtype=torch.float32, device=input.device
            ),
            torch.empty((fc2_expert_weights.shape[0],), dtype=torch.int32, device=input.device),
        ]

    runner = _runner(input.dtype, fc1_expert_weights.dtype, output_dtype)
    top_k = token_selected_experts.shape[1]
    tactics = _tactics_for_shape(
        runner,
        input,
        fc1_expert_weights,
        fc1_expert_biases,
        fc2_expert_weights,
        fc2_expert_biases,
        top_k,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        enable_alltoall,
        tune_max_num_tokens,
        bool(enable_pdl),
        activation_type,
    )

    runner.run_moe(
        output,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc1_expert_biases,
        fc2_expert_weights,
        fc2_expert_biases,
        None,
        input_sf,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        bool(swizzled_input_sf),
        int(tp_size),
        int(tp_rank),
        int(ep_size),
        int(ep_rank),
        int(cluster_size),
        int(cluster_rank),
        bool(enable_alltoall),
        False,
        list(tactics),
        bool(enable_pdl),
        int(activation_type),
    )

    return [
        output,
        torch.empty((1,), dtype=torch.int32, device=input.device),
        torch.empty(
            (fc2_expert_weights.shape[0], input.shape[0]),
            dtype=torch.float32,
            device=input.device,
        ),
        torch.empty((fc2_expert_weights.shape[0],), dtype=torch.int32, device=input.device),
    ]
