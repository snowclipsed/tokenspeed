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

from __future__ import annotations

import os
import threading

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()
next_power_of_2 = lambda value: 1 if value <= 1 else 1 << (value - 1).bit_length()
_AUTOTUNED_BUCKETS_ATTR = "_flashinfer_cutlass_unquant_autotuned_buckets"
_AUTOTUNE_LOCK_ATTR = "_flashinfer_cutlass_unquant_autotune_lock"


def _tune_max_num_tokens(num_tokens: int) -> int:
    return max(8192, next_power_of_2(num_tokens))


def _autotuned_buckets(w: torch.nn.Module) -> set[int]:
    buckets = getattr(w, _AUTOTUNED_BUCKETS_ATTR, None)
    if buckets is None:
        buckets = set()
        setattr(w, _AUTOTUNED_BUCKETS_ATTR, buckets)
    return buckets


def _autotune_lock(w: torch.nn.Module) -> threading.Lock:
    lock = getattr(w, _AUTOTUNE_LOCK_ATTR, None)
    if lock is None:
        lock = threading.Lock()
        setattr(w, _AUTOTUNE_LOCK_ATTR, lock)
    return lock


if platform.is_nvidia:
    from tokenspeed_kernel.thirdparty.cuda.afmoe_cutlass_bf16_moe import (
        ActivationType as LocalActivationType,
        cutlass_fused_moe as local_cutlass_fused_moe,
    )

    try:
        from flashinfer import ActivationType as FlashInferActivationType
        from flashinfer import cutlass_fused_moe as flashinfer_cutlass_fused_moe
        from flashinfer.autotuner import autotune as flashinfer_autotune
    except Exception as exc:
        FlashInferActivationType = None
        flashinfer_cutlass_fused_moe = None
        flashinfer_autotune = None
        _FLASHINFER_IMPORT_ERROR = exc
    else:
        _FLASHINFER_IMPORT_ERROR = None

    def _use_local_afmoe_bf16_cutlass(x: torch.Tensor, w: torch.nn.Module) -> bool:
        if os.environ.get("TOKENSPEED_AFMOE_BF16_MOE_DISABLE_LOCAL"):
            return False
        return (
            x.dtype == torch.bfloat16
            and w.w13_weight.dtype == torch.bfloat16
            and w.w2_weight.dtype == torch.bfloat16
        )

    def _require_flashinfer():
        if flashinfer_cutlass_fused_moe is None or flashinfer_autotune is None:
            raise RuntimeError(
                "FlashInfer Cutlass MoE is unavailable and the local AFMoE BF16 "
                "path does not support this request"
            ) from _FLASHINFER_IMPORT_ERROR

    def flashinfer_cutlass_unquant_moe_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_unquant_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_unquant_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(8, 9),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_unquant_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
    ):
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        tune_max_num_tokens = _tune_max_num_tokens(x.shape[0])

        def call_local_cutlass_fused_moe():
            return local_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights.float(),
                fc1_expert_weights=w.w13_weight,
                fc2_expert_weights=w.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=getattr(w, "ep_size", 1),
                ep_rank=getattr(w, "ep_rank", 0),
                tp_size=getattr(w, "tp_size", 1),
                tp_rank=getattr(w, "tp_rank", 0),
                tune_max_num_tokens=tune_max_num_tokens,
                activation_type=LocalActivationType.Swiglu,
            )[0]

        def call_flashinfer_cutlass_fused_moe():
            _require_flashinfer()
            return flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights.float(),
                fc1_expert_weights=w.w13_weight,
                fc2_expert_weights=w.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=getattr(w, "ep_size", 1),
                ep_rank=getattr(w, "ep_rank", 0),
                tp_size=getattr(w, "tp_size", 1),
                tp_rank=getattr(w, "tp_rank", 0),
                tune_max_num_tokens=tune_max_num_tokens,
                activation_type=FlashInferActivationType.Swiglu,
            )[0]

        if _use_local_afmoe_bf16_cutlass(x, w):
            return call_local_cutlass_fused_moe()

        _require_flashinfer()
        buckets = _autotuned_buckets(w)
        if tune_max_num_tokens not in buckets:
            with _autotune_lock(w):
                if tune_max_num_tokens not in buckets:
                    with flashinfer_autotune():
                        call_flashinfer_cutlass_fused_moe()
                    buckets.add(tune_max_num_tokens)

        return call_flashinfer_cutlass_fused_moe()
