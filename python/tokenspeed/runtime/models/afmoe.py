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

"""Inference-only AFMoE/Trinity model compatible with HuggingFace weights."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
from torch import nn

from tokenspeed.runtime.configs.afmoe_config import AfmoeConfig
from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoELayer,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.utils import get_layer_id
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.utils import add_prefix, make_layers
from tokenspeed.runtime.utils.env import global_server_args_dict


def _is_moe_layer(layer_id: int, config: AfmoeConfig) -> bool:
    return layer_id >= config.num_dense_layers


def _routing_method_type(config: AfmoeConfig) -> RoutingMethodType:
    if config.score_func == "sigmoid" and config.route_norm:
        return RoutingMethodType.SigmoidRenorm
    if config.score_func == "softmax" and config.route_norm:
        return RoutingMethodType.RenormalizeNaive
    return RoutingMethodType.TopK


def _remap_fp8_block_scale_name(
    name: str, quant_config: QuantizationConfig | None
) -> str:
    if (
        isinstance(quant_config, CompressedTensorsConfig)
        and quant_config.is_fp8_block_quantized()
        and name.endswith(".weight_scale")
    ):
        return f"{name}_inv"
    return name


class AfmoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_shared_expert: bool = False,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        if is_shared_expert:
            tp_rank = self.mapping.moe.tp_ep_rank
            tp_size = self.mapping.moe.tp_ep_size
            tp_group = self.mapping.moe.tp_ep_group
        else:
            tp_rank = self.mapping.dense.tp_rank
            tp_size = self.mapping.dense.tp_size
            tp_group = self.mapping.dense.tp_group

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class AfmoeMoE(nn.Module):
    def __init__(
        self,
        config: AfmoeConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_index,
            is_moe=True,
            prev_is_moe=_is_moe_layer(layer_index - 1, config),
        )

        if self.mapping.moe.ep_size > config.num_experts:
            raise ValueError(
                f"EP size {self.mapping.moe.ep_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router.gate", prefix),
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        if config.num_shared_experts > 0:
            self.shared_experts = AfmoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.num_shared_experts
                ),
                hidden_act=config.hidden_act,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                is_shared_expert=True,
            )
        else:
            self.shared_experts = None

        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            routing_config={
                "n_group": getattr(config, "n_group", 1),
                "topk_group": getattr(config, "topk_group", 1),
                "routed_scaling_factor": getattr(config, "route_scale", 1.0),
                "correction_bias": self.expert_bias,
                "routing_method_type": _routing_method_type(config),
            },
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.route_norm,
            use_grouped_topk=False,
            custom_routing_function=self._select_experts,
            correction_bias=None,
            routed_scaling_factor=config.route_scale,
            output_format=self.experts.topk_output_format,
        )

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states
        if self.config.score_func == "sigmoid":
            scores = torch.sigmoid(gating_output.float())
        elif self.config.score_func == "softmax":
            scores = F.softmax(gating_output.float(), dim=-1)
        else:
            raise ValueError(f"Unsupported AFMoE score_func: {self.config.score_func}")

        _, topk_ids = torch.topk(scores + self.expert_bias, k=topk, dim=-1)
        topk_weights = scores.gather(dim=-1, index=topk_ids)
        if self.config.score_func == "sigmoid" and renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def get_moe_routed_weights(self) -> list[torch.Tensor]:
        return [
            param.data
            for name, param in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def forward(self, hidden_states: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(ctx)

        router_logits, _ = self.router(hidden_states)
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        router_logits = self.comm_manager.pre_mlp_comm(router_logits, ctx)

        shared_output = None
        if self.shared_experts is not None and hidden_states.shape[0] > 0:
            shared_output = self.shared_experts(hidden_states)

        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        final_hidden_states, _ = self.comm_manager.post_mlp_comm(
            final_hidden_states, None, ctx
        )
        return final_hidden_states.view(num_tokens, hidden_dim)


class AfmoeAttention(nn.Module):
    def __init__(
        self,
        config: AfmoeConfig,
        mapping: Mapping,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.tp_rank = self.mapping.attn.tp_rank
        self.tp_size = self.mapping.attn.tp_size
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.layer_types[layer_id] == "sliding_attention"

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

        rope_theta = get_rope_theta(config, 10000)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        sliding_window_size = config.sliding_window if self.is_local_attention else -1
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return qk_rmsnorm(
            q,
            k,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
            self.q_norm.variance_epsilon,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate_states, _ = self.gate_proj(hidden_states)
        q, k = self._apply_qk_norm(q, k)
        if self.is_local_attention:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, ctx, out_cache_loc)
        if len(attn_output.size()) == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        attn_output = attn_output * torch.sigmoid(gate_states)
        output, _ = self.o_proj(attn_output)
        return output


class AfmoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: AfmoeConfig,
        mapping: Mapping,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.moe_enabled = _is_moe_layer(layer_id, config)

        self.self_attn = AfmoeAttention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.moe_enabled:
            self.mlp = AfmoeMoE(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = AfmoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.moe_enabled,
            prev_is_moe=_is_moe_layer(layer_id - 1, config),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
        residual = hidden_states

        attn_input = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=attn_input,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        hidden_states, residual = self.comm_manager.post_attn_comm(
            hidden_states, residual, ctx
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        if self.moe_enabled:
            hidden_states = self.mlp(hidden_states, ctx)
        else:
            hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
            hidden_states = self.mlp(hidden_states)
            hidden_states, residual = self.comm_manager.post_mlp_comm(
                hidden_states, residual, ctx
            )
        hidden_states = residual + self.post_mlp_layernorm(hidden_states)
        return hidden_states


class AfmoeModel(nn.Module):
    def __init__(
        self,
        config: AfmoeConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: AfmoeDecoderLayer(
                config=config,
                mapping=self.mapping,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if self.config.mup_enabled:
            hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
            )

        if not ctx.forward_mode.is_idle():
            hidden_states = self.norm(hidden_states)
            hidden_states, _ = self.layers[-1].comm_manager.post_final_norm_comm(
                hidden_states, hidden_states, ctx
            )
        return hidden_states, None

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = self.mapping.attn.tp_size
        tp_rank = self.mapping.attn.tp_rank
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling factor.")


class AfmoeForCausalLM(BaseCausalLM):
    model_cls = AfmoeModel

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> None:
        del kwargs
        qkv_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        mlp_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
            ".weight_shape",
        )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        for name, loaded_weight in weights:
            if "Embedding" in getattr(self.config, "name_or_path", ""):
                name = add_prefix(name, "model")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if ".mlp.router.gate." in name:
                name = name.replace(".mlp.router.gate.", ".mlp.router.")

            loaded = False
            if ".self_attn." in name:
                for param_name, weight_name, shard_id in qkv_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = _remap_fp8_block_scale_name(
                        name.replace(weight_name, param_name), self.quant_config
                    )
                    if mapped_name.endswith(ignore_suffixes) and mapped_name not in params_dict:
                        loaded = True
                        break
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded = True
                    break
                if loaded:
                    continue

            if ".mlp.experts." not in name and ".mlp." in name:
                for param_name, weight_name, shard_id in mlp_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = _remap_fp8_block_scale_name(
                        name.replace(weight_name, param_name), self.quant_config
                    )
                    if mapped_name.endswith(ignore_suffixes) and mapped_name not in params_dict:
                        loaded = True
                        break
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded = True
                    break
                if loaded:
                    continue

            moe_name = _remap_fp8_block_scale_name(name, self.quant_config)
            if moe_loader.matches(moe_name):
                moe_loader.load(moe_name, loaded_weight)
                continue

            param_name = _remap_fp8_block_scale_name(name, self.quant_config)
            if name.endswith(ignore_suffixes) and param_name not in params_dict:
                continue
            if param_name not in params_dict:
                continue
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = AfmoeForCausalLM
