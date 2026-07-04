import json
import unittest

import torch

from tokenspeed.runtime.configs.afmoe_config import AfmoeConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.dense import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import MergedColumnParallelLinear
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY, get_config


def _tiny_afmoe_config() -> AfmoeConfig:
    return AfmoeConfig(
        architectures=["AfmoeForCausalLM"],
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_dense_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=8,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )


def _single_rank_mapping() -> Mapping:
    mapping = Mapping(rank=0, world_size=1)
    global_server_args_dict["mapping"] = mapping
    return mapping


def _fp8_block_compressed_tensors_config() -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "block",
                        "block_structure": [128, 128],
                        "dynamic": False,
                        "symmetric": True,
                    },
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "group",
                        "group_size": 128,
                        "dynamic": True,
                        "symmetric": True,
                    },
                }
            },
            "ignore": [],
        }
    )


class TestAfmoeConfig(unittest.TestCase):
    def test_config_registry(self):
        self.assertEqual(AfmoeConfig.model_type, "afmoe")
        self.assertIs(_CONFIG_REGISTRY["afmoe"], AfmoeConfig)

    def test_get_config_loads_trinity_mini_shape(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["AfmoeForCausalLM"],
                        "attention_dropout": 0.0,
                        "global_attn_every_n_layers": 4,
                        "head_dim": 128,
                        "hidden_act": "silu",
                        "hidden_size": 2048,
                        "intermediate_size": 6144,
                        "layer_types": [
                            "sliding_attention",
                            "sliding_attention",
                            "sliding_attention",
                            "full_attention",
                        ]
                        * 8,
                        "max_position_embeddings": 131072,
                        "model_type": "afmoe",
                        "moe_intermediate_size": 1024,
                        "mup_enabled": True,
                        "n_group": 1,
                        "num_attention_heads": 32,
                        "num_dense_layers": 2,
                        "num_expert_groups": 1,
                        "num_experts": 128,
                        "num_experts_per_tok": 8,
                        "num_hidden_layers": 32,
                        "num_key_value_heads": 4,
                        "num_limited_groups": 1,
                        "num_shared_experts": 1,
                        "rms_norm_eps": 1e-5,
                        "rope_theta": 10000,
                        "route_norm": True,
                        "route_scale": 2.826,
                        "score_func": "sigmoid",
                        "sliding_window": 2048,
                        "tie_word_embeddings": False,
                        "topk_group": 1,
                        "use_grouped_mm": True,
                        "vocab_size": 200192,
                    }
                )
            )

            config = get_config(tmpdir, trust_remote_code=False)

        self.assertIsInstance(config, AfmoeConfig)
        self.assertEqual(config.architectures, ["AfmoeForCausalLM"])
        self.assertEqual(config.num_experts, 128)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.num_shared_experts, 1)
        self.assertEqual(config.layer_types[3], "full_attention")

    def test_model_registry_resolves_afmoe(self):
        from tokenspeed.runtime.models.afmoe import AfmoeForCausalLM
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["AfmoeForCausalLM"])
        self.assertIs(cls, AfmoeForCausalLM)
        self.assertEqual(arch, "AfmoeForCausalLM")

    def test_compressed_tensors_fp8_block_uses_fp8_linear_method(self):
        quant_config = _fp8_block_compressed_tensors_config()

        self.assertTrue(quant_config.is_fp8_block_quantized())
        self.assertEqual(quant_config.weight_block_size, [128, 128])
        self.assertEqual(quant_config.moe_weight_dtype(), "fp8")

        layer = MergedColumnParallelLinear(
            128,
            [128, 128],
            bias=False,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.gate_up_proj",
        )

        self.assertIsInstance(layer.quant_method, Fp8LinearMethod)
        self.assertEqual(tuple(layer.weight.shape), (256, 128))
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (2, 1))

    def test_remaps_compressed_tensors_fp8_block_scale_names(self):
        from tokenspeed.runtime.models.afmoe import _remap_fp8_block_scale_name

        quant_config = _fp8_block_compressed_tensors_config()

        self.assertEqual(
            _remap_fp8_block_scale_name(
                "model.layers.0.mlp.gate_up_proj.weight_scale",
                quant_config,
            ),
            "model.layers.0.mlp.gate_up_proj.weight_scale_inv",
        )
        self.assertEqual(
            _remap_fp8_block_scale_name(
                "model.layers.0.mlp.gate_up_proj.weight",
                quant_config,
            ),
            "model.layers.0.mlp.gate_up_proj.weight",
        )

    def test_constructs_dense_then_moe_layers(self):
        from tokenspeed.runtime.models.afmoe import AfmoeForCausalLM, AfmoeMoE

        model = AfmoeForCausalLM(
            _tiny_afmoe_config(),
            mapping=_single_rank_mapping(),
        )

        self.assertFalse(model.model.layers[0].moe_enabled)
        self.assertTrue(model.model.layers[1].moe_enabled)
        self.assertIsInstance(model.model.layers[1].mlp, AfmoeMoE)
        self.assertTrue(model.model.layers[0].self_attn.is_local_attention)
        self.assertFalse(model.model.layers[1].self_attn.is_local_attention)

    def test_loads_router_and_unfused_expert_weights(self):
        from tokenspeed.runtime.models.afmoe import AfmoeForCausalLM

        model = AfmoeForCausalLM(
            _tiny_afmoe_config(),
            mapping=_single_rank_mapping(),
        )
        weights = [
            (
                "model.layers.1.mlp.router.gate.weight",
                torch.full((4, 16), 3.0),
            ),
            (
                "model.layers.1.mlp.expert_bias",
                torch.arange(4, dtype=torch.float32),
            ),
        ]
        for expert_id in range(4):
            weights.extend(
                [
                    (
                        f"model.layers.1.mlp.experts.{expert_id}.gate_proj.weight",
                        torch.full((8, 16), 1.0 + expert_id),
                    ),
                    (
                        f"model.layers.1.mlp.experts.{expert_id}.up_proj.weight",
                        torch.full((8, 16), 11.0 + expert_id),
                    ),
                    (
                        f"model.layers.1.mlp.experts.{expert_id}.down_proj.weight",
                        torch.full((16, 8), 21.0 + expert_id),
                    ),
                ]
            )

        model.load_weights(weights)

        params = dict(model.named_parameters())
        self.assertEqual(params["model.layers.1.mlp.router.weight"].mean().item(), 3.0)
        torch.testing.assert_close(
            params["model.layers.1.mlp.expert_bias"],
            torch.arange(4, dtype=torch.float32),
        )
        w13 = params["model.layers.1.mlp.experts.w13_weight"]
        w2 = params["model.layers.1.mlp.experts.w2_weight"]
        self.assertEqual(w13[0, :8].mean().item(), 1.0)
        self.assertEqual(w13[0, 8:].mean().item(), 11.0)
        self.assertEqual(w2[0].mean().item(), 21.0)

    def test_moe_forward_uses_explicit_post_comm(self):
        from tokenspeed.runtime.models.afmoe import AfmoeMoE

        class FakeComm:
            def __init__(self):
                self.post_mlp_comm_calls = 0
                self.post_mlp_fused_calls = 0

            def get_num_tokens(self, ctx):
                return 2, 2

            def pre_mlp_comm(self, tensor, ctx):
                return tensor

            def post_mlp_comm(self, tensor, residual, ctx):
                self.post_mlp_comm_calls += 1
                return tensor + 1, residual

            def post_mlp_fused(self, tensor, residual, ctx):
                self.post_mlp_fused_calls += 1
                return tensor + 100, residual

        class FakeRouter(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states.new_zeros((hidden_states.shape[0], 4)), None

        class FakeTopK(torch.nn.Module):
            def forward(self, hidden_states, router_logits):
                return object()

        class FakeExperts(torch.nn.Module):
            def forward(
                self,
                hidden_states,
                topk_output,
                num_global_tokens,
                max_num_tokens_per_gpu,
            ):
                return torch.full_like(hidden_states, 5)

        moe = AfmoeMoE(
            _tiny_afmoe_config(),
            mapping=_single_rank_mapping(),
            layer_index=1,
        )
        fake_comm = FakeComm()
        moe.comm_manager = fake_comm
        moe.router = FakeRouter()
        moe.shared_experts = None
        moe.topk = FakeTopK()
        moe.experts = FakeExperts()

        output = moe(torch.ones(2, 16), ctx=object())

        self.assertEqual(fake_comm.post_mlp_comm_calls, 1)
        self.assertEqual(fake_comm.post_mlp_fused_calls, 0)
        torch.testing.assert_close(output, torch.full((2, 16), 6.0))


if __name__ == "__main__":
    unittest.main()
