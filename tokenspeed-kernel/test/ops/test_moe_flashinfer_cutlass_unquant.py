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

import torch

from tokenspeed_kernel.ops.moe.flashinfer.cutlass_unquant import (
    _autotune_lock,
    _autotuned_buckets,
    _tune_max_num_tokens,
)


def test_unquant_cutlass_tune_bucket_uses_8192_floor_and_power_of_two_ceiling():
    assert _tune_max_num_tokens(1) == 8192
    assert _tune_max_num_tokens(8192) == 8192
    assert _tune_max_num_tokens(8193) == 16384
    assert _tune_max_num_tokens(16384) == 16384


def test_unquant_cutlass_autotune_state_is_per_module_bucket_set():
    first = torch.nn.Module()
    second = torch.nn.Module()

    _autotuned_buckets(first).add(8192)

    assert _autotuned_buckets(first) == {8192}
    assert _autotuned_buckets(second) == set()
    assert _autotune_lock(first) is _autotune_lock(first)
    assert _autotune_lock(first) is not _autotune_lock(second)
