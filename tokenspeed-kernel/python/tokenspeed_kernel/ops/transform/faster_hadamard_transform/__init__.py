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

import logging

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)
platform = current_platform()

_native_hadamard_transform = None
_native_import_error: BaseException | None = None

if platform.is_nvidia:
    try:
        from fast_hadamard_transform import hadamard_transform as _native_hadamard_transform
    except (ImportError, OSError) as exc:
        _native_import_error = exc
        logger.debug(
            "Skipping fast_hadamard_transform backend registration: %s",
            exc,
            exc_info=True,
        )


if _native_hadamard_transform is not None:
    @register_kernel(
        "transform",
        "hadamard_transform",
        name="fast_hadamard_transform",
        solution="fast_hadamard_transform",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        priority=Priority.PERFORMANT,
    )
    def fast_hadamard_transform(
        x: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> torch.Tensor:
        return _native_hadamard_transform(x, scale=scale)
else:

    def fast_hadamard_transform(
        x: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if not platform.is_nvidia:
            raise RuntimeError(
                "fast_hadamard_transform is only available on NVIDIA platforms"
            )
        raise RuntimeError(
            "fast_hadamard_transform is unavailable. Install a "
            "tokenspeed-fast-hadamard-transform wheel built for the active "
            "Torch/CUDA ABI, or use solution='triton' for supported shapes."
        ) from _native_import_error


__all__ = ["fast_hadamard_transform"]
