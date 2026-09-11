# SPDX-License-Identifier: Apache-2.0
"""CPU parity tests for the Ascend-safe causal Wan SDPA path."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.models.dits.causal_wanvideo import CausalWanSelfAttention


class _PlainSDPA(nn.Module):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        return output.transpose(1, 2)


def _attention(local_attn_size: int = -1) -> CausalWanSelfAttention:
    module = CausalWanSelfAttention.__new__(CausalWanSelfAttention)
    nn.Module.__init__(module)
    module.local_attn_size = local_attn_size
    module.attn = _PlainSDPA()
    return module


def _masked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros_like(allowed, dtype=q.dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask[None, None],
    )
    return output.transpose(1, 2)


def test_blockwise_causal_sdpa_matches_explicit_mask() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 12, 2, 4) for _ in range(3))
    frame_seqlen = 2
    block_tokens = 2 * frame_seqlen

    got = _attention()._forward_blockwise_sdpa(
        q,
        k,
        v,
        frame_seqlen=frame_seqlen,
        num_frame_per_block=2,
        teacher_forcing=False,
    )
    indices = torch.arange(q.shape[1])
    block_end = ((indices // block_tokens) + 1) * block_tokens
    allowed = indices[None, :] < block_end[:, None]
    expected = _masked_reference(q, k, v, allowed)

    torch.testing.assert_close(got, expected)


def test_teacher_forcing_sdpa_matches_explicit_mask() -> None:
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 24, 2, 4) for _ in range(3))
    clean_tokens = q.shape[1] // 2
    frame_seqlen = 2
    block_tokens = 2 * frame_seqlen
    allowed = torch.zeros(q.shape[1], q.shape[1], dtype=torch.bool)

    for start in range(0, clean_tokens, block_tokens):
        end = min(start + block_tokens, clean_tokens)
        allowed[start:end, :end] = True
        noisy_start = clean_tokens + start
        noisy_end = clean_tokens + end
        allowed[noisy_start:noisy_end, :start] = True
        allowed[noisy_start:noisy_end, noisy_start:noisy_end] = True

    got = _attention()._forward_blockwise_sdpa(
        q,
        k,
        v,
        frame_seqlen=frame_seqlen,
        num_frame_per_block=2,
        teacher_forcing=True,
    )
    expected = _masked_reference(q, k, v, allowed)

    torch.testing.assert_close(got, expected)
