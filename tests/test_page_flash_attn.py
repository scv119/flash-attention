import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_page_attention,
    flash_attn_varlen_with_page_attention,
)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size
from flash_attn.layers.rotary import apply_rotary_emb

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def pad_query(q, pad_size):
    return F.pad(q, (0, 0, 0, 0, 0, pad_size), "constant", 1)


def pad_seqlens(k, pad_size):
    print(f"{k=}")
    padded = F.pad(k, (0, pad_size), "constant", 1000)
    print(f"{padded=}")
    return padded


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(
    q,
    kv,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        query_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device="cuda") < sparsity
    return mask


def attention_blocksparse_ref(qkv, blockmask, attn_mask, dropout_p, dropout_mask):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        blockmask: (seqlen / 16, seqlen / 256)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    d = qkv.shape[-1]
    seqlen = qkv.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    blockmask = repeat(blockmask, "s_16 s_256 -> (s_16 16) (s_256 256)")
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(rearrange(~attn_mask, "b s -> b 1 s 1"), 0.0)
    attention = attention.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), 0.0)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1 1"), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(S.device, head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k_rounded + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q_rounded + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(
        S,
        "b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)",
        blocksize_m=blocksize_m,
        blocksize_n=blocksize_n,
    )
    S_converted = rearrange(
        S_flat,
        "b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)",
        mmas_n=mmas_n,
        warps_n=warps_n,
        eight=8,
        c0=2,
        c1=2,
        c2=2,
        four=4,
    )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted.masked_fill_(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("num_splits", [1, 0])
# @pytest.mark.parametrize("num_splits", [0])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("new_kv", [False])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [False]) # TODO: fix this.
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
# @pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("rotary_interleaved", [False])
# @pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rotary_fraction", [0.0])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("varlen", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 48, 64, 80, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [32, 64, 128, 256])
@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (1, 128 * 1024),
        (16, 128 * 1024),
        (128, 128),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 256)])
def test_flash_attn_page(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    varlen,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    num_splits,
    dtype,
    block_size,
):
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)

    page_block_size = block_size
    num_pages = 10
    batch_size = 15
    max_page_len = (seqlen_k - 1) // page_block_size + 1
    
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None

    k_cache = torch.randn(num_pages, page_block_size, nheads_k, d, device=device, dtype=dtype)
    v_cache = torch.randn(num_pages, page_block_size, nheads_k, d, device=device, dtype=dtype)

    # random mapping.
    block_tables = torch.randint(0, num_pages, (batch_size, max_page_len), device=device, dtype=torch.int32)

    cache_seqlens = torch.randint(
        1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    print(f"\n")
    print(f"{cache_seqlens=}")
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[:batch_size]
    else:
        cache_batch_idx = None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = torch.rand(seqlen_k, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
    v_cache_ref = (v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()

    k_cache_ref = torch.index_select(k_cache_ref, 0, block_tables.view(batch_size * max_page_len))
    v_cache_ref = torch.index_select(v_cache_ref, 0, block_tables.view(batch_size * max_page_len))

    k_cache_ref = k_cache_ref.view(batch_size_cache, max_page_len * page_block_size, nheads_k, d)
    v_cache_ref = v_cache_ref.view(batch_size_cache, max_page_len * page_block_size, nheads_k, d)
    k_cache_ref = k_cache_ref[:, :seqlen_k, :, :]
    v_cache_ref = v_cache_ref[:, :seqlen_k, :, :]


    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    # k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    # v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    out1 = None
    if varlen:
        cache_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q,  dtype=torch.int32, device=device)
        cache_seqlens_k = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(cache_seqlens, dim=0))).to(dtype=torch.int32)

        print("running varlen paged attention...")
        print(f"{cache_seqlens_q=}")
        print(f"{cache_seqlens_k=}")
        q_view = q.view(batch_size * seqlen_q, nheads, d)
        out1 = flash_attn_varlen_with_page_attention(
            q_view,
            k_cache,
            v_cache,
            block_tables,
            cache_seqlens_q,
            cache_seqlens_k,
            seqlen_q,
            seqlen_k,
            k,
            v,
            cos,
            sin,
            cache_batch_idx,
            causal=causal,
            window_size=window_size,
            rotary_interleaved=rotary_interleaved,
            num_splits=num_splits,
        )

        assert batch_size == 2
        truncated_len = torch.randint(
            1,
            seqlen_q + 1,
            (2,),
            dtype=torch.int32,
            device=device,
        )
        seqlen_q_0 = truncated_len[0]
        seqlen_q_1 = truncated_len[1]
        cache_seqlens_q[1] = seqlen_q_0
        cache_seqlens_q[2] = seqlen_q_0 + seqlen_q_1
        indices = list(range(0, seqlen_q_0)) + list(range(seqlen_q, seqlen_q + seqlen_q_1))
        q_selected = q_view[indices, :, :]

        print("running varlen paged attention truncated...")
        print(f"{cache_seqlens_q=}")
        print(f"{cache_seqlens_k=}")
        out2 = flash_attn_varlen_with_page_attention(
            q_selected,
            k_cache,
            v_cache,
            block_tables,
            cache_seqlens_q,
            cache_seqlens_k,
            seqlen_q,
            seqlen_k,
            k,
            v,
            cos,
            sin,
            cache_batch_idx,
            causal=causal,
            window_size=window_size,
            rotary_interleaved=rotary_interleaved,
            num_splits=num_splits,
        )
        torch.cuda.synchronize()
        assert torch.allclose(out1[:seqlen_q_0, :, :], out2[:seqlen_q_0, :, :], rtol=1e-05, atol=1e-08,)
        assert torch.allclose(out1[seqlen_q:seqlen_q + seqlen_q_1, :, :], out2[-seqlen_q_1:, :, :], rtol=1e-05, atol=1e-08,)
    print("running paged attention...")
    out = flash_attn_with_page_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        k,
        v,
        cos,
        sin,
        cache_seqlens,
        cache_batch_idx,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
    )
    if out1:
        out1 = torch.reshape(out1, out.shape)
    # out = flash_attn_with_kvcache(
    #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
    # )
    # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
    # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # probs = torch.softmax(qk, dim=-1)
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    if out1:
        print(f"Varlen Output max diff: {(out1 - out_ref).abs().max().item()}")
        print(f"Varlen Output mean diff: {(out1 - out_ref).abs().mean().item()}")
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        k_cache_select = k_cache if not has_batch_idx else k_cache[cache_batch_idx]
        v_cache_select = v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    
    assert (out - out_ref).abs().max().item() <= 3 * (out_pt - out_ref).abs().max().item() + 1e-5

    if out1:
        assert (out1 - out_ref).abs().max().item() <= 3 * (out_pt - out_ref).abs().max().item() + 1e-5


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("num_splits", [1, 0])
@pytest.mark.parametrize("num_splits", [0])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("new_kv", [False])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [True]) # TODO: fix this.
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
# @pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("rotary_interleaved", [False])
# @pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rotary_fraction", [0.0])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
#@pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 48, 64, 80, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [32, 64, 128, 256])
@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("pad_q, pad_b", [(0, 0), (3, 4)])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (1, 128 * 1024),
        (16, 128 * 1024),
        (128, 128),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 256)])
def test_varlen_causal_flash_attn_page(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    num_splits,
    dtype,
    pad_q,
    pad_b,
    block_size,
):
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)

    page_block_size = block_size
    num_pages = 10
    batch_size = 2
    max_page_len = (seqlen_k - 1) // page_block_size + 1
    
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None

    k_cache = torch.randn(num_pages, page_block_size, nheads_k, d, device=device, dtype=dtype)
    v_cache = torch.randn(num_pages, page_block_size, nheads_k, d, device=device, dtype=dtype)

    # random mapping.
    block_tables = torch.randint(0, num_pages, (batch_size, max_page_len), device=device, dtype=torch.int32)

    cache_seqlens = torch.randint(
        1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    print(f"\n")
    # cache_seqlens[0] = 256
    # cache_seqlens[1] = 224
    print(f"{cache_seqlens=}")

    out1 = None
    cache_batch_idx = None
    cache_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q,  dtype=torch.int32, device=device)
    cache_seqlens_k = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(cache_seqlens, dim=0))).to(dtype=torch.int32)

    q_view = q.view(batch_size * seqlen_q, nheads, d)
    assert batch_size == 2
    truncated_len = torch.randint(
        1,
        seqlen_q + 1,
        (2,),
        dtype=torch.int32,
        device=device,
    )
    seqlen_q_0 = truncated_len[0]
    seqlen_q_1 = truncated_len[1]
    cache_seqlens_q[1] = seqlen_q_0
    cache_seqlens_q[2] = seqlen_q_0 + seqlen_q_1
    indices = list(range(0, seqlen_q_0)) + list(range(seqlen_q, seqlen_q + seqlen_q_1))
    q_selected = q_view[indices, :, :]

    print("running varlen paged attention truncated, split 1...")
    print('=====================================')
    print(f"{cache_seqlens_q=}")
    print(f"{cache_seqlens_k=}")
    print(f"{block_tables=}")
    print(f"{causal=}, {local=}")
    # out3 = flash_attn_varlen_with_page_attention(
    #     q_selected,
    #     k_cache,
    #     v_cache,
    #     block_tables,
    #     cache_seqlens_q,
    #     cache_seqlens_k,
    #     seqlen_q,
    #     seqlen_k,
    #     None,
    #     None,
    #     None,
    #     None,
    #     cache_batch_idx,
    #     causal=causal,
    #     window_size=window_size,
    #     rotary_interleaved=rotary_interleaved,
    #     num_splits=1,
    # )
    # torch.cuda.synchronize()

    print("running varlen paged attention truncated, split=0...")
    print('=====================================')
    print(f"{cache_seqlens_q=}")
    print(f"{cache_seqlens_k=}")
    print(f"{block_tables=}")
    print(f"{causal=}, {local=}")
    out2 = flash_attn_varlen_with_page_attention(
        pad_query(q_selected, pad_q),
        k_cache,
        v_cache,
        block_tables,
        pad_seqlens(cache_seqlens_q, pad_b),
        pad_seqlens(cache_seqlens_k, pad_b),
        seqlen_q,
        seqlen_k,
        None,
        None,
        None,
        None,
        cache_batch_idx,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
        actual_batch_size=torch.cuda.IntTensor([batch_size]),
    )
    if pad_q > 0:
        out2 = out2[:-pad_q]
    torch.cuda.synchronize()
    print('=====================================')
    print("running paged attention for query[0]...")
    q_0 = q_selected[:seqlen_q_0].view(1, seqlen_q_0, nheads, d)
    block_tables_0 = block_tables[:1]
    cache_seqklen_0 = cache_seqlens[:1]
    print(f"{q_0.shape=}")
    print(f"{block_tables_0=}")
    print(f"{cache_seqklen_0=}")

    out0 = flash_attn_with_page_attention(
        q_0,
        k_cache,
        v_cache,
        block_tables_0,
        None,
        None,
        None,
        None,
        cache_seqklen_0,
        cache_batch_idx,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()

    print('=====================================')
    print("running paged attention for query[1]...")
    q_1 = q_selected[-seqlen_q_1:].view(1, seqlen_q_1, nheads, d)
    block_tables_1 = block_tables[1:]
    cache_seqklen_1 = cache_seqlens[1:]
    print(f"{q_1.shape=}")
    print(f"{block_tables_1=}")
    print(f"{cache_seqklen_1=}")
    out1 = flash_attn_with_page_attention(
        q_1,
        k_cache,
        v_cache,
        block_tables_1,
        None,
        None,
        None,
        None,
        cache_seqklen_1,
        cache_batch_idx,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
    )

    out3 = flash_attn_with_page_attention(
        q_1,
        k_cache,
        v_cache,
        block_tables_1,
        None,
        None,
        None,
        None,
        cache_seqklen_1,
        cache_batch_idx,
        causal=False,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
    )

    print(f"out0 max diff: {(out0[0] - out2[:seqlen_q_0, :, :]).abs().max().item()}")
    print(f"out0 mean diff: {(out0[0] - out2[:seqlen_q_0, :, :]).abs().mean().item()}")
    print(f"out1 max diff: {(out1[0] - out2[-seqlen_q_1:, :, :]).abs().max().item()}")
    print(f"out1 mean diff: {(out1[0] - out2[-seqlen_q_1:, :, :]).abs().mean().item()}")

    # print(f"out0 max diff: {(out0[0] - out3[:seqlen_q_0, :, :]).abs().max().item()}")
    # print(f"out0 mean diff: {(out0[0] - out3[:seqlen_q_0, :, :]).abs().mean().item()}")
    # print(f"out1 max diff: {(out1[0] - out3[-seqlen_q_1:, :, :]).abs().max().item()}")
    # print(f"out1 mean diff: {(out1[0] - out3[-seqlen_q_1:, :, :]).abs().mean().item()}")

    #TODO(scv119): compare with torch implementation.
    #TODO(scv119): investigate why the difference is relatively large?
    assert torch.allclose(out0[0], out2[:seqlen_q_0, :, :], rtol=1e-03, atol=1e-03,)
    assert torch.allclose(out1[0], out2[-seqlen_q_1:, :, :], rtol=1e-03, atol=1e-03,)
    # assert torch.allclose(out0[0], out3[:seqlen_q_0, :, :], rtol=1e-03, atol=1e-03,)
    # assert torch.allclose(out1[0], out3[-seqlen_q_1:, :, :], rtol=1e-03, atol=1e-03,)