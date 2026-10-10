"""Block-layout SO(2) sandwich: rotate into m blocks, one GEMM per block, rotate back.

The pair data of one call live in one flat buffer with one block per m. Block m
holds one row per edge, ``[x_{-m} (C_m) | x_{+m} (C_m)]``, so the SO(2) linear of
the block is a single GEMM against the block-complex weight ``[[A, -B], [B, A]]``
built from the stacked pair weight ``[A; B]``:

    [y_{-m} | y_{+m}] = [x_{-m} | x_{+m}] @ [[A, -B], [B, A]]^T

Rotation into the blocks and back out of them are channel-major kernels: one
thread owns one (edge, irrep channel), keeps its 2l+1 coefficients in registers
and reads or writes every coefficient of the feature row once.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch


MAX_BLOCKS = 16


def _ext():
    from .tensor_product import _load_extension

    return _load_extension()


def _side(all_bases, all_ls, block_members, dim, device):
    """Channel tables of one side (input or output) of a layer.

    ``block_members[m]`` lists the channels (first features) of block m in column
    order, or is None when block m is absent. ``cols[k, m]`` is the column of
    channel k in block m, -1 when the channel is not in that block."""
    bases = [int(b) for b in all_bases.detach().cpu().tolist()]
    levels = [int(v) for v in all_ls.detach().cpu().tolist()]
    index = {b: k for k, b in enumerate(bases)}
    width = len(block_members)
    cols = [[-1] * width for _ in bases]
    widths, strides, prefix = [], [], []
    total = 0
    for m, members in enumerate(block_members):
        prefix.append(total)
        if members is None:
            widths.append(0)
            strides.append(0)
            continue
        members = [int(b) for b in members.detach().cpu().tolist()]
        for c, b in enumerate(members):
            cols[index[b]][m] = c
        widths.append(len(members))
        strides.append(len(members) if m == 0 else 2 * len(members))
        total += strides[-1]
    covered = [0] * int(dim)
    for b, l in zip(bases, levels):
        for f in range(b, b + 2 * l + 1):
            if 0 <= f < int(dim):
                covered[f] += 1

    def as_int(values):
        return torch.tensor(values, dtype=torch.int32, device=device).contiguous()

    return SimpleNamespace(
        base=as_int(bases), l=as_int(levels), cols=as_int([c for row in cols for c in row]),
        prefix=prefix, width=widths, stride=strides, total=total,
        zero_fill=any(c != 1 for c in covered), dim=int(dim))


def sandwich_plan(layout, in_dim, device, *, with_m0=False):
    """Blocks and channel tables of a layout, cached on the layout object."""
    cache = layout.__dict__.setdefault("_sandwich_plans", {})
    key = (int(in_dim), bool(with_m0), str(device))
    hit = cache.get(key)
    if hit is not None:
        return hit
    maps = layout.maps
    in_members, out_members, blocks = [], [], []
    for m, (in_base, _in_l, out_base, _out_l, _offsets) in enumerate(maps):
        present = in_base.numel() > 0 and out_base.numel() > 0 and (m > 0 or with_m0)
        in_members.append(in_base if present else None)
        out_members.append(out_base if present else None)
        if present:
            blocks.append((m, int(in_base.numel()), int(out_base.numel())))
    with torch.inference_mode(False), torch.no_grad():
        hit = SimpleNamespace(
            supported=len(maps) <= MAX_BLOCKS,  # the kernels address at most 16 m blocks
            blocks=tuple(blocks),
            inp=_side(maps[0][0], maps[0][1], in_members, in_dim, device),
            out=_side(maps[0][2], maps[0][3], out_members, layout.out_dim, device),
            offsets=maps[0][-1],
            no_rows=torch.empty(0, dtype=torch.long, device=device),
            no_scale=torch.empty(0, dtype=torch.float32, device=device),
        )
    cache[key] = hit
    return hit


def _rotate(src, side, plan, wigner, rotate, rows=None):
    return _ext().channel_rotate_to_blocks_fp32(
        src, wigner.values, plan.offsets, wigner.compact_offsets, side.base, side.l, side.cols,
        side.prefix, side.width, side.stride, side.total,
        plan.no_rows if rows is None else rows, bool(rotate), int(wigner.mode), int(wigner.stride))


def _gather(src, n, side, plan, wigner, rotate, rows=None, scale=None, into=None):
    return _ext().channel_gather_from_blocks_fp32(
        src, int(n), wigner.values, plan.offsets, wigner.compact_offsets, side.base, side.l, side.cols,
        side.prefix, side.width, side.stride, side.dim, bool(side.zero_fill),
        plan.no_scale if scale is None else scale, plan.no_scale if into is None else into,
        plan.no_rows if rows is None else rows, bool(rotate), int(wigner.mode), int(wigner.stride))


def _view(flat, n, prefix, width):
    return flat[n * prefix:n * (prefix + width)].view(n, width)


class _BlockSandwich(torch.autograd.Function):
    """m>0 blocks of one layer call: rotate in, block-complex GEMM, rotate out.

    Front radial weights [N, C_in,m] scale both pair components of a block before
    its GEMM; the scaled block is recomputed for the weight gradient."""

    @staticmethod
    def forward(ctx, x, plan, layout, wigner, n_weights, *tensors):
        ext = _ext()
        weights = tensors[:n_weights]
        radials = tensors[n_weights:]
        n = x.shape[0]
        inp, out = plan.inp, plan.out
        packed = _rotate(x, inp, plan, wigner, layout.rotate_in)
        block_weights = ext.block_complex_weights_fp32([w.contiguous() for w in weights])
        pairs_out = x.new_empty(n * out.total)
        cursor = 0
        for b, (m, cin, cout) in enumerate(plan.blocks):
            block = _view(packed, n, inp.prefix[m], 2 * cin)
            if radials:
                block = (block.view(n, 2, cin) * radials[b].unsqueeze(1)).view(n, 2 * cin)
            weight = block_weights[cursor:cursor + 4 * cout * cin].view(2 * cout, 2 * cin)
            cursor += 4 * cout * cin
            torch.mm(block, weight.t(), out=_view(pairs_out, n, out.prefix[m], 2 * cout))
        result = _gather(pairs_out, n, out, plan, wigner, layout.rotate_out)
        needs = ctx.needs_input_grad
        keep_packed = any(needs[5:])
        keep_weights = needs[0] or any(needs[5 + n_weights:])
        ctx.save_for_backward(packed if keep_packed else None,
                              block_weights if keep_weights else None, *radials)
        ctx.state = (plan, layout, wigner, n_weights, int(x.shape[1]))
        return result

    @staticmethod
    def backward(ctx, grad_out):
        ext = _ext()
        plan, layout, wigner, n_weights, in_dim = ctx.state
        packed, block_weights, *radials = ctx.saved_tensors
        needs = ctx.needs_input_grad
        n = grad_out.shape[0]
        inp, out = plan.inp, plan.out
        grad_pairs = _rotate(grad_out.contiguous(), out, plan, wigner, layout.rotate_out)
        need_x = needs[0]
        need_w = any(needs[5:5 + n_weights])
        grad_packed = grad_out.new_empty(n * inp.total) if need_x else None
        grad_block_weights = grad_out.new_empty(sum(4 * co * ci for _, ci, co in plan.blocks)) if need_w else None
        grad_radials = [None] * len(radials)
        cursor = 0
        for b, (m, cin, cout) in enumerate(plan.blocks):
            size = 4 * cout * cin
            grad_block = _view(grad_pairs, n, out.prefix[m], 2 * cout)
            radial_grad = bool(radials) and needs[5 + n_weights + b]
            if need_x or radial_grad:
                weight = block_weights[cursor:cursor + size].view(2 * cout, 2 * cin)
                if radials:
                    grad_in = torch.mm(grad_block, weight).view(n, 2, cin)
                    if radial_grad:
                        unscaled = _view(packed, n, inp.prefix[m], 2 * cin).view(n, 2, cin)
                        grad_radials[b] = (grad_in * unscaled).sum(dim=1)
                    if need_x:
                        torch.mul(grad_in, radials[b].unsqueeze(1),
                                  out=_view(grad_packed, n, inp.prefix[m], 2 * cin).view(n, 2, cin))
                else:
                    torch.mm(grad_block, weight, out=_view(grad_packed, n, inp.prefix[m], 2 * cin))
            if need_w:
                block = _view(packed, n, inp.prefix[m], 2 * cin)
                if radials:
                    block = (block.view(n, 2, cin) * radials[b].unsqueeze(1)).view(n, 2 * cin)
                torch.mm(grad_block.t(), block, out=grad_block_weights[cursor:cursor + size].view(2 * cout, 2 * cin))
            cursor += size
        grad_weights = [None] * n_weights
        if need_w:
            grad_weights = ext.block_complex_weight_grads_fp32(
                grad_block_weights, [co for _, _, co in plan.blocks], [ci for _, ci, _ in plan.blocks])
        grad_x = _gather(grad_packed, n, inp, plan, wigner, layout.rotate_in) if need_x else None
        return (grad_x, None, None, None, None, *grad_weights, *grad_radials)


def block_sandwich(x, layout, wigner, weights_by_m, radial_parts=None):
    """m>0 contribution [N, out_dim] of one [2*Cout_m, Cin_m] pair weight per m.

    ``weights_by_m`` and ``radial_parts`` are indexed by m (index 0 unused); radial
    weights are front weights [N, Cin_m]. Returns a zero contribution when no m>0
    block exists; callers check ``sandwich_plan(...).supported`` first."""
    plan = sandwich_plan(layout, x.shape[1], x.device)
    if not plan.blocks:
        return x.new_zeros((x.shape[0], layout.out_dim))
    weights = tuple(weights_by_m[m] for m, _, _ in plan.blocks)
    radials = () if radial_parts is None else tuple(radial_parts[m] for m, _, _ in plan.blocks)
    return _BlockSandwich.apply(x.contiguous(), plan, layout, wigner, len(weights), *weights, *radials)
