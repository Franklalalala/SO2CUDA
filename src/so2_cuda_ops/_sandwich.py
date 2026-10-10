"""Block-layout SO(2) sandwich: rotate into m blocks, one GEMM per block, rotate back.

The data of one call live in one flat buffer with one block per m. Block m > 0
holds one row per edge, ``[x_{-m} (C_m) | x_{+m} (C_m)]``, so the SO(2) linear of
the block is a single GEMM against the block-complex weight ``[[A, -B], [B, A]]``
built from the stacked pair weight ``[A; B]``:

    [y_{-m} | y_{+m}] = [x_{-m} | x_{+m}] @ [[A, -B], [B, A]]^T

Block 0 (optional) holds the m = 0 coefficients and uses the plain [Cout, Cin]
weight and bias. Rotation into the blocks and back out of them are channel-major
kernels: one thread owns one (edge, irrep channel), keeps its 2l+1 coefficients
in registers and reads or writes every coefficient of the feature row once.
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

    table = [v for m in range(width) for v in (prefix[m], widths[m], strides[m])]
    return SimpleNamespace(
        base=as_int(bases), l=as_int(levels), cols=as_int([c for row in cols for c in row]),
        table=torch.tensor(table, dtype=torch.long, device=device),
        prefix=prefix, width=widths, stride=strides, total=total,
        zero_fill=any(c != 1 for c in covered), dim=int(dim))


def sandwich_plan(layout, in_dim, device, *, with_m0=False):
    """Blocks and channel tables of a layout, cached on the layout object."""
    cache = layout.__dict__.setdefault("_sandwich_plans", {})
    key = (int(in_dim), bool(with_m0), str(device))
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Load (on first use, build) the extension before any table reaches the device.
    # Loading it right after the tables' host-to-device copies, in a process that
    # had run a CUDA profiling tool, crashed the first kernel launch in the driver.
    _ext()
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


def _rotate(src, side, plan, wigner, rotate, scale=None, rows=None):
    return _ext().channel_rotate_to_blocks_fp32(
        src, wigner.values, plan.offsets, wigner.compact_offsets, side.base, side.l, side.cols,
        side.table, side.total, plan.no_scale if scale is None else scale,
        plan.no_rows if rows is None else rows, bool(rotate), int(wigner.mode), int(wigner.stride))


def _gather(src, n, side, plan, wigner, rotate, scale=None, rows=None):
    return _ext().channel_gather_from_blocks_fp32(
        src, int(n), wigner.values, plan.offsets, wigner.compact_offsets, side.base, side.l, side.cols,
        side.table, side.dim, bool(side.zero_fill), plan.no_scale if scale is None else scale,
        plan.no_scale, plan.no_rows if rows is None else rows, bool(rotate), int(wigner.mode),
        int(wigner.stride))


def _gemm_ext():
    from ._cublas_grouped_gemm import _load_extension

    return _load_extension()


def _pair_gemm(block, weight, out, spec, transpose=False):
    """out = block @ W^T (or @ W when transpose) for one m block; grouped by spec.ptr rows."""
    if spec.groups:
        from ._cublas_grouped_gemm import _loop_max
        _gemm_ext().grouped_gemm_into_fp32(block, spec.ptr, weight, out, bool(transpose), False,
                                           _loop_max((weight,)))
    else:
        torch.mm(block, weight if transpose else weight.t(), out=out)


def _pair_weight_grad(grad_block, block, out, spec):
    """out = grad_block^T @ block per group (or for the single weight)."""
    if spec.groups:
        from ._cublas_grouped_gemm import _loop_max
        _gemm_ext().grouped_gemm_weight_grad_into_fp32(grad_block, block, spec.ptr, out, False,
                                                       _loop_max((out,)))
    else:
        torch.mm(grad_block.t(), block, out=out)


def _view(flat, n, prefix, width):
    return flat[n * prefix:n * (prefix + width)].view(n, width)


def _scale_block(block, radial, pairs):
    """block [N, w] times radial [N, c] per channel (both pair halves when pairs)."""
    n = block.shape[0]
    if pairs:
        return (block.view(n, 2, -1) * radial.unsqueeze(1)).view(n, -1)
    return block * radial


def _rowdot(a, b, pairs, n):
    """Per-channel product of a and b summed over the pair halves: [N, c]."""
    if pairs:
        return (a.view(n, 2, -1) * b.view(n, 2, -1)).sum(dim=1)
    return a * b


class _Sandwich(torch.autograd.Function):
    """Rotate in, one GEMM per block, rotate out, with the backward written out.

    ``spec`` (static) describes the trailing tensors, in order: the m = 0 weight
    [Cout_0, Cin_0] and bias when present, one stacked pair weight [2*Cout_m, Cin_m]
    per m > 0 block, one radial weight per block (front: [N, Cin_b] scales the
    block's input before its GEMM; back: [N, Cout_b] scales its output), and a
    per-edge gate [N] that scales the whole result.

    With ``spec.groups`` = G > 0 the pair weights are [G, 2*Cout_m, Cin_m]: edge e
    is stored in block row ``spec.rows[e]`` (rows sorted by group, ``spec.order``
    the inverse) and the rows ``spec.ptr[g]:spec.ptr[g+1]`` use weight g."""

    @staticmethod
    def forward(ctx, x, plan, layout, wigner, spec, *tensors):
        ext = _ext()
        n = x.shape[0]
        inp, out = plan.inp, plan.out
        t = list(tensors)
        w0 = t.pop(0) if spec.has_m0 else None
        b0 = t.pop(0) if spec.has_bias else None
        pair_weights = [t.pop(0) for _ in range(spec.n_pairs)]
        radials = [t.pop(0) for _ in range(spec.n_radials)]
        gate = t.pop(0) if spec.has_gate else None
        rows = spec.rows
        # Radial weights follow the block rows (sorted by group when grouped).
        block_radials = [r.index_select(0, spec.order) for r in radials] if spec.order is not None else radials
        packed = _rotate(x, inp, plan, wigner, layout.rotate_in, rows=rows)
        block_weights = (ext.block_complex_weights_fp32([w.contiguous() for w in pair_weights])
                         if pair_weights else None)
        result_blocks = x.new_empty(n * out.total)
        groups = max(spec.groups, 1)
        cursor = 0
        for b, (m, cin, cout) in enumerate(plan.blocks):
            block = _view(packed, n, inp.prefix[m], inp.stride[m])
            if spec.radial_mode == "front":
                block = _scale_block(block, block_radials[b], m > 0)
            target = _view(result_blocks, n, out.prefix[m], out.stride[m])
            if m == 0:
                if b0 is not None:
                    torch.addmm(b0, block, w0.t(), out=target)
                else:
                    torch.mm(block, w0.t(), out=target)
            else:
                size = groups * 4 * cout * cin
                weight = block_weights[cursor:cursor + size]
                weight = weight.view(groups, 2 * cout, 2 * cin) if spec.groups else weight.view(2 * cout, 2 * cin)
                cursor += size
                _pair_gemm(block, weight, target, spec)
        scaled_blocks = result_blocks
        if spec.radial_mode == "back":
            scaled_blocks = torch.empty_like(result_blocks)
            for b, (m, _cin, _cout) in enumerate(plan.blocks):
                _view(scaled_blocks, n, out.prefix[m], out.stride[m]).copy_(
                    _scale_block(_view(result_blocks, n, out.prefix[m], out.stride[m]), block_radials[b], m > 0))
        result = _gather(scaled_blocks, n, out, plan, wigner, layout.rotate_out,
                         scale=None if gate is None else gate.contiguous(), rows=rows)
        needs = ctx.needs_input_grad
        idx = 5
        grad_w0 = bool(spec.has_m0 and needs[idx])
        idx += int(spec.has_m0)
        grad_b0 = bool(spec.has_bias and needs[idx])
        idx += int(spec.has_bias)
        grad_pairs = any(needs[idx:idx + spec.n_pairs])
        idx += spec.n_pairs
        grad_radials = any(needs[idx:idx + spec.n_radials])
        idx += spec.n_radials
        grad_gate = bool(spec.has_gate and needs[idx])
        need_x = needs[0]
        front_grad = spec.radial_mode == "front" and grad_radials
        back_grad = spec.radial_mode == "back" and grad_radials
        keep_packed = grad_w0 or grad_pairs or front_grad
        keep_weights = need_x or front_grad
        keep_result = back_grad or grad_gate
        ctx.flags = (need_x, grad_w0, grad_b0, grad_pairs, grad_radials, grad_gate)
        ctx.save_for_backward(packed if keep_packed else None,
                              block_weights if keep_weights else None,
                              w0 if keep_weights else None,
                              result_blocks if keep_result else None,
                              gate, *block_radials)
        ctx.state = (plan, layout, wigner, spec, n)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        ext = _ext()
        plan, layout, wigner, spec, n = ctx.state
        need_x, grad_w0, grad_b0, grad_pairs, grad_radials, grad_gate = ctx.flags
        packed, block_weights, w0, result_blocks, gate, *radials = ctx.saved_tensors
        inp, out = plan.inp, plan.out
        grad_out = grad_out.contiguous()
        rows = spec.rows
        groups = max(spec.groups, 1)
        gate_grad = None
        if gate is not None and not grad_gate:
            # d result / d blocks carries the gate as a per-edge scale of the rotation.
            grad_blocks = _rotate(grad_out, out, plan, wigner, layout.rotate_out, scale=gate.contiguous(),
                                  rows=rows)
        else:
            grad_blocks = _rotate(grad_out, out, plan, wigner, layout.rotate_out, rows=rows)
            if grad_gate:
                gate_grad = grad_out.new_zeros(n)
                for b, (m, _cin, _cout) in enumerate(plan.blocks):
                    produced = _view(result_blocks, n, out.prefix[m], out.stride[m])
                    if spec.radial_mode == "back":
                        produced = _scale_block(produced, radials[b], m > 0)
                    gate_grad += (_view(grad_blocks, n, out.prefix[m], out.stride[m]) * produced).sum(dim=1)
            if gate is not None:
                for m, _cin, _cout in plan.blocks:
                    _view(grad_blocks, n, out.prefix[m], out.stride[m]).mul_(gate.unsqueeze(1))
        radial_grads = [None] * len(radials)
        if spec.radial_mode == "back":
            for b, (m, _cin, _cout) in enumerate(plan.blocks):
                view = _view(grad_blocks, n, out.prefix[m], out.stride[m])
                if grad_radials:
                    radial_grads[b] = _rowdot(view, _view(result_blocks, n, out.prefix[m], out.stride[m]), m > 0, n)
                view.copy_(_scale_block(view, radials[b], m > 0))
        front_grad = spec.radial_mode == "front" and grad_radials
        grad_packed = grad_out.new_empty(n * inp.total) if need_x else None
        n_pair_entries = sum(groups * 4 * co * ci for m, ci, co in plan.blocks if m > 0)
        grad_block_weights = grad_out.new_empty(n_pair_entries) if grad_pairs else None
        grad_w0_value = grad_b0_value = None
        cursor = 0
        for b, (m, cin, cout) in enumerate(plan.blocks):
            win = inp.stride[m]
            grad_block = _view(grad_blocks, n, out.prefix[m], out.stride[m])
            size = groups * 4 * cout * cin if m > 0 else 0
            shape = (groups, 2 * cout, 2 * cin) if spec.groups else (2 * cout, 2 * cin)
            if need_x or front_grad:
                if m == 0:
                    grad_in = torch.mm(grad_block, w0) if spec.radial_mode == "front" else None
                    if grad_in is None:
                        torch.mm(grad_block, w0, out=_view(grad_packed, n, inp.prefix[m], win))
                else:
                    weight = block_weights[cursor:cursor + size].view(*shape)
                    if spec.radial_mode == "front":
                        grad_in = grad_out.new_empty(n, win)
                        _pair_gemm(grad_block, weight, grad_in, spec, transpose=True)
                    else:
                        _pair_gemm(grad_block, weight, _view(grad_packed, n, inp.prefix[m], win), spec,
                                   transpose=True)
                if spec.radial_mode == "front":
                    if front_grad:
                        radial_grads[b] = _rowdot(grad_in, _view(packed, n, inp.prefix[m], win), m > 0, n)
                    if need_x:
                        _view(grad_packed, n, inp.prefix[m], win).copy_(_scale_block(grad_in, radials[b], m > 0))
            if (m == 0 and grad_w0) or (m > 0 and grad_pairs):
                block = _view(packed, n, inp.prefix[m], win)
                if spec.radial_mode == "front":
                    block = _scale_block(block, radials[b], m > 0)
                if m == 0:
                    grad_w0_value = torch.mm(grad_block.t(), block)
                else:
                    _pair_weight_grad(grad_block, block, grad_block_weights[cursor:cursor + size].view(*shape), spec)
            if m == 0 and grad_b0:
                grad_b0_value = grad_block.sum(dim=0)
            cursor += size
        pair_grads = [None] * spec.n_pairs
        if grad_pairs:
            pair_blocks = [(ci, co) for m, ci, co in plan.blocks if m > 0]
            pair_grads = ext.block_complex_weight_grads_fp32(
                grad_block_weights, [co for _, co in pair_blocks], [ci for ci, _ in pair_blocks], spec.groups)
        if spec.order is not None:
            radial_grads = [None if g is None else g.index_select(0, rows) for g in radial_grads]
        grad_x = _gather(grad_packed, n, inp, plan, wigner, layout.rotate_in, rows=rows) if need_x else None
        grads = [grad_x, None, None, None, None]
        if spec.has_m0:
            grads.append(grad_w0_value)
        if spec.has_bias:
            grads.append(grad_b0_value)
        grads.extend(pair_grads)
        grads.extend(radial_grads)
        if spec.has_gate:
            grads.append(gate_grad)
        return tuple(grads)


def _apply(x, plan, layout, wigner, *, w0=None, b0=None, pair_weights=(), radials=(), gate=None,
           grouping=None):
    groups, ptr, order, rows = grouping if grouping is not None else (0, None, None, None)
    if gate is not None and grouping is not None:
        raise ValueError("a per-edge gate is supported for ungrouped weights only")
    spec = SimpleNamespace(has_m0=w0 is not None, has_bias=b0 is not None, n_pairs=len(pair_weights),
                           radial_mode=("front" if layout.front else "back") if radials else None,
                           n_radials=len(radials), has_gate=gate is not None,
                           groups=int(groups), ptr=ptr, order=order, rows=rows)
    tensors = ([w0] if w0 is not None else []) + ([b0] if b0 is not None else []) + list(pair_weights)
    tensors += list(radials) + ([gate] if gate is not None else [])
    return _Sandwich.apply(x.contiguous(), plan, layout, wigner, spec, *tensors)


def block_sandwich(x, layout, wigner, weights_by_m, radial_parts=None):
    """m>0 contribution [N, out_dim] of one [2*Cout_m, Cin_m] pair weight per m.

    ``weights_by_m`` and ``radial_parts`` are indexed by m (index 0 unused); radial
    weights are [N, Cin_m] for front layouts and [N, Cout_m] otherwise. Callers
    check ``sandwich_plan(...).supported`` first."""
    plan = sandwich_plan(layout, x.shape[1], x.device)
    if not plan.blocks:
        return x.new_zeros((x.shape[0], layout.out_dim))
    weights = tuple(weights_by_m[m] for m, _, _ in plan.blocks)
    radials = () if radial_parts is None else tuple(radial_parts[m] for m, _, _ in plan.blocks)
    return _apply(x, plan, layout, wigner, pair_weights=weights, radials=radials)


def grouped_sandwich(x, layout, wigner, weights_by_m, radial_parts, ptr_edges, order, rows):
    """m>0 contribution of [G, 2*Cout_m, Cin_m] pair weights: edges sorted by group
    (``order``: sorted position -> edge, ``rows``: edge -> sorted position, both None
    when the edges are already sorted) and group g owning the sorted rows
    ``ptr_edges[g]:ptr_edges[g+1]`` (CPU int64)."""
    plan = sandwich_plan(layout, x.shape[1], x.device)
    if not plan.blocks:
        return x.new_zeros((x.shape[0], layout.out_dim))
    weights = tuple(weights_by_m[m] for m, _, _ in plan.blocks)
    radials = () if radial_parts is None else tuple(radial_parts[m] for m, _, _ in plan.blocks)
    groups = int(weights[0].shape[0])
    return _apply(x, plan, layout, wigner, pair_weights=weights, radials=radials,
                  grouping=(groups, ptr_edges, order, rows))


def full_sandwich(x, layout, wigner, w0, b0, pair_weights_by_m, radials_by_m=None, gate=None):
    """Whole layer output [N, out_dim], m = 0 included.

    ``w0`` is [Cout_0, Cin_0] with optional bias ``b0``; ``pair_weights_by_m[m]`` the
    [2*Cout_m, Cin_m] pair weights; ``radials_by_m[m]`` optional radial weights of each
    block (front or back by ``layout.front``); ``gate`` an optional per-edge [N] scale.
    Returns None when the layout has no m = 0 block."""
    plan = sandwich_plan(layout, x.shape[1], x.device, with_m0=True)
    if not plan.blocks or plan.blocks[0][0] != 0:
        return None
    pair_weights = tuple(pair_weights_by_m[m] for m, _, _ in plan.blocks if m > 0)
    radials = () if radials_by_m is None else tuple(radials_by_m[m] for m, _, _ in plan.blocks)
    return _apply(x, plan, layout, wigner, w0=w0, b0=b0, pair_weights=pair_weights, radials=radials,
                  gate=gate)
