"""Feature permutations prepared outside an operator's timed region."""
from __future__ import annotations

from collections import defaultdict

import torch
from e3nn import o3


class FeatureLayout:
    """A pure permutation between canonical mul_ir and a native layout."""

    def __init__(self, irreps, device, *, groups=None, coefficient_shape=False):
        self.irreps = o3.Irreps(irreps)
        self.groups = groups or [[i] for i in range(len(self.irreps))]
        slices = self.irreps.slices()
        indices = [slices[i].start + c * self.irreps[i].ir.dim + k
                   for group in self.groups for i in group
                   for k in range(self.irreps[i].ir.dim) for c in range(self.irreps[i].mul)]
        self.order = torch.tensor(indices, device=device, dtype=torch.long)
        self.inverse = self.order.argsort()
        self.coefficient_shape = coefficient_shape
        if coefficient_shape:
            assert len({mul for mul, _ in self.irreps}) == 1
            self.channels = self.irreps[0].mul

    def to_native(self, x):
        result = x.index_select(-1, self.order)
        return result.reshape(len(x), -1, self.channels) if self.coefficient_shape else result

    def from_native(self, x):
        return x.flatten(1).index_select(-1, self.inverse)


class NativeOperator:
    """Canonical convenience calls; timing calls forward_native directly."""

    def input_to_native(self, x):
        return self.input_layout.to_native(x)

    def output_to_native(self, x):
        return self.output_layout.to_native(x)

    def input_from_native(self, x):
        return self.input_layout.from_native(x)

    def output_from_native(self, x):
        return self.output_layout.from_native(x)

    def forward(self, x):
        return self.output_from_native(self.forward_native(self.input_to_native(x)))


class NativeWigner:
    """Coefficient bmm on native features, with local descriptor ordering.

    Equal multiplicities share a coefficient matrix. Uniform channels require
    one bmm and no feature permutation. General irreps use one split across
    groups and one descriptor permutation, never per-irrep slice writeback.
    """

    def __init__(self, irreps, blocks, *, compact):
        self.irreps = o3.Irreps(irreps)
        groups = defaultdict(list)
        for i, (mul, _) in enumerate(self.irreps):
            groups[mul].append(i)
        self.layout = FeatureLayout(self.irreps, blocks[0].device,
                                    groups=list(groups.values()), coefficient_shape=len(groups) == 1)
        descriptor_order = ([(m, i) for m in range(-self.irreps.lmax, self.irreps.lmax + 1)
                             for i, (_, ir) in enumerate(self.irreps) if abs(m) <= ir.l] if compact else
                            [(m, i) for i, (_, ir) in enumerate(self.irreps)
                             for m in range(-ir.l, ir.l + 1)])
        self.groups, local_order, self.widths = [], [], []
        for mul, indices in groups.items():
            width = sum(self.irreps[i].ir.dim for i in indices)
            matrix = blocks[0].new_zeros(len(blocks[0]), width, width)
            offsets, offset = {}, 0
            for i in indices:
                ir = self.irreps[i].ir
                offsets[i] = offset
                matrix[:, offset:offset+ir.dim, offset:offset+ir.dim] = blocks[ir.l]
                offset += ir.dim
            order = [key for key in descriptor_order if key[1] in indices]
            columns = torch.tensor([offsets[i] + self.irreps[i].ir.l + m for m, i in order],
                                   device=matrix.device)
            self.groups.append((mul, matrix.index_select(-1, columns).transpose(1, 2).contiguous()))
            self.widths.append(width * mul)
            local_order.extend((m, i, c) for m, i in order for c in range(mul))
        locations = {key: j for j, key in enumerate(local_order)}
        order = [locations[m, i, c] for m, i in descriptor_order for c in range(self.irreps[i].mul)]
        self.order = torch.tensor(order, device=blocks[0].device)
        self.inverse = self.order.argsort()
        self.single_group = len(self.groups) == 1

    def rotate(self, x):
        if self.single_group:
            return torch.bmm(self.groups[0][1], x).flatten(1)
        parts = [torch.bmm(matrix, part.reshape(len(x), -1, mul)).flatten(1)
                 for part, (mul, matrix) in zip(x.split(self.widths, dim=-1), self.groups)]
        return torch.cat(parts, dim=-1).index_select(-1, self.order)

    def rotate_inv(self, x):
        if self.single_group:
            mul, matrix = self.groups[0]
            return torch.bmm(matrix.transpose(1, 2), x.reshape(len(x), -1, mul))
        grouped = x.index_select(-1, self.inverse)
        parts = [torch.bmm(matrix.transpose(1, 2), part.reshape(len(x), -1, mul)).flatten(1)
                 for part, (mul, matrix) in zip(grouped.split(self.widths, dim=-1), self.groups)]
        return torch.cat(parts, dim=-1)
