"""Public cuEquivariance eSCN descriptor with explicit SO(2) weight mapping."""
from __future__ import annotations

import torch
from torch import nn
from e3nn import o3


def cue_irreps(irreps):
    import cuequivariance as cue
    return cue.Irreps(cue.SO3, [(mul, cue.SO3(ir.l)) for mul, ir in o3.Irreps(irreps)])


class CueqLocal(nn.Module):
    """Apply escn_tp to mul_ir features, using shared or batched native weights.

    Canonical pair order is (-m,+m), with y-=A*x- - B*x+ and
    y+=A*x+ + B*x-. The native descriptor stores (input,output) weights
    with path-dependent normalization. Mapping is derived from the public
    descriptor paths, so neither normalization nor flattening is guessed.
    """

    def __init__(self, irreps_in, irreps_out, mmax, method="naive", device="cuda",
                 descriptor_name="escn_tp"):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        from cuequivariance.group_theory.experimental.escn import escn_tp, escn_tp_compact
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        self.mmax = mmax
        self.descriptor_name = descriptor_name
        factory = {"escn_tp": escn_tp, "escn_tp_compact": escn_tp_compact}[descriptor_name]
        self.descriptor = factory(cue_irreps(irreps_in), cue_irreps(irreps_out), m_max=mmax)
        polynomial = self.descriptor.polynomial if descriptor_name == "escn_tp" else self.descriptor
        self.executor = cuet.SegmentedPolynomial(polynomial, method=method,
                                                math_dtype=torch.float32).to(device)
        self.method = self.executor.method
        if descriptor_name == "escn_tp":
            self.input_transpose = cuet.TransposeIrrepsLayout(cue_irreps(irreps_in), source=cue.mul_ir,
                                                   target=cue.ir_mul, device=device,
                                                   use_fallback=torch.device(device).type == "cpu")
            self.output_transpose = cuet.TransposeIrrepsLayout(cue_irreps(irreps_out), source=cue.ir_mul,
                                                     target=cue.mul_ir, device=device,
                                                     use_fallback=torch.device(device).type == "cpu")
        else:
            self.register_buffer("input_permutation", compact_permutation(self.irreps_in, device), persistent=False)
            self.register_buffer("output_inverse_permutation", compact_permutation(self.irreps_out, device).argsort(), persistent=False)
        self.weight_shapes = []
        starts = [0]
        for m in range(mmax + 1):
            ni = sum(mul for mul, ir in self.irreps_in if ir.l >= m)
            no = sum(mul for mul, ir in self.irreps_out if ir.l >= m)
            self.weight_shapes.append((no * (2 if m else 1), ni))
            starts.append(starts[-1] + self.weight_shapes[-1][0] * ni)
        self.weight_starts = starts
        # Flattened descriptor feature segments correspond to (irrep block,m).
        segments = lambda irreps: [(i, m) for i, (_, ir) in enumerate(irreps)
                                   for m in range(-ir.l, ir.l + 1)]
        ins, outs = segments(self.irreps_in), segments(self.irreps_out)
        stp = polynomial.operations[0][1]
        mapped = {}
        scales = {}
        for path in stp.paths:
            iw, ix, iy = path.indices
            if descriptor_name == "escn_tp":
                ii, mi = ins[ix]
                oi, mo = outs[iy]
            else:
                mi, mo = ix - self.irreps_in.lmax, iy - self.irreps_out.lmax
            m = abs(mi)
            assert abs(mo) == m and m <= mmax
            if descriptor_name == "escn_tp":
                ni, no = self.irreps_in[ii].mul, self.irreps_out[oi].mul
                ci = sum(mul for mul, ir in self.irreps_in[:ii] if ir.l >= m)
                co = sum(mul for mul, ir in self.irreps_out[:oi] if ir.l >= m)
            else:
                ni = self.weight_shapes[m][1]
                no = self.weight_shapes[m][0] // (2 if m else 1)
                ci = co = 0
            sine = mi != mo
            sign = -1 if sine and mo < 0 else 1
            if sine:
                co += self.weight_shapes[m][0] // 2
            index = torch.arange(co, co + no)[:, None] * self.weight_shapes[m][1]
            index = (index + torch.arange(ci, ci + ni)[None, :] + starts[m]).T.flatten()
            scale = sign / float(path.coefficients)
            if iw in mapped:
                assert torch.equal(mapped[iw], index) and abs(scales[iw] - scale) < 1e-10
            mapped[iw], scales[iw] = index, scale
        indices = torch.cat([mapped[i] for i in range(len(stp.operands[0].segments))])
        factor = torch.cat([torch.full_like(mapped[i], scales[i], dtype=torch.float64)
                            for i in range(len(stp.operands[0].segments))])
        assert len(indices) == starts[-1] and len(indices.unique()) == starts[-1]
        self.register_buffer("map_index", indices.to(device), persistent=False)
        self.register_buffer("map_scale", factor.to(device=device, dtype=torch.float32), persistent=False)

    def map_weights(self, weights):
        """Differentiable canonical -> native mapping; keeps leading batch axes."""
        flat = torch.cat([w.flatten(-2) for w in weights], dim=-1)
        native = flat.index_select(-1, self.map_index) * self.map_scale.to(flat.dtype)
        return native.unsqueeze(0) if native.ndim == 1 else native

    def canonical_gradients(self, native_gradient):
        flat = native_gradient.reshape(-1) * self.map_scale
        canonical = torch.zeros_like(flat).scatter_add(0, self.map_index, flat)
        return tuple(canonical[a:b].view(shape) for a, b, shape in
                     zip(self.weight_starts[:-1], self.weight_starts[1:], self.weight_shapes))

    def to_native(self, x):
        if self.descriptor_name == "escn_tp_compact":
            return x.index_select(-1, self.input_permutation)
        return self.input_transpose(x)

    def from_native(self, x):
        if self.descriptor_name == "escn_tp_compact":
            return x.index_select(-1, self.output_inverse_permutation)
        return self.output_transpose(x)

    def forward_native(self, x, flat_weights, input_indices=None):
        if self.method == "indexed_linear" and input_indices is None:
            if flat_weights.shape[0] == 1:
                input_indices = torch.zeros(len(x), device=x.device, dtype=torch.int32)
            elif flat_weights.shape[0] == len(x):
                input_indices = torch.arange(len(x), device=x.device, dtype=torch.int32)
            else:
                raise ValueError("Indexed weights require explicit indices unless shared or per edge")
        kwargs = {} if input_indices is None else {"input_indices": {0: input_indices}}
        return self.executor([flat_weights, x], **kwargs)[0]

    def forward(self, x, flat_weights, input_indices=None):
        return self.from_native(self.forward_native(self.to_native(x), flat_weights, input_indices))


def compact_permutation(irreps, device):
    """Indices from mul_ir to ascending m, then original block and channel."""
    return torch.tensor([sl.start + c * ir.dim + ir.l + m
                         for m in range(-irreps.lmax, irreps.lmax + 1)
                         for (mul, ir), sl in zip(irreps, irreps.slices()) if ir.l >= abs(m)
                         for c in range(mul)], dtype=torch.long, device=device)
