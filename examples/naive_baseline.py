"""Uncached upstream SO2 and ordinary expert linears for the UniTB benchmark.

SO2 follows deepmodeling/DeePTB's SO2_Linear / SO2_m_Linear at 1dcc7f6.
Each forward rebuilds all-l Wigner D with batch_wigner_D, rotates inputs
with one bmm per l, selects m blocks with boolean masks, combines real and
imaginary outputs with narrow, and rotates each output irrep with einsum.
Radial weights retain their original position before or after the linear.

The expert formula follows fairchem UMA MOLE at 3801dac0. For UniTB's
per-edge routing, distribute the linear mixture as
sum_k c_k(edge) * linear(x_edge, W_k). Each routed expert derives its selected
(edge, slot) pairs from topk_indices, gathers inputs with index_select, calls
ordinary F.linear once, multiplies by the matching gate, and accumulates
with index_add_. Shared experts each apply a separate F.linear to all rows.
PDQ weights W_k = P D_k Q^T are materialized in full on every forward;
router and PDQ parameter gradients remain connected through the mixture.
Expert outputs are mixed before activation.

UniTB-dense uses UMA's weight-space mixture with one coefficient row per
group: einsum("eoi,be->boi", weights, coefficients). Groups follow the
model's graph_index and may have interleaved rows. For each group, nonzero
finds all its rows, index_select gathers them, one F.linear processes them,
and index_copy_ restores their original positions. Contiguous group sizes
use UMA's original segment loop. Per-edge top-k metadata does not expand
the group weight bank. Each linear sublayer makes one F.linear call per
group, irrespective of how many noncontiguous runs those rows form.

There is no sorted-layout cache, grouped GEMM, shared-expert folding,
cross-m packing, low-rank linear evaluation, or SO2CUDA call in this baseline.
Only execution modules are replaced; parameters and checkpoint keys are shared.
"""
from __future__ import annotations

from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from e3nn.o3 import xyz_to_angles

from dptb.nn.pdq_moe import PDQMoE
from dptb.nn.so2_parity import ParityWeightMixin
from dptb.nn.tensor_product import SO2LinearCached, SO2_Linear as DenseSO2
from dptb.nn.tensor_product import batch_wigner_D, _Jd
from dptb.nn.tensor_product_moe_v3 import SO2_Linear as ExpertSO2


class NaiveExpertLinear(ParityWeightMixin, nn.Module):
    """UMA's linear mixture expanded over experts instead of per-edge weights."""

    def __init__(self, source):
        super().__init__()
        for name in ("in_features", "out_features", "num_experts", "num_shared_experts",
                     "mole_expert_parameterization"):
            setattr(self, name, getattr(source, name))
        for name, parameter in source._parameters.items():
            self.register_parameter(name, parameter)
        for name, buffer in source._buffers.items():
            self.register_buffer(name, buffer, persistent=name not in source._non_persistent_buffers_set)
        self.train(source.training)

    def _coefficients(self, x, routing):
        if routing is None or routing.coefficients is None:
            return x.new_full((1, self.num_experts), 1.0 / self.num_experts)
        if getattr(routing, "top1_independent", False):
            raise ValueError("The benchmark baseline supports pre-activation expert mixtures")
        indices, values = routing.topk_indices, routing.topk_values
        # Graph mixtures use one coefficient row per group. Expanded per-edge
        # top-k metadata must not replace this group bank (same contract as PDQMoE).
        if (indices is not None and values is not None
                and indices.shape[0] == routing.coefficients.shape[0]):
            coefficients = values.new_zeros((indices.shape[0], self.num_experts))
            coefficients = coefficients.scatter_add(1, indices, values)
        else:
            coefficients = routing.coefficients
        return coefficients

    def _weight(self, expert):
        if self.mole_expert_parameterization == "shared_core":
            weight = (self.basis_left @ self.core_experts[expert]) @ self.basis_right.T
            return self._parity_value("weight_experts", weight)
        return self.weight_experts[expert]

    def _graph_linear(self, x, routing):
        coefficients = self._coefficients(x, routing)
        bank = torch.stack([self._weight(e) for e in range(self.num_experts)])
        weights = torch.einsum("eoi,be->boi", bank, coefficients)
        biases = (None if self.bias_experts is None else
                  torch.einsum("eo,be->bo", self.bias_experts, coefficients))
        sizes = getattr(routing, "split_sizes", None)
        if sizes is None:
            sizes = getattr(routing, "sizes", None)
        index = getattr(routing, "graph_index", None)
        if index is None:
            if sizes is None:
                if coefficients.shape[0] != 1:
                    raise ValueError("Group routing requires sizes or graph_index")
                sizes = [x.shape[0]]
            sizes = torch.as_tensor(sizes).tolist()
            if len(sizes) != coefficients.shape[0] or sum(sizes) != x.shape[0]:
                raise ValueError("Group sizes must match coefficients and cover all rows")
            # Contiguous structures use UMA's original segment loop.
            parts, start = [], 0
            for group, size in enumerate(sizes):
                end = start + size
                bias = None if biases is None else biases[group]
                parts.append(F.linear(x[start:end], weights[group], bias))
                start = end
            return torch.cat(parts, dim=0)
        index = index.to(device=x.device, dtype=torch.long).reshape(-1)
        if index.numel() != x.shape[0]:
            raise ValueError("graph_index must assign every input row to a group")
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        # A group may be scattered throughout the input (for example bond types).
        # Gather it once, without sorting, caching layouts or splitting it into runs.
        for group in range(coefficients.shape[0]):
            rows = torch.nonzero(index == group, as_tuple=True)[0]
            bias = None if biases is None else biases[group]
            part = F.linear(x.index_select(0, rows), weights[group], bias)
            out.index_copy_(0, rows, part)
        return out

    def _edge_linear(self, x, routing):
        if getattr(routing, "top1_independent", False):
            raise ValueError("The benchmark baseline supports pre-activation expert mixtures")
        indices, values = routing.topk_indices, routing.topk_values
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        for expert in range(self.num_experts):
            if indices is not None and values is not None:
                edge, slot = torch.where(indices == expert)
                gate = values[edge, slot]
            else:
                # Without top-k routing every coefficient remains differentiable.
                edge = torch.arange(x.shape[0], device=x.device)
                gate = routing.coefficients[:, expert]
            weight = self._weight(expert)
            bias = None if self.bias_experts is None else self.bias_experts[expert]
            part = F.linear(x.index_select(0, edge), weight, bias)
            part = part * gate.reshape((-1,) + (1,) * (x.ndim - 1))
            out.index_add_(0, edge, part)
        return out

    def forward(self, x, mole_globals=None):
        branch = getattr(mole_globals, "branch", "all")
        out = None
        if self.num_experts and branch != "shared":
            if getattr(mole_globals, "activation_space", False):
                out = self._edge_linear(x, mole_globals)
            else:
                out = self._graph_linear(x, mole_globals)
        if branch != "routed":
            for expert in range(self.num_shared_experts):
                bias = None if self.bias_shared is None else self.bias_shared[expert]
                part = F.linear(x, self.weight_shared[expert], bias)
                out = part if out is None else out + part
        if out is None:
            out = x.new_zeros(*x.shape[:-1], self.out_features)
        return out


class NaiveSO2MLinear(nn.Module):
    """Use the upstream narrow-based real/imaginary combination verbatim."""

    def __init__(self, source):
        super().__init__()
        self.fc = NaiveExpertLinear(source.fc) if isinstance(source.fc, PDQMoE) else source.fc
        self.num_out_channel = source.num_out_channel
        self.is_mole = isinstance(self.fc, NaiveExpertLinear)
        self.train(source.training)

    def forward(self, x_m, routing=None):
        if self.is_mole:
            x_m = self.fc(x_m, routing)
        elif getattr(routing, "branch", "all") == "routed":
            x_m = x_m.new_zeros(*x_m.shape[:-1], 2 * self.num_out_channel)
        else:
            x_m = self.fc(x_m)
        width = self.num_out_channel
        x_r = x_m.narrow(2, 0, width)
        x_i = x_m.narrow(2, width, width)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        return torch.cat((x_m_r, x_m_i), dim=1)


class NaiveSO2Linear(nn.Module):
    """Upstream SO2 with UniTB rotation flags, routing and output block support."""

    def __init__(self, source):
        super().__init__()
        self.is_mole = isinstance(source, ExpertSO2)
        self.returns_cache = not isinstance(source, DenseSO2)
        for name in ("irreps_in", "irreps_out", "front", "rotate_in", "rotate_out", "m_in_index"):
            setattr(self, name, getattr(source, name))
        self.l_max = max(self.irreps_in.lmax, self.irreps_out.lmax)
        self.m_max = min(self.irreps_in.lmax, self.irreps_out.lmax)
        self.radial_emb = source.radial_emb
        self.fc_m0 = NaiveExpertLinear(source.fc_m0) if self.is_mole else source.fc_m0
        self.m_linear = nn.ModuleList(NaiveSO2MLinear(block) for block in source.m_linear)
        for name in ("m_in_mask", "m_out_mask"):
            value = getattr(source, name)
            if name in source._buffers:
                self.register_buffer(name, value, persistent=name not in source._non_persistent_buffers_set)
            else:
                setattr(self, name, value)
        self.train(source.training)

    def forward(self, x, R, mole_globals=None, latents=None, wigner_D_all=None):
        if not self.is_mole and mole_globals is not None:
            # The cached dense interface places latents in the third position.
            latents = mole_globals
        n = x.shape[0]
        weights = self.radial_emb(latents) if self.radial_emb else None
        angle = xyz_to_angles(R[:, [1, 2, 0]])
        # Rebuild the full block diagonal matrix even when a caller supplies a cache.
        wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)
        x_ = torch.zeros_like(x)
        groups = defaultdict(list)
        for (mul, (l, _)), sl in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, sl))
            if l == 0 or not self.rotate_in:
                x_[:, sl] = x[:, sl]
        if self.rotate_in:
            for l, group in groups.items():
                if l == 0:
                    continue
                muls, slices = zip(*group)
                parts = [x[:, sl].reshape(n, mul, 2 * l + 1) for mul, sl in group]
                rot = wigner_D_all[:, l*l:(l+1)**2, l*l:(l+1)**2]
                transformed = torch.bmm(torch.cat(parts, dim=1), rot)
                for part, sl in zip(transformed.split(muls, dim=1), slices):
                    x_[:, sl] = part.reshape(n, -1)
        out = x.new_zeros(n, self.irreps_out.dim)
        for m in range(self.m_max + 1):
            radial = weights[:, self.m_in_index[m]:self.m_in_index[m+1]] if weights is not None else None
            inp = x_[:, self.m_in_mask[m]]
            if m > 0:
                inp = inp.reshape(n, -1, 2).transpose(1, 2).contiguous()
                if radial is not None:
                    radial = radial.unsqueeze(1)
            if self.front and radial is not None:
                if m > 0:
                    inp.mul_(radial)
                else:
                    inp = inp * radial
            if m == 0:
                value = self.fc_m0(inp, mole_globals) if self.is_mole else self.fc_m0(inp)
            else:
                value = self.m_linear[m-1](inp, mole_globals)
            if not self.front and radial is not None:
                if m > 0:
                    value.mul_(radial)
                else:
                    value = value * radial
            if m > 0:
                value = value.transpose(1, 2).contiguous().reshape(n, -1)
            out[:, self.m_out_mask[m]] += value
        if self.rotate_out:
            for (mul, (l, _)), sl in zip(self.irreps_out, self.irreps_out.slices()):
                if l > 0:
                    rot = wigner_D_all[:, l*l:(l+1)**2, l*l:(l+1)**2]
                    part = out[:, sl].reshape(n, mul, 2*l+1)
                    # Geometry gradients need protection from the slice writeback.
                    # Fixed benchmark geometry uses the upstream view directly.
                    if rot.requires_grad:
                        part = part.clone()
                    out[:, sl] = torch.einsum("nij,nmj->nmi", rot, part).reshape(n, -1)
        result = out.contiguous()
        return (result, wigner_D_all) if self.returns_cache else result


class NaiveExecutionHandle:
    """Restore the original child modules without moving or copying parameters."""

    def __init__(self, replacements):
        self.replacements = replacements

    def restore(self):
        for parent, name, original in reversed(self.replacements):
            original.train(getattr(parent, name).training)
            setattr(parent, name, original)
        self.replacements.clear()


def install_naive_baseline(model):
    """Replace SO2/PDQ children in place; return a reversible execution handle."""
    embedding = getattr(model, "embedding", model)
    if getattr(embedding, "so2_expert_mixing_mode", "pre_activation") not in ("pre_activation", "linear"):
        raise ValueError("This benchmark implements the standard pre-activation UniTB mixture")
    replacements = []

    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, (ExpertSO2, SO2LinearCached)):
                replacement = NaiveSO2Linear(child)
            elif isinstance(child, PDQMoE):
                replacement = NaiveExpertLinear(child)
            else:
                visit(child)
                continue
            replacements.append((parent, name, child))
            setattr(parent, name, replacement)

    visit(model)
    return NaiveExecutionHandle(replacements)
