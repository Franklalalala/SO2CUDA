"""cuEquivariance eSCN tensor products with the ordinary UniTB execution path.

SO(2) tensor products use cuEquivariance, with the per-forward Wigner cache
shared by all layers and split/grouped-bmm feature rotations. Radial modulation,
routers, nonlinearities and the other expert linears follow the ordinary
benchmark baseline. Original Parameter objects and state-dict names remain
unchanged. The canonical-to-descriptor map is rebuilt in each forward so
gradients propagate to full expert weights or their PDQ factors.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from e3nn.o3 import xyz_to_angles

from naive_baseline import (NaiveExecutionHandle, NaiveExpertLinear,
                            NaiveSO2Linear, PDQMoE, ExpertSO2,
                            SO2LinearCached)
from dptb.nn.tensor_product_moe_v3 import _make_wigner_rotation


def _rotation_plan(irreps):
    """Static split sizes and same-l channel groups in original block order."""
    groups = defaultdict(list)
    sizes = []
    for index, (mul, ir) in enumerate(irreps):
        sizes.append(mul * ir.dim)
        groups[ir.l].append((index, mul))
    return tuple(sizes), tuple((l, tuple(group)) for l, group in groups.items())


def _rotate_features(x, plan, wigner, *, inverse=False):
    """One feature split, one bmm per l, then concatenate original blocks.

    split's backward assembles all feature gradients once; repeated feature
    slices/writeback would each allocate and accumulate full-width gradients.
    UniTB's activation/routing boundary retains its canonical mul_ir layout.
    """
    sizes, groups = plan
    parts = list(x.split(sizes, dim=-1))
    for l, group in groups:
        if l == 0:
            continue
        width = 2 * l + 1
        channels = [mul for _, mul in group]
        inputs = [parts[index].reshape(len(x), mul, width) for index, mul in group]
        combined = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=1)
        if hasattr(wigner, "blocks"):
            rotation = wigner.block(l)
        else:
            rotation = wigner[:, l * l:(l + 1) ** 2, l * l:(l + 1) ** 2]
        if inverse:
            rotation = rotation.transpose(-1, -2)
        rotated = torch.bmm(combined, rotation)
        for part, (index, mul) in zip(rotated.split(channels, dim=1), group):
            parts[index] = part.reshape(len(x), mul * width)
    return torch.cat(parts, dim=-1)


class CueqSO2Linear(NaiveSO2Linear):
    """Shared eSCN descriptors applied to graph mixtures or selected experts."""

    def __init__(self, source, *, method="auto", descriptor="escn_tp", selection=None):
        super().__init__(source)
        from operator_cueq import CueqLocal

        self.wigner_apply_mode = getattr(source, "wigner_apply_mode", "compact_blocks")
        self._input_rotation_plan = _rotation_plan(self.irreps_in)
        self._output_rotation_plan = _rotation_plan(self.irreps_out)
        self.wigner_geometry_builds = 0
        self.wigner_cache_hits = 0
        self.forward_calls = 0
        self.last_wigner_layout = None
        linears = [self.fc_m0] + [block.fc for block in self.m_linear]
        if any(not isinstance(linear, (NaiveExpertLinear, nn.Linear)) for linear in linears):
            raise ValueError("The eSCN descriptor requires affine m blocks; nonlinear interpolation is unsupported")
        parameter = next(self.parameters())
        # Descriptor constants are execution state, not checkpoint state. Do not
        # introduce extra parameter/buffer names into an interchangeable model.
        if selection is not None:
            descriptor, method = selection["descriptor"], selection["method"]
            if selection.get("rotation") != "pytorch":
                raise ValueError("The model adapter requires shared Wigner split/grouped-bmm rotation")
            if selection.get("irreps_in", str(self.irreps_in)) != str(self.irreps_in) or \
                    selection.get("irreps_out", str(self.irreps_out)) != str(self.irreps_out):
                raise ValueError("Frozen cuEquivariance selection has different layer irreps")
        # Delay auto construction: the original descriptor has thousands of
        # paths in some layers, while the compact descriptor has only 4*mmax+1.
        local = None if method == "auto" else CueqLocal(
            self.irreps_in, self.irreps_out, self.m_max, descriptor_name=descriptor,
            method=method, device=parameter.device)
        object.__setattr__(self, "_cueq_local", local)
        self.cueq_method_selection = copy.deepcopy(selection.get("method_selection")) if selection else None
        self._cueq_auto_pending = method == "auto"
        self.cueq_selection_frozen = selection is not None
        self.cueq_calls = 0

    @property
    def _linears(self):
        return [self.fc_m0] + [block.fc for block in self.m_linear]

    def _radial(self, value, weights, *, input_side):
        if weights is None:
            return value
        mask = self.m_in_mask if input_side else self.m_out_mask
        result = value.clone()
        n = value.shape[0]
        for m in range(self.m_max + 1):
            radial = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]]
            block = value[:, mask[m]]
            if m:
                block = (block.reshape(n, -1, 2) * radial.unsqueeze(-1)).reshape(n, -1)
            else:
                block = block * radial
            result[:, mask[m]] = block
        return result

    def _local_tp(self, x, weights, bias):
        if self._cueq_auto_pending and x.shape[0]:
            self._select_method(x, weights)
        if self._cueq_local is None:
            return x.new_empty(0, self.irreps_out.dim) + sum(w.sum() for w in weights) * 0
        flat = self._cueq_local.map_weights(weights)
        if flat.ndim == 1:
            flat = flat.unsqueeze(0)
        index = (torch.zeros(x.shape[0], device=x.device, dtype=torch.int32)
                 if self._cueq_local.method == "indexed_linear" else None)
        if x.shape[0]:
            out = self._cueq_local.forward(x, flat, input_indices=index)
        else:
            out = x.new_empty(0, self.irreps_out.dim) + flat.sum() * 0
        self.cueq_calls += 1
        if bias is not None:
            out = out.clone()
            out[:, self.m_out_mask[0]] += bias
        return out

    def _canonical_local(self, x, weights):
        """The upstream SO2 formula, independent of either cueq descriptor."""
        out = x.new_zeros(len(x), self.irreps_out.dim)
        for m, weight in enumerate(weights):
            block = x[:, self.m_in_mask[m]]
            if m:
                block = block.reshape(len(x), -1, 2).transpose(1, 2)
            raw = F.linear(block, weight)
            if m:
                width = weight.shape[0] // 2
                real, imag = raw[:, :, :width], raw[:, :, width:]
                raw = torch.stack((real[:, 0] - imag[:, 1], real[:, 1] + imag[:, 0]), dim=-1).flatten(1)
            out[:, self.m_out_mask[m]] = raw
        return out

    @staticmethod
    def _metric(reference, value):
        delta = (reference.detach().double() - value.detach().double())
        absolute = float(delta.abs().max()) if delta.numel() else 0.0
        relative = float(delta.norm() / reference.detach().double().norm().clamp_min(1e-30))
        finite = bool(torch.isfinite(value).all())
        return {"max_abs": absolute, "relative_l2": relative, "finite": finite,
                "passed": finite and (absolute <= 2e-5 or relative <= 5e-5)}

    def _equivariance(self, executor, x, weights):
        """Check the SO2 sandwich under a deterministic global SO3 rotation.

        Use one descriptor instance and independent geometry for each side.
        Radial scalar modulation and scalar bias are separate model operations.
        """
        from operator_baselines import prepare_geometry, rotate_features
        from e3nn import o3

        count = min(8, len(x))
        t = torch.arange(1, count + 1, device=x.device, dtype=x.dtype)
        vectors = torch.stack((t.sin(), (t * .71).cos(), (t * .37).sin() + .3), dim=-1)
        angles = [torch.tensor(a, dtype=torch.float64) for a in (.37, 1.11, -.51)]
        q = o3.angles_to_matrix(*angles).to(device=x.device, dtype=x.dtype)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            blocks = tuple(o3.wigner_D(l, *angles).to(x.device, x.dtype).unsqueeze(0).expand(count, -1, -1)
                           for l in range(self.l_max + 1))
        finally:
            torch.set_default_dtype(original_dtype)
        first = prepare_geometry(vectors, self.l_max)
        second = prepare_geometry(vectors @ q.T, self.l_max)
        flat = executor.map_weights(weights)

        def sandwich(value, geometry):
            local = rotate_features(value, self.irreps_in, geometry.blocks)
            result = executor(local, flat)
            return rotate_features(result, self.irreps_out, geometry.blocks, inverse=True)

        with torch.no_grad():
            expected = rotate_features(sandwich(x[:count], first), self.irreps_out, blocks, inverse=True)
            actual = sandwich(rotate_features(x[:count], self.irreps_in, blocks, inverse=True), second)
        return self._metric(expected, actual)

    def _select_method(self, x, weights):
        """Bounded descriptor/method selection on at most 64 layer edges.

        Every candidate checks forward, input and canonical weight gradients,
        plus SO3 equivariance, before one warmup and three selection samples.
        The complete model executes only the winner. Selection timings are
        diagnostic and do not enter the published model benchmark.
        """
        from operator_cueq import CueqLocal

        candidates = []
        best, best_ms = None, float("inf")
        rows = min(64, len(x))
        with torch.enable_grad():
            check_x = x[:rows].detach().clone().requires_grad_()
            check_weights = tuple(w.detach().clone().requires_grad_() for w in weights)
            targets = (check_x,) + check_weights
            reference = self._canonical_local(check_x, check_weights)
            upstream = torch.sin(torch.arange(reference.numel(), device=x.device, dtype=x.dtype)
                                 .reshape_as(reference) * .17) / self.irreps_out.dim ** .5
            ref_grads = torch.autograd.grad(reference, targets, upstream)
            reference = reference.detach()
            for descriptor in ("escn_tp", "escn_tp_compact"):
                for method in ("naive", "uniform_1d", "fused_tp", "indexed_linear"):
                    row = {"descriptor": descriptor, "requested_method": method, "rotation": "pytorch"}
                    print(f"cuEquivariance layer selection: {descriptor}/{method}, rows={rows}", flush=True)
                    started = time.perf_counter()
                    executor = value = gradients = output = derivative = None
                    try:
                        if x.device.type != "cuda" and method != "naive":
                            raise NotImplementedError("CUDA executor selection requires a CUDA device")
                        executor = CueqLocal(self.irreps_in, self.irreps_out, self.m_max,
                                             descriptor_name=descriptor, method=method, device=x.device)
                        row["actual_method"] = executor.method
                        value = executor(check_x, executor.map_weights(check_weights))
                        gradients = torch.autograd.grad(value, targets, upstream)
                        metrics = {"output": self._metric(reference, value),
                                   "input_gradient": self._metric(ref_grads[0], gradients[0]),
                                   "canonical_weight_gradients": [self._metric(a, b)
                                      for a, b in zip(ref_grads[1:], gradients[1:])]}
                        metrics["equivariance"] = self._equivariance(executor, check_x.detach(),
                                                                    tuple(w.detach() for w in check_weights))
                        metrics["passed"] = (metrics["output"]["passed"] and metrics["input_gradient"]["passed"]
                            and metrics["equivariance"]["passed"]
                            and all(item["passed"] for item in metrics["canonical_weight_gradients"]))
                        row["equivalence"] = metrics
                        if not metrics["passed"]:
                            raise ValueError("Output, canonical gradient or rotation equivariance check failed")
                        row["initialization_and_check_seconds"] = time.perf_counter() - started
                        value = gradients = None
                        samples, forward_samples = [], []
                        for iteration in range(4):
                            if x.device.type == "cuda":
                                begin, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                                begin.record()
                            else:
                                begin = time.perf_counter()
                            output = executor(check_x, executor.map_weights(check_weights))
                            if x.device.type == "cuda":
                                middle.record()
                            else:
                                middle = time.perf_counter()
                            derivative = torch.autograd.grad(output, targets, upstream)
                            if x.device.type == "cuda":
                                end.record()
                                end.synchronize()
                                fwd, total = begin.elapsed_time(middle), begin.elapsed_time(end)
                            else:
                                end = time.perf_counter()
                                fwd, total = (middle - begin) * 1000, (end - begin) * 1000
                            if iteration:
                                forward_samples.append(fwd)
                                samples.append(total)
                            output = derivative = None
                        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
                        row.update({"status": "passed", "forward_backward_median_ms": statistics.median(samples),
                                    "forward_backward_q1_ms": q1, "forward_backward_q3_ms": q3,
                                    "forward_median_ms": statistics.median(forward_samples),
                                    "forward_samples_ms": forward_samples, "forward_backward_samples_ms": samples})
                        if row["forward_backward_median_ms"] < best_ms:
                            best, best_ms = executor, row["forward_backward_median_ms"]
                    except Exception as error:
                        row.update({"status": "failed_equivalence" if "equivalence" in row else "unsupported",
                                    "error": type(error).__name__ + ": " + str(error)})
                        value = gradients = output = derivative = None
                        executor = None
                        if x.device.type == "cuda":
                            torch.cuda.empty_cache()
                    row["selection_seconds"] = time.perf_counter() - started
                    candidates.append(row)
                    print(f"cuEquivariance layer selection: {descriptor}/{method}, status={row['status']}"
                          + (", " + row["error"] if "error" in row else ""), flush=True)
        if best_ms == float("inf"):
            self.cueq_method_selection = {"timing_rows": rows, "candidates": candidates,
                                          "status": "failed", "model_timing_excludes_selection": True}
            raise RuntimeError("No cuEquivariance executor passed method selection")
        object.__setattr__(self, "_cueq_local", best)
        self._cueq_auto_pending = False
        self.cueq_method_selection = {"selected": best.method, "selected_descriptor": best.descriptor_name,
                                      "timing_rows": rows, "warmup": 1, "iterations": 3, "candidates": candidates,
                                      "selection_scope": "one layer, at most 64 edges; canonical mapping, local TP and input/canonical-weight backward",
                                      "rotation_scope": "shared per-forward Wigner; split/grouped bmm; descriptor-native transpose or index reorder",
                                      "model_timing_excludes_selection": True}
        print(f"Selected cuEquivariance layer: {best.descriptor_name}/{best.method}", flush=True)

    def _expert(self, x, expert, *, shared=False):
        weights, bias = [], None
        for m, linear in enumerate(self._linears):
            if isinstance(linear, NaiveExpertLinear):
                weight = linear.weight_shared[expert] if shared else linear._weight(expert)
                if m == 0:
                    bank = linear.bias_shared if shared else linear.bias_experts
                    bias = None if bank is None else bank[expert]
            elif shared:
                weight = linear.weight
                if m == 0:
                    bias = linear.bias
            else:
                weight = linear.weight.new_zeros(linear.weight.shape)
            weights.append(weight)
        return self._local_tp(x, weights, bias)

    def _edge_tp(self, x, routing):
        branch = getattr(routing, "branch", "all")
        linear = self.fc_m0
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        if linear.num_experts and branch != "shared":
            if getattr(routing, "top1_independent", False):
                raise ValueError("This benchmark implements pre-activation expert mixtures")
            indices, values = routing.topk_indices, routing.topk_values
            for expert in range(linear.num_experts):
                if indices is None or values is None:
                    rows = torch.arange(x.shape[0], device=x.device)
                    gate = routing.coefficients[:, expert]
                else:
                    rows, slot = torch.where(indices == expert)
                    gate = values[rows, slot]
                value = self._expert(x.index_select(0, rows), expert)
                out.index_add_(0, rows, value * gate[:, None])
        if branch != "routed":
            for expert in range(linear.num_shared_experts):
                out = out + self._expert(x, expert, shared=True)
        return out

    def _graph_tp(self, x, routing):
        if not self.is_mole:
            return self._local_tp(x, [linear.weight for linear in self._linears], self.fc_m0.bias)
        coefficients = self.fc_m0._coefficients(x, routing)
        groups = coefficients.shape[0]
        banks, bias = [], None
        branch = getattr(routing, "branch", "all")
        for m, linear in enumerate(self._linears):
            if isinstance(linear, NaiveExpertLinear):
                if linear.num_experts and branch != "shared":
                    bank = torch.stack([linear._weight(e) for e in range(linear.num_experts)])
                    weight = torch.einsum("eoi,be->boi", bank, coefficients)
                    if m == 0 and linear.bias_experts is not None:
                        bias = torch.einsum("eo,be->bo", linear.bias_experts, coefficients)
                else:
                    weight = x.new_zeros(groups, linear.out_features, linear.in_features)
                if branch != "routed" and linear.num_shared_experts:
                    weight = weight + linear.weight_shared.sum(0).unsqueeze(0)
                    if m == 0 and linear.bias_shared is not None:
                        shared_bias = linear.bias_shared.sum(0).unsqueeze(0)
                        bias = shared_bias if bias is None else bias + shared_bias
            else:
                weight = linear.weight.unsqueeze(0).expand(groups, -1, -1)
                if branch == "routed":
                    weight = torch.zeros_like(weight)
                elif m == 0:
                    bias = linear.bias
            banks.append(weight)
        index = getattr(routing, "graph_index", None)
        if index is None:
            sizes = getattr(routing, "split_sizes", None)
            if sizes is None:
                sizes = getattr(routing, "sizes", None)
            if sizes is None:
                if groups != 1:
                    raise ValueError("Group routing needs sizes or graph_index")
                sizes = [x.shape[0]]
            sizes = torch.as_tensor(sizes).tolist()
            if len(sizes) != groups or sum(sizes) != x.shape[0]:
                raise ValueError("Group sizes must cover all rows")
            parts, start = [], 0
            for group, size in enumerate(sizes):
                group_bias = None if bias is None else (bias if bias.ndim == 1 else bias[group])
                parts.append(self._local_tp(x[start:start + size], [w[group] for w in banks], group_bias))
                start += size
            return torch.cat(parts, dim=0)
        index = index.to(device=x.device, dtype=torch.long).reshape(-1)
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        for group in range(groups):
            rows = torch.nonzero(index == group, as_tuple=True)[0]
            group_bias = None if bias is None else (bias if bias.ndim == 1 else bias[group])
            part = self._local_tp(x.index_select(0, rows), [w[group] for w in banks], group_bias)
            out.index_copy_(0, rows, part)
        return out

    def forward(self, x, R, mole_globals=None, latents=None, wigner_D_all=None):
        if not self.is_mole and mole_globals is not None:
            latents = mole_globals
        self.forward_calls += 1
        radial = self.radial_emb(latents) if self.radial_emb else None
        # A later output head may require higher l than a preceding hidden
        # update. Rebuild geometry when its shared cache does not cover that l.
        if wigner_D_all is not None:
            enough_blocks = (len(wigner_D_all.blocks) > self.l_max if hasattr(wigner_D_all, "blocks")
                             else wigner_D_all.shape[-1] >= (self.l_max + 1) ** 2)
            if not enough_blocks:
                wigner_D_all = None
        if wigner_D_all is not None:
            self.wigner_cache_hits += 1
        elif (self.rotate_in or self.rotate_out) and self.l_max > 0:
            angle = xyz_to_angles(R[:, [1, 2, 0]])
            wigner_D_all = _make_wigner_rotation(
                self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), self.wigner_apply_mode)
            self.wigner_geometry_builds += 1
        self.last_wigner_layout = ("compact_blocks" if hasattr(wigner_D_all, "blocks") else
                                   "full_dense" if wigner_D_all is not None else "none")
        local = (_rotate_features(x, self._input_rotation_plan, wigner_D_all)
                 if self.rotate_in else x)
        if self.front:
            local = self._radial(local, radial, input_side=True)
        if self.is_mole and getattr(mole_globals, "activation_space", False):
            out = self._edge_tp(local, mole_globals)
        else:
            out = self._graph_tp(local, mole_globals)
        if not self.front:
            out = self._radial(out, radial, input_side=False)
        if self.rotate_out:
            out = _rotate_features(out, self._output_rotation_plan, wigner_D_all, inverse=True)
        result = out.contiguous()
        return (result, wigner_D_all) if self.returns_cache else result


def install_cueq_baseline(model, *, method="auto", descriptor="escn_tp", selection=None):
    """Swap execution after placing the model on its final device in FP32.

    The descriptor's constants live on that device without changing checkpoint
    keys. Restore this handle before moving the model to a different device.
    ``method='auto'`` selects on at most 64 edges in the first untimed pass.
    ``selection`` freezes the per-layer records returned by
    :func:`cueq_execution_metadata`, without repeating selection.
    """
    embedding = getattr(model, "embedding", model)
    if getattr(embedding, "so2_expert_mixing_mode", "pre_activation") not in ("pre_activation", "linear"):
        raise ValueError("This benchmark implements the standard pre-activation UniTB mixture")
    replacements = []
    used = set()

    def visit(parent, prefix=""):
        for name, child in list(parent.named_children()):
            path = prefix + name
            if isinstance(child, (ExpertSO2, SO2LinearCached)):
                if selection is not None and path not in selection:
                    raise ValueError("Frozen cuEquivariance selection is missing layer " + path)
                replacement = CueqSO2Linear(child, method=method, descriptor=descriptor,
                                           selection=None if selection is None else selection[path])
                used.add(path)
            elif isinstance(child, PDQMoE):
                replacement = NaiveExpertLinear(child)
            else:
                visit(child, path + ".")
                continue
            replacements.append((parent, name, child))
            setattr(parent, name, replacement)

    try:
        visit(model)
        if selection is not None and used != set(selection):
            raise ValueError("Frozen cuEquivariance selection includes unknown layers: " + str(set(selection) - used))
    except Exception:
        NaiveExecutionHandle(replacements).restore()
        raise
    return NaiveExecutionHandle(replacements)


def cueq_execution_metadata(model):
    """Record the descriptor methods and observed execution of every SO2 layer."""
    return {name: {"method": module._cueq_local.method if module._cueq_local is not None else None,
                   "descriptor": module._cueq_local.descriptor_name if module._cueq_local is not None else None,
                   "calls": module.cueq_calls, "selection_frozen": module.cueq_selection_frozen,
                   "method_selection": module.cueq_method_selection,
                   "irreps_in": str(module.irreps_in), "irreps_out": str(module.irreps_out),
                   "radial_position": "input" if module.front else "output",
                   "rotation": "pytorch",
                   "rotation_detail": "shared per-forward Wigner; split/grouped bmm; descriptor-native transpose or index reorder",
                   "model_feature_layout": "mul_ir",
                   "descriptor_feature_layout": ("ascending m, then block and channel" if
                       module._cueq_local is not None and module._cueq_local.descriptor_name == "escn_tp_compact"
                       else "ir_mul"),
                   "wigner_shared_per_forward": True,
                   "wigner_layout": module.last_wigner_layout,
                   "wigner_geometry_builds": module.wigner_geometry_builds,
                   "wigner_cache_hits": module.wigner_cache_hits,
                   "forward_calls": module.forward_calls}
            for name, module in model.named_modules() if isinstance(module, CueqSO2Linear)}
