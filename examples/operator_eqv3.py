"""Adapt the original EquiformerV3 SO(2) convolution without editing it.

Canonical features use e3nn ``mul_ir`` layout.  For positive m, the
canonical pair is (-m, +m), and weights are vertically stacked (A, B):
``y_minus = A x_minus - B x_plus; y_plus = A x_plus + B x_minus``.
EquiformerV3 packs (+m, -m), so its native weight is (A, -B).  Its
``1 / sqrt(2)`` factor is an initialization factor, not a forward factor.

The source must be supplied separately with ``eqv3_root``.  Loading only
the original SO2/rotation modules avoids the repository's training stack
and remains usable on machines without CUDA.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import torch
from torch import nn
from e3nn import o3
from operator_layout import FeatureLayout, NativeOperator


EQV3_COMMIT = "a7300c58df683dc99cb48027d5bfd4c887486c48"
EQV3_REPOSITORY = "https://github.com/atomicarchitects/equiformer_v3"
_PINNED_SOURCE_SHA256 = {
    "so2_ops.py": "215d7e9dc5818dccd62791ad741ad0ab7fe20f0d237b49100a57a3bb84643159",
    "so3.py": "af0910b9cbb035b942168179d130ca08e6ae9c4dab1693e72533a517cefeab97",
    "wigner.py": "ce4f3bf9aa9fc51e659bcfc2545a88a50bb41fa2ac71eb9e5f4289ae38225b49",
    "edge_rot_mat.py": "a836bb183f62e6627386d2c48e27f46cda6d18442425011eb600d68f1c5d5411",
    "Jd.pt": "b4059c45be246dcb6c49c545670b65c56550eb0c2e7a9c92b4b50a92d370dbe2",
}


class UnsupportedConfiguration(ValueError):
    """The original implementation cannot represent the requested layout."""


def _load_original(root):
    """Import original files under an isolated namespace for relative imports."""
    if not root:
        raise ValueError("EquiformerV3 needs --eqv3-root or SO2CUDA_EQV3_ROOT")
    checkout = Path(root).expanduser().resolve()
    source = checkout / "experimental" / "models" / "equiformer_v3"
    required = ("so2_ops.py", "so3.py", "wigner.py", "edge_rot_mat.py", "Jd.pt")
    for filename in required:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"EquiformerV3 source is missing {source / filename}")
    checksums = {filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
                 for filename in required}
    for filename, checksum in checksums.items():
        if checksum != _PINNED_SOURCE_SHA256[filename]:
            raise ValueError(f"EquiformerV3 {filename} differs from the pinned original source")
    try:
        if not (checkout / ".git").exists():
            raise FileNotFoundError("No local Git metadata")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        marker = checkout / ".benchmark_revision"
        revision = marker.read_text().strip() if marker.is_file() else None
    if revision is not None and revision != EQV3_COMMIT:
        raise ValueError(f"EquiformerV3 must be checked out at {EQV3_COMMIT}, got {revision}")
    namespace = "_so2cuda_eqv3_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(source)]
        package.__package__ = namespace
        sys.modules[namespace] = package
    rotation = importlib.import_module(namespace + ".so3")
    linear = importlib.import_module(namespace + ".so2_ops")
    metadata = {
        "repository": EQV3_REPOSITORY,
        "expected_commit": EQV3_COMMIT,
        "commit": revision,
        "source_directory": str(source),
        "source_sha256": checksums,
        "pinned_source_verified": True,
    }
    return rotation.SO3Rotation, linear.SO2Linear, metadata


def _uniform_layout(irreps, label):
    irreps = o3.Irreps(irreps)
    if not irreps:
        raise UnsupportedConfiguration(f"EquiformerV3 cannot express empty {label} irreps")
    lmax = irreps.lmax
    degrees = [ir.l for _, ir in irreps]
    channels = [mul for mul, _ in irreps]
    if sorted(degrees) != list(range(lmax + 1)) or len(set(channels)) != 1:
        raise UnsupportedConfiguration(
            f"EquiformerV3 requires one block for every l=0..lmax and equal "
            f"channel counts across l in {label} irreps"
        )
    return irreps, channels[0], lmax


def _feature_order(irreps, m, device):
    """Indices taking canonical (irrep, channel) order to original (l, channel)."""
    offset = 0
    blocks = []
    for mul, ir in irreps:
        if ir.l >= m:
            blocks.append((ir.l, list(range(offset, offset + mul))))
            offset += mul
    return torch.tensor(
        [index for _, indices in sorted(blocks) for index in indices],
        dtype=torch.long, device=device,
    )


class Eqv3Operator(NativeOperator, nn.Module):
    """Original SO3Rotation -> SO2Linear -> SO3Rotation.rotate_inv.

    ``rotation_matrix`` is the 3x3 rotation whose Wigner matrix the
    canonical implementation right-multiplies into each feature row.
    Original EquiformerV3 left-multiplies feature columns, so its geometry
    is the inverse of this matrix.  Geometry is prepared at construction,
    outside the benchmark's timed region, and has no gradients.

    ``weights`` contains m0 [Nout, Nin] followed by one [2*Nout, Nin]
    tensor per positive m.  The native parameters are detached copies;
    ``canonical_gradients`` maps their gradients back to these shapes.
    """

    def __init__(
        self, irreps_in, irreps_out, mmax, weights, rotation_matrix,
        eqv3_root=None, compile_model=False,
    ):
        super().__init__()
        self.irreps_in, self.channels_in, lmax_in = _uniform_layout(irreps_in, "input")
        self.irreps_out, self.channels_out, lmax_out = _uniform_layout(irreps_out, "output")
        if lmax_in != lmax_out:
            raise UnsupportedConfiguration("EquiformerV3 requires equal input and output lmax")
        self.lmax = lmax_in
        self.mmax = int(mmax)
        if not 0 <= self.mmax <= self.lmax:
            raise ValueError("EquiformerV3 requires 0 <= mmax <= lmax")
        weights = tuple(weights)
        if len(weights) != self.mmax + 1:
            raise ValueError("Provide exactly one canonical weight per retained m")
        device, dtype = weights[0].device, weights[0].dtype
        if dtype != torch.float32:
            raise ValueError("Original EquiformerV3 rotation is evaluated in FP32")
        for m, weight in enumerate(weights):
            nin = sum(mul for mul, ir in self.irreps_in if ir.l >= m)
            nout = sum(mul for mul, ir in self.irreps_out if ir.l >= m)
            shape = ((1 if m == 0 else 2) * nout, nin)
            if tuple(weight.shape) != shape or weight.device != device or weight.dtype != dtype:
                raise ValueError(f"Canonical m={m} weights must have shape {shape}, {dtype}, {device}")
        rotation_class, linear_class, metadata = _load_original(
            eqv3_root or os.environ.get("SO2CUDA_EQV3_ROOT")
        )
        self.rotation = rotation_class(self.lmax, self.mmax).to(device=device, dtype=dtype)
        self.linear = linear_class(
            self.channels_in, self.channels_out, self.lmax, self.mmax,
        ).to(device=device, dtype=dtype)
        self.linear.fc_m0.bias.requires_grad_(False)
        self._weight_shapes = tuple(tuple(weight.shape) for weight in weights)
        for m in range(self.mmax + 1):
            in_order = _feature_order(self.irreps_in, m, device)
            out_order = _feature_order(self.irreps_out, m, device)
            # Original rotate_inv rescales l>mmax by sqrt((2l+1)/(2mmax+1)).
            # Inverse scaling of each output weight row restores D^T L D.
            inverse_rescale = torch.tensor([
                1.0 if l <= self.mmax else ((2 * self.mmax + 1) / (2 * l + 1)) ** 0.5
                for l in range(m, self.lmax + 1)
                for _ in range(self.channels_out)
            ], device=device, dtype=dtype)
            self.register_buffer(f"input_order_{m}", in_order, persistent=False)
            self.register_buffer(f"output_order_{m}", out_order, persistent=False)
            self.register_buffer(f"inverse_rescale_{m}", inverse_rescale, persistent=False)
            nout = len(out_order)
            canonical = weights[m].detach()
            if m == 0:
                native = canonical.index_select(0, out_order).index_select(1, in_order)
            else:
                a = canonical[:nout].index_select(0, out_order).index_select(1, in_order)
                b = canonical[nout:].index_select(0, out_order).index_select(1, in_order)
                native = torch.cat((a, -b), dim=0)
                inverse_rescale = inverse_rescale.repeat(2)
            native = native * inverse_rescale[:, None]
            parameter = self.linear.fc_m0.weight if m == 0 else self.linear.so2_m_linear[m - 1].fc.weight
            with torch.no_grad():
                parameter.copy_(native)
        with torch.no_grad():
            self.linear.fc_m0.bias.zero_()
            self.rotation.set_wigner(
                rotation_matrix.detach().to(device=device, dtype=dtype).transpose(-1, -2).contiguous()
            )
        self.input_layout = FeatureLayout(self.irreps_in, device,
            groups=[[i for i, _ in sorted(enumerate(self.irreps_in), key=lambda item: item[1].ir.l)]],
            coefficient_shape=True)
        self.output_layout = FeatureLayout(self.irreps_out, device,
            groups=[[i for i, _ in sorted(enumerate(self.irreps_out), key=lambda item: item[1].ir.l)]],
            coefficient_shape=True)
        metadata.update({
            "feature_layout": "native [edges, (lmax+1)^2, channels]; conversions outside timing",
            "m_order": "native: m0, +m across l, -m across l; canonical: -m, +m",
            "complex_weights": "native (A,-B) from canonical (A,B)",
            "bias": "m0 bias fixed to zero and excluded from gradients",
            "initialization_scale": "1/sqrt(2) is initialization only; canonical values overwrite it",
            "truncated_m_rescale": "inverse of sqrt((2*l+1)/(2*mmax+1)) on output weight rows for l>mmax",
            "rotation": "original SO3Rotation.set_wigner(inverse canonical rotation), rotate and rotate_inv",
            "compile_requested": bool(compile_model),
            "compile_options": {"dynamic": True} if compile_model else None,
        })
        self.metadata = metadata
        self._execute = torch.compile(self._forward_original, dynamic=True) if compile_model else self._forward_original

    def _forward_original(self, x):
        return self.rotation.rotate_inv(self.linear(self.rotation.rotate(x)))

    def forward_native(self, x):
        return self._execute(x)

    def canonical_gradients(self):
        """Pull native parameter gradients back through the exact weight map."""
        gradients = []
        for m, shape in enumerate(self._weight_shapes):
            parameter = self.linear.fc_m0.weight if m == 0 else self.linear.so2_m_linear[m - 1].fc.weight
            if parameter.grad is None:
                gradients.append(None)
                continue
            in_order = getattr(self, f"input_order_{m}")
            out_order = getattr(self, f"output_order_{m}")
            factor = getattr(self, f"inverse_rescale_{m}")
            rows = len(out_order)
            value = parameter.grad * (factor if m == 0 else factor.repeat(2))[:, None]
            if m > 0:
                value = torch.cat((value[:rows], -value[rows:]), dim=0)
                out_order = torch.cat((out_order, out_order + rows))
            inverse_in = torch.argsort(in_order)
            inverse_out = torch.argsort(out_order)
            gradients.append(value.index_select(0, inverse_out).index_select(1, inverse_in).reshape(shape))
        return tuple(gradients)
