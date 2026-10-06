"""Inspect the optional integration backend without constructing a model."""
import so2_cuda_ops
from so2_cuda_ops import deeptb

print("SO2 CUDA available:", so2_cuda_ops.is_available())
print("SO2CUDA version:", so2_cuda_ops.__version__)
print("Integration entry points:", deeptb.__all__)
