import os

os.environ.setdefault("DPTB_SO2_M_LINEAR_MODE", "indexed_sandwich_materialized_scheduled")
os.environ.setdefault("DPTB_SO2_MATERIALIZED_SCHEDULED_GEMM_STRATEGY", "scheduler")

import so2_cuda_ops

print("SO2 CUDA available:", so2_cuda_ops.is_available())
print("Backend config:", so2_cuda_ops.get_backend_config())
