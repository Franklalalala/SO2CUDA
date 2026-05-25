def test_public_api_imports():
    import so2_cuda_ops

    assert isinstance(so2_cuda_ops.__version__, str)
    assert callable(so2_cuda_ops.is_available)
    assert callable(so2_cuda_ops.get_backend_config)
    assert callable(so2_cuda_ops.set_backend_config)
    assert callable(so2_cuda_ops.indexed_sandwich_multi)
    assert callable(so2_cuda_ops.materialized_scheduler)


def test_backend_config_round_trip(monkeypatch):
    import so2_cuda_ops

    monkeypatch.delenv("SO2_CUDA_BACKEND", raising=False)
    monkeypatch.delenv("SO2_CUDA_MIN_EDGES", raising=False)
    monkeypatch.delenv("SO2_CUDA_MATERIALIZED_MIN_EDGES", raising=False)
    monkeypatch.delenv("SO2_CUDA_GEMM_STRATEGY", raising=False)

    before = so2_cuda_ops.get_backend_config()
    so2_cuda_ops.set_backend_config(
        backend="indexed_sandwich",
        min_edges=128,
        materialized_min_edges=256,
        gemm_strategy="scheduler",
    )
    after = so2_cuda_ops.get_backend_config()

    assert before.backend == "auto"
    assert after.backend == "indexed_sandwich"
    assert after.min_edges == 128
    assert after.materialized_min_edges == 256
    assert after.gemm_strategy == "scheduler"


def test_profiler_records_host_spans_without_cuda(monkeypatch):
    import so2_cuda_ops
    from so2_cuda_ops.profiler import (
        get_profile_summary,
        profile_enabled,
        record_host_span,
        reset_profile_summary,
    )

    assert callable(so2_cuda_ops.get_profile_summary)
    assert callable(so2_cuda_ops.reset_profile_summary)

    monkeypatch.setenv("SO2_CUDA_PROFILE", "1")
    monkeypatch.setenv("SO2_CUDA_PROFILE_PRINT_EVERY", "0")
    reset_profile_summary()

    result = record_host_span("metadata_plan", lambda: "ok")
    summary = get_profile_summary(reset=True)

    assert result == "ok"
    assert profile_enabled()
    assert summary["host_ms"]["metadata_plan"]["count"] == 1
    assert summary["host_ms"]["metadata_plan"]["total"] >= 0.0
