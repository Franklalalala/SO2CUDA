import pytest

torch = pytest.importorskip("torch")


def test_materialized_scheduler_single_route_shape_cpu_nosync():
    from so2_cuda_ops import materialized_scheduler

    out_ptr = torch.tensor([0, 3, 8, 9], dtype=torch.long)

    edge_order, route_ptr, problem_tile_prefix = materialized_scheduler(
        num_rows=17,
        n_problems=3,
        block_m=8,
        block_n=4,
        out_ptr=out_ptr,
        raw_pair_tiles=False,
        nosync=True,
    )

    torch.testing.assert_close(edge_order, torch.arange(17, dtype=torch.long))
    torch.testing.assert_close(route_ptr, torch.tensor([0, 17], dtype=torch.long))
    torch.testing.assert_close(problem_tile_prefix, torch.tensor([0, 3, 9, 12], dtype=torch.long))


def test_materialized_scheduler_raw_pair_tiles_shape_cpu_nosync():
    from so2_cuda_ops.scheduler import prepare_so2_single_route_layout

    out_ptr = torch.tensor([0, 3, 8, 9], dtype=torch.long)

    _, _, problem_tile_prefix = prepare_so2_single_route_layout(
        num_rows=17,
        n_problems=3,
        block_m=8,
        block_n=4,
        out_ptr=out_ptr,
        raw_pair_tiles=True,
        nosync=True,
    )

    torch.testing.assert_close(problem_tile_prefix, torch.tensor([0, 6, 15, 18], dtype=torch.long))
