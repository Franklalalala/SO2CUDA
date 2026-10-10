"""Keep an extension's toolkit headers within the selected CUDA version."""
from so2_cuda_ops import _extension_loader as loader


def test_selected_toolkit_excludes_other_installed_headers(tmp_path, monkeypatch):
    selected, older = tmp_path / "selected", tmp_path / "older"
    for root in (selected, older):
        (root / "include").mkdir(parents=True)
        (root / "include" / "cuda_runtime_api.h").touch()
        (root / "lib64").mkdir()
    monkeypatch.setenv("CUDA_HOME", str(selected))
    monkeypatch.setattr(loader.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(loader, "_candidate_toolkit_roots", lambda: [selected, older])
    includes, libraries = loader._env_cuda_paths()
    assert str(selected / "include") in includes
    assert str(selected / "lib64") in libraries
    assert all(not path.startswith(str(older)) for path in includes + libraries)
