import hashlib
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

from m9 import session as sessions


class Owner:
    pass


@pytest.fixture
def models(tmp_path, monkeypatch):
    for name in ("low", "high", "colorfix"):
        (tmp_path / f"{name}.npz").write_bytes(name.encode())
    network_refs = []
    def load(**kwargs):
        nets = {name: Owner() for name in ("low", "high", "colorfix")}
        network_refs.extend(weakref.ref(net) for net in nets.values())
        return SimpleNamespace(be=SimpleNamespace(wdir=str(tmp_path), nets=nets))
    loader = Mock(side_effect=load)
    monkeypatch.setattr(sessions.styletrans, "load", loader)
    return tmp_path, loader, network_refs


def test_lazy_strict_load_once_per_round_and_fresh_next_round(models):
    directory, load, refs = models
    with sessions.NefSession("cpu") as session:
        load.assert_not_called()
        first = session.get_model()
        assert first is session.get_model() and session.load_count == 1
        load.assert_called_once_with(device="cpu", strict=True)
        assert session.weights_sha256 == {name: hashlib.sha256(name.encode()).hexdigest()
                                         for name in ("low", "high", "colorfix")}
        identity = session.weights_sha256
        identity.clear()
        assert session.weights_sha256
        (directory / "low.npz").write_bytes(b"new file for next round")
        assert session.get_model() is first and session.weights_sha256["low"] != hashlib.sha256(b"new file for next round").hexdigest()
    assert first.be is None and not session.loaded and session.closed
    assert all(ref() is None for ref in refs)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.get_model()
    with sessions.NefSession("cpu") as next_session:
        assert next_session.get_model() is not first
        assert next_session.weights_sha256["low"] == hashlib.sha256(b"new file for next round").hexdigest()
    assert load.call_count == 2


def test_invalidate_allows_reload_and_releases_network_owners(models):
    _, load, refs = models
    with sessions.NefSession("cpu") as session:
        first = session.get_model()
        session.invalidate()
        assert first.be is None and all(ref() is None for ref in refs)
        assert session.weights_sha256 == {} and not session.closed
        assert session.get_model() is not first and session.load_count == 2
    assert load.call_count == 2 and all(ref() is None for ref in refs)


def test_partial_load_failure_clears_unwound_exception_tensor_references(models):
    _, load, _ = models
    refs = []
    def fail(**kwargs):
        allocated = Owner()
        refs.append(weakref.ref(allocated))
        raise RuntimeError("partially loaded network")
    load.side_effect = fail
    with sessions.NefSession("cpu") as session:
        with pytest.raises(RuntimeError, match="partially loaded") as captured:
            session.get_model()
        assert captured.value.__traceback__ is not None
        assert not session.loaded and session.weights_sha256 == {}
        assert refs[0]() is None


def test_interrupted_partial_load_releases_models_and_propagates(models):
    _, load, _ = models
    refs = []
    def interrupted(**kwargs):
        allocated = Owner()
        refs.append(weakref.ref(allocated))
        raise KeyboardInterrupt()
    load.side_effect = interrupted
    with sessions.NefSession("cpu") as session:
        with pytest.raises(KeyboardInterrupt):
            session.get_model()
        assert not session.loaded and refs[0]() is None


def test_failure_hashing_loaded_weights_also_releases_models(models):
    directory, _, refs = models
    (directory / "high.npz").unlink()
    with sessions.NefSession("cpu") as session:
        with pytest.raises(FileNotFoundError):
            session.get_model()
        assert not session.loaded and all(ref() is None for ref in refs)


def test_fixed_device_and_cuda_cleanup_after_detaching_models(models, monkeypatch):
    import torch
    _, _, refs = models
    monkeypatch.setattr(sessions, "resolve_device", lambda device: "cuda")
    empty = Mock(side_effect=lambda: all(ref() is None for ref in refs))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    with sessions.NefSession("auto") as session:
        session.get_model()
        monkeypatch.setattr(sessions, "resolve_device", lambda device: "cpu")
        assert session.device == "cuda"
        with pytest.raises(AttributeError):
            session.device = "cpu"
    empty.assert_called_once_with()
    assert all(ref() is None for ref in refs)


def test_cleanup_failure_does_not_hide_original_failure(models, monkeypatch):
    import torch
    monkeypatch.setattr(sessions, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock(side_effect=RuntimeError("cleanup error")))
    with pytest.raises(ValueError, match="original error") as captured:
        with sessions.NefSession("cuda") as session:
            session.get_model()
            raise ValueError("original error")
    assert session.closed and not session.loaded
    assert any("cleanup" in note for note in captured.value.__notes__)
