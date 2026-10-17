import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from m9 import nef
from m9.frontend import RawInfo
from m9.ae import AEResult
from m9.uhdr import icc_p3


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    source = tmp_path / "中文 目录" / "DSC 0001.NEF"
    source.parent.mkdir()
    source.write_bytes(b"synthetic test input, not NEF")
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("low", "high", "colorfix"):
        (weights / f"{name}.npz").write_bytes(name.encode())
    raw = np.full((33, 49, 3), .5, np.float32)
    info = RawInfo(exposure_time=.01, iso=100, fnumber=2.8, cct=5500, make="NIKON", model="Z 7_2",
                   as_shot_neutral=np.array([.5, 1., .7]))
    ae = AEResult(200, 100, 3, 26, .6, 15.6, .3, 200, 180, .25, 1.2)
    decode = Mock(return_value=(raw, np.eye(3, dtype=np.float32), info))
    monkeypatch.setattr(nef, "load_nef", decode)
    monkeypatch.setattr(nef, "resolve_device", lambda x: "cpu")
    monkeypatch.setattr(nef, "b2y_from_raw", lambda *args, **kw: SimpleNamespace(
        rgb=raw, lin=raw, info=info, cct=5500, lux_index=200, ae=ae))
    st = Mock(side_effect=lambda image, *args, **kwargs: image)
    st.be = SimpleNamespace(wdir=str(weights))
    monkeypatch.setattr(nef.styletrans, "load", Mock(return_value=st))
    monkeypatch.setattr(nef, "LeicaFilter", Mock(return_value=lambda image, **kw: image))
    return source, tmp_path / "输出 成片", decode, st


@pytest.mark.parametrize("write_json", [False, True])
def test_core_writes_p3_complete_record_and_never_overwrites(pipeline, write_json):
    source, output, decode, st = pipeline
    before = source.read_bytes()
    progress = Mock()
    first = nef.process_nef(source, output, progress=progress, write_json=write_json)
    second = nef.process_nef(source, output, method="native", resolution="phone", device="cpu", write_json=write_json)
    third = nef.process_nef(source, output, write_json=write_json)
    assert Path(first["output"]).name == "DSC 0001_M9_dng_original.jpg"
    assert Path(second["output"]).name == "DSC 0001_M9_native_phone.jpg"
    assert Path(third["output"]).name == "DSC 0001_M9_dng_original_2.jpg"
    assert source.read_bytes() == before
    for report in (first, second, third):
        image_path = Path(report["output"])
        with Image.open(image_path) as image:
            assert image.size == (49, 33)
            assert image.info["icc_profile"] == icc_p3()
        if write_json:
            disk = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
            assert disk == report
        else:
            assert report["record"] is None
            assert not image_path.with_suffix(".json").exists()
        assert json.loads(json.dumps(report)) == report
        assert report["jpeg_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert report["models"]["StyleTrans"] == "on"
        assert set(report["models"]["weights_sha256"]) == {"low", "high", "colorfix"}
        assert report["input_size"] == report["output_size"] == [49, 33]
        assert report["exposure"]["short_adjustment_ev"] == -2
        assert report["resources"]["peak_process_rss_bytes_sampled"] > 0
    assert [c.kwargs["method"] for c in decode.call_args_list] == ["dng", "native", "dng"]
    assert st.call_args_list[0].kwargs["resolution"] == "original"
    assert st.call_args_list[1].kwargs["resolution"] == "phone"
    assert progress.call_args.args[0] == "done"
    assert not list(output.glob("*.lock")) and not list(output.glob("*.part"))


def test_core_json_defaults_off_and_toggling_keeps_old_report(pipeline):
    source, output, _, _ = pipeline
    progress = Mock()
    first = nef.process_nef(source, output, progress=progress)
    assert first["record"] is None
    assert sorted(p.name for p in output.iterdir()) == [Path(first["output"]).name]
    assert first["resources"]["peak_process_rss_bytes_sampled"] > 0
    assert progress.call_args.args[0] == "done"
    enabled = nef.process_nef(source, output, write_json=True)
    old = Path(enabled["record"]).read_bytes()
    disabled = nef.process_nef(source, output)
    assert disabled["record"] is None
    assert Path(disabled["output"]).name.endswith("_3.jpg")
    assert Path(enabled["record"]).read_bytes() == old
    assert first["jpeg_sha256"] == enabled["jpeg_sha256"] == disabled["jpeg_sha256"]


def test_default_off_preserves_orphan_json_and_avoids_stale_report_pair(pipeline):
    source, output, _, _ = pipeline
    output.mkdir()
    existing = output / "DSC 0001_M9_dng_original.json"
    existing.write_bytes(b"existing processing evidence")
    report = nef.process_nef(source, output)
    assert Path(report["output"]).name.endswith("_2.jpg")
    assert existing.read_bytes() == b"existing processing evidence"
    assert sorted(p.suffix for p in output.iterdir()) == [".jpg", ".json"]


@pytest.mark.parametrize("stage", ["models", "raw", "style", "filter", "jpeg"])
def test_core_failure_publishes_nothing_and_does_not_retry(pipeline, monkeypatch, stage):
    source, output, decode, st = pipeline
    failure = RuntimeError("deliberate failure")
    if stage == "models":
        nef.styletrans.load.side_effect = failure
    elif stage == "raw":
        decode.side_effect = failure
    elif stage == "style":
        st.side_effect = failure
    elif stage == "filter":
        nef.LeicaFilter.side_effect = failure
    else:
        monkeypatch.setattr(nef, "_encode_jpeg", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match=stage):
        nef.process_nef(source, output, method="native")
    assert not output.exists() or not list(output.iterdir())
    assert decode.call_count <= 1


def test_atomic_publish_rolls_back_when_second_publish_fails(tmp_path, monkeypatch):
    target = tmp_path / "result.jpg"
    real_publish = nef._publish

    def publish(temp, final):
        if final.suffix == ".json":
            raise OSError("simulated disk write failure")
        real_publish(temp, final)

    monkeypatch.setattr(nef, "_publish", publish)
    with pytest.raises(OSError):
        nef._write_result(b"jpeg test", {}, target, write_json=True)
    assert not list(tmp_path.iterdir())


def test_atomic_publish_preserves_concurrent_existing_file(tmp_path):
    target = tmp_path / "result.jpg"
    target.write_bytes(b"existing user file")
    with pytest.raises(FileExistsError):
        nef._write_result(b"new", {}, target)
    assert target.read_bytes() == b"existing user file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.jpg"]


def test_json_publish_collision_rolls_back_only_new_jpeg(tmp_path):
    target = tmp_path / "result.jpg"
    existing = target.with_suffix(".json")
    existing.write_bytes(b"preexisting evidence")
    with pytest.raises(FileExistsError):
        nef._write_result(b"new jpeg", {}, target, write_json=True)
    assert existing.read_bytes() == b"preexisting evidence"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


def test_json_write_failure_before_publish_leaves_no_jpeg(tmp_path, monkeypatch):
    real_temporary = nef.tempfile.NamedTemporaryFile
    calls = []

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream
            self.name = stream.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, payload):
            raise OSError("JSON write failed")

    def temporary(**kwargs):
        calls.append(True)
        stream = real_temporary(**kwargs)
        return FailingStream(stream) if len(calls) == 2 else stream

    monkeypatch.setattr(nef.tempfile, "NamedTemporaryFile", temporary)
    with pytest.raises(OSError, match="JSON write failed"):
        nef._write_result(b"jpeg", {}, tmp_path / "result.jpg", write_json=True)
    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


def test_core_json_publish_failure_has_no_done_event_or_partial_files(pipeline, monkeypatch):
    source, output, _, _ = pipeline
    progress = Mock()
    real_publish = nef._publish

    def publish(temp, final):
        if final.suffix == ".json":
            raise OSError("JSON publish failed")
        real_publish(temp, final)

    monkeypatch.setattr(nef, "_publish", publish)
    with pytest.raises(OSError, match="JSON publish failed"):
        nef.process_nef(source, output, write_json=True, progress=progress)
    assert not list(output.iterdir())
    assert all(call.args[0] != "done" for call in progress.call_args_list)


def test_cli_defaults_and_failures(tmp_path, monkeypatch):
    run = Mock(return_value={"output": "test.jpg"})
    monkeypatch.setattr(nef, "process_nef", run)
    assert nef.main([str(tmp_path / "input.NEF")]) == 0
    assert run.call_args.args[2:5] == ("dng", "original", "auto")
    assert run.call_args.kwargs["write_json"] is False
    assert nef.main([str(tmp_path / "input.NEF"), "--write-json"]) == 0
    assert run.call_args.kwargs["write_json"] is True
    run.side_effect = RuntimeError("missing converter")
    assert nef.main([str(tmp_path / "input.NEF"), "--method", "native"]) == 1


def test_bounded_jpeg_encoding_matches_existing_encoder():
    from m9.uhdr import encode_p3_jpeg
    rgb = np.random.default_rng(17).random((271, 313, 3), dtype=np.float32)
    assert nef._encode_jpeg(rgb) == encode_p3_jpeg(rgb, 95)
