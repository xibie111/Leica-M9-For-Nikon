"""Batch integration with the real JPEG/JSON publisher and existing synthetic RAW fixture."""
import hashlib
import json
from pathlib import Path
import queue

from PIL import Image
import pytest

from m9 import nef
from m9.batch import BatchJob, BatchOptions, PhotoQueue
from m9.uhdr import icc_p3
from test_nef import pipeline  # Reuse the established RAW/network boundary fixture.


def _same_basename_sources(source, tmp_path):
    paths = [source]
    for folder in ("中间 失败", "最后 成功"):
        path = tmp_path / folder / source.name
        path.parent.mkdir()
        path.write_bytes(f"synthetic RAW fixture in {folder}".encode())
        paths.append(path)
    return paths


def _hashes(paths):
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _finish(job, photos):
    job.thread.join(timeout=5)
    assert not job.running
    events = []
    while True:
        try:
            kind, payload = job.events.get_nowait()
        except queue.Empty:
            break
        events.append((kind, payload))
        photos.apply_event(kind, payload)
    return events


@pytest.mark.parametrize("write_json", [False, True])
@pytest.mark.parametrize("failure_stage", ["raw", "publish"])
def test_batch_real_publication_preserves_prior_results_and_continues(
    pipeline, tmp_path, monkeypatch, write_json, failure_stage
):
    source, output, decode, _ = pipeline
    paths = _same_basename_sources(source, tmp_path)
    before = _hashes(paths)
    output.mkdir()
    sentinel = output / f"{source.stem}_M9_native_original.jpg"
    sentinel.write_bytes(b"preexisting user result")
    sentinel.with_suffix(".json").write_bytes(b"preexisting user report")
    initial_files = {path: path.read_bytes() for path in output.iterdir()}
    state = {"current": None, "files_before_failure": {}}
    decoded = decode.return_value

    def load(path, **kwargs):
        state["current"] = Path(path)
        if Path(path) == paths[1]:
            # The first photo must already be fully published before the second starts.
            state["files_before_failure"] = {p: p.read_bytes() for p in output.iterdir()}
            assert len(list(output.glob("*.jpg"))) == 2
            if failure_stage == "raw":
                raise ValueError("synthetic damaged middle RAW")
        return decoded

    decode.side_effect = load
    publish = nef._publish

    def publish_or_fail(temp, final):
        fail_suffix = ".json" if write_json else ".jpg"
        if (failure_stage == "publish" and state["current"] == paths[1]
                and final.suffix == fail_suffix):
            raise OSError("synthetic middle-photo disk failure")
        publish(temp, final)

    monkeypatch.setattr(nef, "_publish", publish_or_fail)
    photos = PhotoQueue()
    items = photos.add_paths(paths)
    job = BatchJob(nef.process_nef)
    job.start(photos.pending_items, BatchOptions("native", output, device="cpu", write_json=write_json))
    events = _finish(job, photos)

    assert [item.status for item in items] == ["succeeded", "failed", "succeeded"]
    assert len({item.id for item in items}) == 3
    assert [call.args[0] for call in decode.call_args_list] == paths
    assert all(call.kwargs["method"] == "native" for call in decode.call_args_list)
    assert _hashes(paths) == before
    assert all(path.read_bytes() == data for path, data in initial_files.items())
    assert all(path.read_bytes() == data for path, data in state["files_before_failure"].items())
    assert items[1].report is None
    assert "middle" in items[1].error
    outputs = [Path(item.report["output"]) for item in (items[0], items[2])]
    assert len(set(outputs)) == 2 and sentinel not in outputs
    assert len(list(output.glob("*.jpg"))) == 3
    assert len(list(output.glob("*.json"))) == (3 if write_json else 1)
    for item, path in zip((items[0], items[2]), outputs):
        report = item.report
        with Image.open(path) as image:
            image.load()
            assert image.size == (49, 33)
            assert image.info["icc_profile"] == icc_p3()
        assert report["input"] == str(item.path)
        assert report["input_sha256"] == before[item.path]
        assert report["jpeg_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert report["models"]["StyleTrans"] == "on"
        if write_json:
            assert json.loads(Path(report["record"]).read_text(encoding="utf-8")) == report
        else:
            assert report["record"] is None and not path.with_suffix(".json").exists()
    assert not list(output.glob("*.lock")) and not list(output.glob("*.part"))
    assert events[-1] == ("finished", {"total": 3, "completed": 3, "succeeded": 2,
                                      "failed": 1, "stopped": False})


def test_stop_during_publication_finishes_active_pair_and_keeps_rest_pending(
    pipeline, tmp_path, monkeypatch
):
    source, output, decode, _ = pipeline
    paths = _same_basename_sources(source, tmp_path)
    before = _hashes(paths)
    photos = PhotoQueue()
    items = photos.add_paths(paths)
    job = BatchJob(nef.process_nef)
    publish = nef._publish

    def stop_after_complete_jpeg(temp, final):
        publish(temp, final)
        if final.suffix == ".jpg":
            job.request_stop()

    monkeypatch.setattr(nef, "_publish", stop_after_complete_jpeg)
    job.start(photos.pending_items, BatchOptions("native", output, device="cpu", write_json=True))
    events = _finish(job, photos)
    assert [item.status for item in items] == ["succeeded", "pending", "pending"]
    assert decode.call_count == 1
    report = items[0].report
    assert json.loads(Path(report["record"]).read_text(encoding="utf-8")) == report
    with Image.open(report["output"]) as image:
        image.load()
        assert image.info["icc_profile"] == icc_p3()
    assert _hashes(paths) == before
    assert sorted(path.suffix for path in output.iterdir()) == [".jpg", ".json"]
    assert events[-1] == ("finished", {"total": 3, "completed": 1, "succeeded": 1,
                                      "failed": 0, "stopped": True})
