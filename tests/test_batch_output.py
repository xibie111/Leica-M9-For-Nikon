"""Batch integration with the real JPEG/JSON publisher and existing synthetic RAW fixture."""
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import queue
from unittest.mock import Mock

from PIL import Image
import pytest

from m9 import frontend, nef
from m9.batch import BatchJob, BatchOptions, PhotoQueue
from m9.session import NefSession
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
    sentinel = output / f"{source.stem}_M9_tele_native_original.jpg"
    sentinel.write_bytes(b"preexisting user result")
    sentinel.with_suffix(".json").write_bytes(b"preexisting user report")
    initial_files = {path: path.read_bytes() for path in output.iterdir()}
    state = {"current": None, "files_before_failure": {}}
    decoded = decode.return_value
    focal_lengths = dict(zip(paths, (75, 98.9, 50)))
    metadata = Mock(side_effect=lambda path: {"FocalLengthIn35mmFilm": focal_lengths[Path(path)]})
    monkeypatch.setattr(frontend, "dng_tags", metadata)

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
    assert [path.name for path in outputs] == [f"{source.stem}_M9_tele_native_original_{n}.jpg" for n in (2, 3)]
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
        assert report["camera_tuning"] == "tele" and report["ev"] == 0.0
        assert report["zoom"] == pytest.approx(75 / 23 if item is items[0] else 3.2)
        if write_json:
            assert json.loads(Path(report["record"]).read_text(encoding="utf-8")) == report
        else:
            assert report["record"] is None and not path.with_suffix(".json").exists()
    focal_paths = [call.args[0] for call in metadata.call_args_list]
    assert focal_paths == [str(path) for path in (paths if failure_stage == "publish" else (paths[0], paths[2]))]
    assert all(call.kwargs == {"ev": 0.0, "camera": "tele"} for call in nef.b2y_from_raw.call_args_list)
    assert [call.kwargs["zoom"] for call in nef.LeicaFilter.return_value.call_args_list] == pytest.approx(
        [75 / 23, 4.3, 3.2] if failure_stage == "publish" else [75 / 23, 3.2])
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
    assert report["camera_tuning"] == "tele" and report["zoom"] == 4.3
    assert Path(report["output"]).name == f"{source.stem}_M9_tele_native_original.jpg"
    assert json.loads(Path(report["record"]).read_text(encoding="utf-8")) == report
    with Image.open(report["output"]) as image:
        image.load()
        assert image.info["icc_profile"] == icc_p3()
    assert _hashes(paths) == before
    assert sorted(path.suffix for path in output.iterdir()) == [".jpg", ".json"]
    assert events[-1] == ("finished", {"total": 3, "completed": 1, "succeeded": 1,
                                      "failed": 0, "stopped": True})


@pytest.mark.parametrize("fail_middle", [False, True])
def test_batch_context_reuses_models_but_keeps_photo_decisions_and_publication_independent(
    pipeline, tmp_path, monkeypatch, fail_middle
):
    source, output, decode, style = pipeline
    paths = _same_basename_sources(source, tmp_path)
    scenes = dict(zip(paths, ((350, 4000), (200, 5500), (350, 4500))))
    for index, path in enumerate(paths):
        # Valid tiny TIFF/Exif headers let the real optional reader retain each
        # original's ISO; RAW pixels and networks remain the existing fakes.
        tags = Image.Exif()
        tags[271], tags[272] = "NIKON", "NIKON Z6_3"
        tags[34665] = {34855: 100 * (index + 1), 36867: f"2026:10:17 12:00:0{index}"}
        Image.new("L", (1, 1)).save(path, format="TIFF", exif=tags.tobytes())
    before = _hashes(paths)
    output.mkdir()
    sentinel = output / f"{source.stem}_M9_tele_native_original.jpg"
    sentinel.write_bytes(b"previous result")
    sentinel.with_suffix(".json").write_bytes(b"previous report")
    old_files = {path: path.read_bytes() for path in output.iterdir()}

    sessions, generations, observed, options_seen, finished_states = [], [], [], [], []
    state = {"path": None, "failed": False}
    raw, matrix, info = decode.return_value
    original_frontend = nef.b2y_from_raw.side_effect
    original_loader = nef.styletrans.load.side_effect

    def load_raw(path, **kwargs):
        state["path"] = Path(path)
        return raw, matrix, replace(info, cct=scenes[Path(path)][1])

    def photo_frontend(*args, **kwargs):
        result = original_frontend(*args, **kwargs)
        result.lux_index, result.cct = scenes[state["path"]]
        return result

    def load_models(**kwargs):
        model = original_loader(**kwargs)
        model.be.nets.update({name: object() for name in ("low", "high", "colorfix")})
        generations.append((model.be, tuple(model.be.nets.values())))
        return model

    def apply_style(image, lux_index, cct, **kwargs):
        active = sessions[-1]
        assert active.loaded and active._model is style
        observed.append({"path": state["path"], "scene": (lux_index, cct),
                         "backend": style.be, "networks": tuple(style.be.nets.values()),
                         "weights": active.weights_sha256})
        if fail_middle and state["path"] == paths[1] and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("synthetic middle-photo style failure")
        if fail_middle and state["path"] == paths[2]:
            assert generations[0][0].nets == {}  # Failure released the old backend first.
        return image

    decode.side_effect = load_raw
    nef.b2y_from_raw.side_effect = photo_frontend
    nef.styletrans.load.side_effect = load_models
    style.side_effect = apply_style

    @contextmanager
    def processor_context(options):
        options_seen.append(options)
        with NefSession(options.device) as session:
            sessions.append(session)
            yield partial(nef.process_nef, session=session)

    unused = Mock(side_effect=AssertionError("the context's processor must handle this round"))
    job = BatchJob(unused, processor_context=processor_context)
    real_put = job.events.put

    def put(event, *args, **kwargs):
        if event[0] == "finished":
            finished_states.append([(session.closed, session.loaded) for session in sessions])
        return real_put(event, *args, **kwargs)

    monkeypatch.setattr(job.events, "put", put)
    photos = PhotoQueue()
    items = photos.add_paths(paths)
    options = BatchOptions("native", output, device="cpu", write_json=True)
    job.start(photos.pending_items, options)
    events = _finish(job, photos)
    assert [item.status for item in items] == (["succeeded", "failed", "succeeded"] if fail_middle
                                              else ["succeeded"] * 3)
    assert nef.styletrans.load.call_count == (2 if fail_middle else 1)
    assert sessions[0].load_count == (2 if fail_middle else 1)
    assert sessions[0].closed and not sessions[0].loaded and not sessions[0].weights_sha256
    assert observed[0]["scene"] == (350, 4000) and observed[1]["scene"] == (200, 5500)
    assert observed[2]["scene"] == (350, 4500)
    if fail_middle:
        assert observed[0]["backend"] is observed[1]["backend"]
        assert observed[2]["backend"] is not observed[0]["backend"]
        assert observed[2]["networks"] != observed[0]["networks"]
        assert "middle-photo" in items[1].error and items[1].report is None
    else:
        assert all(observation["backend"] is observed[0]["backend"] for observation in observed)
        assert all(observation["networks"] == observed[0]["networks"] for observation in observed)
    assert observed[0]["weights"] == observed[1]["weights"] == observed[2]["weights"]
    assert all(backend.nets == {} for backend, _ in generations)
    succeeded = [item for item in items if item.status == "succeeded"]
    assert [item.report["models"]["selected_style"] for item in succeeded] == (
        ["low", "low"] if fail_middle else ["low", "high", "low"])
    first_round_files = {path: path.read_bytes() for path in output.iterdir()}
    assert events[-1] == ("finished", {"total": 3, "completed": 3,
                                      "succeeded": len(succeeded), "failed": int(fail_middle), "stopped": False})

    # A deliberately new round owns a fresh session and must strictly load again.
    photos.clear()
    next_item, = photos.add_paths([paths[0]])
    job.start(photos.pending_items, options)
    second_events = _finish(job, photos)
    assert next_item.status == "succeeded" and sessions[1] is not sessions[0]
    assert sessions[1].load_count == 1 and sessions[1].closed and not sessions[1].loaded
    assert nef.styletrans.load.call_count == (3 if fail_middle else 2)
    assert all(call.kwargs == {"device": "cpu", "strict": True} for call in nef.styletrans.load.call_args_list)
    assert finished_states == [[(True, False)], [(True, False), (True, False)]]
    assert options_seen == [options, options]
    assert second_events[-1] == ("finished", {"total": 1, "completed": 1,
                                             "succeeded": 1, "failed": 0, "stopped": False})
    unused.assert_not_called()
    assert all(path.read_bytes() == content for path, content in first_round_files.items())
    assert all(path.read_bytes() == content for path, content in old_files.items())
    assert _hashes(paths) == before
    for item in [*succeeded, next_item]:
        report = item.report
        assert report["exif"]["success"] and report["exif"]["source_read"]
        assert report["exif"]["source"] == str(item.path)
        assert json.loads(Path(report["record"]).read_text(encoding="utf-8")) == report
        with Image.open(report["output"]) as image:
            image.load()
            root = image.getexif()
            nested = root.get_ifd(34665)
            assert image.info["icc_profile"] == icc_p3() and image.size == (49, 33)
            assert root[274] == 1 and nested[40961] == 65535
            assert [nested[40962], nested[40963]] == [49, 33]
            assert nested[34855] == 100 * (paths.index(item.path) + 1)
    assert not list(output.glob("*.lock")) and not list(output.glob("*.part"))
