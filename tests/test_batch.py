from dataclasses import FrozenInstanceError
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from m9 import batch
from m9.batch import BatchJob, BatchOptions, PhotoQueue


def finish(job, model=None):
    job.thread.join(timeout=3)
    assert not job.running
    events = []
    while True:
        try:
            event = job.events.get_nowait()
        except queue.Empty:
            break
        events.append(event)
        if model is not None:
            model.apply_event(*event)
    assert events[-1][0] == "finished"
    assert sum(kind == "finished" for kind, _ in events) == 1
    return events


def test_queue_deduplicates_normalized_paths_even_after_success_but_not_content(tmp_path):
    first = tmp_path / "照片.NEF"
    second = tmp_path / "same_content.nef"
    first.write_bytes(b"identical source bytes")
    second.write_bytes(first.read_bytes())
    model = PhotoQueue()
    added = model.add_paths([first, tmp_path / "missing_folder" / ".." / first.name, second, first])
    assert [item.path for item in added] == [first.resolve(), second.resolve()]
    assert len({item.id for item in added}) == 2
    assert all(isinstance(item.id, str) and item.id for item in added)
    added[0].status = "succeeded"
    duplicate = first.with_name(first.name.upper()) if batch.os.name == "nt" else first
    assert model.add_paths([duplicate]) == []
    assert model.pending_items == (added[1],)


def test_add_paths_validates_extension_atomically_but_allows_missing_nef(tmp_path):
    model = PhotoQueue()
    with pytest.raises(ValueError, match="NEF"):
        model.add_paths([tmp_path / "valid.NEF", tmp_path / "wrong.jpg"])
    assert model.items == []
    added = model.add_paths([tmp_path / "missing.NeF"])
    assert len(added) == 1 and added[0].status == "pending"
    assert not added[0].path.exists()


def test_folder_current_layer_only_case_insensitive_extension_and_sort(tmp_path):
    for name in ("z.NEF", "B.nef", "a.NeF", "ignore.jpg"):
        (tmp_path / name).write_bytes(b"source")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "first.NEF").write_bytes(b"nested source")
    (tmp_path / "directory.NEF").mkdir()
    model = PhotoQueue()
    assert [item.path.name for item in model.add_folder(tmp_path)] == ["a.NeF", "B.nef", "z.NEF"]
    assert model.add_folder(tmp_path) == []
    with pytest.raises(NotADirectoryError):
        model.add_folder(tmp_path / "missing")


def test_remove_only_pending_and_clear_never_delete_files(tmp_path):
    paths = [tmp_path / f"{i}.NEF" for i in range(4)]
    for path in paths:
        path.write_bytes(b"user input")
    model = PhotoQueue()
    items = model.add_paths(paths)
    for item, status in zip(items, ("succeeded", "failed", "running", "pending")):
        item.status = status
    original_list = model.items
    assert model.remove_pending([item.id for item in items] + ["unknown"]) == [items[3].id]
    assert model.items == items[:3] and model.items is original_list
    model.clear()
    assert original_list == []
    assert all(path.read_bytes() == b"user input" for path in paths)


def test_queue_applies_events_and_requeues_only_failed(tmp_path):
    model = PhotoQueue()
    good, bad, waiting = model.add_paths([tmp_path / f"{i}.NEF" for i in range(3)])
    model.apply_event("item_started", {"id": good.id})
    report = {"output": "good.jpg", "record": None}
    assert model.apply_event("item_done", {"id": good.id, "report": report}) is good
    assert good.status == "succeeded" and good.report is report
    model.apply_event("item_started", {"id": bad.id})
    model.apply_event("item_error", {"id": bad.id, "message": "broken NEF", "traceback": "failure detail"})
    assert (bad.status, bad.report, bad.error, bad.details) == ("failed", None, "broken NEF", "failure detail")
    assert model.apply_event("item_done", {"id": "removed", "report": {}}) is None
    assert model.apply_event("progress", {"id": waiting.id}) is None
    assert model.requeue_failed() == [bad]
    assert (bad.status, bad.report, bad.error, bad.details) == ("pending", None, "", "")
    assert good.status == "succeeded" and good.report is report and waiting.status == "pending"


@pytest.mark.parametrize("write_json", [False, True])
def test_serial_worker_forwards_all_options_continues_failure_and_only_emits_events(tmp_path, write_json):
    model = PhotoQueue()
    items = model.add_paths([tmp_path / name for name in ("first.NEF", "broken.NEF", "last.NEF")])
    entered, release = threading.Event(), threading.Event()
    calls = []
    active = maximum = 0
    caller = threading.get_ident()

    def processor(**kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        calls.append((dict(kwargs), threading.get_ident()))
        try:
            kwargs["progress"]("raw", "decode")
            if kwargs["input_path"].name == "first.NEF":
                entered.set()
                assert release.wait(3)
            if kwargs["input_path"].name == "broken.NEF":
                raise ValueError("invalid source")
            return {"output": str(kwargs["input_path"].with_suffix(".jpg")), "record": None}
        finally:
            active -= 1

    job = BatchJob(processor)
    options = BatchOptions("native", tmp_path / "输出", "phone", "cpu", write_json)
    job.start(model.pending_items, options)
    try:
        assert entered.wait(3) and job.running
        with pytest.raises(RuntimeError, match="队列"):
            job.start(model.pending_items, options)
    finally:
        release.set()
    events = finish(job)
    assert maximum == 1 and len(calls) == 3
    assert all(worker != caller and worker == job.thread.ident for _, worker in calls)
    for kwargs, _ in calls:
        assert kwargs["method"] == "native" and kwargs["resolution"] == "phone"
        assert kwargs["device"] == "cpu" and kwargs["write_json"] is write_json
        assert kwargs["output_dir"] == (tmp_path / "输出").resolve()
    assert [kind for kind, _ in events] == [
        "item_started", "progress", "item_done", "item_started", "progress", "item_error",
        "item_started", "progress", "item_done", "finished"]
    assert [payload["ordinal"] for kind, payload in events if kind == "item_started"] == [1, 2, 3]
    assert all(payload["total"] == 3 for kind, payload in events if kind == "item_started")
    assert "ValueError" in next(payload["traceback"] for kind, payload in events if kind == "item_error")
    assert events[-1][1] == {"total": 3, "completed": 3, "succeeded": 2, "failed": 1, "stopped": False}
    assert all(item.status == "pending" and item.report is None for item in items)
    for event in events:
        model.apply_event(*event)
    assert [item.status for item in items] == ["succeeded", "failed", "succeeded"]


def test_stop_finishes_current_then_resume_does_not_rerun_success(tmp_path):
    model = PhotoQueue()
    items = model.add_paths([tmp_path / f"{i}.NEF" for i in range(3)])
    entered, release = threading.Event(), threading.Event()
    calls = []

    def processor(**kwargs):
        calls.append(kwargs["input_path"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return {"output": "finished.jpg"}

    job = BatchJob(processor)
    options = BatchOptions("native", tmp_path)
    job.start(model.pending_items, options)
    try:
        assert entered.wait(3)
        job.request_stop()
        assert job.running
    finally:
        release.set()
    first = finish(job, model)
    assert first[-1][1] == {"total": 3, "completed": 1, "succeeded": 1, "failed": 0, "stopped": True}
    assert calls == [items[0].path]
    assert [item.status for item in items] == ["succeeded", "pending", "pending"]
    job.start(model.items, options)
    second = finish(job, model)
    assert second[-1][1] == {"total": 2, "completed": 2, "succeeded": 2, "failed": 0, "stopped": False}
    assert calls == [item.path for item in items]
    assert all(item.status == "succeeded" for item in items)


def test_retry_only_failed_and_keeps_success_report(tmp_path):
    model = PhotoQueue()
    good, bad = model.add_paths([tmp_path / "good.NEF", tmp_path / "bad.NEF"])
    calls = []

    def processor(**kwargs):
        path = kwargs["input_path"]
        calls.append(path)
        if path == bad.path and calls.count(path) == 1:
            raise RuntimeError("temporary source error")
        return {"output": f"{path.stem}-{calls.count(path)}.jpg"}

    job = BatchJob(processor)
    options = BatchOptions("dng", tmp_path)
    job.start(model.pending_items, options)
    finish(job, model)
    old_report = good.report
    assert model.requeue_failed() == [bad]
    job.start(model.items, options)
    retry = finish(job, model)
    assert calls == [good.path, bad.path, bad.path]
    assert good.report is old_report and bad.status == "succeeded"
    assert retry[-1][1]["total"] == 1


def test_snapshot_survives_model_and_option_changes(tmp_path):
    model = PhotoQueue()
    items = model.add_paths([tmp_path / "first.NEF", tmp_path / "second.NEF"])
    original = [(item.id, item.path) for item in items]
    entered, release = threading.Event(), threading.Event()
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return {"output": "snapshot.jpg"}

    options = SimpleNamespace(method="native", output_dir=tmp_path, resolution="original", device="cpu", write_json=False)
    job = BatchJob(processor)
    job.start(model.pending_items, options)
    try:
        assert entered.wait(3)
        items[1].id, items[1].path, items[1].status = "changed", tmp_path / "changed.NEF", "failed"
        model.add_paths([tmp_path / "later.NEF"])
        options.method, options.resolution, options.device, options.write_json = "dng", "phone", "cuda", True
        options.output_dir = tmp_path / "changed-output"
    finally:
        release.set()
    events = finish(job)
    assert [call["input_path"] for call in calls] == [path for _, path in original]
    assert [payload["id"] for kind, payload in events if kind == "item_started"] == [item_id for item_id, _ in original]
    assert all(call["method"] == "native" and call["resolution"] == "original" and call["device"] == "cpu"
               and call["write_json"] is False and call["output_dir"] == tmp_path for call in calls)
    with pytest.raises(FrozenInstanceError):
        BatchOptions("native", tmp_path).method = "dng"


def test_thread_start_failure_is_recoverable_and_does_not_change_items(tmp_path, monkeypatch):
    model = PhotoQueue()
    model.add_paths([tmp_path / "photo.NEF"])
    job = BatchJob(lambda **kwargs: {"output": "retry.jpg"})
    options = BatchOptions("native", tmp_path)
    with monkeypatch.context() as context:
        context.setattr(batch.threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("thread startup failed")))
        with pytest.raises(RuntimeError, match="startup"):
            job.start(model.pending_items, options)
    assert not job.running and job.thread is None
    assert model.items[0].status == "pending" and job.events.empty()
    job.start(model.pending_items, options)
    assert finish(job, model)[-1][1]["succeeded"] == 1


@pytest.mark.parametrize("fails", [False, True])
def test_stop_on_last_item_boundary_is_complete_not_stopped(tmp_path, fails):
    model = PhotoQueue()
    model.add_paths([tmp_path / "last.NEF"])

    def processor(**kwargs):
        job.request_stop()
        if fails:
            raise ValueError("last failure")
        return {"output": "last.jpg"}

    job = BatchJob(processor)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    result = finish(job, model)[-1][1]
    assert result == {"total": 1, "completed": 1, "succeeded": int(not fails), "failed": int(fails), "stopped": False}


def test_stop_before_first_boundary_leaves_every_item_pending(tmp_path, monkeypatch):
    model = PhotoQueue()
    model.add_paths([tmp_path / "first.NEF", tmp_path / "second.NEF"])
    entered, release = threading.Event(), threading.Event()
    calls = []
    job = BatchJob(lambda **kwargs: calls.append(kwargs))
    run = job._run

    def delayed(snapshot, options):
        entered.set()
        assert release.wait(3)
        run(snapshot, options)

    monkeypatch.setattr(job, "_run", delayed)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    try:
        assert entered.wait(3)
        job.request_stop()
    finally:
        release.set()
    events = finish(job, model)
    assert events == [("finished", {"total": 2, "completed": 0, "succeeded": 0, "failed": 0, "stopped": True})]
    assert calls == [] and len(model.pending_items) == 2


def test_empty_or_finished_only_start_does_not_create_thread(tmp_path):
    job = BatchJob(lambda **kwargs: {})
    with pytest.raises(ValueError, match="等待"):
        job.start([], BatchOptions("native", tmp_path))
    model = PhotoQueue()
    model.add_paths([tmp_path / "done.NEF"])[0].status = "succeeded"
    with pytest.raises(ValueError):
        job.start(model.items, BatchOptions("native", tmp_path))
    assert job.thread is None and job.events.empty()
