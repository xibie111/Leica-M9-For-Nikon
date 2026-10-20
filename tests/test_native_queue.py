"""Synthetic private queue wiring contracts; never boot a real decoder/helper."""
from contextlib import contextmanager
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

import pytest

from m9.batch import BatchJob, BatchOptions, PhotoQueue
from m9 import native_queue


class Manager:
    def __init__(self, *, ready=True):
        self.ready = ready
        self.calls = []
        self.closed = False
        self.disabled = False
        self.pending = None
        self.discards = []

    def offer(self, request, path):
        self.calls.append(("offer", request, str(path)))
        if self.disabled:
            return False
        self.pending = request
        return True

    def take_nowait(self, request, path):
        self.calls.append(("take", request, str(path)))
        self.pending = None
        if not self.ready:
            self.disabled = True
            return None
        return ("linear", "matrix", "info")

    @property
    def diagnostics(self):
        return {"closed": self.closed, "cleanup_safe": self.closed, "calls": [list(row) for row in self.calls]}

    def discard(self, request):
        self.discards.append(request)
        if self.pending == request:
            self.pending = None
            self.disabled = self.closed = True

    def close(self):
        self.closed = True
        self.calls.append(("close",))
        return {"closed": True, "reader_closed": True, "slot_closed": True, "cleanup_safe": True}


def finish(job, model):
    job.thread.join(3)
    assert not job.running
    events = []
    while True:
        try:
            event = job.events.get_nowait()
        except queue.Empty:
            break
        events.append(event)
        model.apply_event(*event)
    assert [kind for kind, _ in events].count("finished") == 1
    assert events[-1][0] == "finished"
    return events


def planned_job(tmp_path, monkeypatch, manager, processor, count=3):
    model = PhotoQueue()
    model.add_paths([tmp_path / f"{index}.NEF" for index in range(count)])
    monkeypatch.setattr(native_queue, "_make_lookahead", lambda: manager)

    @contextmanager
    def factory(options, *, _plan):
        with native_queue._native_processor_context(processor, object(), _plan) as run:
            yield run

    job = BatchJob(None, factory, _plan_context=factory)
    return job, model


def test_real_batch_plan_gates_once_and_consumes_only_current_ready_request(tmp_path, monkeypatch):
    manager, decoded, public = Manager(), [], []

    def processor(**kwargs):
        path, progress = kwargs["input_path"], kwargs["progress"]
        decoded.append((path, kwargs["_native_decode"](path)))
        progress("frontend", "CPU work")
        if len(decoded) == 1:
            assert not any(row[0] == "offer" for row in manager.calls)
        progress("style_network", "GPU gate")
        progress("style_network", "same public stage again")
        return {"output": str(path.with_suffix(".jpg"))}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor)
    original_put = job.events.put

    def put(event, *args, **kwargs):
        public.append(event[0])
        return original_put(event, *args, **kwargs)

    job.events.put = put
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    events = finish(job, model)
    ids = [item.id for item in model.items]
    assert [row[:2] for row in manager.calls] == [("offer", ids[1]), ("take", ids[1]),
                                                          ("offer", ids[2]), ("take", ids[2]), ("close",)]
    assert decoded[0][1] is None and all(row[1] == ("linear", "matrix", "info") for row in decoded[1:])
    assert [kind for kind, _ in events] == ["item_started", "progress", "progress", "progress", "item_done"] * 3 + ["finished"]
    assert public[-1] == "finished" and manager.closed
    assert job._plan_receipt["close"]["slot_closed"]
    json.dumps(job._plan_receipt)  # The receipt owns primitives, not manager/model/image objects.


def test_helper_miss_stays_private_and_formal_processor_runs_once_per_item(tmp_path, monkeypatch):
    manager, formal = Manager(ready=False), []

    def processor(**kwargs):
        assert kwargs["_native_decode"](kwargs["input_path"]) is None
        formal.append(kwargs["input_path"])
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "formal.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    events = finish(job, model)
    assert formal == [item.path for item in model.items]
    assert not any(kind in ("item_error", "batch_error") for kind, _ in events)
    assert [row[0] for row in manager.calls].count("take") == 2
    assert events[-1][1]["succeeded"] == 3


def test_stop_before_gpu_gate_never_offers_next_decode(tmp_path, monkeypatch):
    manager = Manager()
    entered, release = threading.Event(), threading.Event()

    def processor(**kwargs):
        kwargs["progress"]("frontend", "CPU")
        entered.set()
        assert release.wait(3)
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "first.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    try:
        assert entered.wait(3)
        job.request_stop()
    finally:
        release.set()
    events = finish(job, model)
    assert manager.calls == [("close",)]
    assert [item.status for item in model.items] == ["succeeded", "pending", "pending"]
    assert events[-1][1] == dict(total=3, completed=1, succeeded=1, failed=0, stopped=True)


def test_stop_waits_for_helper_close_before_finished_without_starting_next(tmp_path, monkeypatch):
    offered, release_photo = threading.Event(), threading.Event()
    cleaning, release_cleanup = threading.Event(), threading.Event()

    class ClosingManager(Manager):
        def close(self):
            cleaning.set()
            assert release_cleanup.wait(3)
            return super().close()

    manager = ClosingManager()

    def processor(**kwargs):
        kwargs["progress"]("style_network", "GPU gate")
        offered.set()
        assert release_photo.wait(3)
        return {"output": "first.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor, count=2)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    try:
        assert offered.wait(3)
        job.request_stop()
        release_photo.set()
        assert cleaning.wait(3)
        assert job.running and not manager.closed
        assert not any(kind == "finished" for kind, _ in list(job.events.queue))
    finally:
        release_photo.set()
        release_cleanup.set()
    events = finish(job, model)
    assert manager.closed and [item.status for item in model.items] == ["succeeded", "pending"]
    assert [kind for kind, _ in events].count("item_started") == 1


def test_slow_offer_cleanup_never_holds_gui_stop_boundary(tmp_path, monkeypatch):
    closing, release = threading.Event(), threading.Event()

    class SlowOffer(Manager):
        def offer(self, request, path):
            closing.set()
            assert release.wait(3)
            self.disabled = True
            return False

    manager = SlowOffer()

    def processor(**kwargs):
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "first.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor, count=2)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    stopped = threading.Event()
    stopper = threading.Thread(target=lambda: (job.request_stop(), stopped.set()))
    try:
        assert closing.wait(3)
        stopper.start()
        assert stopped.wait(1), "Stop was blocked by offer/IPC cleanup under the boundary lock"
        assert not release.is_set() and job.running
    finally:
        release.set()
        stopper.join(3)
    events = finish(job, model)
    assert events[-1][1] == dict(total=2, completed=1, succeeded=1, failed=0, stopped=True)
    assert manager.closed and model.items[1].status == "pending"


def test_false_offer_is_taken_and_pre_raw_failure_discards_only_its_own_request(tmp_path, monkeypatch):
    class DeferredFailure(Manager):
        def offer(self, request, path):
            super().offer(request, path)
            return False  # A failed send can still need cleanup before formal raw.

    manager = DeferredFailure()
    calls = []

    def processor(**kwargs):
        path = kwargs["input_path"]
        calls.append(path)
        if len(calls) == 2:
            raise ValueError("formal item failed before raw")
        kwargs["_native_decode"](path)
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "formal.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    events = finish(job, model)
    assert model.items[1].id in manager.discards and manager.pending is None
    assert [item.status for item in model.items] == ["succeeded", "failed", "succeeded"]
    assert events[-1][1]["failed"] == 1 and not any(kind == "batch_error" for kind, _ in events)


def test_false_offer_is_consumed_at_raw_boundary_before_original_fallback(tmp_path, monkeypatch):
    class FailedOffer(Manager):
        def offer(self, request, path):
            self.pending = request
            return False

        def take_nowait(self, request, path):
            self.calls.append(("take", request, str(path)))
            self.pending = None
            return None

    manager = FailedOffer()

    def processor(**kwargs):
        assert kwargs["_native_decode"](kwargs["input_path"]) is None
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "original.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor, count=2)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    finish(job, model)
    assert [row[1] for row in manager.calls if row[0] == "take"] == [model.items[1].id]


@pytest.mark.parametrize("interrupt", [False, True])
def test_close_error_visible_and_original_base_exception_preserved(tmp_path, monkeypatch, interrupt):
    class BrokenClose(Manager):
        def close(self):
            super().close()
            raise RuntimeError("helper close failed")

    manager = BrokenClose()

    def processor(**kwargs):
        if interrupt:
            raise KeyboardInterrupt("original interruption")
        return {"output": "first.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor, count=1)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    events = finish(job, model)
    error = next(payload for kind, payload in events if kind == "batch_error")
    assert error["phase"] == ("processing" if interrupt else "cleanup")
    assert "helper close failed" in error["traceback"]
    assert ("KeyboardInterrupt" in error["traceback"]) == interrupt
    assert job._plan_receipt["cleanup_error"] == "helper close failed"
    assert events[-1][1]["failed"] == int(interrupt)


def test_factory_identity_opt_in_leaves_replaced_context_and_custom_processor_untouched(tmp_path):
    calls = []

    class Processor:
        def __getattr__(self, name):
            raise AssertionError(f"Do not inspect custom processor attributes: {name}")

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {"output": "custom.jpg"}

    def original(options, *, _plan):
        pytest.fail("Replaced original factory must not be used")

    @contextmanager
    def replacement(options):
        yield Processor()

    model = PhotoQueue()
    model.add_paths([tmp_path / "custom.NEF"])
    job = BatchJob(None, original, _plan_context=original)
    job.processor_context = replacement
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    finish(job, model)
    assert set(calls[0]) == {"input_path", "output_dir", "method", "resolution", "device", "write_json", "progress"}
    assert job._plan_receipt == {}


def test_critical_take_cleanup_stops_batch_instead_of_formal_fallback_or_next_start(tmp_path, monkeypatch):
    class CleanupFailure(BaseException):
        pass

    class CriticalManager(Manager):
        def take_nowait(self, request, path):
            raise CleanupFailure("could not prove helper stopped")

    manager = CriticalManager()

    def processor(**kwargs):
        kwargs["_native_decode"](kwargs["input_path"])
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "first.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    events = finish(job, model)
    assert manager.closed
    assert [item.status for item in model.items] == ["succeeded", "failed", "pending"]
    assert events[-1][1] == dict(total=3, completed=2, succeeded=1, failed=1, stopped=False)
    error = next(payload for kind, payload in events if kind == "batch_error")
    assert error["phase"] == "processing" and "CleanupFailure" in error["traceback"]


def test_unproven_special_cleanup_emits_error_without_finished_and_latches_restart(tmp_path, monkeypatch):
    from m9.native_prefetch import NativePrefetchCleanupError

    class UnsafeManager(Manager):
        def close(self):
            raise NativePrefetchCleanupError("reader/slot ownership unresolved")

    manager = UnsafeManager()

    def processor(**kwargs):
        return {"output": "first.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor, count=1)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    job.thread.join(3)
    assert not job.running and job._cleanup_unsafe
    events = list(job.events.queue)
    assert any(kind == "batch_error" for kind, _ in events)
    assert not any(kind == "finished" for kind, _ in events)
    assert job._plan_receipt["cleanup_safe"] is False
    with pytest.raises(RuntimeError, match="清理未完成"):
        job.start(model.pending_items, BatchOptions("native", tmp_path))



def test_partial_constructor_cleanup_failure_never_finishes_or_restarts(tmp_path, monkeypatch):
    from m9.native_prefetch import NativePrefetchCleanupError
    acquired, release = threading.Event(), threading.Event()
    ownership = []

    def construct():
        ownership.extend(("slot_created", "job_created"))
        acquired.set()
        assert release.wait(3)
        raise NativePrefetchCleanupError("partial constructor ownership unresolved")

    job, model = planned_job(tmp_path, monkeypatch, Manager(), lambda **kwargs: pytest.fail("No photo may start"))
    monkeypatch.setattr(native_queue, "_make_lookahead", construct)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    try:
        assert acquired.wait(3) and ownership == ["slot_created", "job_created"]
        assert not any(kind == "finished" for kind, _ in list(job.events.queue))
    finally:
        release.set()
        job.thread.join(3)
    assert not job.running and job._cleanup_unsafe
    events = list(job.events.queue)
    assert not any(kind in ("item_started", "finished") for kind, _ in events)
    error = next(payload for kind, payload in events if kind == "batch_error")
    assert error["phase"] == "setup" and "partial constructor" in error["message"]
    assert job._plan_receipt["cleanup_safe"] is False
    with pytest.raises(RuntimeError, match="清理未完成"):
        job.start(model.pending_items, BatchOptions("native", tmp_path))


def test_disabled_constructor_with_proven_safe_close_preserves_formal_fallback(tmp_path, monkeypatch):
    class DisabledManager(Manager):
        def __init__(self):
            super().__init__()
            self.disabled = True

        def take_nowait(self, request, path):
            return None

    manager, formal = DisabledManager(), []

    def processor(**kwargs):
        assert kwargs["_native_decode"](kwargs["input_path"]) is None
        formal.append(kwargs["input_path"])
        kwargs["progress"]("style_network", "GPU gate")
        return {"output": "formal.jpg"}

    job, model = planned_job(tmp_path, monkeypatch, manager, processor, count=2)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    events = finish(job, model)
    assert formal == [item.path for item in model.items]
    assert events[-1][1]["succeeded"] == 2 and not job._cleanup_unsafe
    assert job._plan_receipt["cleanup_safe"] is True
    assert not any(kind in ("item_error", "batch_error") for kind, _ in events)


def test_plan_snapshot_and_worker_identity_survive_model_and_option_mutation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    seen = []
    model = PhotoQueue()
    model.add_paths([tmp_path / "first.NEF", tmp_path / "second.NEF"])
    original = tuple((item.id, item.path) for item in model.items)

    @contextmanager
    def factory(options, *, _plan):
        assert _plan.items == original
        seen.append(("factory", threading.get_ident()))
        try:
            def processor(**kwargs):
                seen.append((_plan.current_id, _plan.ordinal, kwargs["input_path"], options.method))
                if _plan.ordinal == 1:
                    entered.set()
                    assert release.wait(3)
                assert _plan.items == original
                return {"output": "snapshot.jpg"}
            yield processor
        finally:
            seen.append(("exit", threading.get_ident()))

    job = BatchJob(None, factory, _plan_context=factory)
    job.start(model.pending_items, BatchOptions("native", tmp_path))
    try:
        assert entered.wait(3)
        model.items[1].id, model.items[1].path = "changed", tmp_path / "changed.NEF"
        model.add_paths([tmp_path / "later.NEF"])
    finally:
        release.set()
    events = finish(job, model)
    assert seen[0] == ("factory", job.thread.ident) and seen[-1] == ("exit", job.thread.ident)
    assert seen[1:-1] == [(item_id, ordinal, path, "native") for ordinal, (item_id, path) in enumerate(original, 1)]
    assert events[-1][1]["total"] == 2


def test_batch_and_gui_imports_do_not_load_scientific_or_helper_modules():
    source = Path(__file__).resolve().parents[1]
    script = (f"import sys;sys.path.insert(0,{str(source)!r});import m9.batch,m9.gui;"
              "assert not any(n in sys.modules for n in ('numpy','torch','rawpy','m9.native_prefetch'))")
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=10)
