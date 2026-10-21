"""First-item scheduling and real framed IPC with a tiny synthetic CPU helper.

No NEF decoder, network, CUDA, real Windows Job, or timing speed assertion.
The fake Job substitutes only process ownership; NativeLookahead keeps its real
reader, authentication, request validation, slot consumer and cleanup paths.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from m9 import native_prefetch as prefetch, native_queue
from m9.batch import BatchJob, BatchOptions, PhotoQueue
from m9.frontend import RawInfo
from m9.native_payload import (array_layout, encode_matrix, encode_raw_info,
                               path_identity, recv_message, send_message, sha256, slot_array)


def _boot_packet(job, packet, *, parent_closing):
    job.first_packet = dict(kind=packet.get("kind"), owner_match=packet.get("owner") == job.owner,
                            parent_closing=parent_closing)
    if packet.get("kind") == "STOP" and packet.get("owner") == job.owner and parent_closing:
        # The unchanged real worker rejects this pre-ACK cancellation. Model a
        # protocol abort in this thread, never a normal BYE or process exit0.
        job.protocol_abort = "owner_close_before_BOOT_ACK"
        return False
    assert packet == dict(kind="BOOT_ACK", owner=job.owner, generation=0), job.first_packet
    return True


@pytest.fixture
def wire(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.NEF"
    source.write_bytes(b"synthetic bytes, never decoded as NEF")
    control = SimpleNamespace(hello=threading.Event(), ready=threading.Event(),
        offered=threading.Event(), sent=[], jobs=[], mode="good", unsafe=False)
    control.hello.set()
    control.ready.set()

    class Job:
        def __init__(self, command, *, cwd, env, owner):
            self.owner, self.pid = owner, os.getpid()
            self.command, self.cwd, self.env = command, cwd, env
            self.stop = threading.Event()
            self.closed = False
            self.errors = []
            self.error_details = []
            self.first_packet = None
            self.protocol_abort = None
            self.normal_bye_sent = False
            self.observed = []
            control.jobs.append(self)
            self.thread = threading.Thread(target=self.run, name="test-synthetic-native-helper")
            self.thread.start()

        def run(self):
            from multiprocessing.shared_memory import SharedMemory
            option = lambda key: self.command[self.command.index(key) + 1]
            try:
                with socket.create_connection((option("--host"), int(option("--port"))), timeout=2) as conn:
                    conn.settimeout(.05)
                    while not control.hello.wait(.01):
                        if self.stop.is_set():
                            return
                    send_message(conn, dict(kind="HELLO", owner=self.owner,
                        token=option("--token") if control.mode != "bad_hello" else "wrong",
                        pid=self.pid, worker_module=str(Path(self.cwd)/"m9/native_decode_worker.py"), torch_absent=True))
                    if control.mode == "bad_hello":
                        return
                    packet = recv_message(conn, self.stop)
                    parent = next((m for m in managers if m.owner == self.owner), None)
                    if not _boot_packet(self, packet, parent_closing=parent is not None and parent._closing):
                        return
                    while not self.stop.is_set():
                        message = recv_message(conn, self.stop)
                        control.sent.append(message)
                        if message["kind"] == "STOP":
                            send_message(conn, dict(kind="BYE", owner=self.owner, pid=self.pid, torch_absent=True))
                            self.normal_bye_sent = True
                            return
                        if message["kind"] == "RELEASE":
                            continue
                        assert message["kind"] == "OFFER"
                        control.offered.set()
                        while not control.ready.wait(.01):
                            if self.stop.is_set():
                                return
                        if control.mode == "error":
                            send_message(conn, dict(kind="ERROR", owner=self.owner, request_id=message["request_id"],
                                generation=message["generation"], message="synthetic bad RAW"))
                            continue
                        # F-order makes the K-layout contract observable, not just values.
                        rgb = np.array(np.arange(18, dtype=np.uint16).reshape(3, 2, 3), order="F")
                        layout = array_layout(rgb)
                        slot = SharedMemory(name=message["slot_name"])
                        view = slot_array(slot.buf, layout)
                        view[:] = rgb
                        del view
                        slot.close()
                        identity, digest = path_identity(source), sha256(source)
                        info = RawInfo(exposure_time=.01, iso=100., fnumber=2.8, cct=5500.,
                            make="NIKON", model="synthetic", as_shot_neutral=np.array([.5,1.,.7], dtype=np.float64),
                            extra={"method":"native", "decoded_dimensions":[2,3]})
                        packet = dict(kind="READY", owner=self.owner, request_id=message["request_id"],
                            generation=message["generation"], path=message["path"], identity_before=identity,
                            identity_after=identity, sha256_before=digest, sha256_after=digest, layout=layout,
                            matrix=encode_matrix(np.eye(3,dtype=np.float64)), raw_info=encode_raw_info(info), torch_absent=True)
                        if control.mode == "generation": packet["generation"] += 1
                        if control.mode == "hash": packet["sha256_before"] = "0"*64
                        if control.mode == "layout": packet["layout"]["nbytes"] += 2
                        send_message(conn, packet)
            except (EOFError, InterruptedError, ConnectionError, OSError):
                pass  # Parent close intentionally interrupts this synthetic transport.
            except BaseException as error:
                self.errors.append(repr(error))
                self.error_details.append(dict(error=repr(error), traceback=traceback.format_exc(limit=8)))

        def poll(self):
            # NativeLookahead uses only None/non-None here; this is not a real
            # Windows process and has no observed operating-system exit code.
            return None if self.thread.is_alive() else "synthetic_thread_stopped"

        def observe(self, *, expected_pid):
            assert expected_pid == self.pid
            self.observed.append(expected_pid)

        def audit(self):
            return dict(cleanup_safe=self.closed and not control.unsafe, first_packet=self.first_packet,
                        protocol_abort=self.protocol_abort, normal_bye_sent=self.normal_bye_sent,
                        errors=list(self.errors), error_details=list(self.error_details), real_process_exit_code=None)

        def close(self, deadline):
            self.stop.set()
            self.thread.join(max(0, deadline-time.perf_counter()))
            assert not self.thread.is_alive()
            self.closed = True
            return self.audit()

    monkeypatch.setattr(prefetch, "WindowsJobProcess", Job)
    monkeypatch.setattr(prefetch, "SLOT_BYTES", 4096)
    monkeypatch.setattr(prefetch, "FIRST_RAW_WAIT_SECONDS", .5)
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8*1024**3))
    managers = []
    def make():
        manager = prefetch.NativeLookahead()
        managers.append(manager)
        return manager
    control.source, control.make = source, make
    control.managers = managers
    yield control
    control.hello.set(); control.ready.set()
    control.unsafe = False
    for manager in managers:
        if not manager.diagnostics["closed"]:
            manager.close()
    assert all(not job.thread.is_alive() and not job.errors for job in control.jobs), [
        dict(thread_alive=job.thread.is_alive(), **job.audit()) for job in control.jobs]


def test_pending_first_waits_for_verified_hello_without_blocking_models(wire):
    wire.hello.clear()
    manager = wire.make()
    assert manager.offer_first("first", wire.source)
    assert not wire.offered.is_set() and manager.diagnostics["state"] == "pending_first"
    assert not manager.offer_first("duplicate", wire.source)
    wire.hello.set()
    result = manager.take_first("first", wire.source)
    expected = np.array(np.arange(18, dtype=np.uint16).reshape(3,2,3), order="F").astype(np.float32) / 65535.
    assert np.array_equal(result[0].view(np.uint32), expected.view(np.uint32))
    assert result[0].flags.owndata and result[0].flags.f_contiguous
    assert wire.jobs[0].observed == [os.getpid()]
    offers = [m for m in wire.sent if m["kind"] == "OFFER"]
    assert len(offers) == 1 and offers[0]["request_id"] == "first"
    manager.close()
    assert manager.diagnostics["hits"] == 1 and manager.diagnostics["cleanup_safe"]


@pytest.mark.parametrize("mode", ["error", "bad_hello", "generation", "hash", "layout"])
def test_helper_failure_is_cleaned_before_formal_fallback(wire, mode):
    wire.mode = mode
    manager = wire.make()
    manager.offer_first("first", wire.source)
    assert manager.take_first("first", wire.source) is None
    assert manager.diagnostics["cleanup_safe"] and manager.diagnostics["slot_closed"]
    assert manager.diagnostics["reader_joined"] and wire.jobs[0].closed


@pytest.mark.parametrize("late", ["hello", "ready"])
def test_never_ready_deadline_closes_owned_work_before_fallback(wire, late):
    getattr(wire, late).clear()
    manager = wire.make()
    assert manager.offer_first("first", wire.source)
    if late == "ready": assert wire.offered.wait(2)
    assert manager.take_first("first", wire.source) is None
    diag = manager.diagnostics
    assert diag["disabled_reason"] == "first_raw_timeout" and diag["misses"] == 1
    assert diag["cleanup_safe"] and wire.jobs[0].closed
    getattr(wire, late).set()  # A late wakeup cannot publish or resurrect this result.
    assert manager.take_first("first", wire.source) is None and manager.diagnostics["hits"] == 0


def test_source_changed_after_offer_cannot_be_consumed(wire):
    wire.ready.clear()
    manager = wire.make()
    manager.offer_first("first", wire.source)
    assert wire.offered.wait(2)
    wire.source.write_bytes(b"changed source while helper was pending")
    wire.ready.set()
    assert manager.take_first("first", wire.source) is None
    assert manager.diagnostics["disabled_reason"] == "invalid_or_changed_ready"
    assert manager.diagnostics["cleanup_safe"]


def test_late_ready_after_condition_reacquires_lock_is_still_timeout(wire,monkeypatch):
    manager=wire.make()
    manager.offer_first("first",wire.source)
    wait=manager._changed.wait_for
    late=[False]
    real_clock=time.perf_counter
    def clock():return real_clock()+(1. if late[0] else 0.)
    def delayed_return(predicate,timeout):
        assert wait(predicate,timeout=timeout)
        assert manager._state=="ready"
        late[0]=True  # Controlled host scheduling delay after READY, no 5s sleep.
        return True
    monkeypatch.setattr(manager._changed,"wait_for",delayed_return)
    monkeypatch.setattr(prefetch,"time",SimpleNamespace(perf_counter=clock,
        perf_counter_ns=time.perf_counter_ns,time_ns=time.time_ns))
    assert manager.take_first("first",wire.source) is None
    diag=manager.diagnostics
    assert diag["disabled_reason"]=="first_raw_timeout" and diag["hits"]==0
    assert diag["cleanup_safe"]


def test_models_failure_can_discard_pending_boot_before_next_item(wire):
    wire.hello.clear()
    manager=wire.make()
    assert manager.offer_first("first",wire.source)
    assert manager.diagnostics["state"]=="pending_first"
    manager.discard("first")
    diag=manager.diagnostics
    assert diag["cleanup_safe"] and diag["reader_joined"] and diag["slot_closed"]
    assert not wire.offered.is_set() and wire.jobs[0].closed


def test_concurrent_close_wakes_wait_but_fallback_waits_for_safe_cleanup(wire,monkeypatch):
    wire.hello.clear()
    manager=wire.make();manager.offer_first("first",wire.source)
    waiting=threading.Event();results=[]
    original=manager._changed.wait_for
    def wait(predicate,timeout):
        waiting.set()
        return original(predicate,timeout=timeout)
    monkeypatch.setattr(manager._changed,"wait_for",wait)
    thread=threading.Thread(target=lambda:results.append(manager.take_first("first",wire.source)))
    thread.start()
    try:
        assert waiting.wait(2)
        manager.close()
    finally:
        thread.join(2)
    assert not thread.is_alive() and results==[None]
    assert manager.diagnostics["cleanup_safe"]


def test_close_already_in_progress_does_not_allow_early_fallback(wire,monkeypatch):
    wire.hello.clear()
    manager=wire.make();manager.offer_first("first",wire.source)
    closing,release=threading.Event(),threading.Event()
    original=wire.jobs[0].close
    def held_close(deadline):
        closing.set();assert release.wait(2)
        return original(deadline)
    monkeypatch.setattr(wire.jobs[0],"close",held_close)
    closer=threading.Thread(target=manager.close)
    results=[]
    consumer=threading.Thread(target=lambda:results.append(manager.take_first("first",wire.source)))
    closer.start()
    try:
        assert closing.wait(2)
        consumer.start();consumer.join(.05)
        assert consumer.is_alive() and results==[]
    finally:
        release.set();closer.join(2);consumer.join(2)
    assert not closer.is_alive() and not consumer.is_alive() and results==[None]
    assert manager.diagnostics["cleanup_safe"]


def test_close_between_wait_decision_and_consume_must_finish_before_fallback(wire,monkeypatch):
    manager=wire.make();manager.offer_first("first",wire.source)
    with manager._changed:
        assert manager._changed.wait_for(lambda:manager._state=="ready",timeout=2)
    closing,release,returned_none=threading.Event(),threading.Event(),threading.Event()
    close_job=wire.jobs[0].close
    def held_close(deadline):
        closing.set();assert release.wait(2)
        return close_job(deadline)
    monkeypatch.setattr(wire.jobs[0],"close",held_close)
    closer=threading.Thread(target=manager.close)
    take=manager.take_nowait
    def interleaved_take(request,path):
        closer.start();assert closing.wait(2)
        value=take(request,path)
        assert value is None
        returned_none.set()
        return value
    monkeypatch.setattr(manager,"take_nowait",interleaved_take)
    results=[]
    consumer=threading.Thread(target=lambda:results.append(manager.take_first("first",wire.source)))
    consumer.start()
    try:
        assert returned_none.wait(2)
        consumer.join(.05)
        assert consumer.is_alive() and results==[]
    finally:
        release.set();consumer.join(2);closer.join(2)
    assert results==[None] and not consumer.is_alive() and not closer.is_alive()
    assert manager.diagnostics["cleanup_safe"]


@pytest.mark.parametrize("available", [0, prefetch.ENTER_RAM-1, prefetch.OFFER_RAM-1])
def test_low_ram_never_bypasses_original_formal_route(wire, monkeypatch, available):
    import psutil
    if available < prefetch.OFFER_RAM:
        manager = wire.make()
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    else:
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
        manager = wire.make()
    assert not manager.offer_first("first", wire.source)
    assert manager.take_first("first", wire.source) is None
    manager.close()
    assert manager.diagnostics["cleanup_safe"] and manager.diagnostics["hits"] == 0


@pytest.mark.parametrize("available", [0, prefetch.OFFER_RAM-1])
def test_owned_close_before_boot_ack_is_explicit_protocol_abort(wire, monkeypatch, available):
    ack_waiting, stop_sent = threading.Event(), threading.Event()
    calls = []
    original_send, original_close = prefetch.NativeLookahead._send, prefetch.NativeLookahead.close

    def send(self, message):
        kind = message["kind"]
        calls.append(kind)
        if kind == "BOOT_ACK":
            ack_waiting.set()
            assert stop_sent.wait(2), "STOP must precede the released ACK"
            calls.append("ACK_released")
        try:
            return original_send(self, message)
        finally:
            if kind == "STOP":
                calls.append("STOP_sent")
                stop_sent.set()

    def close(self):
        if not self._closed:
            # The original low-RAM test must already have rejected the offer.
            # No HELLO wait is added before this admission/error boundary.
            assert self._failure is not None and self._audit["disable_pending_reason"]
            assert ack_waiting.wait(2), "Reader must reach the controlled ACK barrier"
        return original_close(self)

    monkeypatch.setattr(prefetch.NativeLookahead, "_send", send)
    monkeypatch.setattr(prefetch.NativeLookahead, "close", close)
    test_low_ram_never_bypasses_original_formal_route(wire, monkeypatch, available)
    job = wire.jobs[0]
    assert job.first_packet == dict(kind="STOP", owner_match=True, parent_closing=True)
    assert job.protocol_abort == "owner_close_before_BOOT_ACK"
    assert not job.errors and not job.normal_bye_sent and job.audit()["real_process_exit_code"] is None
    assert job.poll() == "synthetic_thread_stopped"
    assert calls.index("STOP_sent") < calls.index("ACK_released")
    assert "OFFER" not in calls and not wire.sent
    assert job.closed and not job.thread.is_alive()
    from multiprocessing.shared_memory import SharedMemory
    assert len(wire.managers) == 1
    diag = wire.managers[0].diagnostics
    assert diag["cleanup_safe"] and diag["reader_joined"] and not diag["reader_alive"] and diag["slot_closed"]
    with pytest.raises(FileNotFoundError):
        slot = SharedMemory(name=diag["slot_name"], create=False)
        slot.close()


@pytest.mark.parametrize("packet,parent_closing", [
    ({"kind":"STOP", "owner":"foreign"}, True),
    ({"kind":"STOP", "owner":"owner"}, False),
    ({"kind":"BOOT_ACK", "owner":"foreign", "generation":0}, True),
    ({"kind":"READY", "owner":"owner"}, True),
    ({"kind":"BOOT_ACK", "owner":"owner", "generation":1}, False),
])
def test_other_first_packets_remain_assertion_failures(packet, parent_closing):
    job = SimpleNamespace(owner="owner", first_packet=None, protocol_abort=None)
    with pytest.raises(AssertionError):
        _boot_packet(job, packet, parent_closing=parent_closing)
    assert job.protocol_abort is None
    assert job.first_packet["kind"] == packet["kind"]


def test_ordinary_boot_ack_remains_strict_and_does_not_claim_protocol_abort():
    job = SimpleNamespace(owner="owner", first_packet=None, protocol_abort=None)
    assert _boot_packet(job, dict(kind="BOOT_ACK",owner="owner",generation=0), parent_closing=False)
    assert job.protocol_abort is None and job.first_packet["kind"] == "BOOT_ACK"


def test_cleanup_unknown_raises_baseexception_instead_of_fallback(wire):
    wire.ready.clear(); wire.unsafe = True
    manager = wire.make()
    manager.offer_first("first", wire.source)
    with pytest.raises(prefetch.NativePrefetchCleanupError):
        manager.take_first("first", wire.source)
    assert not manager.diagnostics["cleanup_safe"]


def test_subsequent_item_keeps_nonwaiting_miss_policy(wire):
    manager = wire.make()
    manager.offer_first("first", wire.source)
    assert manager.take_first("first", wire.source) is not None
    wire.ready.clear(); wire.offered.clear()
    manager.offer("second", wire.source)
    assert wire.offered.wait(2)
    assert manager.take_nowait("second", wire.source) is None
    diag = manager.diagnostics
    assert diag["disabled_reason"] == "not_ready_at_formal_boundary"
    assert len([e for e in diag["events"] if e["kind"] == "first_wait_begin"]) == 1


class QueueManager:
    def __init__(self):
        self.calls, self.pending = [], None
    def offer_first(self, request, path):
        self.calls.append(("first", request)); self.pending = request
        return False  # Deferred helper failure must not mask model validation.
    def offer(self, request, path):
        self.calls.append(("next", request)); self.pending = request
        return True
    def take_first(self, request, path):
        self.calls.append(("wait", request)); self.pending = None
        return None
    def take_nowait(self, request, path):
        self.calls.append(("nowait", request)); self.pending = None
        return None
    def discard(self, request):
        self.calls.append(("discard", request)); self.pending = None
    def close(self):
        self.calls.append(("close",)); return {"cleanup_safe":True}
    @property
    def diagnostics(self):
        return {"calls":[list(v) for v in self.calls]}


def queue_job(tmp_path, monkeypatch, core, session, manager):
    model = PhotoQueue(); model.add_paths([tmp_path/"a.NEF",tmp_path/"b.NEF"])
    monkeypatch.setattr(native_queue, "_make_lookahead", lambda: manager)
    @contextmanager
    def factory(options, *, _plan):
        with native_queue._native_processor_context(core, session, _plan) as run:
            yield run
    return BatchJob(None, factory, _plan_context=factory), model


@pytest.mark.parametrize("loaded", [False, True])
def test_first_models_notification_is_one_shot_and_later_items_never_use_wait(tmp_path, monkeypatch, loaded):
    manager, session = QueueManager(), SimpleNamespace(loaded=loaded)
    formal = []
    def core(**kw):
        kw["progress"]("models", "strict")
        kw["progress"]("models", "duplicate notification")
        session.loaded = True
        assert kw["_native_decode"](kw["input_path"]) is None
        formal.append(kw["input_path"])
        kw["progress"]("style_network", "GPU")
        return {"output":"formal.jpg"}
    job, model = queue_job(tmp_path,monkeypatch,core,session,manager)
    job.start(model.pending_items,BatchOptions("native",tmp_path)); job.thread.join(2)
    assert not job.running and len(formal)==2
    assert [c[0] for c in manager.calls].count("first") == int(not loaded)
    assert [c[0] for c in manager.calls].count("wait") == int(not loaded)
    assert [c[0] for c in manager.calls].count("nowait") == 1
    assert list(job.events.queue)[-1][1]["succeeded"] == 2


def test_model_error_precedes_helper_failure_and_discards_first_without_raw(tmp_path,monkeypatch):
    manager, session = QueueManager(), SimpleNamespace(loaded=False)
    def core(**kw):
        kw["progress"]("models","strict")
        raise RuntimeError("models 阶段失败：strict weight mismatch")
    job,model=queue_job(tmp_path,monkeypatch,core,session,manager)
    job.start(model.pending_items,BatchOptions("native",tmp_path));job.thread.join(2)
    assert not job.running
    events=list(job.events.queue)
    errors=[v for k,v in events if k=="item_error"]
    assert len(errors)==2 and all("strict weight mismatch" in e["message"] for e in errors)
    assert not any(c[0] in ("wait","nowait") for c in manager.calls)
    assert manager.pending is None and manager.calls[-1]==("close",)


def test_stop_after_started_finishes_current_first_and_leaves_next_pending(tmp_path,monkeypatch):
    manager,session=QueueManager(),SimpleNamespace(loaded=False)
    entered,release=threading.Event(),threading.Event()
    def core(**kw):
        kw["progress"]("models","strict");entered.set();assert release.wait(2)
        kw["_native_decode"](kw["input_path"])
        kw["progress"]("style_network","GPU")
        return {"output":"first.jpg"}
    job,model=queue_job(tmp_path,monkeypatch,core,session,manager)
    job.start(model.pending_items,BatchOptions("native",tmp_path))
    try:
        assert entered.wait(2);job.request_stop()  # Actual GUI close remains a later functional gate.
    finally:
        release.set();job.thread.join(2)
    events=list(job.events.queue)
    for event in events:model.apply_event(*event)
    assert [i.status for i in model.items]==["succeeded","pending"]
    assert events[-1][0]=="finished" and events[-1][1]["stopped"]
    assert [c[0] for c in manager.calls]==["first","wait","discard","close"]


def test_second_queue_owns_new_first_request_and_manager(tmp_path,monkeypatch):
    managers=[]
    def make():
        manager=QueueManager();managers.append(manager);return manager
    monkeypatch.setattr(native_queue,"_make_lookahead",make)
    @contextmanager
    def factory(options,*,_plan):
        session=SimpleNamespace(loaded=False)
        def core(**kw):
            kw["progress"]("models","strict");session.loaded=True
            kw["_native_decode"](kw["input_path"])
            kw["progress"]("style_network","GPU")
            return {"output":"formal.jpg"}
        with native_queue._native_processor_context(core,session,_plan) as run:yield run
    job=BatchJob(None,factory,_plan_context=factory)
    for _ in range(2):
        model=PhotoQueue();model.add_paths([tmp_path/"a.NEF",tmp_path/"b.NEF"])
        job.start(model.pending_items,BatchOptions("native",tmp_path));job.thread.join(2)
        assert not job.running and not job._cleanup_unsafe
    assert len(managers)==2 and all(sum(c[0]=="first" for c in m.calls)==1 for m in managers)


@pytest.mark.parametrize("invalid", ["missing_source", "method", "resolution"])
def test_real_process_validation_happens_before_first_request(tmp_path,monkeypatch,invalid):
    from m9.nef import process_nef
    manager=QueueManager()
    monkeypatch.setattr(native_queue,"_make_lookahead",lambda:manager)
    source=tmp_path/"input.NEF";source.write_bytes(b"not actually decoded")
    if invalid=="missing_source":source.unlink()
    session=SimpleNamespace(loaded=False,device="cpu",closed=False,invalidate=lambda:None)
    plan=SimpleNamespace(ordinal=1,current_id="first",receipt={},cleanup_safe=True,offer_next=lambda offer:False)
    with native_queue._native_processor_context(process_nef,session,plan) as run:
        with pytest.raises(ValueError):
            run(input_path=source,output_dir=tmp_path,method="bad" if invalid=="method" else "native",
                resolution="bad" if invalid=="resolution" else "original",device="auto")
    assert not any(c[0]=="first" for c in manager.calls)


def test_real_strict_model_failure_never_reaches_formal_raw(tmp_path,monkeypatch):
    from m9 import nef
    manager=QueueManager()
    monkeypatch.setattr(native_queue,"_make_lookahead",lambda:manager)
    source=tmp_path/"input.NEF";source.write_bytes(b"not actually decoded")
    calls=[]
    def get_model():
        calls.append("strict_model")
        raise ValueError("strict weight mismatch")
    session=SimpleNamespace(loaded=False,device="cpu",closed=False,get_model=get_model,
        invalidate=lambda:calls.append("invalidate"))
    monkeypatch.setattr(nef,"load_nef",lambda *a,**kw:pytest.fail("RAW must not run after model failure"))
    plan=SimpleNamespace(ordinal=1,current_id="first",receipt={},cleanup_safe=True,offer_next=lambda offer:False)
    with native_queue._native_processor_context(nef.process_nef,session,plan) as run:
        with pytest.raises(RuntimeError,match="models.*strict weight mismatch"):
            run(input_path=source,output_dir=tmp_path,method="native",resolution="original",device="auto")
    assert calls==["strict_model","invalidate"]
    assert [c[0] for c in manager.calls]==["first","discard","close"]
