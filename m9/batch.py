"""Path-based photo queue and serial worker; UI state changes only on the UI thread."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from uuid import uuid4


@dataclass
class QueueItem:
    id: str
    path: Path
    status: str = "pending"
    report: dict | None = None
    error: str = ""
    details: str = ""


class PhotoQueue:
    def __init__(self):
        self.items: list[QueueItem] = []

    def add_paths(self, paths):
        normalized = [Path(path).expanduser().resolve() for path in paths]
        if any(path.suffix.lower() != ".nef" for path in normalized):
            raise ValueError("只能加入 .NEF 照片。")
        known = {os.path.normcase(str(item.path)) for item in self.items}
        added = []
        for path in normalized:
            key = os.path.normcase(str(path))
            if key not in known:
                item = QueueItem(uuid4().hex, path)
                self.items.append(item)
                added.append(item)
                known.add(key)
        return added

    def add_folder(self, path):
        folder = Path(path).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"照片目录不存在或不是文件夹：{folder}")
        files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".nef"),
                       key=lambda p: (p.name.casefold(), p.name))
        return self.add_paths(files)

    @property
    def pending_items(self):
        return tuple(item for item in self.items if item.status == "pending")

    def remove_pending(self, ids):
        requested = set(ids)
        removed = [item.id for item in self.items if item.id in requested and item.status == "pending"]
        removed_set = set(removed)
        self.items[:] = [item for item in self.items if item.id not in removed_set]
        return removed

    def clear(self):
        self.items.clear()

    def requeue_failed(self):
        retry = [item for item in self.items if item.status == "failed"]
        for item in retry:
            item.status, item.report, item.error, item.details = "pending", None, "", ""
        return retry

    def apply_event(self, kind, payload):
        if kind not in ("item_started", "item_done", "item_error"):
            return None
        item = next((item for item in self.items if item.id == payload.get("id")), None)
        if item is None:
            return None
        if kind == "item_started":
            item.status, item.report, item.error, item.details = "running", None, "", ""
        elif kind == "item_done":
            item.status, item.report, item.error, item.details = "succeeded", payload["report"], "", ""
        else:
            item.status, item.report = "failed", None
            item.error, item.details = payload["message"], payload["traceback"]
        return item


@dataclass(frozen=True)
class BatchOptions:
    method: str
    output_dir: Path
    resolution: str = "original"
    device: str = "auto"
    write_json: bool = False


@dataclass
class _BatchPlan:
    items: tuple[tuple[str, Path], ...]
    offer_next: Callable
    is_stopping: Callable[[], bool]
    ordinal: int = 0
    current_id: str | None = None
    receipt: dict = field(default_factory=dict)
    cleanup_safe: bool = True


class BatchJob:
    def __init__(self, processor, processor_context=None, *, _plan_context=None):
        self.processor = processor
        self.processor_context = processor_context
        self._plan_context = _plan_context
        self._plan_receipt = {}
        self._cleanup_unsafe = False
        self.events = queue.Queue()
        self.thread = None
        self._boundary = threading.Lock()
        self._stop_requested = False

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, items, options: BatchOptions):
        with self._boundary:
            if self._cleanup_unsafe:
                raise RuntimeError("原生预解码资源清理未完成；此队列已锁定，不能开始新一轮。")
            if self.running:
                raise RuntimeError("已有队列正在处理，请等待完成。")
            snapshot = tuple((str(item.id), Path(item.path).expanduser().resolve())
                             for item in items if item.status == "pending")
            if not snapshot:
                raise ValueError("没有等待处理的照片。")
            settings = BatchOptions(options.method, Path(options.output_dir).expanduser().resolve(),
                                    options.resolution, options.device, options.write_json)
            self._stop_requested = False
            self._plan_receipt = {}
            self.thread = threading.Thread(target=self._run, args=(snapshot, settings), name="m9-nef-batch-worker")
            try:
                self.thread.start()
            except Exception:
                self.thread = None
                raise

    def request_stop(self):
        with self._boundary:
            self._stop_requested = True

    def _run(self, snapshot, options):
        total = len(snapshot)
        completed = succeeded = failed = 0
        active_item_id = None
        phase = "setup"
        plan = None
        batch_error_emitted = False
        try:
            factory = self.processor_context
            if factory is not None and factory is self._plan_context:
                def is_stopping():
                    with self._boundary:
                        return self._stop_requested

                def offer_next(offer):
                    # Reserve before stop, but never hold the UI boundary during IPC/cleanup.
                    with self._boundary:
                        if self._stop_requested or not plan.ordinal or plan.ordinal >= total:
                            return False
                        item_id, path = plan.items[plan.ordinal]
                        admission = {"id": item_id, "ordinal": plan.ordinal + 1,
                                     "reserved_perf_ns": time.perf_counter_ns(), "cancelled_before_offer": False}
                        plan.receipt.setdefault("admissions", []).append(admission)
                    if is_stopping():
                        admission["cancelled_before_offer"] = True
                        return False
                    # A stop after reservation may leave this admitted operation in flight.
                    return offer(item_id, path)

                plan = _BatchPlan(snapshot, offer_next, is_stopping)
                self._plan_receipt = plan.receipt
                context = factory(options, _plan=plan)
            else:
                context = factory(options) if factory is not None else nullcontext(self.processor)
            with context as processor:
                if not callable(processor):
                    raise TypeError("队列处理上下文必须返回可调用的处理器。")
                phase = "processing"
                for ordinal, (item_id, path) in enumerate(snapshot, 1):
                    # The started event is the boundary: stop and the next start are mutually exclusive.
                    with self._boundary:
                        if self._stop_requested:
                            break
                        if plan is not None:
                            plan.ordinal, plan.current_id = ordinal, item_id
                        self.events.put(("item_started", {"id": item_id, "input_path": path,
                                                          "ordinal": ordinal, "total": total}))
                        active_item_id = item_id
                    try:
                        report = processor(
                            input_path=path, output_dir=options.output_dir, method=options.method,
                            resolution=options.resolution, device=options.device, write_json=options.write_json,
                            progress=lambda stage, message, current_id=item_id: self.events.put(
                                ("progress", {"id": current_id, "stage": stage, "message": message})),
                        )
                    except Exception as exc:
                        failed += 1
                        self.events.put(("item_error", {"id": item_id, "message": str(exc),
                                                        "traceback": traceback.format_exc()}))
                    else:
                        succeeded += 1
                        self.events.put(("item_done", {"id": item_id, "report": report}))
                    completed += 1
                    active_item_id = None
                    if plan is not None:
                        plan.current_id = None
                phase = "cleanup"
        except BaseException as exc:
            details = traceback.format_exc()
            if active_item_id is not None:
                failed += 1
                completed += 1
                self.events.put(("item_error", {"id": active_item_id, "message": str(exc), "traceback": details}))
            self.events.put(("batch_error", {"phase": phase, "message": str(exc),
                                             "traceback": details}))
            batch_error_emitted = True
        finally:
            with self._boundary:
                if plan is not None and not plan.cleanup_safe:
                    self._cleanup_unsafe = True
                    plan.receipt["cleanup_safe"] = False
                    if not batch_error_emitted:
                        self.events.put(("batch_error", {"phase": "cleanup", "message": "原生预解码资源清理未完成；队列已锁定。",
                                                         "traceback": "Native lookahead ownership cleanup was not proven safe."}))
                else:
                    self.events.put(("finished", {"total": total, "completed": completed, "succeeded": succeeded,
                                                  "failed": failed, "stopped": self._stop_requested and completed < total}))
