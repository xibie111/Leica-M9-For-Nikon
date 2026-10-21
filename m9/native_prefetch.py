"""One optional native uint16 lookahead slot owned by one serial queue run."""
from __future__ import annotations

import copy
import hmac
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time

from .native_job import WindowsJobProcess

SLOT_BYTES = 512 * 1024**2
ENTER_RAM = 4 * 1024**3
OFFER_RAM = 3 * 1024**3
CLOSE_SECONDS = 8.0
FIRST_RAW_WAIT_SECONDS = 5.0


class NativePrefetchCleanupError(BaseException):
    """Unsafe ownership cleanup stops the whole queue, not just one photo."""


class NativeLookahead:
    def __init__(self, *, source_root=None, _worker_script=None):
        self.source_root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
        self._worker_script = None if _worker_script is None else Path(_worker_script).resolve()
        self.owner = secrets.token_hex(16)
        self._token = secrets.token_hex(32)
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._close_lock = threading.Lock()
        self._consume_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._bye = threading.Event()
        self._listener = self._conn = self._job = self._slot = self._reader_thread = None
        self._request = self._ready = self._failure = None
        self._first_request_id = None
        self._helper_pid = None
        self._attempted = set()
        self._generation = 0
        self._state = "booting"
        self._closing = self._closed = False
        self._cleanup_error = None
        self._shutdown_started = self._shutdown_deadline = None
        self._audit = {"owner": self.owner, "events": [], "disabled_reason": "", "hits": 0,
                       "misses": 0, "offers": 0, "slot_capacity": 0, "slot_closed": True,
                       "reader_joined": True, "closed": False, "cleanup_errors": [],
                       "cleanup_safe": True, "disable_pending_reason": ""}
        try:
            import psutil
            available = int(psutil.virtual_memory().available)
            self._audit["enter_available_ram_bytes"] = available
            if available < ENTER_RAM:
                self._state = "disabled"
                self._audit["disabled_reason"] = "insufficient_enter_ram"
                return
            from multiprocessing.shared_memory import SharedMemory
            self._slot = SharedMemory(create=True, size=SLOT_BYTES)
            self._audit.update(slot_capacity=SLOT_BYTES, slot_name=self._slot.name, slot_closed=False, cleanup_safe=False)
            self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener.bind(("127.0.0.1", 0))
            self._listener.listen(1)
            self._listener.settimeout(.1)
            environment = dict(os.environ)
            environment.update(PYTHONPATH=str(self.source_root), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            entry = ["-m", "m9.native_decode_worker"] if self._worker_script is None else [str(self._worker_script)]
            command = [sys.executable, "-B", "-u", *entry, "--host", "127.0.0.1",
                       "--port", str(self._listener.getsockname()[1]), "--token", self._token, "--owner", self.owner]
            self._job = WindowsJobProcess(command, cwd=self.source_root, env=environment, owner=self.owner)
            self._reader_thread = threading.Thread(target=self._reader, name="m9-native-reader-" + self.owner[:8], daemon=False)
            self._audit["reader_joined"] = False
            self._reader_thread.start()
            self._event("boot_launched", launcher_pid=self._job.pid)
        except Exception as error:
            self._mark_failure("boot_error", repr(error))
            self._disable("boot_error")

    def _event(self, kind, **values):
        with self._lock:
            self._audit["events"].append({"kind": kind, "perf_ns": time.perf_counter_ns(),
                                          "wall_ns": time.time_ns(), **values})
            self._changed.notify_all()

    def _mark_failure(self, reason, message):
        with self._lock:
            self._failure = {"reason": reason, "message": str(message)}
            if not self._closing:
                self._state = "error"
                self._event("helper_error", **self._failure)

    def _send(self, message):
        from .native_payload import send_message
        if not self._send_lock.acquire(timeout=.1):
            raise TimeoutError("Native helper control send is busy")
        try:
            with self._lock:
                connection = self._conn
            if connection is None:
                raise RuntimeError("Native helper is not connected")
            send_message(connection, message)
        finally:
            self._send_lock.release()

    def _reader(self):
        from .native_payload import recv_message
        try:
            while not self._stop.is_set():
                try:
                    connection, address = self._listener.accept()
                    break
                except socket.timeout:
                    if self._job.poll() is not None:
                        raise RuntimeError("Native helper exited before HELLO")
            else:
                return
            connection.settimeout(.1)
            with self._lock:
                self._conn = connection
            hello = recv_message(connection, stop_event=self._stop)
            if hello.get("kind") != "HELLO" or hello.get("owner") != self.owner or not hmac.compare_digest(str(hello.get("token", "")), self._token):
                raise ValueError("Invalid native helper HELLO")
            expected_module = self._worker_script or self.source_root / "m9/native_decode_worker.py"
            if Path(hello["worker_module"]).resolve() != expected_module or hello.get("torch_absent") is not True:
                raise ValueError("Native helper module or CPU-only identity differs")
            self._job.observe(expected_pid=int(hello["pid"]))
            self._helper_pid = int(hello["pid"])
            self._send({"kind": "BOOT_ACK", "owner": self.owner, "generation": 0})
            with self._lock:
                if not self._closing:
                    self._event("hello", helper_pid=int(hello["pid"]), worker_module=str(expected_module))
                    if self._state == "pending_first":
                        request = self._request
                        self._state = "busy"
                        self._send(request)
                        self._event("first_sent", request_id=request["request_id"], generation=request["generation"])
                    elif self._state == "booting":
                        self._state = "idle"
            while not self._stop.is_set():
                packet = recv_message(connection, stop_event=self._stop)
                kind = packet.get("kind")
                if packet.get("owner") != self.owner:
                    raise ValueError("Native helper owner differs")
                if kind == "BYE":
                    if packet.get("pid") != self._helper_pid or packet.get("torch_absent") is not True:
                        raise ValueError("Native helper BYE identity/CPU-only state differs")
                    self._event("bye", helper_pid=self._helper_pid, torch_absent=True)
                    self._bye.set()
                    return
                with self._lock:
                    request = self._request
                    if kind in ("ACK", "RELEASED"):
                        continue
                    if request is None or any(packet.get(field) != request[field] for field in ("request_id", "generation")):
                        raise ValueError("Native helper request/generation differs")
                    if kind == "ERROR":
                        self._mark_failure("helper_error", packet.get("traceback") or packet.get("message", "Unknown helper error"))
                    elif kind == "READY" and self._state == "busy":
                        if packet.get("torch_absent") is not True:
                            raise ValueError("Native helper imported a GPU/model dependency")
                        self._ready = packet
                        self._state = "ready"
                        self._event("ready", request_id=request["request_id"], generation=request["generation"],
                                    decode_begin_perf_ns=packet.get("postprocess_begin_perf_ns"),
                                    decode_end_perf_ns=packet.get("postprocess_end_perf_ns"))
                    else:
                        raise ValueError("Unexpected native helper control packet")
        except (Exception,) as error:
            if not self._stop.is_set() and not self._closing:
                self._mark_failure("reader_error", repr(error))
        finally:
            self._event("reader_exit")

    def offer(self, request_id, path):
        """Only the current photo's GPU gate may offer the next immutable item."""
        return self._offer(request_id, path, first=False)

    def offer_first(self, request_id, path):
        """Reserve one started first item without waiting for authenticated BOOT."""
        with self._lock:
            if self._first_request_id is not None:
                return False
            self._first_request_id = str(request_id)
        return self._offer(request_id, path, first=True)

    def _offer(self, request_id, path, *, first):
        request_id = str(request_id)
        with self._lock:
            if self._closing or self._closed or self._audit["disabled_reason"] or self._audit["disable_pending_reason"]:
                return False
            self._attempted.add(request_id)
        try:
            from .native_payload import path_identity
            import psutil
            path = Path(path).expanduser().resolve()
            with self._lock:
                if self._closing or self._closed or self._audit["disabled_reason"]:
                    return False
                pending_boot = first and self._state == "booting"
                if (self._state != "idle" and not pending_boot) or self._request is not None:
                    self._event("offer_miss", request_id=request_id, reason="helper_not_idle")
                    raise RuntimeError("Native helper is not idle at GPU gate")
                available = int(psutil.virtual_memory().available)
                if available < OFFER_RAM:
                    self._failure = {"reason": "insufficient_offer_ram", "message": str(available)}
                    self._state = "error"
                    raise RuntimeError("Insufficient available RAM for native lookahead offer")
                identity = path_identity(path)
                self._generation += 1
                request = {"kind": "OFFER", "owner": self.owner, "request_id": request_id,
                           "generation": self._generation, "path": str(path), "identity": identity,
                           "sha256": None, "slot_name": self._slot.name, "slot_capacity": SLOT_BYTES}
                self._request = request
                self._ready = None
                self._state = "pending_first" if pending_boot else "busy"
                self._audit["offers"] += 1
            if not pending_boot:
                self._send(request)
            self._event("offered", request_id=request_id, generation=request["generation"], path=str(path),
                        available_ram_bytes=available, first=first, pending_boot=pending_boot)
            return True
        except Exception as error:
            self._mark_failure("offer_error", repr(error))
            with self._lock:
                self._audit["disable_pending_reason"] = "offer_error"
            return False

    def _disable(self, reason):
        with self._lock:
            if not self._audit["disabled_reason"]:
                self._audit["disabled_reason"] = reason
        self.close()

    def discard(self, request_id):
        """Abandon this formal item's unconsumed work, outside the stop boundary."""
        request_id = str(request_id)
        with self._lock:
            request = self._request
            matches = ((request is not None and request["request_id"] == request_id)
                       or (request is None and request_id in self._attempted
                           and bool(self._audit["disable_pending_reason"])))
            if not matches or self._closed:
                return None
            self._event("discard", request_id=request_id)
        self._disable("discarded_before_consume")
        return self.diagnostics

    def take_nowait(self, request_id, path):
        """At the formal raw boundary use READY only; misses never wait for RAW."""
        request_id = str(request_id)
        with self._lock:
            if self._closing or self._closed or self._audit["disabled_reason"]:
                return None
            # The first photo has no lookahead request. Idle boot must survive it.
            if request_id not in self._attempted and self._request is None:
                self._event("not_requested", request_id=request_id)
                return None
            request, packet = self._request, self._ready
            ready = (self._state == "ready" and request is not None and packet is not None
                     and request["request_id"] == request_id)
            if not ready:
                self._audit["misses"] += 1
                reason = self._failure["reason"] if self._failure else "not_ready_at_formal_boundary"
                self._event("miss", request_id=request_id, reason=reason)
        if not ready:
            self._disable(reason)
            return None
        result = None
        reason = None
        with self._consume_lock:
            rgb = linear = None
            try:
                import numpy as np
                from . import frontend
                from .native_payload import path_identity, sha256, slot_array, decode_matrix, decode_raw_info
                path = Path(path).expanduser().resolve()
                if packet.get("owner") != self.owner or any(packet.get(field) != request[field] for field in ("request_id", "generation")):
                    raise ValueError("Native lookahead ownership/generation differs")
                if str(path) != request["path"] or packet.get("path") != str(path):
                    raise ValueError("Native lookahead source path differs")
                if packet["identity_before"] != packet["identity_after"] or packet["identity_before"] != request["identity"]:
                    raise ValueError("Native lookahead source identity changed during decode")
                if packet["sha256_before"] != packet["sha256_after"]:
                    raise ValueError("Native lookahead source bytes changed during decode")
                if path_identity(path) != request["identity"] or sha256(path) != packet["sha256_after"] or path_identity(path) != request["identity"]:
                    raise ValueError("Native lookahead source changed before consumption")
                layout = packet["layout"]
                if not 0 < layout["nbytes"] <= SLOT_BYTES:
                    raise ValueError("Native lookahead slot capacity differs")
                rgb = slot_array(self._slot.buf, layout)
                if rgb.dtype != np.dtype(np.uint16) or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError("Native lookahead pixels are not uint16 RGB")
                matrix, info = decode_matrix(packet["matrix"]), decode_raw_info(packet["raw_info"])
                if matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                    raise ValueError("Native lookahead matrix differs")
                if any(not np.isfinite(getattr(info, field)) or getattr(info, field) <= 0 for field in ("exposure_time", "iso", "fnumber", "cct")):
                    raise ValueError("Native lookahead exposure/CCT differs")
                if any(not np.isfinite(getattr(info, field)) for field in ("ev100", "baseline_exposure")):
                    raise ValueError("Native lookahead exposure metadata is not finite")
                if not isinstance(info.make, str) or not isinstance(info.model, str) or "NIKON" not in info.make.upper():
                    raise ValueError("Native lookahead camera metadata differs")
                neutral = info.as_shot_neutral
                if not isinstance(neutral, np.ndarray) or neutral.dtype != np.dtype(np.float64) or neutral.shape != (3,) or not np.isfinite(neutral).all() or np.any(neutral <= 0):
                    raise ValueError("Native lookahead neutral metadata differs")
                if not isinstance(info.extra, dict) or info.extra.get("method") != "native" or info.extra.get("decoded_dimensions") != [rgb.shape[1], rgb.shape[0]]:
                    raise ValueError("Native lookahead geometry metadata differs")
                with self._lock:
                    self._state = "consuming"
                linear = frontend._normalize_raw_rgb(rgb)
                if not linear.flags.owndata or np.shares_memory(linear, rgb) or linear.strides != tuple(2 * value for value in rgb.strides):
                    raise ValueError("Native lookahead normalization layout/ownership differs")
                result = linear, matrix, info
                rgb = None
                self._send({"kind": "RELEASE", "owner": self.owner, "request_id": request_id, "generation": request["generation"]})
                with self._lock:
                    self._request = self._ready = None
                    self._state = "idle"
                    self._audit["hits"] += 1
                    self._event("consumed", request_id=request_id, generation=request["generation"])
            except Exception as error:
                result = None
                linear = None
                reason = "invalid_or_changed_ready"
                self._mark_failure(reason, repr(error))
                with self._lock:
                    self._audit["misses"] += 1
                    self._event("miss", request_id=request_id, reason=reason, message=repr(error))
            finally:
                rgb = None
        if reason is not None:
            self._disable(reason)
        return result

    def take_first(self, request_id, path):
        """One bounded READY wait at first formal RAW; cleanup precedes fallback."""
        request_id = str(request_id)
        with self._lock:
            if request_id != self._first_request_id:
                raise ValueError("First native request differs")
            needs_close = self._closing or self._closed or bool(self._audit["disabled_reason"])
        if needs_close:
            self.close()
            return None
        with self._changed:
            deadline = time.perf_counter() + FIRST_RAW_WAIT_SECONDS
            self._event("first_wait_begin", request_id=request_id, deadline_perf_ns=int(deadline * 1e9),
                        limit_seconds=FIRST_RAW_WAIT_SECONDS)
            terminal = self._changed.wait_for(
                lambda: self._closing or self._closed or self._state not in ("booting", "pending_first", "busy"),
                timeout=max(0, deadline - time.perf_counter()))
            # wait_for may see a late READY when reacquiring the lock after its
            # timeout. The decision itself must still belong to this window.
            terminal = terminal and time.perf_counter() <= deadline
            closing = self._closing or self._closed
            self._event("first_wait_end", request_id=request_id, terminal=terminal, state=self._state)
            if not terminal:
                # Claim the timeout under the same lock as READY. A late packet
                # cannot turn this timed-out attempt into a successful consume.
                self._state = "error"
                self._failure = {"reason": "first_raw_timeout", "message": "First RAW READY deadline reached"}
                self._audit["misses"] += 1
                self._event("miss", request_id=request_id, reason="first_raw_timeout")
        if not terminal:
            self._disable("first_raw_timeout")
            return None
        if closing:
            self.close()  # A concurrent close must prove cleanup before fallback.
            return None
        result = self.take_nowait(request_id, path)
        if result is None:
            self.close()  # Also cover close starting just before take_nowait.
        return result

    @property
    def diagnostics(self):
        with self._lock:
            result = copy.deepcopy(self._audit)
            result.update(state=self._state, generation=self._generation,
                          reader_alive=bool(self._reader_thread and self._reader_thread.is_alive()))
            if self._job is not None:
                result["job"] = self._job.audit()
            return result

    def close(self):
        started = time.perf_counter()
        with self._lock:
            if self._shutdown_deadline is None:
                self._shutdown_started = started
                self._shutdown_deadline = started + CLOSE_SECONDS
            deadline = self._shutdown_deadline
        if not self._close_lock.acquire(timeout=max(0, deadline - time.perf_counter())):
            raise NativePrefetchCleanupError("Native lookahead concurrent close exceeded8s")
        try:
            if self._closed and self._audit["cleanup_safe"]:
                if self._cleanup_error:
                    raise NativePrefetchCleanupError(self._cleanup_error)
                return self.diagnostics
            if time.perf_counter() >= deadline:
                self._audit.update(close_seconds=time.perf_counter() - self._shutdown_started,
                                   close_within_budget=False)
                self._cleanup_error = "Native lookahead original8s shutdown deadline exhausted; cleanup remains unsafe"
                raise NativePrefetchCleanupError(self._cleanup_error)
            with self._lock:
                self._closing = True
                self._generation += 1
                self._event("close_begin")
            errors = []
            try:
                if self._conn is not None:
                    self._send({"kind": "STOP", "owner": self.owner})
                    grace_end = min(deadline, time.perf_counter() + .5)
                    while time.perf_counter() < grace_end and not self._bye.is_set() and self._job.poll() is None:
                        self._bye.wait(min(.02, max(0, grace_end - time.perf_counter())))
            except Exception as error:
                self._event("normal_quit_unavailable", message=repr(error))
            self._stop.set()
            for connection in (self._conn, self._listener):
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        connection.close()
                    except Exception as error:
                        errors.append({"phase": "socket_close", "error": repr(error)})
            if self._job is not None:
                try:
                    self._audit["job"] = self._job.close(max(time.perf_counter(), deadline - .25))
                except Exception as error:
                    errors.append({"phase": "job_close", "error": repr(error)})
                    self._audit["job"] = self._job.audit()
            reader = self._reader_thread
            if reader is not None and reader.ident is not None:
                reader.join(timeout=max(0, deadline - time.perf_counter()))
            self._audit["reader_joined"] = reader is None or not reader.is_alive()
            if not self._audit["reader_joined"]:
                errors.append({"phase": "reader_join", "error": "reader remains alive"})
            acquired = self._consume_lock.acquire(timeout=max(0, deadline - time.perf_counter()))
            if acquired:
                try:
                    if self._slot is not None:
                        try:
                            self._slot.close()
                            self._slot.unlink()
                            self._slot = None
                            self._audit["slot_closed"] = True
                        except Exception as error:
                            errors.append({"phase": "slot_close", "error": repr(error)})
                finally:
                    self._consume_lock.release()
            else:
                errors.append({"phase": "consumer_join", "error": "consumer still holds shared view"})
            with self._lock:
                self._request = self._ready = None
                self._closed = True
                job_safe = self._job is None or self._job.audit().get("cleanup_safe") is True
                safe = job_safe and self._audit["reader_joined"] and self._slot is None
                self._state = "closed" if safe else "cleanup_failed"
                self._audit.update(closed=True, close_seconds=time.perf_counter() - self._shutdown_started,
                                   close_attempt_seconds=time.perf_counter() - started,
                                   close_within_budget=time.perf_counter() <= deadline, cleanup_safe=safe)
                self._event("close_end")
            if not self._audit["close_within_budget"]:
                errors.append({"phase": "close_deadline", "error": "total close exceeded8s"})
            self._audit["cleanup_errors"].extend(errors)
            if errors:
                self._cleanup_error = f"Native lookahead critical cleanup failed: {errors}"
                raise NativePrefetchCleanupError(self._cleanup_error)
            if not safe:
                self._cleanup_error = "Native lookahead resources are not proved closed"
                raise NativePrefetchCleanupError(self._cleanup_error)
            self._cleanup_error = None
            return self.diagnostics
        finally:
            self._close_lock.release()
