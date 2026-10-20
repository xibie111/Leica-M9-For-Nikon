"""Windows ownership for one hidden CPU helper and its launcher descendants."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import copy
import os
import subprocess
import threading
import time


class _BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("minimum_ws", ctypes.c_size_t),
                ("maximum_ws", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", _BasicLimits), ("io", _IoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in
                ("user", "kernel", "period_user", "period_kernel")] + [
                (name, wintypes.DWORD) for name in
                ("page_faults", "total_processes", "active_processes", "terminated_processes")]


class WindowsJobProcess:
    """Assign the suspended .venv launcher before it can create a real child."""

    def __init__(self, command, *, cwd, env, owner):
        if os.name != "nt":
            raise OSError("Native lookahead requires Windows Job ownership")
        import _winapi
        self._winapi = _winapi
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        self._lock = threading.RLock()
        self._job = self._process = None
        self.pid = None
        self._identities = {}
        self._closed = False
        self._record = {"owner": owner, "command": list(command), "cwd": str(cwd),
                        "job_assigned_before_resume": False, "kill_on_job_close": True,
                        "breakaway_allowed": False, "job_handle_inherited": False,
                        "identity_errors": [], "cleanup_errors": [], "closed": False,
                        "cleanup_safe": False}
        thread = None
        try:
            self._job = self._kernel.CreateJobObjectW(None, None)
            if not self._job:
                self._raise("CreateJobObjectW")
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE only.
            if not self._kernel.SetInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                self._raise("SetInformationJobObject")
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            process, thread, self.pid, thread_id = _winapi.CreateProcess(
                str(command[0]), subprocess.list2cmdline([str(arg) for arg in command]),
                None, None, False, 0x00000004 | 0x08000000, dict(env), str(cwd), startup)
            self._process = process
            self._record.update(launcher_pid=self.pid, primary_thread_id=thread_id)
            if not self._kernel.AssignProcessToJobObject(self._job, process):
                self._raise("AssignProcessToJobObject")
            self._record["job_assigned_before_resume"] = True
            self.observe()
            if self._kernel.ResumeThread(thread) == 0xFFFFFFFF:
                self._raise("ResumeThread")
            self._record["resumed_perf_ns"] = time.perf_counter_ns()
        except BaseException:
            # An unassigned suspended launcher is still owned by its exact handle.
            if self._process is not None:
                self._kernel.TerminateProcess(self._process, 1)
                _winapi.WaitForSingleObject(self._process, 2000)
            for handle in (self._process, self._job):
                if handle:
                    self._kernel.CloseHandle(handle)
            self._process = self._job = None
            raise
        finally:
            if thread is not None:
                _winapi.CloseHandle(thread)

    def _configure_api(self):
        api = self._kernel
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("TerminateProcess", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("ResumeThread", [wintypes.HANDLE], wintypes.DWORD),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, result

    @staticmethod
    def _raise(operation):
        raise OSError(ctypes.get_last_error(), operation)

    def _pids(self):
        count = 8
        while count <= 4096:
            class PidList(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("listed", wintypes.DWORD),
                            ("pids", ctypes.c_size_t * count)]
            value = PidList()
            if self._kernel.QueryInformationJobObject(self._job, 3, ctypes.byref(value), ctypes.sizeof(value), None):
                return [int(pid) for pid in value.pids[:value.listed]]
            if ctypes.get_last_error() != 234:
                self._raise("QueryInformationJobObject process list")
            count = max(count * 2, int(value.assigned))
        raise RuntimeError("Unexpected native helper process tree size")

    def _active(self):
        value = _Accounting()
        if not self._kernel.QueryInformationJobObject(self._job, 1, ctypes.byref(value), ctypes.sizeof(value), None):
            self._raise("QueryInformationJobObject accounting")
        self._record["last_accounting"] = {name: int(getattr(value, name)) for name, _ in value._fields_}
        return int(value.active_processes)

    def observe(self, expected_pid=None):
        import psutil
        with self._lock:
            if self._closed:
                return dict(self._identities)
            pids = self._pids()
            if expected_pid is not None and int(expected_pid) not in pids:
                raise RuntimeError("HELLO process is outside the owned Job")
            for pid in pids:
                try:
                    process = psutil.Process(pid)
                    identity = {"pid": pid, "creation": process.create_time(), "cmdline": process.cmdline()}
                    prior = self._identities.get(pid)
                    if prior is not None and prior != identity:
                        raise RuntimeError("Owned PID identity changed")
                    self._identities[pid] = identity
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    self._record["identity_errors"].append({"pid": pid, "error": repr(error)})
                    raise
            self._record["last_active_pids"] = pids
            self._active()
            return dict(self._identities)

    def poll(self):
        if self._process is None:
            return self._record.get("launcher_exit_code")
        if self._winapi.WaitForSingleObject(self._process, 0) == 0:
            return self._winapi.GetExitCodeProcess(self._process)
        return None

    def audit(self):
        with self._lock:
            return copy.deepcopy({**self._record, "identities": {str(pid): value for pid, value in self._identities.items()}})

    def close(self, deadline):
        """One absolute budget for kill, identity-guarded fallback and all waits."""
        import psutil
        with self._lock:
            if self._closed:
                return self.audit()
            errors = []
            try:
                self.observe()
            except Exception as error:
                errors.append({"phase": "observe", "error": repr(error)})
            try:
                if self._active() and not self._kernel.TerminateJobObject(self._job, 1):
                    self._raise("TerminateJobObject")
            except Exception as error:
                errors.append({"phase": "terminate_job", "error": repr(error)})
            # Fallback touches only identities captured from this Job, never names.
            targets = []
            for identity in self._identities.values():
                try:
                    process = psutil.Process(identity["pid"])
                    if process.create_time() != identity["creation"]:
                        continue  # Original process exited; its PID has been reused.
                    if process.cmdline() != identity["cmdline"]:
                        raise RuntimeError("Owned command changed; refusing fallback terminate")
                    process.terminate()
                    targets.append(process)
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    errors.append({"phase": "fallback_terminate", "pid": identity["pid"], "error": repr(error)})
            _, alive = psutil.wait_procs(targets, timeout=min(.5, max(0, deadline - time.perf_counter())))
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    errors.append({"phase": "fallback_kill", "pid": process.pid, "error": repr(error)})
            _, alive = psutil.wait_procs(alive, timeout=max(0, deadline - time.perf_counter()))
            active = None
            while time.perf_counter() < deadline:
                try:
                    active = self._active()
                except Exception as error:
                    errors.append({"phase": "terminal_accounting", "error": repr(error)})
                    break
                if active == 0:
                    break
                time.sleep(min(.02, max(0, deadline - time.perf_counter())))
            self._record.update(terminal_active_processes=active, fallback_remaining_pids=[process.pid for process in alive],
                                launcher_exit_code=self.poll(), close_perf_ns=time.perf_counter_ns())
            # Keep owned handles available for a later bounded retry if zero
            # active processes could not yet be proved. Parent death still
            # closes its KILL_ON_JOB_CLOSE handle.
            if active == 0 and not alive:
                for name in ("_process", "_job"):
                    handle = getattr(self, name)
                    if handle and not self._kernel.CloseHandle(handle):
                        errors.append({"phase": "close_handle", "handle": name, "error": ctypes.get_last_error()})
                    else:
                        setattr(self, name, None)
            safe = active == 0 and not alive and self._process is None and self._job is None
            self._closed = safe
            self._record.update(closed=safe, cleanup_safe=safe)
            self._record["cleanup_errors"].extend(errors)
            if active != 0 or alive or errors:
                raise RuntimeError(f"Native helper cleanup failed: {self.audit()}")
            return self.audit()
