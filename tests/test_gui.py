import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest

from m9.gui import NefWindow


@pytest.fixture(scope="module")
def tk_app():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def tk_window(tk_app):
    import tkinter as tk
    window = tk.Toplevel(tk_app)
    window.withdraw()
    yield window
    for callback in tk_app.tk.call("after", "info"):
        window.after_cancel(callback)
    if window.winfo_exists():
        window.destroy()


def pump(root, until, seconds=3):
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:
        root.update()
        time.sleep(0.005)
    assert until(), "Tk/worker did not reach the expected state"


def make_window(root, tmp_path, processor, count=2):
    window = NefWindow(root, processor=processor)
    paths = [tmp_path / f"照片 {i}.NEF" for i in range(count)]
    for path in paths:
        path.write_bytes(b"synthetic test input")
    window.add_paths(paths)
    window.output_dir.set(str(tmp_path / "输出 成片"))
    return window, paths


def test_native_first_defaults_and_keyboard_order(tk_window):
    window = NefWindow(tk_window, processor=lambda **kwargs: {})
    tk_window.deiconify()
    tk_window.update()
    assert window.native_button.cget("text") == "直接 NEF 处理"
    assert window.dng_button.cget("text") == "自动转 DNG 处理"
    assert window.native_button.winfo_rooty() < window.dng_button.winfo_rooty()
    assert str(window.native_button.tk_focusNext()) == str(window.dng_button)
    assert window.resolution.get() == "original"
    assert window.device.get() == "auto"
    assert window.write_json.get() is False
    assert str(window.progressbar.cget("mode")) == "determinate"
    assert tk_window.winfo_width() == 1000
    assert window.tree.winfo_height() > 100
    assert window.summary.get() == "本轮 0/0 · 成功 0 · 失败 0"


def test_photo_and_current_folder_import_remove_and_clear(tmp_path, tk_window, monkeypatch):
    from tkinter import filedialog
    folder = tmp_path / "照片 目录"
    folder.mkdir()
    a, b = folder / "A.NEF", folder / "b.nef"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    (folder / "ignore.jpg").write_bytes(b"jpeg")
    sub = folder / "子文件夹"
    sub.mkdir()
    (sub / "nested.NEF").write_bytes(b"nef")
    window = NefWindow(tk_window, processor=lambda **kwargs: {})
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **kwargs: (str(b), str(a), str(b)))
    window.choose_photos()
    assert [item.path.name for item in window.photo_queue.items] == ["b.nef", "A.NEF"]
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kwargs: str(folder))
    window.choose_folder()
    assert len(window.tree.get_children()) == 2
    assert window.output_dir.get() == str(folder / "M9_output")
    first, second = window.photo_queue.items
    window.photo_queue.apply_event("item_done", {"id": first.id, "report": {"output": "result.jpg"}})
    window.refresh_queue()
    window.tree.selection_set((first.id, second.id))
    window.remove_selected()
    assert [item.id for item in window.photo_queue.items] == [first.id]
    assert window.tree.item(first.id, "values")[2:] == ("成功", "result.jpg")
    window.clear_queue()
    assert not window.photo_queue.items and not window.tree.get_children()
    assert a.read_bytes() == b"a" and b.read_bytes() == b"b"


def test_loop_responsive_one_worker_and_busy_mutations_blocked(tmp_path, tk_window):
    entered, release = threading.Event(), threading.Event()
    worker_ids = []

    def processor(*, progress, input_path, **kwargs):
        worker_ids.append(threading.get_ident())
        progress("decode", "读取中文 路径")
        entered.set()
        assert release.wait(3)
        return {"output": str(input_path.with_suffix(".jpg")), "record": None}

    window, paths = make_window(tk_window, tmp_path, processor)
    main_id = threading.get_ident()
    applied_ids = []
    real_apply = window.photo_queue.apply_event

    def apply(*args):
        applied_ids.append(threading.get_ident())
        return real_apply(*args)

    window.photo_queue.apply_event = apply
    ticks = []

    def tick():
        ticks.append(True)
        tk_window.after(10, tick)

    tk_window.after(10, tick)
    window.native_button.invoke()
    try:
        assert entered.wait(2)
        pump(tk_window, lambda: len(ticks) >= 5 and "decode" in window.current_stage.get())
        assert window.busy
        for button in (window.native_button, window.dng_button, window.add_photos_button,
                       window.add_folder_button, window.remove_button, window.clear_button,
                       window.retry_button, window.write_json_check):
            assert button.instate(["disabled"])
        assert window.stop_button.instate(["!disabled"])
        window.tree.selection_set(window.tree.get_children())
        window.remove_selected()
        window.clear_queue()
        window.retry_failed()
        assert window.add_paths([tmp_path / "other.NEF"]) == []
        window.start("dng")
        assert len(window.photo_queue.items) == 2
        assert len(worker_ids) == 1
        assert window.photo_queue.items[0].status == "running"
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert len(worker_ids) == 2 and len(set(worker_ids)) == 1
    assert worker_ids[0] != main_id
    assert set(applied_ids) == {main_id}
    assert window.last_summary == dict(total=2, completed=2, succeeded=2, failed=0, stopped=False)
    assert window.summary.get() == "本轮 2/2 · 成功 2 · 失败 0"
    assert float(window.progressbar.cget("value")) == 2
    assert all(item.status == "succeeded" for item in window.photo_queue.items)


@pytest.mark.parametrize("method", ["native", "dng"])
@pytest.mark.parametrize("resolution", ["original", "phone"])
@pytest.mark.parametrize("write_json", [False, True])
def test_options_snapshot_and_optional_json_all_routes(
    tmp_path, tk_window, method, resolution, write_json
):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(3)
        return {"output": "result.jpg", "record": "result.json" if kwargs["write_json"] else None}

    window, paths = make_window(tk_window, tmp_path, processor)
    original_output = Path(window.output_dir.get()).resolve()
    window.resolution.set(resolution)
    assert window.write_json.get() is False
    if write_json:
        window.write_json_check.invoke()
    (window.native_button if method == "native" else window.dng_button).invoke()
    try:
        assert entered.wait(2)
        # Programmatic variable changes cannot alter the captured batch settings.
        window.resolution.set("phone" if resolution == "original" else "original")
        window.device.set("cuda")
        window.write_json.set(not write_json)
        window.output_dir.set(str(tmp_path / "changed"))
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert len(calls) == 2
    for call in calls:
        assert call["method"] == method
        assert call["resolution"] == resolution
        assert call["device"] == "auto"
        assert call["write_json"] is write_json
        assert call["output_dir"] == original_output
    assert window.write_json_check.instate(["!disabled"])
    assert window.last_summary["succeeded"] == 2


def test_stop_finishes_current_then_resume_only_remaining_and_resets_run_count(tmp_path, tk_window):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def processor(*, input_path, **kwargs):
        calls.append(input_path)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return {"output": str(input_path.with_suffix(".jpg"))}

    window, paths = make_window(tk_window, tmp_path, processor, count=3)
    window.start("native")
    try:
        assert entered.wait(2)
        window.request_stop()
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert calls == paths[:1]
    assert [item.status for item in window.photo_queue.items] == ["succeeded", "pending", "pending"]
    assert window.last_summary == dict(total=3, completed=1, succeeded=1, failed=0, stopped=True)
    assert window.summary.get() == "本轮 1/3 · 成功 1 · 失败 0"
    window.start("native")
    pump(tk_window, lambda: not window.busy)
    assert calls == paths
    assert window.last_summary == dict(total=2, completed=2, succeeded=2, failed=0, stopped=False)
    assert window.summary.get() == "本轮 2/2 · 成功 2 · 失败 0"
    assert "成功 3" in window.queue_status.get()


def test_error_continues_with_summary_and_failed_retry_is_manual(tmp_path, tk_window, monkeypatch):
    from tkinter import messagebox
    calls = []
    fail_once = [True]

    def processor(*, input_path, **kwargs):
        calls.append(input_path.name)
        if input_path.name == "照片 1.NEF" and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("bad NEF fixture")
        return {"output": str(input_path.with_suffix(".jpg"))}

    def no_per_photo_popup(*args, **kwargs):
        raise AssertionError("per-photo failure must not block the queue with a popup")

    monkeypatch.setattr(messagebox, "showerror", no_per_photo_popup)
    window, paths = make_window(tk_window, tmp_path, processor, count=3)
    window.start("native")
    pump(tk_window, lambda: not window.busy)
    assert [item.status for item in window.photo_queue.items] == ["succeeded", "failed", "succeeded"]
    assert window.last_summary == dict(total=3, completed=3, succeeded=2, failed=1, stopped=False)
    failed = window.photo_queue.items[1]
    assert "bad NEF fixture" in window.tree.item(failed.id, "values")[3]
    assert "本轮失败汇总" in window.log.get("1.0", "end")
    assert "RuntimeError" in failed.details
    window.retry_button.invoke()
    assert len(calls) == 3
    assert failed.status == "pending" and not failed.error and not failed.details
    window.start("dng")
    pump(tk_window, lambda: not window.busy)
    assert calls == [path.name for path in paths] + [paths[1].name]
    assert window.last_summary == dict(total=1, completed=1, succeeded=1, failed=0, stopped=False)


def test_terminal_event_does_not_reenable_controls_before_thread_exit(tmp_path, tk_window):
    tail_entered, release = threading.Event(), threading.Event()
    window, _ = make_window(tk_window, tmp_path, lambda **kwargs: {"output": "done.jpg"}, count=1)
    real_run = window.job._run

    def run_with_tail(*args):
        real_run(*args)
        tail_entered.set()
        assert release.wait(3)

    window.job._run = run_with_tail
    window.start("native")
    try:
        assert tail_entered.wait(2)
        window.poll()
        assert window.busy and window.last_summary is None
        assert window._pending_finish is not None
        assert window.native_button.instate(["disabled"])
        window.start("dng")
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert window.last_summary["completed"] == 1


def test_thread_start_failure_restores_ui_and_queue(tmp_path, tk_window, monkeypatch):
    from tkinter import messagebox
    errors = []
    window, _ = make_window(tk_window, tmp_path, lambda **kwargs: {})
    monkeypatch.setattr(messagebox, "showerror", lambda *args, **kwargs: errors.append(args))

    def fail(*args):
        raise RuntimeError("cannot start a new thread")

    monkeypatch.setattr(window.job, "start", fail)
    window.start("native")
    assert not window.busy
    assert window.native_button.instate(["!disabled"])
    assert window.write_json_check.instate(["!disabled"])
    assert all(item.status == "pending" for item in window.photo_queue.items)
    assert "启动失败" in window.status.get()
    assert "cannot start a new thread" in window.log.get("1.0", "end")
    assert len(errors) == 1


def test_close_return_keeps_processing_without_requesting_stop(tmp_path, tk_window):
    entered, release = threading.Event(), threading.Event()
    calls, stops = [], []

    def processor(*, input_path, **kwargs):
        calls.append(input_path)
        entered.set()
        assert release.wait(3)
        return {"output": "done.jpg"}

    window, _ = make_window(tk_window, tmp_path, processor)
    real_stop = window.job.request_stop
    window.job.request_stop = lambda: (stops.append(True), real_stop())
    window.start("native")
    try:
        assert entered.wait(2)
        window.close()
        assert window.close_finish_button.cget("text") == "完成当前张后关闭"
        assert window.close_return_button.cget("text") == "返回"
        window.close_return_button.invoke()
        assert not window.close_requested and window.close_dialog is None
        assert not stops and not window.closed
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert len(calls) == 2 and not window.last_summary["stopped"]


def test_close_after_current_waits_for_publish_and_worker_exit(tmp_path, tk_window):
    entered, release = threading.Event(), threading.Event()
    finished = []

    def processor(**kwargs):
        entered.set()
        assert release.wait(3)
        finished.append(True)
        return {"output": "complete.jpg"}

    window, _ = make_window(tk_window, tmp_path, processor)
    window.start("native")
    try:
        assert entered.wait(2)
        window.close()
        window.close_finish_button.invoke()
        assert window.close_requested and not window.closed and window.job.running
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: window.closed)
    assert finished == [True]
    assert window.last_summary == dict(total=2, completed=1, succeeded=1, failed=0, stopped=True)
    assert [item.status for item in window.photo_queue.items] == ["succeeded", "pending"]


@pytest.mark.parametrize("choice", ["return", "finish"])
def test_worker_can_finish_while_close_modal_is_open(tmp_path, tk_window, choice):
    entered, release = threading.Event(), threading.Event()

    def processor(**kwargs):
        entered.set()
        assert release.wait(3)
        return {"output": "done.jpg"}

    window, _ = make_window(tk_window, tmp_path, processor, count=1)
    window.start("native")
    assert entered.wait(2)
    window.close()
    release.set()
    window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert window.close_dialog is not None and not window.closed
    assert not window.last_summary["stopped"]
    if choice == "return":
        window.close_return_button.invoke()
        assert not window.closed and not window.close_requested
    else:
        window.close_finish_button.invoke()
        assert window.closed


def test_stop_during_last_photo_counts_complete_not_stopped(tmp_path, tk_window):
    entered, release = threading.Event(), threading.Event()

    def processor(**kwargs):
        entered.set()
        assert release.wait(3)
        return {"output": "done.jpg"}

    window, _ = make_window(tk_window, tmp_path, processor, count=1)
    window.start("native")
    try:
        assert entered.wait(2)
        window.stop_button.invoke()
    finally:
        release.set()
        window.job.thread.join(timeout=3)
    pump(tk_window, lambda: not window.busy)
    assert window.last_summary == dict(total=1, completed=1, succeeded=1, failed=0, stopped=False)
    assert float(window.progressbar.cget("value")) == 1


def test_open_output_directory(tmp_path, tk_window, monkeypatch):
    opened = []
    window = NefWindow(tk_window, processor=lambda **kwargs: {})
    window.output_dir.set(str(tmp_path))
    monkeypatch.setattr(os, "startfile", lambda path: opened.append(path), raising=False)
    window.open_output_button.invoke()
    assert opened == [str(tmp_path.resolve())]


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_batch_launcher_handles_chinese_space_paths_and_forwards_arguments(tmp_path):
    launch_dir = tmp_path / "中文 启动目录"
    scripts = launch_dir / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(sys.executable, scripts / "python.exe")
    (scripts.parent / "pyvenv.cfg").write_text(
        f"home = {sys.base_prefix}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    shutil.copy2(Path(__file__).parents[1] / "process_nef.cmd", launch_dir)
    package = launch_dir / "m9"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "nef.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}, ensure_ascii=True))\n",
        encoding="utf-8",
    )
    source = tmp_path / "照片 空格.NEF"
    output = tmp_path / "输出 成片"
    launcher = launch_dir / "process_nef.cmd"
    command = f'cmd /d /s /c ""{launcher}" "{source}" --output-dir "{output}" --method native"'
    run = subprocess.run(command, capture_output=True, text=True, timeout=10, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    actual = json.loads(run.stdout.strip())
    assert Path(actual["cwd"]) == launch_dir
    assert actual["args"] == [str(source), "--output-dir", str(output), "--method", "native"]
