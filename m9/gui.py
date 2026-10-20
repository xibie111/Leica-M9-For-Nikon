"""Responsive, session-only Tk photo queue for the strict NEF pipeline."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import queue
import traceback

from .batch import BatchJob, BatchOptions, PhotoQueue


@contextmanager
def _nef_processor_context(options, *, _plan=None):
    from .nef import process_nef, _NATIVE_QUEUE_PROCESSOR
    from .session import NefSession
    with NefSession(options.device) as session:
        if (_plan is not None and len(_plan.items) > 1 and options.method == "native"
                and process_nef is _NATIVE_QUEUE_PROCESSOR and session.device == "cuda"
                and not _plan.is_stopping()):
            from .native_queue import _native_processor_context
            with _native_processor_context(process_nef, session, _plan) as processor:
                yield processor
        else:
            if _plan is not None:
                _plan.receipt.update(enabled=False, reason="single, CPU, DNG or replaced core")
            yield partial(process_nef, session=session)


class NefWindow:
    def __init__(self, root, processor=None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.photo_queue = PhotoQueue()
        self.job = BatchJob(processor, processor_context=_nef_processor_context if processor is None else None,
                            _plan_context=_nef_processor_context if processor is None else None)
        self.busy = self.closed = self.close_requested = False
        self.close_dialog = None
        self.last_summary = self.run_options = self._pending_finish = None
        self.batch_errors = []
        self._run_ids = set()
        self._counts = dict(total=0, completed=0, succeeded=0, failed=0)
        self.output_dir = tk.StringVar(root)
        self.resolution = tk.StringVar(root, value="original")
        self.device = tk.StringVar(root, value="auto")
        self.write_json = tk.BooleanVar(root, value=False)
        self.status = tk.StringVar(root, value="添加 NEF 照片，选择输出目录，然后开始处理。")
        self.current_photo = tk.StringVar(root, value="当前照片：无")
        self.current_stage = tk.StringVar(root, value="当前阶段：等待")
        self.summary = tk.StringVar(root, value="本轮 0/0 · 成功 0 · 失败 0")
        self.queue_status = tk.StringVar(root, value="队列共 0 张 · 等待 0 · 成功 0 · 失败 0")
        root.title("OffLineM9 · NEF 本地处理")
        root.minsize(900, 700)
        root.geometry("1000x760")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame = ttk.Frame(root, padding=12)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=2)
        frame.rowconfigure(12, weight=1)

        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.add_photos_button = ttk.Button(toolbar, text="添加照片…", command=self.choose_photos)
        self.add_folder_button = ttk.Button(toolbar, text="添加文件夹…", command=self.choose_folder)
        self.remove_button = ttk.Button(toolbar, text="移除选中等待项", command=self.remove_selected)
        self.clear_button = ttk.Button(toolbar, text="清空队列", command=self.clear_queue)
        self.retry_button = ttk.Button(toolbar, text="失败项重新入队", command=self.retry_failed)
        for button in (self.add_photos_button, self.add_folder_button, self.remove_button,
                       self.clear_button, self.retry_button):
            button.pack(side="left", padx=(0, 6))

        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=("name", "parent", "status", "result"),
                                 show="headings", selectmode="extended", height=8)
        for column, text, width in (("name", "文件名", 150), ("parent", "来源目录", 230),
                                    ("status", "状态", 80), ("result", "输出结果 / 错误", 390)):
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, minwidth=60)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

        output = ttk.Frame(frame)
        output.grid(row=2, column=0, sticky="ew", pady=7)
        output.columnconfigure(1, weight=1)
        ttk.Label(output, text="输出目录").grid(row=0, column=0)
        self.output_entry = ttk.Entry(output, textvariable=self.output_dir)
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self.output_button = ttk.Button(output, text="选择目录…", command=self.choose_output)
        self.output_button.grid(row=0, column=2)
        self.open_output_button = ttk.Button(output, text="打开输出目录", command=self.open_output)
        self.open_output_button.grid(row=0, column=3, padx=(6, 0))

        options = ttk.Frame(frame)
        options.grid(row=3, column=0, sticky="w", pady=3)
        ttk.Label(options, text="分辨率").pack(side="left")
        self.resolution_box = ttk.Combobox(options, textvariable=self.resolution,
                                          values=("original", "phone"), state="readonly", width=11)
        self.resolution_box.pack(side="left", padx=(8, 20))
        ttk.Label(options, text="设备").pack(side="left")
        self.device_box = ttk.Combobox(options, textvariable=self.device,
                                      values=("auto", "cuda", "cpu"), state="readonly", width=8)
        self.device_box.pack(side="left", padx=8)
        self.write_json_check = ttk.Checkbutton(options, text="导出 JSON 处理报告", variable=self.write_json)
        self.write_json_check.pack(side="left", padx=12)
        self.tuning_label = ttk.Label(options, text="调校：tele（长焦）")
        self.tuning_label.pack(side="left", padx=12)
        self.native_button = ttk.Button(frame, text="直接 NEF 处理", command=lambda: self.start("native"))
        self.native_button.grid(row=4, column=0, sticky="ew", pady=(6, 3))
        self.dng_button = ttk.Button(frame, text="自动转 DNG 处理", command=lambda: self.start("dng"))
        self.dng_button.grid(row=5, column=0, sticky="ew", pady=3)
        self.stop_button = ttk.Button(frame, text="停止队列（完成当前张）", command=self.request_stop,
                                      state="disabled")
        self.stop_button.grid(row=6, column=0, sticky="w", pady=3)
        ttk.Label(frame, textvariable=self.status, wraplength=950).grid(row=7, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.current_photo).grid(row=8, column=0, sticky="w", pady=2)
        ttk.Label(frame, textvariable=self.current_stage).grid(row=9, column=0, sticky="w")
        self.progressbar = ttk.Progressbar(frame, mode="determinate", maximum=1, value=0)
        self.progressbar.grid(row=10, column=0, sticky="ew", pady=5)
        totals = ttk.Frame(frame)
        totals.grid(row=11, column=0, sticky="ew")
        ttk.Label(totals, textvariable=self.summary).pack(side="left")
        ttk.Label(totals, textvariable=self.queue_status).pack(side="right")
        log_frame = ttk.Frame(frame)
        log_frame.grid(row=12, column=0, sticky="nsew", pady=(5, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=7, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_id = root.after(100, self.poll)

    def append_log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", str(message).rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh_queue(self):
        known = {item.id for item in self.photo_queue.items}
        for item_id in self.tree.get_children():
            if item_id not in known:
                self.tree.delete(item_id)
        labels = {"pending": "等待", "running": "正在处理", "succeeded": "成功", "failed": "失败"}
        for item in self.photo_queue.items:
            result = item.report["output"] if item.report else item.error
            values = (item.path.name, str(item.path.parent), labels[item.status], result)
            if self.tree.exists(item.id):
                self.tree.item(item.id, values=values)
            else:
                self.tree.insert("", "end", iid=item.id, values=values)
        counts = {status: sum(item.status == status for item in self.photo_queue.items)
                  for status in ("pending", "succeeded", "failed")}
        self.queue_status.set(f"队列共 {len(known)} 张 · 等待 {counts['pending']} · "
                              f"成功 {counts['succeeded']} · 失败 {counts['failed']}")

    def _added(self, added):
        if added and not self.output_dir.get().strip():
            self.output_dir.set(str(added[0].path.parent / "M9_output"))
        self.refresh_queue()
        self.status.set(f"已添加 {len(added)} 张新照片；同一路径不会重复添加。")
        return added

    def add_paths(self, paths):
        if self.busy:
            return []
        return self._added(self.photo_queue.add_paths(paths))

    def choose_photos(self):
        from tkinter import filedialog, messagebox
        if self.busy:
            return
        paths = filedialog.askopenfilenames(parent=self.root, title="添加 Nikon NEF 照片",
                                           filetypes=(("Nikon NEF", "*.nef *.NEF"),))
        if paths:
            try:
                self.add_paths(paths)
            except (OSError, ValueError) as exc:
                messagebox.showerror("无法添加照片", str(exc), parent=self.root)

    def choose_folder(self):
        from tkinter import filedialog, messagebox
        if self.busy:
            return
        folder = filedialog.askdirectory(parent=self.root, title="添加当前文件夹的 NEF（不含子文件夹）")
        if folder:
            try:
                self._added(self.photo_queue.add_folder(folder))
            except (OSError, ValueError) as exc:
                messagebox.showerror("无法添加文件夹", str(exc), parent=self.root)

    def remove_selected(self):
        if not self.busy:
            self.photo_queue.remove_pending(self.tree.selection())
            self.refresh_queue()

    def clear_queue(self):
        if not self.busy:
            self.photo_queue.clear()
            self.refresh_queue()

    def retry_failed(self):
        if not self.busy:
            count = len(self.photo_queue.requeue_failed())
            self.refresh_queue()
            self.status.set(f"已将 {count} 张失败照片重新入队；请选择处理方式开始。")

    def choose_output(self):
        from tkinter import filedialog
        if self.busy:
            return
        path = filedialog.askdirectory(parent=self.root, title="选择 JPEG 输出目录")
        if path:
            self.output_dir.set(path)

    def open_output(self):
        from tkinter import messagebox
        path = self.run_options.output_dir if self.busy else Path(self.output_dir.get().strip()).expanduser()
        try:
            if not self.output_dir.get().strip() or not path.is_dir():
                raise NotADirectoryError(f"输出目录尚不存在：{path}")
            os.startfile(str(path.resolve()))
        except OSError as exc:
            messagebox.showerror("无法打开输出目录", str(exc), parent=self.root)

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.add_photos_button, self.add_folder_button, self.remove_button,
                       self.clear_button, self.retry_button, self.output_entry, self.output_button,
                       self.native_button, self.dng_button, self.write_json_check):
            widget.configure(state=state)
        for widget in (self.resolution_box, self.device_box):
            widget.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def _show_counts(self):
        c = self._counts
        self.summary.set(f"本轮 {c['completed']}/{c['total']} · 成功 {c['succeeded']} · 失败 {c['failed']}")
        self.progressbar.configure(maximum=max(1, c["total"]), value=c["completed"])

    def start(self, method):
        from tkinter import messagebox
        if self.busy:
            return
        pending = self.photo_queue.pending_items
        if not pending:
            messagebox.showinfo("没有等待项", "请添加照片或将失败项重新入队。", parent=self.root)
            return
        output = self.output_dir.get().strip()
        if not output:
            messagebox.showerror("请选择输出目录", "请先选择或填写输出目录。", parent=self.root)
            return
        self.run_options = BatchOptions(method, Path(output).expanduser(), self.resolution.get(),
                                        self.device.get(), bool(self.write_json.get()))
        self.last_summary = self._pending_finish = None
        self.batch_errors = []
        self._run_ids = {item.id for item in pending}
        self._counts = dict(total=len(pending), completed=0, succeeded=0, failed=0)
        self._show_counts()
        self.set_busy(True)
        self.status.set("正在开始本轮队列…")
        self.current_stage.set("当前阶段：准备")
        self.append_log(f"开始本轮 {len(pending)} 张；方法={method}；分辨率={self.run_options.resolution}；"
                        f"设备={self.run_options.device}；JSON报告={'开' if self.run_options.write_json else '关'}")
        try:
            self.job.start(pending, self.run_options)
        except Exception as exc:
            self.set_busy(False)
            self.status.set("队列启动失败；详细信息见日志。")
            self.current_stage.set("当前阶段：启动失败")
            self.append_log(traceback.format_exc())
            messagebox.showerror("NEF 队列启动失败", str(exc), parent=self.root)

    def request_stop(self):
        if self.busy:
            self.job.request_stop()
            self.stop_button.configure(state="disabled")
            self.status.set("已请求停止后续任务；正在处理的照片将正常完成。")

    def poll(self):
        self._poll_id = None
        while True:
            try:
                kind, payload = self.job.events.get_nowait()
            except queue.Empty:
                break
            item = self.photo_queue.apply_event(kind, payload)
            if item is not None:
                self.refresh_queue()
            if kind == "item_started":
                self.current_photo.set(f"当前照片 {payload['ordinal']}/{payload['total']}：{Path(payload['input_path']).name}")
                self.current_stage.set("当前阶段：开始")
                self.append_log(f"开始照片：{payload['input_path']}")
            elif kind == "progress":
                self.current_stage.set(f"当前阶段：{payload['stage']} · {payload['message']}")
                self.append_log(f"[{payload['stage']}] {payload['message']}")
            elif kind in ("item_done", "item_error"):
                self._counts["completed"] += 1
                self._counts["succeeded" if kind == "item_done" else "failed"] += 1
                self._show_counts()
                if kind == "item_done":
                    self.append_log(f"成功：{payload['report']['output']}")
                else:
                    self.append_log(f"失败：{item.path if item else payload['id']} · {payload['message']}\n{payload['traceback']}")
            elif kind == "batch_error":
                self.batch_errors.append(dict(payload))
                phase = {"setup": "准备", "processing": "执行", "cleanup": "清理"}.get(payload["phase"], payload["phase"])
                self.current_stage.set(f"当前阶段：队列{phase}失败")
                self.status.set("队列发生错误；详细信息见日志。")
                self.append_log(f"队列{phase}失败：{payload['message']}\n{payload['traceback']}")
            elif kind == "finished":
                self._pending_finish = payload
        # A terminal event can arrive just before its worker returns. Both must finish.
        if self.job._cleanup_unsafe:
            self.status.set("原生预解码资源清理未完成；队列已锁定，详细信息见日志。")
            self.current_stage.set("当前阶段：资源清理失败")
            self.stop_button.configure(state="disabled")
        if self._pending_finish is not None and not self.job.running and not self.job._cleanup_unsafe:
            self._finish(self._pending_finish)
            self._pending_finish = None
        if not self.closed:
            self._poll_id = self.root.after(100, self.poll)

    def _finish(self, payload):
        self.last_summary = dict(payload)
        self._counts = {key: payload[key] for key in ("total", "completed", "succeeded", "failed")}
        self._show_counts()
        self.set_busy(False)
        self.current_photo.set("当前照片：无")
        if self.batch_errors:
            self.current_stage.set("当前阶段：队列错误")
            self.status.set("本轮队列发生错误；照片结果见列表，详细信息见日志。")
            conclusion = " · 队列错误"
        else:
            self.current_stage.set("当前阶段：已停止" if payload["stopped"] else "当前阶段：本轮完成")
            self.status.set("本轮已停止，未开始的照片仍在等待。" if payload["stopped"] else "本轮处理完成。")
            conclusion = " · 已停止" if payload["stopped"] else " · 已完成"
        self.append_log(self.summary.get() + conclusion)
        failed = [item for item in self.photo_queue.items if item.id in self._run_ids and item.status == "failed"]
        if failed:
            self.append_log("本轮失败汇总：\n" + "\n".join(f"{item.path}：{item.error}" for item in failed))
        if self.close_requested:
            self._destroy_window()

    def close(self):
        if not self.busy:
            self._destroy_window()
            return
        if self.close_dialog is not None:
            self.close_dialog.lift()
            return
        import tkinter as tk
        from tkinter import ttk
        dialog = self.close_dialog = tk.Toplevel(self.root)
        dialog.title("正在处理队列")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="选择“完成当前张后关闭”将停止后续任务，并等待正在处理的照片完成。",
                  wraplength=420).pack(pady=(0, 12))
        self.close_finish_button = ttk.Button(body, text="完成当前张后关闭", command=self.confirm_close_after_current)
        self.close_finish_button.pack(side="left", padx=(0, 10))
        self.close_return_button = ttk.Button(body, text="返回", command=self.cancel_close)
        self.close_return_button.pack(side="left")
        dialog.protocol("WM_DELETE_WINDOW", self.cancel_close)
        dialog.grab_set()
        self.close_return_button.focus_set()

    def cancel_close(self):
        if self.close_dialog is not None:
            self.close_dialog.grab_release()
            self.close_dialog.destroy()
            self.close_dialog = None

    def confirm_close_after_current(self):
        self.cancel_close()
        self.close_requested = True
        if self.busy:
            self.request_stop()
        else:
            self._destroy_window()

    def _destroy_window(self):
        self.closed = True
        self.cancel_close()
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.root.destroy()


def main():
    import tkinter as tk
    root = tk.Tk()
    NefWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
