"""Assemble/seal a candidate from official Python archives and a locked wheelhouse.

Run with the development Python 3.12 x64. MSI files are read as archives only;
this script never installs/registers Python, drivers or Windows components.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "packaging" / "windows"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def records(root):
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()
            and path.name not in ("FILE_MANIFEST.json", "SHA256SUMS.txt")]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_tcltk(msi_path, destination):
    """Read the MSI cabinet and file tables without invoking Windows Installer."""
    import _msi

    destination.mkdir()
    database = _msi.OpenDatabase(str(msi_path), _msi.MSIDBOPEN_READONLY)

    def rows(sql, columns):
        view = database.OpenView(sql)
        view.Execute(None)
        result = []
        while (row := view.Fetch()) is not None:
            result.append([row.GetString(i) for i in range(1, columns + 1)])
        view.Close()
        return result

    cabinets = rows("SELECT Cabinet FROM Media", 1)
    if cabinets != [["#cab1.cab"]]:
        raise RuntimeError(f"Unexpected official Tcl/Tk cabinet layout: {cabinets}")
    # _msi exposes table queries but not RecordReadStream. Use the same read-only
    # Windows API for the embedded cabinet bytes; no MSI actions are executed.
    api = ctypes.WinDLL("msi", use_last_error=True)
    handle = wintypes.UINT
    api.MsiOpenDatabaseW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(handle)]
    api.MsiDatabaseOpenViewW.argtypes = [handle, wintypes.LPCWSTR, ctypes.POINTER(handle)]
    api.MsiViewExecute.argtypes = [handle, handle]
    api.MsiViewFetch.argtypes = [handle, ctypes.POINTER(handle)]
    api.MsiRecordReadStream.argtypes = [handle, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    api.MsiCloseHandle.argtypes = [handle]

    def checked(result):
        if result:
            raise OSError(f"Read-only MSI archive API returned {result}")

    handles = []
    try:
        db, view, record = handle(), handle(), handle()
        checked(api.MsiOpenDatabaseW(str(msi_path), None, ctypes.byref(db)))
        handles.append(db)
        checked(api.MsiDatabaseOpenViewW(db, "SELECT `Data` FROM `_Streams` WHERE `Name`='cab1.cab'", ctypes.byref(view)))
        handles.append(view)
        checked(api.MsiViewExecute(view, 0))
        checked(api.MsiViewFetch(view, ctypes.byref(record)))
        handles.append(record)
        cabinet = destination / "cab1.cab"
        with cabinet.open("xb") as stream:
            while True:
                buffer = ctypes.create_string_buffer(65536)
                count = wintypes.DWORD(len(buffer))
                checked(api.MsiRecordReadStream(record, 1, buffer, ctypes.byref(count)))
                if not count.value:
                    break
                stream.write(buffer.raw[:count.value])
    finally:
        for item in reversed(handles):
            api.MsiCloseHandle(item)
    expanded = destination / "cabinet"
    expanded.mkdir()
    subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32" / "expand.exe"),
                    str(cabinet), "-F:*", str(expanded)], check=True, stdout=subprocess.DEVNULL)
    directories = {key: (parent, name) for key, parent, name in rows(
        "SELECT Directory,Directory_Parent,DefaultDir FROM Directory", 3)}
    components = dict(rows("SELECT Component,Directory_ FROM Component", 2))

    def directory(key):
        if key in ("TARGETDIR", "InstallDirectory"):
            return Path()
        parent, name = directories[key]
        return directory(parent) / name.split(":")[0].split("|")[-1]

    layout = destination / "layout"
    for key, component, name in rows("SELECT File,Component_,FileName FROM File", 3):
        relative = directory(components[component]) / name.split("|")[-1]
        if not (relative.parts[0] in ("tcl", "DLLs") or relative.parts[:2] == ("Lib", "tkinter")):
            continue
        target = (layout / relative).resolve()
        if not target.is_relative_to(layout.resolve()):
            raise ValueError("Invalid archive path")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(expanded / key, target)
    return layout


def build(output, downloads, version):
    if os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Build with Windows x64 Python 3.12")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-candidate\.[0-9]+", version):
        raise ValueError("Use a version such as 0.1.0-candidate.1")
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"LeicaM9ForNikon-{version}-win-x64-nvidia"
    bundle.mkdir()  # Refuse to overwrite a previous candidate.
    runtime = bundle / "runtime"
    archive = downloads / "python-3.12.4-embed-amd64.zip"
    with zipfile.ZipFile(archive) as source:
        source.extractall(runtime)
    tcl_layout = extract_tcltk(downloads / "tcltk.msi", output / f"tcltk-{version}")
    shutil.copytree(tcl_layout / "Lib" / "tkinter", runtime / "Lib" / "tkinter")
    shutil.copytree(tcl_layout / "tcl", runtime / "tcl")
    for name in ("_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "zlib1.dll"):
        shutil.copyfile(tcl_layout / "DLLs" / name, runtime / name)
    # _pth isolates both GUI and -m m9.native_decode_worker from system Python,
    # user site-packages, PYTHONPATH and registry-installed interpreters.
    (runtime / "python312._pth").write_text(
        "python312.zip\n.\nLib\nLib/site-packages\n..\nimport site\n", encoding="ascii")
    source_files = json.loads((CONFIG / "source-manifest.json").read_text(encoding="utf-8"))
    for relative in source_files:
        source = (REPO / relative).resolve()
        if not source.is_relative_to(REPO) or not source.is_file():
            raise ValueError(f"Invalid source manifest entry: {relative}")
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("启动程序.cmd", "环境诊断.cmd"):
        # cmd.exe requires CRLF; UTF-8 text is enabled before non-ASCII commands.
        (bundle / name).write_bytes((CONFIG / name).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    for name in ("使用说明.txt", "THIRD_PARTY_NOTICES.txt", "requirements-runtime.lock"):
        shutil.copyfile(CONFIG / name, bundle / name)
    packages = runtime / "Lib" / "site-packages"
    environment = dict(os.environ, TEMP=str(downloads), TMP=str(downloads), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-compile",
                    "--no-cache-dir", "--only-binary=:all:", "--find-links", str(downloads / "wheels"),
                    "--target", str(packages), "-r", str(CONFIG / "requirements-runtime.lock")],
                   env=environment, check=True)
    # Generated CLI wrappers embed the build interpreter path; no such tools are
    # needed by this application. Preserve package code, metadata and licenses.
    for name in ("bin", "Scripts"):
        target = (packages / name).resolve()
        if target.exists():
            if not target.is_relative_to(bundle.resolve()):
                raise ValueError("Invalid generated tools path")
            shutil.rmtree(target)
    dependencies = {}
    for line in (CONFIG / "requirements-runtime.lock").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version_text = line.split("==")
            dependencies[name] = version_text
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in importlib.metadata.distributions(path=[str(packages)])}
    for name, version_text in dependencies.items():
        if installed.get(name.lower().replace("_", "-")) != version_text:
            raise RuntimeError(f"Installed wheel differs from lock: {name}")
    licenses = bundle / "licenses"
    license_index = []
    for path in sorted(runtime.rglob("*")):
        if not path.is_file() or not any(word in path.name.lower() for word in ("license", "copyright", "notice", "eula")):
            continue
        relative = path.relative_to(runtime)
        target = licenses / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        license_index.append(relative.as_posix())
    write_json(licenses / "INDEX.json", license_index)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    release = {"version": version, "status": "LOCAL_CANDIDATE_NOT_FOR_REDISTRIBUTION",
               "created_utc": datetime.now(timezone.utc).isoformat(), "source_head": head,
               "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
               "python": "3.12.4", "platform": "Windows x64 NVIDIA", "dependencies": dependencies,
               "pipeline": {"camera": "tele", "white_balance": "D50", "extra_ev": 0,
                            "jpeg_quality": 95, "default_resolution": "original", "device": "cuda"},
               "second_machine_offline_acceptance": "NOT_RUN",
               "redistribution_permission": "BLOCKED_PENDING_CONFIRMATION"}
    write_json(bundle / "RELEASE.json", release)
    write_json(output / "build-inputs.json", {
        "archives": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in [archive, downloads / "tcltk.msi", *sorted((downloads / "wheels").glob("*.whl"))]],
        "source_files": [{"path": p, "sha256": sha256(REPO / p)} for p in source_files],
        "source_head": head})
    seal(bundle, make_zip=False)
    print(bundle, flush=True)


def seal(bundle, *, make_zip=True):
    if not (bundle / "RELEASE.json").is_file() or not (bundle / "runtime" / "python.exe").is_file():
        raise ValueError("Not a portable candidate directory")
    # Cache files would contain local paths and are not part of a release.
    bad = [p for p in bundle.rglob("*") if p.name in ("__pycache__", "pyvenv.cfg", ".git")
           or p.suffix.lower() in (".nef", ".dng", ".jpg", ".jpeg", ".log", ".pyc")]
    if bad:
        raise RuntimeError(f"Unexpected development/user files: {bad[:5]}")
    manifest = records(bundle)
    write_json(bundle / "FILE_MANIFEST.json", manifest)
    sums = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in manifest)
    sums += f"{sha256(bundle / 'FILE_MANIFEST.json')}  FILE_MANIFEST.json\n"
    (bundle / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    if make_zip:
        archive = bundle.parent / (bundle.name + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as target:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    target.write(path, (Path(bundle.name) / path.relative_to(bundle)).as_posix())
        (archive.parent / (archive.name + ".sha256")).write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
        print(f"ZIP: {archive} ({archive.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--downloads", type=Path, required=True)
    create.add_argument("--version", default="0.1.0-candidate.1")
    finalize = commands.add_parser("seal")
    finalize.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output.resolve(), args.downloads.resolve(), args.version)
    else:
        seal(args.bundle.resolve())


if __name__ == "__main__":
    main()
