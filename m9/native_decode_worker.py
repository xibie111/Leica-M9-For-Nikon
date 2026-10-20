"""One CPU-only native decoder, launched as -m m9.native_decode_worker.

The parent owns its suspended launch/Windows Job and the shared slot lifetime.
No decode occurs until an authenticated owner sends OFFER. READY owns no RGB view.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time
import traceback
sys.dont_write_bytecode = True

from .native_payload import (MAX_SLOT_BYTES, array_layout, encode_matrix, encode_raw_info,
                             path_identity, recv_message, send_message, sha256, slot_array)


def _torch_absent():
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("The native decoder must never import torch")
    return True


def _stamp():
    return dict(perf_ns=time.perf_counter_ns(), wall_ns=time.time_ns())


def _request(message, owner, last_generation):
    if message.get("owner") != owner:
        raise ValueError("Native decoder owner mismatch")
    request, generation = message.get("request_id"), message.get("generation")
    if (type(request) is not str or not 0 < len(request) <= 256
            or type(generation) is not int or generation <= last_generation):
        raise ValueError("Native decoder request/generation mismatch")
    path = Path(message["path"]).expanduser().resolve()
    if str(path) != message["path"]:
        raise ValueError("Native decoder requires a canonical input path")
    capacity = message.get("slot_capacity")
    name = message.get("slot_name")
    if type(capacity) is not int or not 0 < capacity <= MAX_SLOT_BYTES or type(name) is not str or not 0 < len(name) <= 256:
        raise ValueError("Invalid parent-owned native slot")
    digest = message.get("sha256")
    if digest is not None and (type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Invalid optional native source hash")
    return request, generation, path


def _decode(message, timings):
    # Delayed imports keep the BOOT/auth phase idle. These modules have CPU-only dependencies.
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np
    from . import raw_input
    _torch_absent()
    path = Path(message["path"])
    before = path_identity(path)
    if before != message.get("identity"):
        raise ValueError("Native source identity changed before decode")
    before_sha = sha256(path)
    if message.get("sha256") is not None and before_sha != message["sha256"]:
        raise ValueError("Native source hash differs from the offered file")
    if path_identity(path) != before:
        raise ValueError("Native source identity changed while hashing")
    slot = SharedMemory(name=message["slot_name"], create=False)
    rgb = view = matrix = info = None
    original_imread = raw_input.rawpy.imread
    try:
        if slot.size != message["slot_capacity"]:
            raise ValueError("Native shared-slot capacity differs from the offer")
        class RawProxy:
            def __init__(self, raw):
                self.raw = raw
            def __enter__(self):
                self.raw = self.raw.__enter__()
                return self
            def __exit__(self, *args):
                return self.raw.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.raw, name)
            def postprocess(self, *args, **kwargs):
                call = dict(begin=_stamp())
                timings.append(call)
                try:
                    return self.raw.postprocess(*args, **kwargs)
                finally:
                    call["end"] = _stamp()
        def observed_imread(*args, **kwargs):
            return RawProxy(original_imread(*args, **kwargs))
        raw_input.rawpy.imread = observed_imread
        try:
            rgb, matrix, info = raw_input.load_native_nef_u16(path)
        finally:
            raw_input.rawpy.imread = original_imread
        if len(timings) != 1 or "end" not in timings[0]:
            raise ValueError("Expected exactly one actual native postprocess call")
        after = path_identity(path)
        after_sha = sha256(path)
        if before != after or path_identity(path) != after or before_sha != after_sha:
            raise ValueError("Native source changed during decode")
        layout = array_layout(rgb)
        if layout["nbytes"] > message["slot_capacity"]:
            raise ValueError("Decoded native pixels exceed the offered slot")
        # Encode all metadata before touching the slot; no full-size float32 prototype.
        reply = dict(kind="READY", owner=message["owner"], request_id=message["request_id"],
                     generation=message["generation"], path=str(path),
                     identity_before=before, identity_after=after, sha256_before=before_sha,
                     sha256_after=after_sha, layout=layout, matrix=encode_matrix(matrix), raw_info=encode_raw_info(info),
                     postprocess_begin_perf_ns=timings[0]["begin"]["perf_ns"],
                     postprocess_end_perf_ns=timings[0]["end"]["perf_ns"],
                     postprocess_begin_wall_ns=timings[0]["begin"]["wall_ns"],
                     postprocess_end_wall_ns=timings[0]["end"]["wall_ns"],
                     postprocess_calls=1, torch_absent=_torch_absent())
        view = slot_array(slot.buf, layout)
        np.copyto(view, rgb, casting="no")
        return reply
    finally:
        raw_input.rawpy.imread = original_imread
        # READY is sent only after this finally. No private RGB or slot export survives it.
        view = rgb = matrix = info = None
        slot.close()


def main(args):
    sys.dont_write_bytecode = True
    if args.host != "127.0.0.1" or not 0 < args.port < 65536 or not args.token or not args.owner:
        raise ValueError("Native decoder accepts only an authenticated localhost owner")
    _torch_absent()
    with socket.create_connection((args.host, args.port), timeout=5.0) as control:
        control.settimeout(0.1)
        send_message(control, dict(kind="HELLO", token=args.token, owner=args.owner, pid=os.getpid(),
                                  worker_module=str(Path(__file__).resolve()), torch_absent=_torch_absent()))
        ack = recv_message(control)
        if ack != dict(kind="BOOT_ACK", owner=args.owner, generation=0):
            raise ValueError("Invalid native decoder BOOT acknowledgement")
        active = None
        last_generation = 0
        slot_offer = None
        while True:
            try:
                message = recv_message(control)
            except EOFError:
                _torch_absent()
                return 0
            except Exception as error:
                send_message(control, dict(kind="ERROR", owner=args.owner, request_id=None, generation=None,
                                          message=str(error), traceback=traceback.format_exc()[-12000:],
                                          torch_absent=_torch_absent()))
                return 1
            timings = []
            try:
                kind = message.get("kind")
                if message.get("owner") != args.owner:
                    raise ValueError("Native decoder owner mismatch")
                if kind == "STOP":
                    send_message(control, dict(kind="BYE", owner=args.owner, torch_absent=_torch_absent(), pid=os.getpid()))
                    return 0
                if kind == "RELEASE":
                    if active != (message.get("request_id"), message.get("generation")):
                        raise ValueError("Native decoder release does not match the pending request")
                    active = None  # No slot write; a later valid OFFER is required.
                    continue
                if kind != "OFFER" or active is not None:
                    raise ValueError("Native decoder needs one OFFER and matching RELEASE at a time")
                request, generation, path = _request(message, args.owner, last_generation)
                offered_slot = (message["slot_name"], message["slot_capacity"])
                if slot_offer is not None and slot_offer != offered_slot:
                    raise ValueError("Native decoder cannot change its parent-owned slot")
                slot_offer = offered_slot
                active = (request, generation)
                last_generation = generation
                reply = _decode(message, timings)
            except Exception as error:
                # Private miss only. The parent decides formal item timing and synchronous retry.
                detail = traceback.format_exc()
                traceback.clear_frames(error.__traceback__)
                reply = dict(kind="ERROR", owner=args.owner, request_id=message.get("request_id"),
                             generation=message.get("generation"), path=message.get("path"),
                             message=str(error), traceback=detail[-12000:], postprocess=timings,
                             torch_absent=_torch_absent())
                send_message(control, reply)
                return 1  # Invalid/error work never authorizes another slot write.
            send_message(control, reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token", required=True)
    parser.add_argument("--owner", required=True)
    raise SystemExit(main(parser.parse_args()))
