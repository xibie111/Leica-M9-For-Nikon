"""Private built-in queue wiring; CPU helper admission follows the GPU gate."""
from contextlib import contextmanager


def _make_lookahead():
    from .native_prefetch import NativeLookahead
    return NativeLookahead()


@contextmanager
def _native_processor_context(processor, session, plan):
    from .native_prefetch import NativePrefetchCleanupError
    # Construction can acquire ownership before it returns a manager to this frame.
    plan.cleanup_safe = False
    plan.receipt.update(enabled=True, cleanup_safe=False)
    manager = _make_lookahead()
    offered = set()
    first_request = None
    error = None

    def diagnostics():
        plan.receipt["diagnostics"] = manager.diagnostics

    def consume(path):
        request = plan.current_id
        if request not in offered:
            return None
        offered.remove(request)
        try:
            if request == first_request:
                return manager.take_first(request, path)
            return manager.take_nowait(request, path)
        finally:
            diagnostics()

    def offer(request, path):
        # False offers can still carry deferred failure cleanup to this formal item.
        offered.add(request)
        accepted = manager.offer(request, path)
        diagnostics()
        return accepted

    def run(**kwargs):
        nonlocal first_request
        public_progress = kwargs.get("progress")
        gate_seen = False
        models_seen = False

        def progress(stage, message):
            nonlocal gate_seen, models_seen, first_request
            if public_progress is not None:
                public_progress(stage, message)
            # process_nef has already validated its parameters and source. The
            # started first item owns this request even if stop arrives now.
            if stage == "models" and not models_seen:
                models_seen = True
                if plan.ordinal == 1 and not session.loaded:
                    first_request = plan.current_id
                    offered.add(first_request)
                    manager.offer_first(first_request, kwargs["input_path"])
                    diagnostics()
            if stage == "style_network" and not gate_seen:
                gate_seen = True
                plan.offer_next(offer)

        request = plan.current_id
        try:
            return processor(**dict(kwargs, session=session, _native_decode=consume, progress=progress))
        except NativePrefetchCleanupError:
            plan.cleanup_safe = False
            raise
        finally:
            try:
                # A formal failure before raw must not strand this item's READY result.
                manager.discard(request)
            except NativePrefetchCleanupError:
                plan.cleanup_safe = False
                raise
            finally:
                offered.discard(request)
                diagnostics()

    try:
        diagnostics()
        yield run
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            audit = manager.close()
            plan.receipt["close"] = audit
            plan.cleanup_safe = audit.get("cleanup_safe") is True
            if not plan.cleanup_safe:
                raise NativePrefetchCleanupError("Native lookahead close did not prove ownership cleanup safe")
        except BaseException as cleanup_error:
            if isinstance(cleanup_error, NativePrefetchCleanupError):
                plan.cleanup_safe = False
            plan.receipt["cleanup_error"] = str(cleanup_error)
            if error is None:
                raise
            error.add_note(f"Native lookahead cleanup also failed: {cleanup_error}")
        finally:
            diagnostics()
            if plan.receipt["diagnostics"].get("cleanup_safe") is True:
                plan.cleanup_safe = True
            plan.receipt["cleanup_safe"] = plan.cleanup_safe
