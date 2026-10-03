"""Cooperative CLI cancellation. SIGKILL and machine loss cannot run cleanup."""
from contextlib import contextmanager
from functools import wraps
import signal
import threading


_cancellation = threading.local()


class TerminationRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(signum)


@contextmanager
def termination_handlers():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    old_state = getattr(_cancellation, "state", None)
    state = {"depth": 0, "pending": None}
    _cancellation.state = state
    received = False
    def handle(signum, frame):
        nonlocal received
        if not received:
            received = True
            if state["depth"]:
                state["pending"] = signum
            else:
                raise TerminationRequested(signum)
    try:
        for sig in previous: signal.signal(sig, handle)
        yield
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
        _cancellation.state = old_state


@contextmanager
def defer_cancellation():
    """Register process ownership before delivering a pending cancellation.

    Keep Python handlers active: blocking OS signals would propagate the mask
    to newly exec'd children. Nesting delivers only at the outer boundary.
    """
    state = getattr(_cancellation, "state", None)
    if state is None:
        yield
        return
    state["depth"] += 1
    try:
        yield
    finally:
        state["depth"] -= 1
        if not state["depth"] and state["pending"] is not None:
            signum = state["pending"]
            state["pending"] = None
            raise TerminationRequested(signum)


@contextmanager
def finish_cleanup():
    """Repeated Ctrl-C/TERM must not interrupt bounded cleanup or status writes."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def cancellation_exit(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with termination_handlers():
            try:
                return function(*args, **kwargs)
            except TerminationRequested as exc:
                return 128 + exc.signum
    return wrapped
