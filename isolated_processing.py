"""Limit worker memory and elapsed time for trusted application functions.

This is not a security sandbox. The worker has no filesystem or network
restrictions, and a parser exploit could compromise the parent through the
pickled result. Handle untrusted files in a separate, restricted environment.
The 1 GiB default was not validated on a 100 MB CSV. The large stress runs
used 1.5 GiB and 2 GiB via stress_processing.py --memory-mib.
"""

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import pickle
import tempfile
import time

import psutil


@dataclass(frozen=True)
class ProcessLimits:
    memory_bytes: int = 1024 ** 3
    timeout_seconds: float = 120


@dataclass(frozen=True)
class ProcessUsage:
    elapsed_seconds: float
    peak_rss_bytes: int


def _windows_job(process, memory_bytes):
    import ctypes
    from ctypes import wintypes as w

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", w.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_processes", w.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x100 | 0x200 | 0x2000  # process and job memory, kill on close
    limits.process_memory = limits.job_memory = memory_bytes
    if (not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(job, process.sentinel)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    return lambda: kernel.CloseHandle(job)


def _peak_rss(process):
    memory = process.memory_info()
    return max(memory.rss, getattr(memory, "peak_wset", 0))


def _worker(ready, request, response, status, memory_bytes):
    import os

    try:
        if not ready.wait(30):
            return
        if os.name != "nt":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        with open(request, "rb") as handle:
            function, args = pickle.load(handle)
        result = function(*args)
        with open(response, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        status.send((True, "", _peak_rss(psutil.Process())))
    except Exception as exc:
        message = "Limite mémoire du processus atteinte." if isinstance(exc, MemoryError) else str(exc)
        status.send((False, message[:2000], 0))
    finally:
        status.close()


def run_isolated(function, *args, limits=ProcessLimits(), with_usage=False):
    """Run a trusted callable in a fresh process and enforce wall time and memory.

    Pickle is private IPC between application processes. User files are passed as
    arguments to the parser and are never used as pickle IPC payloads.
    """
    import os

    if limits.memory_bytes <= 0 or not 0 < limits.timeout_seconds < float("inf"):
        raise ValueError("Les limites de mémoire et de temps doivent être positives.")
    context = mp.get_context("spawn")
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="autoprep-") as directory:
        request = Path(directory, "request")
        response = Path(directory, "response")
        with request.open("wb") as handle:
            pickle.dump((function, args), handle, protocol=pickle.HIGHEST_PROTOCOL)
        ready = context.Event()
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(ready, request, response, sender, limits.memory_bytes))
        close_job = None
        try:
            process.start()
            sender.close()
            if os.name == "nt":
                close_job = _windows_job(process, limits.memory_bytes)
            child = psutil.Process(process.pid)
            ready.set()
            peak = 0
            while True:
                remaining = limits.timeout_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError(f"Délai du processus dépassé ({limits.timeout_seconds:g} s).")
                try:
                    peak = max(peak, _peak_rss(child))
                except psutil.NoSuchProcess:
                    pass
                if receiver.poll(min(.05, remaining)):
                    try:
                        ok, message, reported_peak = receiver.recv()
                    except EOFError as exc:
                        raise RuntimeError("Le processus s'est arrêté sans résultat (limite ou erreur native).") from exc
                    break
            if not ok:
                raise ValueError(message)
            if response.stat().st_size > limits.memory_bytes:
                raise ValueError("Résultat isolé trop volumineux pour le budget mémoire.")
            with response.open("rb") as handle:
                result = pickle.load(handle)
            usage = ProcessUsage(time.monotonic() - start, max(peak, reported_peak))
            return (result, usage) if with_usage else result
        finally:
            receiver.close()
            sender.close()
            if process.pid is not None:
                process.join(.1)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            if close_job is not None:
                close_job()
