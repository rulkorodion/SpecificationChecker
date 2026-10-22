"""Низкоуровневые метрики памяти процесса без обязательных зависимостей."""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - модуль отсутствует на Windows.
    resource = None  # type: ignore[assignment]


def process_memory_info() -> dict[str, Any]:
    """Вернуть текущий RSS и peak RSS в MiB, если это поддерживает ОС."""

    if os.name == "nt":
        return _windows_process_memory()
    try:
        if resource is None:
            raise RuntimeError("resource module is unavailable")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_bytes = int(usage.ru_maxrss)
        # Linux сообщает KiB, macOS — bytes.
        if os.uname().sysname != "Darwin":
            peak_bytes *= 1024
        return {
            "process_rss_mib": None,
            "process_peak_rss_mib": round(peak_bytes / (1024**2), 3),
        }
    except Exception:
        return {"process_rss_mib": None, "process_peak_rss_mib": None}


def _windows_process_memory() -> dict[str, Any]:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return {
            "process_rss_mib": round(counters.WorkingSetSize / (1024**2), 3),
            "process_peak_rss_mib": round(
                counters.PeakWorkingSetSize / (1024**2), 3
            ),
        }
    except Exception:
        try:
            import psutil

            memory = psutil.Process(os.getpid()).memory_info()
            return {
                "process_rss_mib": round(memory.rss / (1024**2), 3),
                "process_peak_rss_mib": round(
                    memory.peak_wset / (1024**2), 3
                ),
            }
        except Exception:
            return {"process_rss_mib": None, "process_peak_rss_mib": None}


class NvidiaMemorySampler:
    """Периодически измерять VRAM текущего процесса через nvidia-smi."""

    def __init__(self, interval_seconds: float = 0.75) -> None:
        self.interval_seconds = interval_seconds
        self.peak_process_vram_mib: int | None = None
        self.peak_device_vram_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 3))
        return {
            "peak_process_vram_mib": self.peak_process_vram_mib,
            "peak_device_vram_mib": self.peak_device_vram_mib,
        }

    def _poll(self) -> None:
        process_id = os.getpid()
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                used = 0
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 2 and int(parts[0]) == process_id:
                        used += int(float(parts[1]))
                self.peak_process_vram_mib = max(
                    self.peak_process_vram_mib or 0, used
                )
            except Exception:
                pass
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "--id=0",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                used = int(float(completed.stdout.strip().splitlines()[0]))
                self.peak_device_vram_mib = max(
                    self.peak_device_vram_mib or 0, used
                )
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)
