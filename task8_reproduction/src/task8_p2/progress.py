from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import os
import threading
import time


def log(message):
    worker = os.environ.get("TASK8_WORKER", "main")
    print(f"{datetime.now().isoformat(timespec='seconds')} [{worker}] {message}", flush=True)


class Progress:
    def __init__(self, label, total, interval=10):
        self.label, self.total, self.interval = label, int(total), float(interval)
        self.start = time.monotonic()
        self.last = 0.0
        self.update(0, force=True)

    def update(self, done, detail="", force=False):
        now = time.monotonic()
        if not force and done != self.total and now - self.last < self.interval:
            return
        elapsed = now - self.start
        fraction = done / max(self.total, 1)
        filled = min(20, int(20 * fraction))
        bar = "#" * filled + "-" * (20 - filled)
        eta = f"{elapsed * (self.total - done) / done:.0f}s" if done else "?"
        log(f"{self.label} [{bar}] {done}/{self.total} ({100*fraction:.1f}%) elapsed={elapsed:.0f}s ETA={eta} {detail}")
        self.last = now


@contextmanager
def heartbeat(label, seconds=20):
    stop = threading.Event()
    start = time.monotonic()
    log(f"START {label}")

    def beat():
        while not stop.wait(seconds):
            log(f"RUNNING {label} elapsed={time.monotonic()-start:.0f}s")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


class ProgressLoader:
    def __init__(self, loader, label, interval=10):
        self.loader, self.label, self.interval = loader, label, interval

    def __iter__(self):
        progress = Progress(self.label, len(self.loader), self.interval)
        for index, batch in enumerate(self.loader):
            yield batch
            progress.update(index + 1)

    def __len__(self):
        return len(self.loader)
