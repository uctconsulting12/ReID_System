from __future__ import annotations

import threading
import time

import cv2

from .config import CaptureConfig


class RTSPCapture:
    def __init__(self, src: str, name: str, capture_cfg: CaptureConfig):
        self.src = src
        self.name = name
        self.capture_cfg = capture_cfg
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> "RTSPCapture":
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def _open(self):
        src = self.src
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.capture_cfg.buffer_size)
        return cap

    def _loop(self):
        delay = self.capture_cfg.reconnect_initial_delay_sec
        max_delay = self.capture_cfg.reconnect_max_delay_sec
        while self.running:
            cap = self._open()
            if not cap.isOpened():
                print(f"[{self.name}] Cannot open {self.src} - retry in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 1.5, max_delay)
                continue

            print(f"[{self.name}] Connected -> {self.src}")
            delay = self.capture_cfg.reconnect_initial_delay_sec
            failures = 0
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures > self.capture_cfg.max_read_failures:
                        print(f"[{self.name}] Stream lost - reconnecting...")
                        break
                    time.sleep(0.01)
                    continue

                failures = 0
                with self.lock:
                    self.frame = frame
            cap.release()
        print(f"[{self.name}] Capture stopped.")

