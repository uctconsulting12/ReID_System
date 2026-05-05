from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("hls.ffmpeg")


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class FFmpegHLSWriter:
    """Single ffmpeg subprocess that ingests raw BGR frames over stdin and
    writes a rolling HLS playlist + segments to ``out_dir``.

    The subprocess starts lazily on the first ``write()`` call so that the
    output resolution can be pinned to whatever the source actually delivered.
    """

    def __init__(
        self,
        out_dir: Path,
        fps: int = 25,
        use_nvenc: bool = False,
        hls_time: int = 2,
        hls_list_size: int = 5,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.fps = max(1, int(fps))
        self.use_nvenc = bool(use_nvenc)
        self.hls_time = int(hls_time)
        self.hls_list_size = int(hls_list_size)

        self.proc: subprocess.Popen | None = None
        self.size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._frames_written = 0

    @property
    def playlist_path(self) -> Path:
        return self.out_dir / "stream.m3u8"

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            return
        if frame is None or frame.size == 0:
            return

        with self._lock:
            if self.proc is None:
                self._start(frame.shape[1], frame.shape[0])
            if self.proc is None or self.proc.stdin is None:
                return
            try:
                self.proc.stdin.write(frame.tobytes())
                self._frames_written += 1
            except (BrokenPipeError, OSError) as exc:
                logger.warning("[%s] ffmpeg pipe broken (%s); shutting writer down",
                               self.out_dir.name, exc)
                self._teardown_locked()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._teardown_locked()

    def _teardown_locked(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
        finally:
            self.proc = None

    def _start(self, width: int, height: int) -> None:
        if not _ffmpeg_available():
            raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to enable HLS output")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        # purge any leftover segments from a previous run with the same id
        for stale in self.out_dir.glob("*.ts"):
            try: stale.unlink()
            except Exception: pass
        if self.playlist_path.exists():
            try: self.playlist_path.unlink()
            except Exception: pass

        seg_pattern = str(self.out_dir / "seg_%05d.ts")
        gop = self.fps * 2

        # NOTE: matches the ffmpeg_test README — `use_nvenc` is accepted in the
        # API but the encoder always runs as software libx264 here. Switch to
        # h264_nvenc only if the operator explicitly asks for it.
        if self.use_nvenc:
            video_args = [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-tune", "ll",
                "-rc", "cbr_ld_hq",
            ]
        else:
            video_args = [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
            ]

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",
            *video_args,
            "-pix_fmt", "yuv420p",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time", str(self.hls_time),
            "-hls_list_size", str(self.hls_list_size),
            "-hls_flags", "delete_segments+append_list+omit_endlist",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", seg_pattern,
            str(self.playlist_path),
        ]

        logger.info("[%s] starting ffmpeg %dx%d @ %d fps -> %s",
                    self.out_dir.name, width, height, self.fps, self.playlist_path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.size = (width, height)
