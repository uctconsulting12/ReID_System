"""
dwell.py
========
Average dwell time across visible people, with a fallback to the average
dwell of the last 10 exits when nobody is visible.
"""

from __future__ import annotations

import time

from .entry_exit import EntryExitTracker


def _format_hms(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class DwellTracker:
    """Thin helper that computes the dwell snapshot from an EntryExitTracker.

    All authoritative state (entry_time, last_seen, recent_exits) lives in
    EntryExitTracker; this class is stateless other than reading from it.
    """

    @staticmethod
    def per_person_dwell(entry_time_epoch: float, now: float | None = None) -> str:
        now = time.time() if now is None else now
        return _format_hms(now - entry_time_epoch)

    @staticmethod
    def lifetime(
        tracker: EntryExitTracker,
        sid: str,
        now: float | None = None,
    ) -> str:
        """Accumulated dwell for `sid` across all visits on this camera,
        formatted as HH:MM:SS. Includes the in-progress visit if active."""
        return _format_hms(tracker.lifetime_dwell_sec(sid, now=now))

    @staticmethod
    def average(tracker: EntryExitTracker, now: float | None = None) -> str:
        """Mean dwell across every SID that has ever visited *this camera*.

        Per-SID total = completed visits on this cam + any in-progress visit
        on this cam. The mean is taken over the count of unique visiting
        SIDs, so each person contributes once regardless of how many times
        they re-entered.
        """
        now = time.time() if now is None else now
        durations = tracker.cam_visitor_dwell_sec(now=now)
        if not durations:
            return "00:00:00"
        return _format_hms(sum(durations) / len(durations))
