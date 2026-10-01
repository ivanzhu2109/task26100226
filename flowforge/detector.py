from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

WINDOW_MS = 900_000


@dataclass(frozen=True)
class Event:
    sensor_id: str
    timestamp: int
    value: float


@dataclass(frozen=True)
class Result:
    sensor_id: str
    timestamp: int
    value: float
    mean: Optional[float]
    std: Optional[float]
    is_anomaly: bool


class AnomalyDetector:
    def __init__(self, watermark: int) -> None:
        self._watermark = watermark
        self._values: Dict[str, Dict[int, float]] = {}
        self._sorted_ts: Dict[str, List[int]] = {}
        self._pending: Set[Tuple[str, int]] = set()

    @property
    def watermark(self) -> int:
        return self._watermark

    def push(self, events: Iterable[Event]) -> None:
        for e in events:
            if e.timestamp <= self._watermark:
                continue
            sensor_values = self._values.setdefault(e.sensor_id, {})
            if e.timestamp in sensor_values:
                continue
            sensor_values[e.timestamp] = e.value
            bisect.insort(self._sorted_ts.setdefault(e.sensor_id, []), e.timestamp)
            self._pending.add((e.sensor_id, e.timestamp))

    def advance_watermark(self, w: int) -> List[Result]:
        if w < self._watermark:
            raise ValueError("watermark cannot move backward")
        due = sorted(
            (ts, sid) for (sid, ts) in self._pending if ts <= w
        )
        results: List[Result] = []
        for ts, sid in due:
            results.append(self._evaluate(sid, ts))
            self._pending.discard((sid, ts))
        self._watermark = w
        self._prune(w - WINDOW_MS)
        return results

    def _evaluate(self, sensor_id: str, timestamp: int) -> Result:
        ts_list = self._sorted_ts[sensor_id]
        lo = bisect.bisect_left(ts_list, timestamp - WINDOW_MS)
        hi = bisect.bisect_left(ts_list, timestamp)
        sensor_values = self._values[sensor_id]
        window = [sensor_values[t] for t in ts_list[lo:hi]]
        value = sensor_values[timestamp]
        if len(window) < 2:
            mean = statistics.fmean(window) if window else None
            std = statistics.pstdev(window) if window else None
            return Result(sensor_id, timestamp, value, mean, std, False)
        mean = statistics.fmean(window)
        std = statistics.pstdev(window)
        is_anomaly = abs(value - mean) > 3.0 * std
        return Result(sensor_id, timestamp, value, mean, std, is_anomaly)

    def _prune(self, min_exclusive_ts: int) -> None:
        for sid, ts_list in self._sorted_ts.items():
            cut = bisect.bisect_right(ts_list, min_exclusive_ts)
            if cut:
                sensor_values = self._values[sid]
                for t in ts_list[:cut]:
                    del sensor_values[t]
                del ts_list[:cut]
