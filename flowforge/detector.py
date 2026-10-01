"""FlowForge 事件时间异常检测内存库：15 分钟滚动基线 z-score 检测。"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

WINDOW_MS = 900_000  # 15 分钟滚动基线窗口
Z_THRESHOLD = 3.0
MIN_SAMPLES = 2


@dataclass(frozen=True)
class Event:
    sensor_id: str
    timestamp: int  # 整数毫秒
    value: float  # float64


@dataclass(frozen=True)
class AnomalyResult:
    sensor_id: str
    timestamp: int
    value: float
    mean: float
    stddev: float
    is_anomaly: bool


class EventTimeAnomalyDetector:
    """按传感器隔离的事件时间 z-score 异常检测器。"""

    def __init__(self, watermark: int) -> None:
        self._watermark = watermark
        # 每个传感器：已排序时间戳列表 + 时间戳->值（去重后的全部已接收样本）
        self._times: Dict[str, List[int]] = {}
        self._values: Dict[str, Dict[int, float]] = {}
        # 每个传感器：尚未输出结果的事件（时间戳->值）
        self._pending: Dict[str, Dict[int, float]] = {}

    @property
    def watermark(self) -> int:
        return self._watermark

    def push(self, events: Iterable[Event]) -> None:
        for event in events:
            self._accept(event.sensor_id, int(event.timestamp), float(event.value))

    def _accept(self, sensor_id: str, timestamp: int, value: float) -> None:
        if timestamp <= self._watermark:
            return  # 已过水位线，直接忽略
        values = self._values.get(sensor_id)
        if values is None:
            values = {}
            self._values[sensor_id] = values
            self._times[sensor_id] = []
            self._pending[sensor_id] = {}
        if timestamp in values:
            return  # 重复指纹，只贡献一次
        bisect.insort(self._times[sensor_id], timestamp)
        values[timestamp] = value
        self._pending[sensor_id][timestamp] = value

    def advance_watermark(self, w: int) -> List[AnomalyResult]:
        due: List[Tuple[str, int, float]] = []
        for sensor_id, pending in self._pending.items():
            ready = [ts for ts in pending if ts <= w]
            for ts in ready:
                due.append((sensor_id, ts, pending.pop(ts)))
        due.sort(key=lambda item: (item[1], item[0]))
        results = [
            self._evaluate(sensor_id, ts, value) for sensor_id, ts, value in due
        ]
        if w > self._watermark:
            self._watermark = w
        return results

    def _evaluate(self, sensor_id: str, timestamp: int, value: float) -> AnomalyResult:
        times = self._times[sensor_id]
        values = self._values[sensor_id]
        lo = bisect.bisect_left(times, timestamp - WINDOW_MS)
        hi = bisect.bisect_left(times, timestamp)
        baseline = [values[times[i]] for i in range(lo, hi)]
        n = len(baseline)
        if n < MIN_SAMPLES:
            mean = math.fsum(baseline) / n if n else 0.0
            return AnomalyResult(sensor_id, timestamp, value, mean, 0.0, False)
        mean = math.fsum(baseline) / n
        variance = math.fsum((x - mean) ** 2 for x in baseline) / n
        stddev = math.sqrt(variance)
        if stddev == 0.0:
            is_anomaly = value != mean
        else:
            is_anomaly = abs(value - mean) > Z_THRESHOLD * stddev
        return AnomalyResult(sensor_id, timestamp, value, mean, stddev, is_anomaly)
