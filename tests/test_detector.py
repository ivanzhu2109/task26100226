import math
import random
import unittest

from flowforge import Event, EventTimeAnomalyDetector
from flowforge.detector import WINDOW_MS


def as_key(result):
    return (
        result.sensor_id,
        result.timestamp,
        result.value,
        result.mean,
        result.stddev,
        result.is_anomaly,
    )


class EquivalenceTest(unittest.TestCase):
    """同一事件集合：任意批次/乱序/重复/推进粒度产生相同结果集合。"""

    def make_events(self):
        rng = random.Random(42)
        events = []
        ts = 1_000
        for i in range(60):
            ts += rng.randint(500, 5_000)
            sensor = f"s{i % 3}"
            value = 10.0 + rng.uniform(-1.0, 1.0)
            if i in (20, 45):
                value = 500.0  # 注入异常
            events.append(Event(sensor, ts, value))
        return events

    def feed(self, events, batch_size, shuffle_seed, advance_step):
        detector = EventTimeAnomalyDetector(0)
        rng = random.Random(shuffle_seed)
        events = list(events)
        rng.shuffle(events)
        # 随机复制部分事件（重复指纹，value 相同）
        duplicated = events + rng.sample(events, k=len(events) // 4)
        rng.shuffle(duplicated)
        results = []
        max_ts = max(e.timestamp for e in events)
        w = 0
        for i in range(0, len(duplicated), batch_size):
            detector.push(duplicated[i : i + batch_size])
            remaining = [e.timestamp for e in duplicated[i + batch_size :]]
            cap = (min(remaining) - 1) if remaining else max_ts
            w = min(w + advance_step, cap)
            results.extend(detector.advance_watermark(w))
        results.extend(detector.advance_watermark(max_ts))
        return {as_key(r) for r in results}

    def test_batch_shuffle_duplicate_advance_equivalence(self):
        events = self.make_events()
        reference = self.feed(events, batch_size=7, shuffle_seed=1, advance_step=60_000)
        variants = [
            self.feed(events, batch_size=1, shuffle_seed=2, advance_step=60_000),
            self.feed(events, batch_size=13, shuffle_seed=3, advance_step=10_000),
            self.feed(events, batch_size=1000, shuffle_seed=4, advance_step=10**9),
            self.feed(events, batch_size=5, shuffle_seed=1, advance_step=1),
        ]
        for i, got in enumerate(variants):
            self.assertEqual(reference, got, f"variant {i} mismatch")

    def test_stepwise_vs_single_jump(self):
        events = self.make_events()
        max_ts = max(e.timestamp for e in events)

        stepwise = EventTimeAnomalyDetector(0)
        stepwise.push(events)
        step_results = []
        w = 0
        while w < max_ts:
            w += 30_000
            step_results.extend(stepwise.advance_watermark(w))

        jump = EventTimeAnomalyDetector(0)
        jump.push(events)
        jump_results = jump.advance_watermark(max_ts + 1)

        self.assertEqual(
            {as_key(r) for r in step_results}, {as_key(r) for r in jump_results}
        )


class WindowBoundaryTest(unittest.TestCase):
    def test_left_boundary_inclusive(self):
        # t-WINDOW_MS 处的样本必须计入基线
        d = EventTimeAnomalyDetector(0)
        t = 10_000_000
        d.push(
            [
                Event("s", t - WINDOW_MS, 100.0),
                Event("s", t - WINDOW_MS + 1, 100.0),
                Event("s", t, 100.0),
            ]
        )
        result = [r for r in d.advance_watermark(t) if r.timestamp == t][0]
        self.assertAlmostEqual(result.mean, 100.0)
        self.assertAlmostEqual(result.stddev, 0.0)
        self.assertFalse(result.is_anomaly)

    def test_just_outside_left_boundary_excluded(self):
        # t-WINDOW_MS-1 处的极端样本不得计入基线
        d = EventTimeAnomalyDetector(0)
        t = 10_000_000
        d.push(
            [
                Event("s", t - WINDOW_MS - 1, 1e9),
                Event("s", t - 2, 10.0),
                Event("s", t - 1, 10.0),
                Event("s", t, 10.0),
            ]
        )
        result = [r for r in d.advance_watermark(t) if r.timestamp == t][0]
        self.assertAlmostEqual(result.mean, 10.0)
        self.assertAlmostEqual(result.stddev, 0.0)
        self.assertFalse(result.is_anomaly)

    def test_current_event_excluded_from_baseline(self):
        # 基线不含当前事件：相同值序列 stddev=0 且不报警
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", 1_000, 5.0), Event("s", 2_000, 5.0), Event("s", 3_000, 5.0)])
        results = d.advance_watermark(3_000)
        self.assertFalse(results[2].is_anomaly)
        self.assertAlmostEqual(results[2].stddev, 0.0)

    def test_min_samples_required(self):
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", 1_000, 1e9)])
        (r1,) = d.advance_watermark(1_000)
        self.assertFalse(r1.is_anomaly)  # 0 条基线样本
        d.push([Event("s", 2_000, -1e9)])
        (r2,) = d.advance_watermark(2_000)
        self.assertFalse(r2.is_anomaly)  # 仅 1 条基线样本


class WatermarkTest(unittest.TestCase):
    def test_late_events_ignored(self):
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", 1_000, 1.0), Event("s", 2_000, 2.0)])
        finalized = d.advance_watermark(2_000)
        # 越过水位线的事件被忽略，不改状态、不出结果
        d.push([Event("s", 1_500, 999.0), Event("s", 2_000, 999.0)])
        self.assertEqual(d.advance_watermark(2_000), [])
        d.push([Event("s", 3_000, 3.0)])
        (r,) = d.advance_watermark(3_000)
        # 基线只有 1.0 和 2.0，迟到事件未污染
        self.assertAlmostEqual(r.mean, 1.5)
        self.assertAlmostEqual(r.stddev, 0.5)
        self.assertEqual(len(finalized), 2)

    def test_watermark_monotonic(self):
        d = EventTimeAnomalyDetector(100)
        self.assertEqual(d.advance_watermark(50), [])  # 倒退视为保持不变
        self.assertEqual(d.watermark, 100)
        d.push([Event("s", 80, 1.0)])  # 已低于水位线，忽略
        d.push([Event("s", 150, 2.0)])
        self.assertEqual(len(d.advance_watermark(150)), 1)
        self.assertEqual(d.watermark, 150)

    def test_results_output_exactly_once(self):
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", 1_000, 1.0), Event("s", 1_000, 1.0)])  # 重复指纹
        first = d.advance_watermark(1_000)
        self.assertEqual(len(first), 1)
        self.assertEqual(d.advance_watermark(1_000), [])
        self.assertEqual(d.advance_watermark(2_000), [])

    def test_sensor_isolation(self):
        d = EventTimeAnomalyDetector(0)
        d.push(
            [
                Event("a", 1_000, 1.0),
                Event("a", 2_000, 1.0),
                Event("b", 1_500, 1e9),  # 不影响 a 的基线
                Event("a", 3_000, 1.0),
            ]
        )
        results = {
            r.sensor_id: r
            for r in d.advance_watermark(3_000)
            if r.timestamp == 3_000 or r.sensor_id == "b"
        }
        ra = results["a"]
        self.assertAlmostEqual(ra.mean, 1.0)
        self.assertAlmostEqual(ra.stddev, 0.0)
        self.assertFalse(ra.is_anomaly)


class StatisticsTest(unittest.TestCase):
    def test_constant_window_zero_stddev(self):
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", t, 7.5) for t in (1_000, 2_000, 3_000)])
        d.push([Event("s", 4_000, 7.5)])
        (r_same,) = d.advance_watermark(4_000)[-1:]
        self.assertAlmostEqual(r_same.stddev, 0.0)
        self.assertFalse(r_same.is_anomaly)
        d.push([Event("s", 5_000, 7.5000001)])
        (r_diff,) = d.advance_watermark(5_000)[-1:]
        self.assertAlmostEqual(r_diff.stddev, 0.0)
        self.assertTrue(r_diff.is_anomaly)  # 零标准差下不同于均值即报警

    def test_population_stddev_and_threshold(self):
        # 基线 [0, 10]：均值 5，总体标准差 5，3σ = 15
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", 1_000, 0.0), Event("s", 2_000, 10.0)])
        d.push([Event("s", 3_000, 20.0)])  # |20-5| = 15，不严格大于，不报警
        (r_border,) = d.advance_watermark(3_000)[-1:]
        self.assertAlmostEqual(r_border.mean, 5.0)
        self.assertAlmostEqual(r_border.stddev, 5.0)
        self.assertFalse(r_border.is_anomaly)
        d.push([Event("s2", 4_000, 0.0), Event("s2", 5_000, 10.0)])
        d.push([Event("s2", 6_000, 20.0000001)])  # 严格大于 3σ，报警
        r_over = [r for r in d.advance_watermark(6_000) if r.sensor_id == "s2" and r.timestamp == 6_000][0]
        self.assertTrue(r_over.is_anomaly)

    def test_recovery_after_extreme_sample_expires(self):
        # 极端样本在窗口内时抬高均值/标准差；过期后窗口恢复为常量
        d = EventTimeAnomalyDetector(0)
        base = 10_000_000
        d.push([Event("s", base, 1e6)])  # 极端样本
        d.push([Event("s", base + 1_000, 1.0), Event("s", base + 2_000, 1.0)])
        d.push([Event("s", base + 3_000, 1.0)])
        (r_in,) = d.advance_watermark(base + 3_000)[-1:]
        self.assertFalse(r_in.is_anomaly)  # 极端样本把 3σ 抬得很高
        self.assertGreater(r_in.stddev, 1e5)
        # 极端样本过期后：窗口只剩常量 1.0
        t2 = base + WINDOW_MS + 1_000
        d.push([Event("s", t2 - 2_000, 1.0), Event("s", t2 - 1_000, 1.0)])
        d.push([Event("s", t2, 1.0)])
        (r_out,) = d.advance_watermark(t2)[-1:]
        self.assertAlmostEqual(r_out.mean, 1.0)
        self.assertAlmostEqual(r_out.stddev, 0.0)
        self.assertFalse(r_out.is_anomaly)
        # 恢复后的常量窗口对微小偏离敏感
        d.push([Event("s", t2 + 1_000, 1.0), Event("s", t2 + 2_000, 1.0001)])
        (r_dev,) = d.advance_watermark(t2 + 2_000)[-1:]
        self.assertTrue(r_dev.is_anomaly)

    def test_anomalous_sample_enters_future_baseline(self):
        # 异常样本仍进入后续事件的基线
        d = EventTimeAnomalyDetector(0)
        d.push([Event("s", 1_000, 0.0), Event("s", 2_000, 0.0)])
        d.push([Event("s", 3_000, 100.0)])  # 对 [0,0] 是异常
        (r_anom,) = d.advance_watermark(3_000)[-1:]
        self.assertTrue(r_anom.is_anomaly)
        d.push([Event("s", 4_000, 0.0)])
        (r_next,) = d.advance_watermark(4_000)[-1:]
        # 基线为 [0, 0, 100]，均值 100/3
        self.assertAlmostEqual(r_next.mean, 100.0 / 3.0)
        expected_std = math.sqrt(((0 - 100 / 3) ** 2 * 2 + (100 - 100 / 3) ** 2) / 3)
        self.assertAlmostEqual(r_next.stddev, expected_std)


if __name__ == "__main__":
    unittest.main()
