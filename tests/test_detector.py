import random
import statistics

import pytest

from flowforge import AnomalyDetector, Event
from flowforge.detector import WINDOW_MS


def run_script(detector, steps):
    results = []
    for action, payload in steps:
        if action == "push":
            detector.push(payload)
        else:
            results.extend(detector.advance_watermark(payload))
    return results


def as_tuples(results):
    return sorted(
        (r.sensor_id, r.timestamp, r.value, r.mean, r.std, r.is_anomaly)
        for r in results
    )


def make_stream():
    rng = random.Random(20261002)
    events = []
    for sensor in ("alpha", "beta"):
        base = 10.0 if sensor == "alpha" else 50.0
        for i in range(60):
            ts = 1_000_000 + i * 60_000
            value = base + rng.uniform(-0.5, 0.5)
            if i in (25, 47):
                value = base + 100.0
            events.append(Event(sensor, ts, value))
    return events


def test_batching_shuffling_and_watermark_granularity_are_equivalent():
    events = make_stream()
    max_ts = max(e.timestamp for e in events)

    reference = as_tuples(run_script(AnomalyDetector(0), [
        ("push", events),
        ("advance", max_ts),
    ]))

    ordered = sorted(events, key=lambda e: e.timestamp)
    chunk = (len(ordered) + 4) // 5
    groups = [ordered[i * chunk:(i + 1) * chunk] for i in range(5)]
    group_max = [max(e.timestamp for e in g) for g in groups]

    stepwise = []
    for group, mark in zip(groups, group_max):
        shuffled = group[:]
        random.Random(len(group)).shuffle(shuffled)
        stepwise.append(("push", shuffled + shuffled[:3]))
        stepwise.append(("advance", mark))
    stepwise.append(("advance", max_ts))
    assert as_tuples(run_script(AnomalyDetector(0), stepwise)) == reference

    interleaved = []
    for pair in zip(events[::2], events[1::2]):
        interleaved.extend([pair[1], pair[0], pair[0]])
    coarse = [
        ("push", interleaved),
        ("advance", group_max[2]),
        ("advance", max_ts),
    ]
    assert as_tuples(run_script(AnomalyDetector(0), coarse)) == reference


def test_window_left_boundary_inclusive_and_expiry():
    d = AnomalyDetector(0)
    d.push([
        Event("s", 100_000, 2.0),
        Event("s", 200_000, 4.0),
        Event("s", 100_000 + WINDOW_MS, 3.0),
        Event("s", 100_000 + WINDOW_MS + 1, 3.0),
    ])
    results = {r.timestamp: r for r in d.advance_watermark(1_000_001)}

    boundary = results[100_000 + WINDOW_MS]
    assert boundary.mean == 3.0
    assert boundary.std == 1.0
    assert not boundary.is_anomaly

    expired = results[100_000 + WINDOW_MS + 1]
    assert expired.mean == 3.5
    assert expired.std == 0.5
    assert not expired.is_anomaly


def test_current_event_excluded_from_own_baseline():
    d = AnomalyDetector(0)
    d.push([
        Event("s", 100, 0.0),
        Event("s", 200, 0.0),
        Event("s", 300, 100.0),
    ])
    results = {r.timestamp: r for r in d.advance_watermark(300)}
    spike = results[300]
    assert spike.mean == 0.0
    assert spike.std == 0.0
    assert spike.is_anomaly


def test_duplicate_fingerprint_counts_once():
    d = AnomalyDetector(0)
    d.push([
        Event("s", 100, 0.0),
        Event("s", 100, 0.0),
        Event("s", 100, 0.0),
        Event("s", 200, 30.0),
    ])
    results = d.advance_watermark(200)
    assert [r.timestamp for r in results] == [100, 200]
    spike = results[1]
    assert spike.mean == 0.0
    assert not spike.is_anomaly


def test_events_behind_watermark_are_ignored():
    d = AnomalyDetector(0)
    d.push([Event("s", 1_000, 1.0), Event("s", 2_000, 1.0)])
    first = d.advance_watermark(2_000)
    assert len(first) == 2

    d.push([Event("s", 500, 999.0), Event("s", 1_500, 999.0)])
    d.push([Event("s", 3_000, 1.0)])
    second = d.advance_watermark(3_000)
    assert [r.timestamp for r in second] == [3_000]
    assert second[0].mean == 1.0
    assert second[0].std == 0.0

    assert d.advance_watermark(3_000) == []
    assert d.advance_watermark(4_000) == []


def test_watermark_cannot_move_backward():
    d = AnomalyDetector(100)
    d.advance_watermark(200)
    with pytest.raises(ValueError):
        d.advance_watermark(199)
    assert d.watermark == 200


def test_constant_window_zero_std():
    d = AnomalyDetector(0)
    d.push([Event("s", ts, 7.0) for ts in range(1_000, 6_000, 1_000)])
    d.push([Event("s", 6_000, 7.0), Event("s", 7_000, 8.0)])
    results = {r.timestamp: r for r in d.advance_watermark(7_000)}

    same = results[6_000]
    assert same.std == 0.0
    assert same.mean == 7.0
    assert not same.is_anomaly

    different = results[7_000]
    assert different.std == 0.0
    assert different.is_anomaly


def test_std_recovers_after_extreme_samples_expire():
    d = AnomalyDetector(0)
    d.push([
        Event("s", 1_000, 0.0),
        Event("s", 2_000, 1_000.0),
        Event("s", 3_000, -5_000.0),
    ])
    hot = {r.timestamp: r for r in d.advance_watermark(3_000)}[3_000]
    assert hot.std == pytest.approx(statistics.pstdev([0.0, 1_000.0]))
    assert hot.is_anomaly

    calm_ts = [3_000 + WINDOW_MS + 1_000 + i * 1_000 for i in range(5)]
    d.push([Event("s", ts, 5.0) for ts in calm_ts])
    calm = {r.timestamp: r for r in d.advance_watermark(calm_ts[-1])}
    last = calm[calm_ts[-1]]
    assert last.mean == 5.0
    assert last.std == 0.0
    assert not last.is_anomaly


def test_tiny_fluctuation_window_std_matches_pstdev():
    d = AnomalyDetector(0)
    values = [5.0, 5.001, 5.0, 5.001, 5.0]
    d.push([Event("s", 10_000 + i * 1_000, v) for i, v in enumerate(values)])
    d.push([Event("s", 15_000, 5.0), Event("s", 16_000, 6.0)])
    results = {r.timestamp: r for r in d.advance_watermark(16_000)}

    expected_std = statistics.pstdev(values)
    in_line = results[15_000]
    assert in_line.std == pytest.approx(expected_std)
    assert not in_line.is_anomaly

    out_of_line = results[16_000]
    assert out_of_line.std == pytest.approx(statistics.pstdev(values + [5.0]))
    assert out_of_line.is_anomaly


def test_sensors_are_isolated():
    d = AnomalyDetector(0)
    d.push([
        Event("a", 1_000, 1.0),
        Event("b", 1_000, 9.0),
        Event("a", 2_000, 1.0),
        Event("b", 2_000, 9.0),
        Event("a", 3_000, 500.0),
        Event("b", 3_000, 9.0),
    ])
    results = {(r.sensor_id, r.timestamp): r for r in d.advance_watermark(3_000)}
    assert results[("a", 3_000)].is_anomaly
    b_result = results[("b", 3_000)]
    assert b_result.mean == 9.0
    assert b_result.std == 0.0
    assert not b_result.is_anomaly


def test_results_emitted_exactly_once():
    d = AnomalyDetector(0)
    d.push([Event("s", 100, 1.0)])
    assert len(d.advance_watermark(100)) == 1
    assert d.advance_watermark(10_000) == []
