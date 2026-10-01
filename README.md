# task26100226

FlowForge 事件时间异常检测内存库：15 分钟（900000 ms）滚动基线 z-score 检测。

## 用法

```python
from flowforge import AnomalyDetector, Event

detector = AnomalyDetector(watermark=0)          # 初始水位线，单位 ms
detector.push([Event("sensor-1", 1000, 42.0)])   # 任意顺序、可重复，不产生输出
results = detector.advance_watermark(2000)       # 返回 timestamp <= 2000 的结果，只输出一次
```

- 基线：同传感器、去重后、时间戳在 `[t-900000, t)` 内的事件（不含当前事件），使用总体标准差。
- 报警：样本数 >= 2 且 `|value - mean| > 3 * std`；零标准差时值不同于均值即报警。
- `timestamp <= 当前水位线` 的事件在 push 时被忽略；水位线只能前进或不变，后退抛 `ValueError`。

## 运行测试

```bash
python3 -m pytest tests/ -q
```
