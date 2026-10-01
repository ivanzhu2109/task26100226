# FlowForge 事件时间异常检测内存库

15 分钟（900000 ms）滚动基线的 z-score 异常检测，纯 Python 标准库实现，无外部依赖。

## 接口

- `EventTimeAnomalyDetector(watermark)`：以初始水位线 W（整数毫秒）创建检测器。
- `push(events)`：接收任意顺序的 `Event(sensor_id, timestamp, value)`，不输出结果。
  - `timestamp <= 当前 W` 的事件直接忽略；重复指纹（sensor_id + timestamp）只计一次。
- `advance_watermark(w)`：返回所有尚未输出且 `timestamp <= w` 的事件结果（按时间戳排序），随后 `W = max(W, w)`。结果只输出一次。

检测规则：对时间戳 t 的事件，基线为同传感器已接收去重后、`t - 900000 <= ts < t` 的样本（不含当前事件，异常样本仍进入后续基线）。样本不足 2 条不报警；否则 `|value - mean| > 3 * 总体标准差` 时报警；标准差为零时，值不同于均值即报警。不同传感器完全隔离。

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 使用示例

```python
from flowforge import Event, EventTimeAnomalyDetector

detector = EventTimeAnomalyDetector(watermark=0)
detector.push([Event("sensor-1", 1000, 1.0), Event("sensor-1", 2000, 2.0)])
detector.push([Event("sensor-1", 3000, 100.0)])  # 乱序/回放同样直接调用
for result in detector.advance_watermark(3000):
    if result.is_anomaly:
        print("anomaly:", result)
```
