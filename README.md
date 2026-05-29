# Future V2V Adaptive Timing

独立的新研究工程：强化学习决定 V2V 平台何时触发批量匹配，约束优化器决定订单与车辆如何匹配。

第一版默认主环境使用 NYC TLC 黄出租数据改造出的 Manhattan taxi-zone 级 Future V2V 场景；黄出租数据只提供真实时空需求、OD 热点、行程时间和价格强度，V2V 的电量、等待窗口、报价、保留电量和车辆供给仍由业务模型生成。

准备 TLC Manhattan 缓存：

```bash
python scripts/prepare_tlc_manhattan.py --month 2025-10
```

快速运行：

```bash
python scripts/run_experiment.py --stage all --scale smoke --episodes 5
```

验证：

```bash
python -m pytest tests -q
python -m ruff check future_v2v scripts tests
```

更多说明见：

- `docs/research_design_zh.md`
- `docs/environment_design_zh.md`
- `docs/runbook_zh.md`
