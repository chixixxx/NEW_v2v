# Future V2V Adaptive Timing

独立的新研究工程：强化学习决定 V2V 平台何时触发批量匹配，约束优化器决定订单与车辆如何匹配。

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

