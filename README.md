# Future V2V Adaptive Timing

独立的新研究工程：强化学习决定 V2V 平台何时触发批量匹配，约束优化器决定订单与车辆如何匹配。

默认主环境使用 NYC TLC 黄出租数据改造出的 Manhattan taxi-zone 级 Future V2V 场景。黄出租数据只提供真实时空需求、OD 热点、行程时间和价格强度；V2V 的电量、等待窗口、报价、保留电量和车辆供给仍由业务模型生成。

当前默认：

- 1 tick = 3 分钟。
- `main`：4 小时决策窗口 + 42 分钟 terminal buffer。
- `smoke`：3 小时决策窗口 + 24 分钟 terminal buffer。
- DQN 训练支持并行 rollout workers。

准备 TLC Manhattan 缓存：

```bash
python scripts/prepare_tlc_manhattan.py --month 2025-10
```

快速运行：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2
```

主实验：

```bash
python scripts/run_experiment.py --stage all --scale main --rollout-workers 4 --eval-workers 4
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

## Episode demand scale

`scales.<scale>.total_orders` is a fixed episode order count, not an upper bound. TLC rows calibrate temporal-spatial demand, OD structure, travel time, and price strength. Sparse sampled windows are resampled with zone-pressure weights so evaluation does not mix different order scales.

`outputs/<run_name>/env_health/eval_scenario_manifest.csv` stores fixed evaluation windows. When present, `eval` reuses those scenarios so repeated evaluations compare policies on the same Manhattan demand windows.
