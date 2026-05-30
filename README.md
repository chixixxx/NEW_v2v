# Future V2V Adaptive Timing

独立研究工程：强化学习决定 V2V 平台何时触发批量匹配，约束优化器决定订单与车辆如何匹配。

默认主环境使用 NYC TLC 黄出租数据改造出的 Manhattan taxi-zone 级 Future V2V 场景。黄出租数据只提供真实时空需求、OD 热点、行程时间和价格强度；V2V 的电量、等待窗口、报价、保留电量、SOC、电池健康和车辆供给仍由业务模型生成。

当前主线：

- 1 tick = 3 分钟。
- 主算法动作空间：立即匹配、延迟 1 步后匹配、延迟 2 步后匹配、延迟 3 步后匹配。
- 旧三动作 `WAIT / MATCH_TOP_BATCH / MATCH_FULL` 仅保留为诊断基线。
- `main`：约 4 小时决策窗口 + 42 分钟 terminal buffer。
- `smoke`：约 3 小时决策窗口 + 24 分钟 terminal buffer。
- DQN 训练支持并行 rollout workers。
- 默认训练使用 PBRS 奖励塑造和 `compact_v2v` 紧凑观测；最终评估指标不因奖励塑造改变。
- 电池健康与双边价格机制默认开启：传输效率、最低 SOC、安全可供电量、退化成本、卖方补偿和平台边际收益都会进入匹配约束与评估指标。

准备 TLC Manhattan 缓存：

```bash
python scripts/prepare_tlc_manhattan.py --month 2025-10
```

快速运行：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2 --reward-shaping pbrs --observation-profile compact_v2v
```

PBRS 对照：

```bash
python scripts/run_pbrs_ablation.py --scale smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2 --run-name smoke_pbrs_ablation_v1
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

更多说明：

- `docs/research_design_zh.md`
- `docs/environment_design_zh.md`
- `docs/runbook_zh.md`

## Episode demand scale

`scales.<scale>.total_orders` is a fixed episode order count, not an upper bound. TLC rows calibrate temporal-spatial demand, OD structure, travel time, and price strength. Sparse sampled windows are resampled with zone-pressure weights so evaluation does not mix different order scales.

`outputs/<run_name>/env_health/eval_scenario_manifest.csv` stores fixed evaluation windows. When present, `eval` reuses those scenarios so repeated evaluations compare policies on the same Manhattan demand windows.
