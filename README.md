# Future V2V Adaptive Timing

本项目研究双边 V2V 平台中的自适应批量匹配时机：强化学习决定当前 tick 是继续等待还是触发完整匹配，约束优化器统一决定具体 CV-DV 匹配边。默认主环境使用 NYC TLC 黄出租数据改造出的 Manhattan taxi-zone 级 Future V2V 场景；出租车数据只提供时空需求、OD 热点、行程时间和价格强度，V2V 的电量、等待窗口、报价、保留电量、SOC、电池健康和车辆供给仍由业务模型生成。

当前主线：
- 1 tick = 3 分钟。
- 主算法是 `adaptive_timing_ppo`：二元动作 `WAIT / MATCH_FULL`，动态匹配间隔由连续 `WAIT` 后的下一次 `MATCH_FULL` 统计出来。
- 底层匹配边只由同一个完整正收益约束优化器决定；强化学习不直接挑订单、车辆或匹配边。
- `adaptive_interval_dqn` 保留为显式对照，不再作为默认主方法。
- 主评估表只保留固定 1/2/3/4 步完整匹配、手写强规则和 `adaptive_timing_ppo`。
- 默认训练使用有限时域 PBRS 和 `compact_v2v` 紧凑观测；奖励塑造只影响训练信号，不改变最终评估指标。
- 电池健康与双边价格机制默认开启，传输效率、最低 SOC、安全可供电量、退化成本、卖方补偿和平台边际收益都会进入匹配约束与评估指标。

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

摩擦灵敏度单独运行，不默认混入主实验：

```bash
python scripts/run_friction_sensitivity.py --scale main --run-name main_latest --eval-workers 4
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

## Episode Demand Scale

`scales.<scale>.total_orders` 是固定的单轮订单数，不是上界。TLC 行程用于校准时空需求、OD 结构、旅行时间和价格强度；稀疏采样窗口会按区域压力重采样，因此评估不会混入不同订单规模。

`outputs/<run_name>/env_health/eval_scenario_manifest.csv` 保存固定评估窗口。主规模默认固定 80 个评估场景；存在足够数量的 manifest 和环境健康摘要时，`generate` 与 `eval` 会复用同一批场景，保证策略比较使用相同 Manhattan 需求窗口。
