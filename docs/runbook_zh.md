# Future V2V Adaptive Timing 运行说明

## 环境准备

进入项目目录：

```bash
cd C:\Sioux\future_v2v_adaptive_timing
```

依赖：

- Python 3.10+
- numpy
- scipy
- torch
- pandas
- pyarrow
- pytest
- ruff
- tqdm

## TLC Manhattan 数据准备

把 NYC TLC 黄出租月度 parquet 和 taxi zone lookup 放到项目根目录，默认文件名：

```text
yellow_tripdata_2025-10.parquet
taxi_zone_lookup.csv
```

预处理：

```bash
python scripts/prepare_tlc_manhattan.py --month 2025-10
```

当前默认 tick 为 3 分钟，预处理输出会带 `tick3m` 后缀。

## 运行顺序

完整 smoke：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2
```

完整 main：

```bash
python scripts/run_experiment.py --stage all --scale main --rollout-workers 4 --eval-workers 4
```

分阶段：

```bash
python scripts/run_experiment.py --stage generate --scale main --eval-episodes 3
python scripts/run_experiment.py --stage train --scale main --rollout-workers 4
python scripts/run_experiment.py --stage eval --scale main --eval-workers 4
python scripts/run_experiment.py --stage report --scale main
```

## 阶段含义

- `generate`：生成环境健康报告，检查供需比、活跃订单/车辆和可行边密度。
- `train`：训练 DQN timing policy，RL 学习 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`，匹配边由约束优化器决定。
- `eval`：评估 DQN 与 fixed interval、queue threshold、deadline trigger、supply-demand pressure、short lookahead 等策略。
- `report`：汇总环境、训练和评估报告。
- `all`：顺序执行 `generate -> train -> eval -> report`。

## 默认规模

当前配置：

- tick：3 分钟。
- `smoke`：60 ticks + 8 buffer，约 3 小时 + 24 分钟，80 单，120 候选车。
- `main`：80 ticks + 14 buffer，约 4 小时 + 42 分钟，1600 单，1900 候选车，基础入池率 0.36，供给缩放 0.50。
- 服务功率：`2.7 kWh/tick`，约等于 54 kW。
- 每单候选车辆上限：20。
- 每次 dispatch 固定成本：12.0。
- top-batch 容量：活跃订单的 55%，并限制在 8 到 80 之间。

## 常调参数

集中修改：

```text
configs/default.json
```

常用参数：

- `environment.dispatch_fixed_cost`
- `environment.dispatch_capacity_ratio`
- `environment.queue_threshold_ratio`
- `environment.wait_penalty_per_order_tick`
- `environment.supply_scale`
- `environment.max_candidate_vehicles_per_order`
- `scales.main.total_orders`
- `scales.main.candidate_vehicles`
- `scales.main.vehicle_join_probability`
- `training.rollout_workers`
- `training.validation_episodes`

## 结果解读

优先看：

- `outputs/<run_name>/env_health/env_health_summary.csv`
- `outputs/<run_name>/train/train_history.csv`
- `outputs/<run_name>/train/validation_history.csv`
- `outputs/<run_name>/train/action_distribution.csv`
- `outputs/<run_name>/eval/eval_summary.csv`
- `outputs/<run_name>/eval/timing_policy_comparison.csv`
- `outputs/<run_name>/eval/action_trace_by_policy.csv`
- `outputs/<run_name>/eval/dispatch_trace_by_policy.csv`
- `outputs/<run_name>/eval/wait_tradeoff_trace.csv`
- `outputs/<run_name>/reports/run_report.md`

`eval_summary.csv` 中：

- `future_v2v_score_mean` 是主排序指标。
- `mean_batch_interval_mean` 越接近 1，越像一步一匹配。
- `timing_degenerate_risk=True` 表示策略可能退化为过于频繁匹配。
- `environment_target_band=True` 表示服务率和过期/取消压力落在建议区间。

## 验证

```bash
python -m ruff check future_v2v scripts tests
python -m pytest tests -q
```
