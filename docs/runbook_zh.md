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

分阶段运行：

```bash
python scripts/run_experiment.py --stage generate --scale main --eval-episodes 8
python scripts/run_experiment.py --stage train --scale main --rollout-workers 4
python scripts/run_experiment.py --stage eval --scale main --eval-workers 4
python scripts/run_experiment.py --stage report --scale main
```

## 阶段含义

- `generate`：生成固定评估 manifest 和环境健康报告，检查供需比、活跃订单/车辆、可行边密度、健康可供电量和 SOC 安全线绑定率。
- `train`：训练 DQN timing policy。RL 学习 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`，匹配边由约束优化器决定。
- `eval`：评估 DQN 与 fixed interval、queue threshold、deadline trigger、supply-demand pressure、short lookahead 等策略。
- `report`：汇总环境、训练和评估报告。
- `all`：顺序执行 `generate -> train -> eval -> report`。

## 默认规模

- `smoke`: 60 ticks + 8 buffer，80 单，120 候选车辆。
- `main`: 80 ticks + 14 buffer，1600 单，1900 候选车辆，基础入池率 0.36，供给缩放 0.55。
- 每单候选车辆上限：20。
- `MATCH_TOP_BATCH` 固定成本：18.0。
- `MATCH_FULL` 固定成本：38.5。
- 连续派单惩罚：最近 1 tick 内已经派单时，额外扣 10.0。
- top-batch 容量：活跃订单的 55%，并限制在 8 到 80 之间。

## 电池健康与价格参数

集中配置在 `configs/default.json`：

```text
environment.battery_health.donor_min_soc_ratio
environment.battery_health.transfer_efficiency
environment.battery_health.degradation_cost_per_kwh
environment.battery_health.max_discharge_power_kw
```

默认值：

```text
donor_min_soc_ratio = 0.25
transfer_efficiency = 0.90
degradation_cost_per_kwh = 0.08
max_discharge_power_kw = 50.0
```

订单 `demand_kwh` 是 CV 实际获得的有效电量。DV 输出电量、传输损耗、卖方补偿、退化成本和平台边际收益会在 dispatch trace 和 eval summary 中单独输出。

## 结果解读

优先看：

- `outputs/<run_name>/env_health/env_health_summary.csv`
- `outputs/<run_name>/eval/eval_summary.csv`
- `outputs/<run_name>/eval/timing_policy_comparison.csv`
- `outputs/<run_name>/eval/paired_policy_delta_summary.csv`
- `outputs/<run_name>/eval/environment_acceptance_summary.csv`
- `outputs/<run_name>/eval/dispatch_trace_by_policy.csv`
- `outputs/<run_name>/eval/wait_tradeoff_trace.csv`

关键指标：

- `future_v2v_score_mean`：主排序指标。
- `platform_profit_mean`：扣除 dispatch 固定成本和连续派单惩罚后的平台利润。
- `platform_margin_per_served_order`：不含 dispatch 成本的单服务边际收益。
- `energy_loss_rate`：V2V 传输损耗率，默认应约 10%。
- `seller_compensation_share`：卖方补偿占买方支付比例。
- `donor_soc_violation_count_mean`：必须为 0。
- `battery_health_rejection_count_mean`：电池健康约束造成的候选拒绝规模。
- `mean_batch_interval_mean`：越接近 1，越像一步一匹配。
- `capacity_bind_rate`：top-batch 容量是否真的起到截断作用。
- `paired_policy_delta_summary.csv`：优先看同场景 score delta 和 win rate，而不是只看 raw std。
- `dynamic_timing_ready`：策略差异、batch interval、fixed_2 服务率下降和 top-batch 可用性是否同时满足 stress benchmark 条件。

## 验证

```bash
python -m ruff check future_v2v scripts tests
python -m pytest tests -q
```
