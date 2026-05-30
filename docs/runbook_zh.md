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
- `eval`：评估 DQN 与 fixed interval、queue threshold、deadline trigger、supply-demand pressure、short lookahead 等策略，并额外运行 friction sensitivity。
- `report`：汇总环境、训练和评估报告。
- `all`：顺序执行 `generate -> train -> eval -> report`。

## 默认规模

- `smoke`: 60 ticks + 8 buffer，80 单，120 候选车辆。
- `main`: 80 ticks + 14 buffer，1600 单，1900 候选车辆，基础入池率 0.36，供给缩放 0.55。
- 每单候选车辆上限：20。
- top-batch 容量：活跃订单的 55%，并限制在 8 到 80 之间。

## Dispatch Friction 参数

集中配置在 `configs/default.json` 的 `environment.dispatch_friction`：

```text
enabled = true
setup_cost = 18.0
pair_coordination_cost = 0.75
full_mode_extra_pair_cost = 0.20
refresh_cost = 16.0
refresh_decay_ticks = 1.5
```

评估会额外输出 `friction_sensitivity_summary.csv`，包括：

- `decomposed_transaction_cost`：默认正式口径。
- `no_refresh_friction`：去掉平滑刷新摩擦。
- `common_fixed_cost`：去掉 full mode 额外 pair cost。
- `no_dispatch_friction`：完全去掉 dispatch friction，仅作反事实诊断。

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
- `outputs/<run_name>/eval/friction_sensitivity_summary.csv`
- `outputs/<run_name>/eval/environment_acceptance_summary.csv`
- `outputs/<run_name>/eval/dispatch_trace_by_policy.csv`
- `outputs/<run_name>/eval/wait_tradeoff_trace.csv`

关键指标：

- `future_v2v_score_mean`：主排序指标。
- `platform_profit_mean`：扣除 dispatch friction 后的平台利润。
- `dispatch_friction_cost_mean`：交易摩擦总成本。
- `friction_share_of_gross_profit_mean`：交易摩擦占 dispatch gross profit 的比例。
- `platform_margin_per_served_order`：不含 dispatch friction 的单服务边际收益。
- `energy_loss_rate`：V2V 传输损耗率，默认应约 10%。
- `donor_soc_violation_count_mean`：必须为 0。
- `mean_batch_interval_mean`：越接近 1，越像一步一匹配。
- `capacity_bind_rate`：top-batch 容量是否真的起到截断作用。
- `friction_robust_ready`：动态时机优势是否在 `no_refresh_friction` 下仍然保留。
- `friction_sensitive_risk`：若为 True，说明环境差异仍过度依赖 dispatch friction。

## 验证

```bash
python -m ruff check future_v2v scripts tests
python -m pytest tests -q
```
## DQN 训练与验证更新

DQN 观测已经包含 dispatch 间隔、top/full 预估交易摩擦、time-of-day bucket、临期订单占比和服务风险。训练 reward 使用 service-risk delta shaping：奖励动作后服务风险潜势下降，惩罚风险潜势上升，避免 WAIT 被 absolute service gap 持续压低。

WAIT 动作会根据 one-step wait tradeoff 获得轻量 opportunity bonus。若 WAIT 后候选边收益增量大于过期、取消、等待和刷新摩擦风险，则奖励 WAIT；若临期损失大，则不奖励。

`teacher_prefill_episodes` 会混入同一动作空间下的 `WaitOpportunityTeacherPolicy`、`fixed_2_tick_full_match`、deadline rescue、supply-demand pressure 和 short-lookahead 样本。teacher 不直接输出匹配边，仍只输出 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`。

checkpoint selection 使用分层 validation：`training.validation_time_buckets` 默认覆盖 `morning_peak / midday / evening_peak / off_peak`。`train/validation_history.csv` 会输出各 bucket 的 score mean、`validation_bucket_min_score_mean`、`validation_off_peak_floor_penalty` 和 `checkpoint_selection_score`。
## 最小验证闸门

每次修改训练信号后，先用三层小成本验证，不要直接跑完整 main：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2 --run-name smoke_gate_v1
python scripts/summarize_timing_run.py --run-name smoke_gate_v1 --scale smoke
```

```bash
python scripts/run_experiment.py --stage all --scale main --episodes 40 --eval-episodes 8 --rollout-workers 4 --eval-workers 4 --run-name mini_main_gate_v1
python scripts/summarize_timing_run.py --run-name mini_main_gate_v1 --scale main
```

```bash
python scripts/run_experiment.py --stage all --scale main --episodes 80 --eval-episodes 16 --rollout-workers 4 --eval-workers 4 --run-name candidate_main_v1
python scripts/summarize_timing_run.py --run-name candidate_main_v1 --scale main
```

`summarize_timing_run.py` 只读取已有 CSV，不重新仿真。重点看 `dqn_vs_fixed1_delta`、`dqn_gap_to_best`、`dqn_wait_rate`、`dqn_full_match_rate`、`dqn_mean_batch_interval`、`no_refresh_delta` 和 `diagnosis`。

最小通过线：DQN 相比 `fixed_1_tick_full_match` 的 paired delta 为正，`WAIT rate` 在 15%-45%，`MATCH_FULL rate` 在 10%-50%，`mean_batch_interval` 在 1.15-1.70，且 off-peak 不出现明显负分。
