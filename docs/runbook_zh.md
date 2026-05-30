# Future V2V 运行说明

## 准备

```bash
cd C:\Sioux\future_v2v_adaptive_timing
```

依赖包括 `numpy`、`scipy`、`torch`、`pandas`、`pyarrow`、`pytest`、`ruff`、`tqdm`。

若使用 TLC Manhattan 主环境，请将数据放到项目根目录：

```text
yellow_tripdata_2025-10.parquet
taxi_zone_lookup.csv
```

预处理：

```bash
python scripts/prepare_tlc_manhattan.py --month 2025-10
```

没有 TLC 数据时，smoke 可回退到合成环境。

## 主实验

完整 smoke：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2 --run-name smoke_adaptive_interval_v1
```

完整 main：

```bash
python scripts/run_experiment.py --stage all --scale main --rollout-workers 4 --eval-workers 4 --run-name adaptive_interval_main_v1
```

分阶段运行：

```bash
python scripts/run_experiment.py --stage generate --scale main --eval-episodes 8
python scripts/run_experiment.py --stage train --scale main --rollout-workers 4
python scripts/run_experiment.py --stage eval --scale main --eval-workers 4
python scripts/run_experiment.py --stage report --scale main
```

默认主算法为 `adaptive_interval_dqn`，即强化学习选择匹配间隔，约束优化器统一完成匹配。旧三动作深度 Q 网络只作为诊断基线。

默认训练口径：

```text
reward_shaping_mode = pbrs
observation_profile = compact_v2v
```

普通训练可以临时覆盖奖励塑造和观测配置：

```bash
python scripts/run_experiment.py --stage all --scale main --reward-shaping pbrs --observation-profile compact_v2v --run-name main_pbrs_compact_v1
```

PBRS 对照实验会顺序运行 `none`、`legacy_delta` 和 `pbrs` 三组，并汇总训练曲线与评估摘要：

```bash
python scripts/run_pbrs_ablation.py --scale main --episodes 80 --eval-episodes 16 --rollout-workers 4 --eval-workers 4 --run-name pbrs_ablation_v1
```

主评估表默认只保留 6 类策略：

```text
fixed_1_tick_full_match
fixed_2_tick_full_match
fixed_3_tick_full_match
fixed_4_tick_full_match
handcrafted_deadline_rule
adaptive_interval_dqn
```

旧的 top-batch、queue、pressure、short-lookahead 策略和旧三动作 DQN 已下线；主表只保留固定 1/2/3/4 步完整匹配、手写强规则和匹配间隔 DQN。

摩擦灵敏度不再随主实验默认运行，需要单独调用：

```bash
python scripts/run_friction_sensitivity.py --scale main --run-name main_latest --eval-workers 4
```

## 匹配步长诊断

先看环境是否支持不同状态选择不同匹配步长：

```bash
python scripts/run_interval_envelope.py --scale smoke --eval-episodes 3 --run-name smoke_interval_envelope_v1
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --run-name interval_envelope_v1
```

关键输出：

```text
outputs/<run_name>/env_diagnostics/interval_bucket_envelope.csv
outputs/<run_name>/env_diagnostics/interval_diversity_summary.csv
outputs/<run_name>/env_diagnostics/pickup_distance_summary.csv
```

重点看：

- `fixed2_dominance_rate` 是否低于 70%。
- `unique_best_interval_count` 是否至少为 2。
- 不同时间、供需、风险、距离桶是否出现不同最佳步长。

## 2 分钟粒度诊断

不直接替换主环境，先做对比：

```bash
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --run-name tick3_interval_diag
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --config configs/tick2_diagnostic.json --run-name tick2_interval_diag
```

若 2 分钟粒度明显提升步长多样性，且运行成本可接受，再考虑升为正式主环境。

## 结果总结

训练评估后快速汇总：

```bash
python scripts/summarize_timing_run.py --run-name adaptive_interval_main_v1 --scale main
```

该脚本只读已有 CSV，不重新仿真。新主算法会优先读取 `adaptive_interval_dqn`、`interval_policy_trace.csv` 和 `interval_action_distribution.csv`。

## 常看输出

```text
outputs/<run_name>/env_health/env_health_summary.csv
outputs/<run_name>/train/train_history.csv
outputs/<run_name>/train/reward_shaping_history.csv
outputs/<run_name>/train/training_curve_comparison.csv
outputs/<run_name>/train/pbrs_ablation_summary.csv
outputs/<run_name>/train/interval_action_distribution.csv
outputs/<run_name>/eval/eval_summary.csv
outputs/<run_name>/eval/distance_adjusted_eval_summary.csv
outputs/<run_name>/eval/timing_policy_comparison.csv
outputs/<run_name>/eval/paired_policy_delta_summary.csv
outputs/<run_name>/eval/friction_sensitivity_summary.csv
outputs/<run_name>/eval/environment_acceptance_summary.csv
outputs/<run_name>/eval/interval_policy_trace.csv
outputs/<run_name>/eval/interval_action_distribution_by_policy.csv
outputs/<run_name>/eval/interval_distribution_histogram.csv
outputs/<run_name>/eval/state_action_policy_trace.csv
outputs/<run_name>/eval/state_action_bucket_summary.csv
outputs/<run_name>/eval/dispatch_trace_by_policy.csv
```

## 验证

```bash
python -m ruff check future_v2v scripts tests
python -m pytest tests -q
```

最小验收建议：先跑 smoke 包络诊断，再跑 main 小样本包络诊断；环境有步长多样性后，再跑 `adaptive_interval_dqn` 训练。
