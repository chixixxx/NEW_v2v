# Future V2V 运行手册

## 快速开始

准备 TLC Manhattan 缓存：

```bash
python scripts/prepare_tlc_manhattan.py --month 2025-10
```

烟测：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2 --run-name smoke_ppo_v1
```

主实验：

```bash
python scripts/run_experiment.py --stage all --scale main --rollout-workers 4 --eval-workers 4 --run-name main_ppo_v1
```

## 主方法

默认主算法为 `adaptive_timing_ppo`。动作空间是二元：

```text
WAIT
MATCH_FULL
```

PPO 只决定此刻是否触发匹配；具体 CV-DV 匹配边始终由同一个约束优化器决定。动态匹配间隔由连续 `WAIT` 后触发 `MATCH_FULL` 的长度统计出来，而不是直接把 1/2/3/4 步做成动作。

`adaptive_interval_dqn` 仍可通过配置 `training.agent_type = dqn` 或命令行 `--agent dqn` 运行，用作旧方法对照。主实验默认不评估 DQN 检查点，避免主表混杂。

默认训练口径：

```text
agent_type = ppo
reward_shaping_mode = pbrs
pbrs_terminal_mode = finite_horizon_correction
observation_profile = compact_v2v
ppo_eval_deterministic = false
```

评估时 PPO 默认按固定随机种子从策略分布采样，而不是纯贪心取最大概率动作。原因是二元 PPO 的策略本身是随机策略，若用贪心评估，容易把 0.51 的匹配概率硬化成“每步匹配”。导出的 `match_probability`、`wait_probability`、动作分布和间隔分布用于解释策略。

## 对照与消融

PBRS 对照实验会顺序运行 `none`、`legacy_delta` 和 `pbrs` 三组，并汇总训练曲线与评估摘要：

```bash
python scripts/run_pbrs_ablation.py --scale main --episodes 80 --eval-episodes 16 --rollout-workers 4 --eval-workers 4 --run-name pbrs_ablation_v1
```

摩擦灵敏度不再随主实验默认运行，需要单独调用：

```bash
python scripts/run_friction_sensitivity.py --scale main --run-name main_latest --eval-workers 4
```

候选车辆 cap 灵敏度也单独调用。`0` 表示不限制每个订单预筛车辆数：

```bash
python scripts/run_cap_sensitivity.py --scale main --run-name main_latest --eval-workers 4 --caps 20,32,48,64,0
```

主评估表默认保留 6 类策略：

```text
fixed_1_tick_full_match
fixed_2_tick_full_match
fixed_3_tick_full_match
fixed_4_tick_full_match
handcrafted_deadline_rule
adaptive_timing_ppo
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
python scripts/summarize_timing_run.py --run-name main_ppo_v1 --scale main
```

该脚本只读已有 CSV，不重新仿真。它会优先读取 `adaptive_timing_ppo`，并报告二元动作占比、动态间隔分布、相对固定 1 步的配对差值、与最佳策略差距、离峰风险和摩擦灵敏度状态。

## 常看输出

```text
outputs/<run_name>/env_health/env_health_summary.csv
outputs/<run_name>/train/train_history.csv
outputs/<run_name>/train/reward_shaping_history.csv
outputs/<run_name>/train/training_curve_comparison.csv
outputs/<run_name>/train/pbrs_ablation_summary.csv
outputs/<run_name>/train/interval_action_distribution.csv
outputs/<run_name>/train/validation_history.csv
outputs/<run_name>/eval/eval_summary.csv
outputs/<run_name>/eval/eval_summary_zh.csv
outputs/<run_name>/eval/distance_adjusted_eval_summary.csv
outputs/<run_name>/eval/timing_policy_comparison.csv
outputs/<run_name>/eval/paired_policy_delta_summary.csv
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

最小验收建议：先跑 smoke，再跑 main 小样本。如果 PPO 的二元动作或动态间隔再次塌缩，再优先检查教师样本、熵系数、PBRS 势函数和 checkpoint 动作分布约束，不先改环境惩罚。

如果同时配置 `train_days` 和 `eval_days`，训练环境只从 `train_days` 采样，主评估和 manifest 只从 `eval_days` 采样；不要再依赖生成器内部的隐式 fallback。
