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

结果快速汇总：

```bash
python scripts/summarize_timing_run.py --run-name main_ppo_v1 --scale main
```

## 主实验流程

主实验默认只做四类工作：

1. 生成或复用固定评估场景，并输出环境健康报告。
2. 使用手写临期规则进行教师预填充。
3. 训练 `adaptive_timing_ppo`。
4. 在同一批固定评估场景上评估主策略和基线策略。

摩擦灵敏度、PBRS 消融、候选车辆上限灵敏度和步长包络诊断不随主实验默认运行，避免主表混入过多诊断结果。

## 主方法配置

默认主算法为 `adaptive_timing_ppo`。动作空间是二元：

```text
WAIT
MATCH_FULL
```

PPO 只决定此刻是否触发匹配；具体 CV-DV 匹配边始终由同一个约束优化器决定。动态匹配间隔由连续 `WAIT` 后触发 `MATCH_FULL` 的长度统计出来。

默认训练口径：

```text
agent_type = ppo
reward_shaping_mode = pbrs
pbrs_terminal_mode = finite_horizon_correction
observation_profile = compact_v2v
ppo_eval_deterministic = false
ppo_reward_scale = 1000.0
ppo_value_clip_range = 0.20
```

评估时 PPO 默认按固定随机种子从策略分布采样，而不是纯贪心取最大概率动作。二元 PPO 的策略本身是随机策略，贪心评估容易把接近 0.5 的匹配概率硬化成每步匹配。导出的匹配概率、等待概率、动作分布和动态间隔分布用于解释策略。

DQN 可通过 `--agent dqn` 或配置项运行，但只作为历史对照，不进入默认主表。

## 主评估策略

主评估表默认保留 6 类策略：

```text
fixed_1_tick_full_match
fixed_2_tick_full_match
fixed_3_tick_full_match
fixed_4_tick_full_match
handcrafted_deadline_rule
adaptive_timing_ppo
```

固定 1/2/3/4 步用于比较不同固定匹配间隔。手写临期规则是强启发式。PPO 是主方法。旧三动作和其它规则策略不再作为默认主实验内容。

## 独立诊断脚本

PBRS 对照：

```bash
python scripts/run_pbrs_ablation.py --scale main --episodes 80 --eval-episodes 16 --rollout-workers 4 --eval-workers 4 --run-name pbrs_ablation_v1
```

摩擦灵敏度：

```bash
python scripts/run_friction_sensitivity.py --scale main --run-name main_ppo_v1 --eval-workers 4
```

候选车辆上限灵敏度：

```bash
python scripts/run_cap_sensitivity.py --scale main --run-name main_ppo_v1 --eval-workers 4 --caps 20,32,48,64,0
```

步长包络诊断：

```bash
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --eval-workers 4 --run-name interval_envelope_v1
```

2 分钟粒度诊断：

```bash
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --eval-workers 4 --run-name tick3_interval_diag
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --eval-workers 4 --config configs/tick2_diagnostic.json --run-name tick2_interval_diag
```

## 固定场景复用

主规模默认使用固定评估场景。存在足够数量的：

```text
outputs/<run_name>/env_health/eval_scenario_manifest.csv
```

时，后续生成环境健康报告和主评估会复用同一批场景。这样可以减少不同日期窗口、需求强度和供需错配造成的额外方差。

如果同时配置 `train_days` 和 `eval_days`，训练环境只从 `train_days` 采样，评估和 manifest 只从 `eval_days` 采样。不要再依赖生成器内部的隐式回退。

## 常看输出

环境和评估：

```text
outputs/<run_name>/env_health/env_health_summary.csv
outputs/<run_name>/env_health/eval_scenario_manifest.csv
outputs/<run_name>/eval/eval_summary.csv
outputs/<run_name>/eval/eval_summary_zh.csv
outputs/<run_name>/eval/paired_policy_delta_summary.csv
outputs/<run_name>/eval/timing_policy_comparison.csv
outputs/<run_name>/eval/distance_adjusted_eval_summary.csv
outputs/<run_name>/eval/environment_acceptance_summary.csv
```

训练和解释：

```text
outputs/<run_name>/train/train_history.csv
outputs/<run_name>/train/reward_shaping_history.csv
outputs/<run_name>/train/training_curve_comparison.csv
outputs/<run_name>/train/pbrs_ablation_summary.csv
outputs/<run_name>/train/interval_action_distribution.csv
outputs/<run_name>/train/validation_history.csv
outputs/<run_name>/eval/interval_policy_trace.csv
outputs/<run_name>/eval/interval_action_distribution_by_policy.csv
outputs/<run_name>/eval/interval_distribution_histogram.csv
outputs/<run_name>/eval/state_action_policy_trace.csv
outputs/<run_name>/eval/state_action_bucket_summary.csv
```

环境诊断：

```text
outputs/<run_name>/env_diagnostics/interval_bucket_envelope.csv
outputs/<run_name>/env_diagnostics/interval_diversity_summary.csv
outputs/<run_name>/env_diagnostics/pickup_distance_summary.csv
```

`eval_summary_zh.csv` 是主评估表的中文字段版本，便于论文表格阅读。正式分析仍建议保留原始 `eval_summary.csv`，方便脚本继续处理。

## 结果不好时先看什么

如果 PPO 接近每步匹配，先看动作分布、动态间隔分布、熵系数、教师样本和 checkpoint 选择约束，不先改环境惩罚。

如果 PPO 过度等待，先看临期订单服务率、过期取消、PBRS 势函数、奖励尺度和价值函数损失。

如果固定 3 步利润高但服务率低，不要只看利润。应同时看平台得分、服务率、过期取消、单次派单利润、总接驾距离和单服务接驾距离。

如果不同运行的标准差很大，先确认是否使用同一批 manifest，再看配对差值和标准误。标准差大往往来自 Manhattan 不同窗口的需求强度和供需错配差异。

如果策略排名对候选车辆上限敏感，运行候选车辆上限灵敏度。高供给场景中，过低的上限可能排除稍远但高利润的车辆。

如果动态策略只在交易摩擦较强时占优，运行摩擦灵敏度。主结论应优先依赖默认可解释摩擦和弱化摩擦下仍成立的差异。

## 验证

文档修改只需检查格式和关键词；代码修改后再运行完整检查：

```bash
git diff --check
python -m ruff check future_v2v scripts tests
python -m pytest tests -q
```

最小实验验收建议：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5 --eval-episodes 3 --rollout-workers 2 --eval-workers 2 --run-name smoke_gate_v1
python scripts/run_experiment.py --stage all --scale main --episodes 40 --eval-episodes 8 --rollout-workers 4 --eval-workers 4 --run-name mini_main_gate_v1
```

通过小样本后再扩大训练轮数和评估场景数。
