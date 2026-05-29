# Future V2V Adaptive Timing 运行说明

## 环境准备

在项目目录运行：

```bash
cd C:\Sioux\future_v2v_adaptive_timing
```

当前实现依赖：

- Python 3.10+
- numpy
- scipy
- torch
- pytest
- ruff

## Smoke 运行

```bash
python scripts/run_experiment.py --stage all --scale smoke --episodes 5
```

或使用便捷 smoke stage：

```bash
python scripts/run_experiment.py --stage smoke --episodes 5
```

输出目录：

```text
outputs/smoke_latest/
  env_health/
  train/
  eval/
  reports/
```

## 主实验运行

```bash
python scripts/run_experiment.py --stage all --scale main
```

分阶段运行：

```bash
python scripts/run_experiment.py --stage generate --scale main
python scripts/run_experiment.py --stage train --scale main
python scripts/run_experiment.py --stage eval --scale main
python scripts/run_experiment.py --stage report --scale main
```

## 参数调整

集中修改：

```text
configs/default.json
```

常调参数：

- `scales.smoke.total_orders`
- `scales.smoke.candidate_vehicles`
- `scales.main.total_orders`
- `scales.main.candidate_vehicles`
- `environment.pickup_cap_minutes`
- `environment.service_kwh_per_tick`
- `training.teacher_prefill_episodes`
- `training.learning_rate`
- `scales.<scale>.train_episodes`

## 验证

```bash
python -m pytest tests -q
python -m ruff check future_v2v scripts tests
```

## 结果解读

优先看：

- `outputs/<run_name>/eval/eval_summary.csv`
- `outputs/<run_name>/eval/eval_report.md`
- `outputs/<run_name>/reports/run_report.md`

主表按 `future_v2v_score_mean` 排序。若 DQN 分数不高，先看 `dispatch_epoch_count_mean` 和 `mean_batch_interval_mean`，判断它是否学成了过度等待或过度频繁匹配。
