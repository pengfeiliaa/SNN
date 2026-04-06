# 第五章 基于完整日志证据的 Sleep-EDF-20 轻量 SNN 主线重构

## 5.1 任务与硬约束

本章只保留一个主线协议：

- 数据集：`Sleep-EDF-20`
- 任务：`W / N1 / N2 / N3 / REM` 五分类
- 协议：`subject-wise 5-fold cross validation`
- 最终部署模型：必须仍然是轻量 `SNN` 主框架

本轮不允许退回普通重型 `CNN/LSTM/Transformer` 路线，也不允许继续把训练期 `val` 指标当作最终 `test` 结果汇报。

## 5.2 为什么旧方案必须停用

已经确认失败的旧默认 recipe 为：

- recipe=`real_strategy_logit_adjust_threshold`
- active loss=`LogitAdjustedCrossEntropyLoss`
- `temporal_consistency_enable=False`
- `tc_loss=0.0000`
- `learnable_threshold_enable=True`

这条路线失败的核心原因有四个：

1. 它没有解决 `macro_f1`，尤其没有解决 `N1`。
2. 它让 temporal consistency 只停留在命名和日志层，不在真实训练主链里发挥作用。
3. 它的 best checkpoint 规则容忍 `N1` 明显塌陷。
4. 它容易出现“验证指标看起来还行，但最终测试并不稳”的现象。

旧失败 recipe 的完整 5-fold 结果为：

- mean_acc=`0.7697`
- mean_macro_f1=`0.6662`
- mean_kappa=`0.6832`
- mean_N1_f1=`0.1689`
- mean_N1_recall=`0.1326`

因此它已被明确降级为失败参考，不再作为默认主方案。

## 5.3 这次保留的新轻量 SNN 主框架

最终保留模型：`ContextPicoSNN`

最终保留配置：

- preset=`context_pico`
- recipe=`context_pico_v1_ldam`
- active loss=`LDAMLoss`
- temporal consistency=`disabled`
- short-context sequence modeling=`enabled`
- context_len=`3`

它仍属于明确的 `SNN-first` 设计：

- 原始 EEG 先经过 very-light multi-scale conv stem 压缩
- epoch 表征由脉冲层完成
- 前向/后向短上下文聚合由脉冲递归层完成
- 最终分类决策仍位于脉冲主干而非普通 dense 时序主干

## 5.4 指标记录、best ckpt 与 N1 保护

本轮重写后的训练日志显式区分：

- `train_acc`
- `val_acc`
- `val_macro_f1`
- `val_kappa`
- `val_N1_f1`
- `val_N1_recall`
- `val_REM_f1`
- `val_pred_ratio`

最终评估阶段只报告 `test` 指标，并单独输出：

- `eval/per_fold_test_metrics.csv`
- `eval/per_fold_val_test_compare.csv`
- `eval/summary_metrics.csv`

新的 best checkpoint 规则为：

1. 第一优先级：`val_macro_f1`
2. 第二优先级：`val_kappa`
3. 第三优先级：`val_N1_f1`
4. 第四优先级：`val_N1_recall`
5. `val_acc` 仅作更后位参考

并且加入硬门槛：

- `val_N1_f1 < 0.15` 不允许成为 best ckpt
- `val_N1_recall < 0.12` 不允许成为 best ckpt
- 若配置声称启用 temporal consistency，但连续 `2` 个 epoch 的 `tc_loss` 都等效为 `0`，训练直接报错退出

本轮保留 run 的所有 split 都通过了 N1 floor，且 `N1` 预测比例没有跌破 `0.03` 抑制线。

## 5.5 工程清理

本轮已删除的非主线文件：

- `scripts/diagnose_sleep_edf_truth_tests.py`
- `src/eco_sleep/models/debug_teacher_cnn.py`

此外：

- wearable 专用 `WearableTinyTCN` 不再保留在 active model registry 中
- `WearableSNN` 仅保留为导入兼容别名，避免自检命令失效

## 5.6 结构化实验与决策

本轮用于决策的候选包括：

- `current_failed_recipe_reference`
- `ContextPicoSNN + LDAM`
- `ContextPicoSNN + cb_focal + temporal_consistency`
- `ContextPicoSNNV2 + class_balanced_focal`
- `ContextPicoSNNV2 + LDAM-DRW`
- `ContextPicoSNNV2 + LDAM-DRW + short_context`

决策结果如下：

- `real_strategy_logit_adjust_threshold`：淘汰。原因是 `macro_f1 / N1` 失效，且 temporal consistency 实际未接入训练主链。
- `ContextPicoSNN + cb_focal + temporal_consistency`：淘汰。历史完整 5-fold 指标为 `acc=0.7812 / macro_f1=0.7297 / kappa=0.7100 / N1_f1=0.3562`，仍弱于保留方案的 `N1_f1`。
- `ContextPicoSNNV2 + class_balanced_focal`：淘汰。2-fold 筛选均值 `acc=0.7619 / macro_f1=0.7007`。
- `ContextPicoSNNV2 + LDAM-DRW`：淘汰。2-fold 筛选均值 `acc=0.7466 / macro_f1=0.6971`。
- `ContextPicoSNNV2 + LDAM-DRW + short_context`：淘汰。2-fold 筛选均值 `acc=0.7427 / macro_f1=0.7008`，且 `MACs` 超出预算。
- `ContextPicoSNN + LDAM`：保留。它在当前已验证候选中给出了更好的 `N1_f1` 与更低的复杂度。

## 5.7 完整 5-fold 的真实结果

最终保留 run：

- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold`

完整 5-fold `selected` 结果：

- mean_acc=`0.7743`
- mean_macro_f1=`0.7281`
- mean_kappa=`0.7023`
- mean_N1_f1=`0.3708`
- mean_N1_recall=`0.5576`
- mean_REM_f1=`0.7822`

其中最明显的泛化风险出现在 split `0`：

- `val_macro_f1_at_ckpt=0.8465`
- `test_macro_f1=0.7270`
- `val_N1_f1_at_ckpt=0.5934`
- `test_N1_f1=0.3618`

因此该 split 被自动记录为 `overfit_or_validation_mismatch`。

## 5.8 是否达到目标

未达到：

- `acc >= 0.85`：未达到
- `macro_f1 >= 0.80`：未达到

主要差距仍来自：

- `N1` 过渡阶段的泛化上限
- 轻量 SNN 预算下，短上下文仍不足以完全稳定 `W-N1-REM` 边界
- 某些 split 仍存在验证集选点过于乐观的问题

## 5.9 复杂度是否仍在 PicoSleepNet 级别

最终保留 student：

- params=`31,164`
- MACs=`5,124,372`
- FLOPs=`10,248,744`
- timesteps `T`=`43`
- estimated CPU latency=`5.90 ms`

相对 in-repo `PicoSleepNetPlusSNN` baseline proxy：

- params ratio=`0.329x`
- MACs ratio=`1.241x`
- latency ratio=`0.086x`

结论：

- 参数量明显更低
- MACs 略高，但仍处于允许预算内
- 没有演化成重型 dense sequence model

而 `ContextPicoSNNV2` 的复杂度为：

- params=`46,495`
- MACs=`10,135,172`

因此 `v2` 已超出当前优先预算，不予保留。

## 5.10 复现实验

编译检查：

```powershell
python .\scripts\compile_check.py
```

导入检查：

```powershell
python -c "import eco_sleep; from eco_sleep.models import WearableSNN, EdfSNN"
```

快速 smoke：

```powershell
python .\scripts\quick_smoke.py
```

训练最终保留方案：

```powershell
python .\scripts\train_sleep_edf.py --allow_cpu --preset context_pico --recipe context_pico_v1_ldam --run_dir runs\20260405_context_pico_v1_ldam_guarded_5fold
```

评估完整 5-fold：

```powershell
python .\scripts\eval_sleep_edf.py --run_dir E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold
```

证据文件目录：

- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold\change_proof\`
- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold\eval\`
