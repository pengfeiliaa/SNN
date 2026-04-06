# SNN Sleep Stage Guide

## 默认模型

- 配置项：`model.name: picosleepnet_baseline`
- 类名：`PicoSleepNetBaseline`
- 可选增强：`PicoSleepNetPlusSNN`
- 当前默认目标：先稳定复现 baseline，再谈增强分支

## 当前数据协议

- 数据集：Sleep-EDF
- 标签映射：`W=0, N1=1, N2=2, N3/N4=3, REM=4`
- 单通道：`Fpz-Cz`
- 每个 epoch：30 秒
- 采样率：100 Hz
- 每个 epoch 长度：3000 点
- LCS 编码输出：`raw_epoch / lcs_pos_count / lcs_neg_count / lcs_pos / lcs_neg / epoch_stage_desc`

## 当前训练主线

- sampler：`weighted`
- loss：`ce`
- `use_class_weights=false`
- 默认不开 QAT
- 每个 batch 前会 `reset_state()`
- audit 先检查 logits、softmax、bias、firing rate、split class counts，再允许训练

## 为什么之前会全部预测成类 1

这次定位到的根因不是单一因素，而是多个工程错误叠加：

1. `raw_dir` 配置是乱码路径，preprocess 经常根本没读到真实 Sleep-EDF EDF 文件。
2. 旧版 PSG/Hypnogram 配对逻辑错误，找不到同名 Hypnogram 时直接取目录中的第一份 `*Hypnogram.edf`，导致大量 PSG 共用同一份 Hypnogram。旧 processed 的 153 条记录出现完全相同标签分布，就是这个 bug 的证据。
3. preprocess 即使 `saved=0` 也不会阻断后续步骤，训练继续吃旧缓存，因此表面上看像“preprocess 失败但 train 还在跑”。
4. 训练默认同时使用 weighted sampler、类别权重 CE 和未归一化的 `mask_l1` 正则，旧版 `mask_l1` 量级过大，会主导 loss，并把模型推向少数类 N1 的单类塌陷。

## 现在 audit 会检查什么

`scripts/audit_sleep_edf_pipeline.py` 会检查：

- preprocess 是否真的写入了本轮有效样本
- 是否显式选择了 `reuse_existing` 或 `force_rebuild`
- processed cache 是否为新 schema
- 原始 stage 文本、映射标签、dataset 标签三者是否一致
- split 后 train/val/test 每类样本数是否为 0
- 初始 logits / softmax / bias 是否异常
- 前 3 个 batch 的 firing rate / spike ratio / membrane stats 是否异常

如果上述任一项失败，audit 直接退出，训练不会继续。

## 运行方式

全流程：

```powershell
.\scripts\run_all_sleep_edf.ps1
```

显式复用已有 processed：

```powershell
.\scripts\run_all_sleep_edf.ps1 -ReuseExisting
```

仅 smoke：

```powershell
python .\scripts\train_sleep_edf.py --config configs\sleep_edf_5class.yaml --allow_cpu --smoke --smoke_epochs 3 --smoke_splits 1
```

单独 audit：

```powershell
python .\scripts\audit_sleep_edf_pipeline.py --config configs\sleep_edf_5class.yaml
```

样本可视化：

```powershell
python .\scripts\debug_sleep_edf_sample.py --config configs\sleep_edf_5class.yaml
```

## 如何判断链路已经不再是旧错误态

至少要满足：

1. `audit_summary.json` 的 `blocking_errors` 为空。
2. `label_mapping_summary.json` 中 `mismatch_count=0`。
3. `processed_dir` 不再是 legacy cache。
4. smoke 训练时 `pred_ratio` 不再在每个 epoch 固定成 `[0,1,0,0,0]`。
5. `eval_sleep_edf.py` 不再出现 checkpoint mismatch。
