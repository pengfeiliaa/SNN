# ECO Sleep Lightweight 5-Class

## Project Scope

This repository now keeps a single active line:

- Dataset: `Sleep-EDF-20`
- Task: `W / N1 / N2 / N3 / REM`
- Protocol: `subject-wise 5-fold cross validation`
- Final deploy model family: lightweight `SNN-first`

Removed from the active source tree:

- cross-domain Sleep-EDF evaluation configs/scripts
- unused EEGNet-style side branches
- diagnostic teacher-only script: `scripts/diagnose_sleep_edf_truth_tests.py`
- diagnostic teacher model: `src/eco_sleep/models/debug_teacher_cnn.py`
- wearable-specific `WearableTinyTCN` path from the active model registry

## Deprecated Failed Recipe

Deprecated default:

- recipe: `real_strategy_logit_adjust_threshold`
- run dir: `E:\papercode\eco_sleep_lightweight_5class\runs\20260404_115045_sleep_edf_real_strategy_change`
- active loss: `LogitAdjustedCrossEntropyLoss`
- temporal consistency: `disabled` in the actual train chain
- learnable threshold: `enabled`

Why it failed:

- it raised accuracy without solving `macro_f1`
- it still collapsed `N1`
- it logged temporal consistency concepts while `tc_loss` stayed outside the real train path
- it allowed best checkpoints whose `N1` quality was still too weak
- its train-time validation metrics were not stable proxies for final test behavior

Deprecated failed 5-fold result:

- mean_acc: `0.7697`
- mean_macro_f1: `0.6662`
- mean_kappa: `0.6832`
- mean_N1_f1: `0.1689`
- mean_N1_recall: `0.1326`

## Final Kept Model

Final kept student:

- `ContextPicoSNN`
- default preset: `context_pico`
- default recipe: `context_pico_v1_ldam`
- run dir: `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold`
- active loss: `LDAMLoss`
- temporal consistency: `disabled`
- short-context sequence modeling: `enabled` with `context_len=3`

Why it is still SNN-first:

- the main epoch encoder is spike-driven
- the short-context forward/backward aggregation path is spike-driven
- the classification path stays inside the spiking trunk
- the convolutional front-end only compresses raw EEG into compact tokens

## Full 5-Fold Result

Final kept run:

- mean_acc: `0.7743`
- mean_macro_f1: `0.7281`
- mean_kappa: `0.7023`
- mean_N1_f1: `0.3708`
- mean_N1_recall: `0.5576`
- mean_REM_f1: `0.7822`

Target check:

- `acc >= 0.85`: not reached
- `macro_f1 >= 0.80`: not reached

Main remaining gap:

- `N1` is still the dominant bottleneck under the current lightweight SNN budget
- split `0` shows a visible validation/test mismatch: `val_macro_f1=0.8465` but `test_macro_f1=0.7270`
- the redesigned short-context SNN is materially better than the failed old default, but still below the sprint target

## Complexity

Final kept student complexity:

- params: `31,164`
- MACs: `5,124,372`
- FLOPs: `10,248,744`
- timesteps `T`: `43`
- estimated CPU inference latency: `5.90 ms` at batch size `32`

Against the in-repo `PicoSleepNetPlusSNN` baseline proxy:

- params ratio: `0.329x`
- MACs ratio: `1.241x`
- latency ratio: `0.086x`
- complexity budget: `within budget`

Against the screened `ContextPicoSNNV2` branch:

- `ContextPicoSNNV2` params: `46,495`
- `ContextPicoSNNV2` MACs: `10,135,172`
- `ContextPicoSNNV2` budget status: `rejected`

## Structured Experiments

Machine-readable ablation file:

- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold\change_proof\ablation_decision_report.csv`

Decision summary:

- deprecated `real_strategy_logit_adjust_threshold`: kept only as failure reference
- historical `ContextPicoSNN + cb_focal + temporal_consistency`: rejected after full 5-fold because `N1_f1` and `macro_f1` were worse than the kept route
- `ContextPicoSNNV2 + class_balanced_focal`: rejected in 2-fold screening
- `ContextPicoSNNV2 + LDAM-DRW`: rejected in 2-fold screening
- `ContextPicoSNNV2 + LDAM-DRW + short_context`: rejected in 2-fold screening and exceeded the MAC budget
- `ContextPicoSNN + LDAM`: kept as the new default

## Main Entry Points

- `scripts/preprocess_sleep_edf.py`
- `scripts/diagnose_data.py`
- `scripts/train_sleep_edf.py`
- `scripts/eval_sleep_edf.py`
- `scripts/analyze_model_complexity.py`
- `scripts/run_all_sleep_edf.ps1`
- `scripts/compile_check.py`
- `scripts/quick_smoke.py`

## Reproduce

Compile check:

```powershell
python .\scripts\compile_check.py
```

Import check:

```powershell
python -c "import eco_sleep; from eco_sleep.models import WearableSNN, EdfSNN"
```

Quick smoke:

```powershell
python .\scripts\quick_smoke.py
```

Train final kept model:

```powershell
python .\scripts\train_sleep_edf.py --allow_cpu --preset context_pico --recipe context_pico_v1_ldam --run_dir runs\20260405_context_pico_v1_ldam_guarded_5fold
```

Evaluate full run:

```powershell
python .\scripts\eval_sleep_edf.py --run_dir E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold
```

The machine-readable evidence lives under:

- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold\change_proof\`
- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_context_pico_v1_ldam_guarded_5fold\eval\`

## Current Incremental Optimization Recipe

Locked reference baseline for the current lightweight line:

- `E:\papercode\eco_sleep_lightweight_5class\runs\20260405_162343_sleep_edf_context_pico`
- model: `ContextPicoSNN`
- baseline recipe: `context_pico_v1_ldam`

Current incremental recipe for the next full run:

- recipe: `context_pico_v1_ldam`
- keeps the same `ContextPicoSNN` backbone, parameter count, context window, and `t_steps`
- train-time loss schedule: `LDAM -> LDAM-DRW`
- weight averaging: `EMA` single-checkpoint export
- checkpoint selection: `0.60 * val_macro_f1 + 0.20 * val_N1_f1 + 0.10 * val_kappa + 0.10 * val_REM_f1`
- hard gates: `val_N1_f1 >= 0.30` and `val_N1_recall >= 0.25`
- fallback: top-3 selection-score checkpoint averaging
- temporal consistency: removed from the active mainline until non-zero train loss is proven
- validation-only calibration: temperature scaling + 5D class-bias calibration, with raw fallback when test outputs/metrics do not actually change

Short smoke verification:

```powershell
python .\scripts\train_sleep_edf.py --allow_cpu --preset context_pico --recipe context_pico_v1_ldam --smoke --smoke_epochs 8 --smoke_splits 1 --smoke_train_per_class 8 --smoke_eval_per_class 4 --only_splits 0 --run_dir runs\20260407_context_pico_v1_ldam_smoke
python .\scripts\eval_sleep_edf.py --run_dir E:\papercode\eco_sleep_lightweight_5class\runs\20260407_context_pico_v1_ldam_smoke --only_splits 0 --postprocess auto --smoke --smoke_eval_per_class 4
```

Full run for the user-owned 5-fold experiment:

```powershell
python .\scripts\train_sleep_edf.py --allow_cpu --preset context_pico --recipe context_pico_v1_ldam --run_dir runs\20260407_context_pico_v1_ldam_5fold
python .\scripts\eval_sleep_edf.py --run_dir E:\papercode\eco_sleep_lightweight_5class\runs\20260407_context_pico_v1_ldam_5fold --postprocess auto
```
