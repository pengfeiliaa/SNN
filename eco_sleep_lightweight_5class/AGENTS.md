# AGENTS

- 任务的 `num_classes` 必须与标签映射一致，修改任务或映射时请同步校验。
- run_all 流程若未生成 `best.ckpt` 必须提示并退出，不得继续评估。
- 推荐自检命令：`python .\scripts\compile_check.py`、`python -c "import eco_sleep; from eco_sleep.models import WearableSNN, EdfSNN"`、`python .\scripts\quick_smoke.py`。
