# Transformer 与数据整理学习项目

本项目包含一个可运行的编码器—解码器 Transformer，以及一个可复现的数据清洗、去重、候选生成与评测实验。

## 文件

- `transformer_translation.py`：Transformer 模型、注意力、mask、teacher forcing 和贪心生成。
- `train_transformer.py`：可复现的训练与验证脚本。
- `data/translation_sample.tsv`：仓库内置的小型英中样例数据。
- `TRANSFORMER_GUIDE.md`：shape 流程和关键模块说明。
- `notes/week1-transformer.md`：第 1 周 PyTorch、Transformer、mask、训练与推理总结。
- `experiments/transformer_baseline.ipynb`：可从头运行的 Transformer 翻译基线 notebook。
- `data_curation_experiment.py`：文本清洗、精确去重、Jaccard 候选、倒排索引预筛和评测检查。
- `DATA_CURATION_REPORT.md`：数据整理实验指标、阈值选择、失败案例和结论边界。
- `notes/week2-pretraining-and-data.md`：第 2 周预训练、Scaling Laws、数据整理与实验总结。
- `tiny_lm_experiment.py`：tiny decoder-only language model 与两种训练数据处理方法的受控对照实验。
- `scaling_law_experiment.py`：固定模型与数据、改变训练步数的多种子计算量实验。
- `LEARNING_PROGRESS.md`：当前掌握情况、薄弱点和下次复测项目。
- `outputs/training_curve.svg`：最近一次训练曲线。
- `outputs/validation_results.json`：逐 epoch 验证结果。
- `outputs/transformer_demo.pt`：最佳验证 epoch 的模型权重。
- `outputs/tiny_lm_comparison.json`：两种数据处理方法的 loss、训练时间和生成样例。
- `outputs/scaling_law_results.json`：多个训练步数下的验证 loss 均值、标准差与时间。

## 运行

```powershell
python transformer_translation.py
python train_transformer.py
python data_curation_experiment.py
python tiny_lm_experiment.py
python scaling_law_experiment.py
```

第一条命令运行 Transformer 模块、mask 和 shape 测试，第二条命令重新训练并覆盖对应的 Transformer 输出，第三条命令运行数据整理与候选评测检查，第四条命令运行 tiny language model 的数据处理对照实验，第五条命令运行固定模型与数据的训练计算量实验并保存结果。

## Jupyter 环境

`experiments/transformer_baseline.ipynb` 使用已注册的 `machine-learning-two` kernel，显示名称为 `Python 3.13 (two)`。

该 kernel 对应以下现有解释器：

```text
C:\Users\20571\PyCharmMiscProject\.venv\Scripts\python.exe
```

需要单独启动 JupyterLab 时，可在 PowerShell 中运行：

```powershell
& 'C:\Users\20571\PyCharmMiscProject\.venv\Scripts\python.exe' -m jupyter lab
```

Transformer notebook 为保证跨 CUDA 环境的基线一致性，会显式使用 CPU；这不会影响该解释器在其他项目中使用 GPU。
