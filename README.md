# Transformer 中英翻译学习项目

本项目从原中英翻译实验中单独整理了一个可运行的编码器—解码器 Transformer，并包含数据、训练、验证、自回归生成与 shape 测试。

## 文件

- `transformer_translation.py`：Transformer 模型、注意力、mask、teacher forcing 和贪心生成。
- `train_transformer.py`：可复现的训练与验证脚本。
- `data/translation_sample.tsv`：仓库内置的小型英中样例数据。
- `TRANSFORMER_GUIDE.md`：shape 流程和关键模块说明。
- `LEARNING_PROGRESS.md`：当前掌握情况、薄弱点和下次复测项目。
- `outputs/training_curve.svg`：最近一次训练曲线。
- `outputs/validation_results.json`：逐 epoch 验证结果。
- `outputs/transformer_demo.pt`：最佳验证 epoch 的模型权重。

## 运行

```powershell
python transformer_translation.py
python train_transformer.py
```

第一条命令运行模块、mask 和 shape 测试，第二条命令重新训练并覆盖 `outputs` 中的实验结果。
