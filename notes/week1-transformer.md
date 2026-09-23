# 第 1 周：PyTorch 与 Transformer 基础

## 学习目标

本周目标是从中英翻译项目中整理出可独立运行的编码器—解码器 Transformer，并建立从数据、mask、训练到推理的完整理解。

## 1. Dataset 与 DataLoader

`Dataset` 定义单个样本怎样读取和编码，`DataLoader` 负责批处理、打乱顺序和迭代。

翻译数据中的单个样本包含英文源序列和中文目标序列。目标序列包含 `BOS` 与 `EOS`，用于 teacher forcing 和自回归生成。

## 2. 训练与验证

训练阶段执行以下顺序：

1. `model.train()`；
2. `optimizer.zero_grad()`；
3. 前向传播并计算 loss；
4. `loss.backward()`；
5. `optimizer.step()`。

验证阶段使用 `model.eval()` 和 `torch.no_grad()`，只计算指标，不更新参数。

`model.train()` 与 `model.eval()` 控制 dropout 等模块的行为，`torch.no_grad()` 控制是否记录梯度，两者职责不同。

## 3. Teacher forcing

设完整目标序列 shape 为 `[B,T+1]`，其中 `B` 是 batch size，`T+1` 是包含边界 token 的目标长度。

通过错位切片得到：

```python
decoder_input = target_tokens[:, :-1]
target = target_tokens[:, 1:]
```

两者 shape 都是 `[B,T]`。模型输出 logits 的 shape 为 `[B,T,V]`，其中 `V` 是目标词表大小。

## 4. Q、K、V 与缩放点积注意力

- Q 表示当前位置正在查询什么。
- K 表示每个位置提供哪些可匹配特征。
- V 表示匹配后实际汇总的内容。

缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

其中 $d_k$ 是每个注意力头的键向量维度。除以 $\sqrt{d_k}$ 可以减轻维度增大导致的点积过大和 softmax 饱和。

mask 必须在 softmax 前应用，使不可见位置的注意力权重变为零。

## 5. 多头注意力 shape

设模型维度为 $D$，注意力头数为 $H$，每个头的维度为 $d_{head}=D/H$。

shape 变化为：

1. 输入 `[B,L,D]`；
2. 拆头 `[B,H,L,d_head]`；
3. 注意力分数 `[B,H,L_q,L_k]`；
4. 每个头的输出 `[B,H,L_q,d_head]`；
5. 合并多头 `[B,L_q,D]`；
6. 输出投影后保持 `[B,L_q,D]`。

合并多头前必须交换头维与序列维，避免把同一头的不同 token 错误拼接。

## 6. 编码器与解码器

编码器层由源端自注意力、残差连接、LayerNorm 和位置前馈网络组成。编码器自注意力的 Q、K、V 都来自源序列表示。

解码器层依次执行：

1. 带 causal mask 的目标端自注意力；
2. 查询编码器 `memory` 的交叉注意力；
3. 位置前馈网络。

交叉注意力的 Q 来自解码器当前隐藏状态，K、V 来自编码器 `memory`。

## 7. 两种 mask

padding mask 遮住 `<PAD>`，防止填充位置参与注意力。源 padding mask 的典型 shape 为 `[B,1,1,S]`。

causal mask 遮住当前目标位置之后的 token，防止训练时看到未来答案。目标 mask 的典型 shape 为 `[B,1,T,T]`。

解码器自注意力需要目标 padding mask 与 causal mask，交叉注意力使用源 padding mask。

## 8. 训练与推理的差异

训练时使用真实目标前缀，可以并行预测每个位置的下一个 token。

推理时没有真实目标答案，只能从 `BOS` 开始，把刚生成的 token 追加回输入，直到生成 `EOS` 或达到最大长度。

批量异步解码需要累计保存每个样本的 `finished` 状态。已经结束的样本后续保持输出 `EOS`，不能只检查当前一步是否所有样本同时产生 `EOS`。

## 9. 可复现实验结果

训练使用固定随机种子 `42`、10 条训练样本、2 条验证样本和 20 个 epoch。

- 训练 loss 从 `4.8233` 下降到 `1.9668`。
- 最佳验证 loss 为 `4.5734`，出现在第 5 个 epoch。
- 第 20 个 epoch 的验证 loss 为 `4.7680`。
- 训练 loss 持续下降而验证 loss 在第 5 个 epoch 后回升，说明微型数据集已经过拟合。

训练曲线位于 `outputs/training_curve.svg`，逐 epoch 结果位于 `outputs/validation_results.json`，最佳模型权重位于 `outputs/transformer_demo.pt`。

## 10. 可复现运行

```powershell
python transformer_translation.py
python train_transformer.py
```

第一条命令验证注意力、mask、shape 和批量贪心解码，第二条命令重新训练并保存曲线、验证结果和模型权重。

Notebook 版本位于 `experiments/transformer_baseline.ipynb`。

## 11. 结论与边界

本实验已经打通数据读取、teacher forcing、Transformer 前向传播、训练、验证、模型选择和自回归生成流程。

由于样本量极小，当前 loss 和生成结果只能证明代码流程正确，不能代表真实翻译质量。扩大实验时应增加数据、使用更稳定的翻译指标并报告多个随机种子。
