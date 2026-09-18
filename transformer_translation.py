import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(query, key, value, mask=None):
    """计算缩放点积注意力并返回输出与注意力权重。"""
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights=torch.softmax(scores, dim=-1)
    output=torch.matmul(attention_weights,value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """把模型维度拆成多个注意力头，再将各头输出合并。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.q_proj(query).view(
            batch_size, -1, self.num_heads, self.d_head
        ).transpose(1, 2)

        k = self.k_proj(key).view(
            batch_size, -1, self.num_heads, self.d_head
        ).transpose(1, 2)

        v = self.v_proj(value).view(
            batch_size, -1, self.num_heads, self.d_head
        ).transpose(1, 2)
        head_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        merged = head_output.transpose(1, 2).reshape(
            batch_size, -1, self.d_model
        )
        output = self.out_proj(merged)

        return output, attention_weights


class FeedForward(nn.Module):
    """对每个序列位置独立执行两层前馈变换。"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    """Transformer 编码器层：自注意力后接位置前馈网络。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attention_output, _ = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x


class DecoderLayer(nn.Module):
    """Transformer 解码器层：因果自注意力、交叉注意力和前馈网络。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        self_attention_output, _ = self.self_attention(
            x, x, x, tgt_mask
        )
        x = self.norm1(x + self.dropout(self_attention_output))

        cross_attention_output, _ = self.cross_attention(
            x, memory, memory, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_forward_output))

        return x


def make_src_mask(src_tokens, pad_id):
    """生成源序列 padding mask，目标 shape 为 [B, 1, 1, S]。"""
    mask = src_tokens != pad_id
    return mask.unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt_tokens, pad_id):
    """合并目标序列 padding mask 与 causal mask。"""
    padding_mask = (tgt_tokens != pad_id).unsqueeze(1).unsqueeze(2)

    target_length = tgt_tokens.size(1)
    causal_mask = torch.tril(
        torch.ones(
            target_length,
            target_length,
            dtype=torch.bool,
            device=tgt_tokens.device,
        )
    ).unsqueeze(0).unsqueeze(0)

    return padding_mask & causal_mask


class PositionalEncoding(nn.Module):
    """为 token embedding 注入固定的正弦位置信息。"""

    def __init__(self, d_model, max_length=128):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]


class TransformerSeq2Seq(nn.Module):
    """用于序列到序列翻译的编码器—解码器 Transformer。"""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout=0.1,
        pad_id=0,
        max_length=128,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=pad_id
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=pad_id
        )
        self.position = PositionalEncoding(d_model, max_length)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_tokens, tgt_tokens):
        src_mask = make_src_mask(src_tokens, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_tokens, self.pad_id)

        src = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        src = self.position(src)

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)

        memory = src

        tgt = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt = self.position(tgt)

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, memory, tgt_mask, src_mask)

        logits = self.output_projection(tgt)
        return logits


def make_teacher_forcing_batch(full_targets):
    """把完整目标序列错位成解码器输入与监督标签。"""
    decoder_input = full_targets[:, :-1]
    labels = full_targets[:, 1:]
    return decoder_input, labels


def greedy_decode(model, src_tokens, bos_id, eos_id, max_length):
    """从 BOS 开始，每一步选择概率最大的 token 继续生成。"""
    model.eval()

    generated = torch.full(
        (src_tokens.size(0), 1),
        bos_id,
        dtype=torch.long,
        device=src_tokens.device,
    )
    finished = torch.zeros(
        src_tokens.size(0),
        dtype=torch.bool,
        device=src_tokens.device,
    )

    with torch.no_grad():
        for _ in range(max_length - 1):
            logits = model(src_tokens, generated)

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eos_id),
                next_token,
            )
            finished = finished | (next_token.squeeze(1) == eos_id)

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if finished.all():
                break

    return generated

def run_shape_check():
    torch.manual_seed(42)
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 6, 8)
    value = torch.randn(2, 4, 6, 8)

    output, attention_weights = scaled_dot_product_attention(query, key, value)

    assert attention_weights.shape == (2, 4, 5, 6)
    assert output.shape == (2, 4, 5, 8)
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        torch.ones(2, 4, 5),
        atol=1e-6,
    )
    mask = torch.ones(2, 1, 5, 6, dtype=torch.bool)
    mask[..., -1] = False
    _, masked_weights = scaled_dot_product_attention(query, key, value, mask)
    assert torch.all(masked_weights[..., -1] == 0)

    multi_head_attention = MultiHeadAttention(d_model=8, num_heads=2)
    sequence = torch.randn(2, 5, 8)
    multi_head_output, multi_head_weights = multi_head_attention(
        sequence,
        sequence,
        sequence,
    )
    assert multi_head_output.shape == (2, 5, 8)
    assert multi_head_weights.shape == (2, 2, 5, 5)

    decoder_state = torch.randn(2, 3, 8)
    encoder_memory = torch.randn(2, 5, 8)
    cross_output, cross_weights = multi_head_attention(
        decoder_state,
        encoder_memory,
        encoder_memory,
    )
    assert cross_output.shape == (2, 3, 8)
    assert cross_weights.shape == (2, 2, 3, 5)

    encoder_layer = EncoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )
    encoder_output = encoder_layer(encoder_memory)
    assert encoder_output.shape == (2, 5, 8)

    decoder_layer = DecoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )
    decoder_output = decoder_layer(decoder_state, encoder_memory)
    assert decoder_output.shape == (2, 3, 8)

    src_tokens = torch.tensor([[4, 5, 0], [6, 0, 0]])
    src_mask = make_src_mask(src_tokens, pad_id=0)
    assert src_mask.shape == (2, 1, 1, 3)
    assert src_mask.tolist() == [
        [[[True, True, False]]],
        [[[True, False, False]]],
    ]

    tgt_tokens = torch.tensor([[2, 7, 8, 0]])
    tgt_mask = make_tgt_mask(tgt_tokens, pad_id=0)
    assert tgt_mask.shape == (1, 1, 4, 4)
    assert torch.all(tgt_mask[..., -1] == 0)
    assert torch.all(torch.triu(tgt_mask, diagonal=1) == 0)

    model = TransformerSeq2Seq(
        src_vocab_size=32,
        tgt_vocab_size=40,
        d_model=8,
        num_heads=2,
        num_layers=2,
        d_ff=16,
        dropout=0.0,
        pad_id=0,
        max_length=16,
    )
    src_batch = torch.tensor(
        [[4, 5, 3, 0, 0, 0], [6, 7, 8, 9, 3, 0]]
    )
    tgt_batch = torch.tensor(
        [[2, 10, 11, 12, 3], [2, 13, 14, 3, 0]]
    )
    logits = model(src_batch, tgt_batch)
    assert logits.shape == (2, 5, 40)
    assert torch.isfinite(logits).all()

    full_targets = torch.tensor(
        [[2, 10, 11, 3, 0], [2, 13, 14, 15, 3]]
    )
    decoder_input, labels = make_teacher_forcing_batch(full_targets)
    assert decoder_input.tolist() == [
        [2, 10, 11, 3],
        [2, 13, 14, 15],
    ]
    assert labels.tolist() == [
        [10, 11, 3, 0],
        [13, 14, 15, 3],
    ]

    generated = greedy_decode(
        model,
        src_batch[:1],
        bos_id=2,
        eos_id=3,
        max_length=6,
    )
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 6
    assert generated[0, 0].item() == 2

    class ScriptedGreedyModel(nn.Module):
        def forward(self, src_tokens, tgt_tokens):
            batch_size = src_tokens.size(0)
            target_length = tgt_tokens.size(1)
            logits = torch.full(
                (batch_size, target_length, 6),
                -1.0,
                device=tgt_tokens.device,
            )
            next_ids = torch.full(
                (batch_size,),
                4,
                dtype=torch.long,
                device=tgt_tokens.device,
            )
            if target_length == 2:
                next_ids[0] = 3
            if target_length == 4:
                next_ids[1] = 3
            logits[:, -1, :].scatter_(1, next_ids.unsqueeze(1), 1.0)
            return logits

    batch_generated = greedy_decode(
        ScriptedGreedyModel(),
        torch.tensor([[4], [5]]),
        bos_id=2,
        eos_id=3,
        max_length=8,
    )
    assert batch_generated.tolist() == [
        [2, 4, 3, 3, 3],
        [2, 4, 4, 4, 3],
    ]
    print("Scaled dot-product attention checks passed.")


if __name__ == "__main__":
    run_shape_check()
