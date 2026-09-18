import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformer_translation import (
    TransformerSeq2Seq,
    greedy_decode,
    make_teacher_forcing_batch,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "translation_sample.tsv"
OUTPUT_DIR = ROOT / "outputs"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MAX_LENGTH = 32
EPOCHS = 20


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_english(text):
    return re.findall(r"[a-z']+|[.,!?;:]", text.lower())


def tokenize_chinese(text):
    return [character for character in text if not character.isspace()]


def load_pairs(path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        english, chinese = line.split("\t", maxsplit=1)
        pairs.append((tokenize_english(english), tokenize_chinese(chinese)))
    return pairs


def build_vocab(token_sequences):
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}
    counts = Counter(token for sequence in token_sequences for token in sequence)
    for token in sorted(counts):
        vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab, add_bos=False, add_eos=True):
    reserved = int(add_bos) + int(add_eos)
    body = [vocab.get(token, UNK_ID) for token in tokens[: MAX_LENGTH - reserved]]
    ids = ([BOS_ID] if add_bos else []) + body + ([EOS_ID] if add_eos else [])
    ids += [PAD_ID] * (MAX_LENGTH - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.examples = [
            (
                encode(src_tokens, src_vocab, add_bos=False),
                encode(tgt_tokens, tgt_vocab, add_bos=True),
            )
            for src_tokens, tgt_tokens in pairs
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def compute_loss(model, src, full_targets, loss_fn):
    decoder_input, labels = make_teacher_forcing_batch(full_targets)
    logits = model(src, decoder_input)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return loss, logits, labels


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for src, targets in loader:
            src = src.to(next(model.parameters()).device)
            targets = targets.to(src.device)

            if training:
                optimizer.zero_grad()

            loss, logits, labels = compute_loss(model, src, targets, loss_fn)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            valid = labels != PAD_ID
            total_correct += ((predictions == labels) & valid).sum().item()
            total_tokens += valid.sum().item()

    return {
        "loss": total_loss / len(loader),
        "token_accuracy": total_correct / max(total_tokens, 1),
    }


def decode_tokens(token_ids, id_to_token):
    result = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id not in {PAD_ID, BOS_ID}:
            result.append(id_to_token.get(token_id, "<UNK>"))
    return "".join(result)


def save_curve(train_losses, val_losses):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height, margin = 800, 500, 60
    values = train_losses + val_losses
    value_min = min(values)
    value_max = max(values)
    value_range = max(value_max - value_min, 1e-9)

    def point(index, value):
        x = margin + index * (width - 2 * margin) / max(len(train_losses) - 1, 1)
        y = height - margin - (value - value_min) * (height - 2 * margin) / value_range
        return f"{x:.1f},{y:.1f}"

    train_points = " ".join(
        point(index, value) for index, value in enumerate(train_losses)
    )
    val_points = " ".join(
        point(index, value) for index, value in enumerate(val_losses)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-size="20">Transformer training curve</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{train_points}"/>
<polyline fill="none" stroke="#dc2626" stroke-width="3" points="{val_points}"/>
<text x="{width - 210}" y="55" fill="#2563eb">train loss</text>
<text x="{width - 110}" y="55" fill="#dc2626">validation loss</text>
<text x="{width / 2}" y="{height - 15}" text-anchor="middle">epoch</text>
<text x="15" y="{height / 2}" transform="rotate(-90 15 {height / 2})" text-anchor="middle">cross-entropy loss</text>
<text x="{margin - 8}" y="{margin + 5}" text-anchor="end" font-size="12">{value_max:.3f}</text>
<text x="{margin - 8}" y="{height - margin + 5}" text-anchor="end" font-size="12">{value_min:.3f}</text>
</svg>
"""
    (OUTPUT_DIR / "training_curve.svg").write_text(svg, encoding="utf-8")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(DATA_PATH)
    train_pairs = pairs[:10]
    val_pairs = pairs[10:]

    src_vocab = build_vocab(src for src, _ in train_pairs)
    tgt_vocab = build_vocab(tgt for _, tgt in train_pairs)
    tgt_id_to_token = {index: token for token, index in tgt_vocab.items()}

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        dropout=0.1,
        pad_id=PAD_ID,
        max_length=MAX_LENGTH,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    train_losses = []
    val_losses = []
    history = []
    best_val_loss = float("inf")
    best_state = None
    best_validation = None
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn)
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_token_accuracy": train_metrics["token_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_token_accuracy": val_metrics["token_accuracy"],
            }
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_validation = history[-1].copy()
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    model.load_state_dict(best_state)
    model.eval()
    val_src, val_target = val_dataset[0]
    generated = greedy_decode(
        model,
        val_src.unsqueeze(0).to(device),
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_length=MAX_LENGTH,
    )[0].cpu()

    results = {
        "seed": 42,
        "device": str(device),
        "epochs": EPOCHS,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "best_validation": best_validation,
        "final_train": history[-1],
        "generated_validation_text": decode_tokens(generated, tgt_id_to_token),
        "reference_validation_text": decode_tokens(val_target, tgt_id_to_token),
        "history": history,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_curve(train_losses, val_losses)
    (OUTPUT_DIR / "validation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), OUTPUT_DIR / "transformer_demo.pt")
    print(f"saved={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
