import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "outputs" / "tiny_lm_comparison.json"
SEED = 42
EXPERIMENT_SEEDS = [11, 22, 33, 44, 55]
CONTEXT_LENGTH = 5
TRAIN_STEPS = 60
BATCH_SIZE = 4

RAW_TRAIN_DOCUMENTS = [
    "language models predict the next token",
    "language models predict the next token",
    "  language   models predict the next token  ",
    "transformers use attention to combine context",
    "attention lets models compare token representations",
    "clean data improves language model training",
    "diverse data helps models learn robust patterns",
    "validation loss measures generalization on held out data",
    "causal masks prevent models from seeing future tokens",
    "tokenizers convert text into token identifiers",
    "training updates parameters with gradient descent",
    "training updates parameters with gradient descent",
]

VALIDATION_DOCUMENTS = [
    "attention helps models combine context",
    "clean training data improves models",
    "validation loss measures model generalization",
]


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def baseline_process(documents):
    return [
        normalize_whitespace(document)
        for document in documents
        if normalize_whitespace(document)
    ]


def curated_process(documents):
    seen = set()
    curated = []
    for document in baseline_process(documents):
        if document not in seen:
            seen.add(document)
            curated.append(document)
    return curated


class VocabularyTokenizer:
    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self, training_texts):
        vocabulary = sorted({
            token
            for text in training_texts
            for token in self.tokenize(text)
        })
        tokens = self.SPECIAL_TOKENS + vocabulary
        self.token_to_id = {
            token: index for index, token in enumerate(tokens)
        }
        self.id_to_token = {
            index: token for token, index in self.token_to_id.items()
        }

    @staticmethod
    def tokenize(text):
        return re.findall(r"[a-z]+|[^\w\s]", text.lower())

    @property
    def pad_id(self):
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self):
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self):
        return self.token_to_id["<EOS>"]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def encode(self, text, include_eos=True):
        token_ids = [
            self.token_to_id.get(token, self.token_to_id["<UNK>"])
            for token in self.tokenize(text)
        ]
        encoded = [self.bos_id, *token_ids]
        if include_eos:
            encoded.append(self.eos_id)
        return encoded

    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if token == "<EOS>":
                break
            if token not in {"<PAD>", "<BOS>"}:
                tokens.append(token)
        return " ".join(tokens)


def build_next_token_examples(token_ids, context_length):
    """Build fixed-length sliding windows and their one-token-shifted labels."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    examples = []
    for start in range(len(token_ids) - context_length):
        model_input = token_ids[start:start + context_length]
        target = token_ids[start + 1:start + context_length + 1]
        examples.append((model_input, target))
    return examples


class LanguageModelDataset(Dataset):
    def __init__(self, texts, tokenizer, context_length):
        examples = []
        for text in texts:
            examples.extend(build_next_token_examples(
                tokenizer.encode(text),
                context_length,
            ))
        if not examples:
            raise ValueError("No language-model examples were created")
        self.inputs = torch.tensor(
            [model_input for model_input, _ in examples],
            dtype=torch.long,
        )
        self.targets = torch.tensor(
            [target for _, target in examples],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class TinyDecoderOnlyLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        dropout=0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError("Input is longer than context_length")
        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)
        hidden = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.layers(hidden, mask=causal_mask)
        return self.output_projection(hidden)


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for model_input, target in data_loader:
            logits = model(model_input)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count


def generate(model, tokenizer, prompt, max_new_tokens=12):
    model.eval()
    generated = tokenizer.encode(prompt, include_eos=False)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            model_input = torch.tensor(
                [generated[-model.context_length:]],
                dtype=torch.long,
            )
            next_token = int(model(model_input)[0, -1].argmax())
            generated.append(next_token)
            if next_token == tokenizer.eos_id:
                break
    return tokenizer.decode(generated)


def run_variant(
    name,
    training_texts,
    validation_texts,
    tokenizer,
    seed,
    train_steps=TRAIN_STEPS,
):
    set_seed(seed)
    train_dataset = LanguageModelDataset(
        training_texts,
        tokenizer,
        CONTEXT_LENGTH,
    )
    validation_dataset = LanguageModelDataset(
        validation_texts,
        tokenizer,
        CONTEXT_LENGTH,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    model = TinyDecoderOnlyLM(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    iterator = iter(train_loader)
    recent_losses = []
    start_time = time.perf_counter()
    model.train()
    for _ in range(train_steps):
        try:
            model_input, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            model_input, target = next(iterator)
        optimizer.zero_grad()
        logits = model(model_input)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        recent_losses.append(loss.item())
    training_seconds = time.perf_counter() - start_time

    return {
        "name": name,
        "seed": seed,
        "train_steps": train_steps,
        "processed_token_count_proxy": (
            train_steps * BATCH_SIZE * CONTEXT_LENGTH
        ),
        "training_document_count": len(training_texts),
        "training_token_count": sum(
            len(tokenizer.encode(text)) for text in training_texts
        ),
        "training_example_count": len(train_dataset),
        "last_10_step_train_loss": (
            sum(recent_losses[-10:]) / len(recent_losses[-10:])
        ),
        "validation_loss": evaluate(model, validation_loader, loss_fn),
        "training_seconds": training_seconds,
        "generated_sample": generate(model, tokenizer, "language models"),
    }


def summarize_paired_runs(paired_runs):
    baseline_losses = [
        run["normalized_with_duplicates"]["validation_loss"]
        for run in paired_runs
    ]
    curated_losses = [
        run["normalized_and_deduplicated"]["validation_loss"]
        for run in paired_runs
    ]
    paired_differences = [
        curated_loss - baseline_loss
        for baseline_loss, curated_loss
        in zip(baseline_losses, curated_losses)
    ]
    return {
        "normalized_with_duplicates_validation_loss_mean": (
            statistics.fmean(baseline_losses)
        ),
        "normalized_with_duplicates_validation_loss_std": (
            statistics.pstdev(baseline_losses)
        ),
        "normalized_and_deduplicated_validation_loss_mean": (
            statistics.fmean(curated_losses)
        ),
        "normalized_and_deduplicated_validation_loss_std": (
            statistics.pstdev(curated_losses)
        ),
        "curated_minus_baseline_mean_difference": (
            statistics.fmean(paired_differences)
        ),
        "curated_minus_baseline_difference_std": (
            statistics.pstdev(paired_differences)
        ),
        "curated_win_rate": sum(
            difference < 0 for difference in paired_differences
        ) / len(paired_differences),
        "paired_validation_loss_differences": [
            {
                "seed": run["seed"],
                "curated_minus_baseline": difference,
            }
            for run, difference in zip(paired_runs, paired_differences)
        ],
    }


def run_checks():
    assert build_next_token_examples([10, 11], 2) == []
    assert build_next_token_examples([10, 11, 12], 2) == [
        ([10, 11], [11, 12]),
    ]
    assert build_next_token_examples([10, 11, 12, 13], 2) == [
        ([10, 11], [11, 12]),
        ([11, 12], [12, 13]),
    ]
    try:
        build_next_token_examples([10, 11], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("context_length=0 must raise ValueError")
    print("Next-token example checks passed.")

    baseline = baseline_process(RAW_TRAIN_DOCUMENTS)
    curated = curated_process(RAW_TRAIN_DOCUMENTS)
    assert len(curated) < len(baseline)
    assert set(curated) == set(baseline)
    assert VALIDATION_DOCUMENTS == list(VALIDATION_DOCUMENTS)
    print("Data-processing variant checks passed.")

    model = TinyDecoderOnlyLM(vocab_size=20, context_length=5)
    logits = model(torch.tensor([[1, 2, 3, 4, 5]]))
    assert logits.shape == (1, 5, 20)
    print("Tiny decoder-only shape checks passed.")

    paired_summary = summarize_paired_runs([
        {
            "seed": 1,
            "normalized_with_duplicates": {"validation_loss": 3.0},
            "normalized_and_deduplicated": {"validation_loss": 2.5},
        },
        {
            "seed": 2,
            "normalized_with_duplicates": {"validation_loss": 2.0},
            "normalized_and_deduplicated": {"validation_loss": 2.5},
        },
    ])
    assert paired_summary["curated_win_rate"] == 0.5
    assert paired_summary["curated_minus_baseline_mean_difference"] == 0.0
    assert paired_summary["curated_minus_baseline_difference_std"] == 0.5
    print("Paired multi-seed summary checks passed.")


def run_experiment():
    baseline_texts = baseline_process(RAW_TRAIN_DOCUMENTS)
    curated_texts = curated_process(RAW_TRAIN_DOCUMENTS)
    tokenizer = VocabularyTokenizer(baseline_texts)
    paired_runs = []
    for seed in EXPERIMENT_SEEDS:
        paired_runs.append({
            "seed": seed,
            "normalized_with_duplicates": run_variant(
                "normalized_with_duplicates",
                baseline_texts,
                VALIDATION_DOCUMENTS,
                tokenizer,
                seed,
            ),
            "normalized_and_deduplicated": run_variant(
                "normalized_and_deduplicated",
                curated_texts,
                VALIDATION_DOCUMENTS,
                tokenizer,
                seed,
            ),
        })
    results = {
        "controlled_variables": {
            "paired_seeds": EXPERIMENT_SEEDS,
            "context_length": CONTEXT_LENGTH,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "tokenizer_vocabulary_size": tokenizer.vocab_size,
            "validation_documents": VALIDATION_DOCUMENTS,
            "model": "one-layer decoder-only Transformer",
        },
        "changed_variable": "training data processing method",
        "paired_runs": paired_runs,
        "summary": summarize_paired_runs(paired_runs),
        "limitations": [
            "The corpus is intentionally tiny and cannot support general claims.",
            "Five seeds reduce but do not eliminate random uncertainty.",
            "Training time on a tiny CPU experiment is noisy.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved={OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_checks()
    if not args.check_only:
        run_experiment()


if __name__ == "__main__":
    main()
