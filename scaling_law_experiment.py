import json
import statistics
from pathlib import Path

import torch

from tiny_lm_experiment import (
    RAW_TRAIN_DOCUMENTS,
    VALIDATION_DOCUMENTS,
    VocabularyTokenizer,
    curated_process,
    run_variant,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "outputs" / "scaling_law_results.json"
STEP_BUDGETS = [20, 60, 120]
EXPERIMENT_SEEDS = [11, 22, 33, 44, 55]


def summarize_step_budgets(runs):
    summary = []
    for train_steps in STEP_BUDGETS:
        matching_results = [
            run["results_by_step"][str(train_steps)]
            for run in runs
        ]
        validation_losses = [
            result["validation_loss"] for result in matching_results
        ]
        summary.append({
            "train_steps": train_steps,
            "processed_token_count_proxy": matching_results[0][
                "processed_token_count_proxy"
            ],
            "validation_loss_mean": statistics.fmean(
                validation_losses
            ),
            "validation_loss_std": statistics.pstdev(
                validation_losses
            ),
            "mean_training_seconds": statistics.fmean(
                result["training_seconds"]
                for result in matching_results
            ),
        })
    return summary


def run_checks():
    fake_runs = [
        {
            "seed": seed,
            "results_by_step": {
                str(train_steps): {
                    "validation_loss": float(train_steps + seed),
                    "training_seconds": float(train_steps),
                    "processed_token_count_proxy": train_steps * 20,
                }
                for train_steps in STEP_BUDGETS
            },
        }
        for seed in [1, 2]
    ]
    summary = summarize_step_budgets(fake_runs)
    assert [row["train_steps"] for row in summary] == STEP_BUDGETS
    assert summary[0]["validation_loss_mean"] == 21.5
    assert summary[0]["processed_token_count_proxy"] == 400
    print("Scaling summary checks passed.")


def run_experiment():
    training_texts = curated_process(RAW_TRAIN_DOCUMENTS)
    tokenizer = VocabularyTokenizer(training_texts)
    runs = []
    for seed in EXPERIMENT_SEEDS:
        results_by_step = {}
        for train_steps in STEP_BUDGETS:
            results_by_step[str(train_steps)] = run_variant(
                name="normalized_and_deduplicated",
                training_texts=training_texts,
                validation_texts=VALIDATION_DOCUMENTS,
                tokenizer=tokenizer,
                seed=seed,
                train_steps=train_steps,
            )
        runs.append({
            "seed": seed,
            "results_by_step": results_by_step,
        })

    results = {
        "research_question": (
            "How does additional training compute affect validation loss "
            "for one fixed tiny language model?"
        ),
        "controlled_variables": {
            "seeds": EXPERIMENT_SEEDS,
            "training_data": "normalized and deduplicated",
            "model": "one-layer decoder-only Transformer",
            "tokenizer": "fixed word-level vocabulary",
            "validation_documents": VALIDATION_DOCUMENTS,
        },
        "changed_variable": "train_steps",
        "runs": runs,
        "summary": summarize_step_budgets(runs),
        "limitations": [
            "Training steps are only a compute proxy, not measured FLOPs.",
            "The corpus and model are too small for general scaling claims.",
            "The experiment does not compare model sizes or data sizes.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"saved={OUTPUT_PATH}")


def main():
    torch.set_num_threads(1)
    run_checks()
    run_experiment()


if __name__ == "__main__":
    main()
