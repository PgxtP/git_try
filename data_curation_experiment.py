import re


GOLD_EVALUATION_TEXTS = [
    "model predicts next token from context",
    "model uses context to predict next token",
    "model predicts previous token from future context",
    "tokenizer converts text into token ids",
    "text becomes token ids through tokenizer",
]

GOLD_DUPLICATE_LABELS = {
    (0, 1): True,
    (0, 2): False,
    (0, 3): False,
    (0, 4): False,
    (1, 2): False,
    (1, 3): False,
    (1, 4): False,
    (2, 3): False,
    (2, 4): False,
    (3, 4): True,
}


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_exact(texts):
    """保留每个字符串第一次出现的位置，删除后续完全相同的字符串。"""
    seen = set()
    unique_texts = []

    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    return unique_texts


def normalize_and_deduplicate(texts):
    """先规范化每条文本的空白，再保持原顺序进行精确去重。"""
    return deduplicate_exact([normalize_whitespace(text) for text in texts])


def deduplicate_records_by_context_and_text(records):
    """按规范化后的上下文与文本去重，并保留第一次出现的记录。"""
    seen = set()
    unique_records = []

    for record in records:
        key = (
            normalize_whitespace(record["context"]),
            normalize_whitespace(record["text"]),
        )
        if key not in seen:
            seen.add(key)
            unique_records.append(record)

    return unique_records


def deduplicate_records_and_merge_sources(records):
    """按上下文与文本去重，并把重复记录的来源合并为列表。"""
    merged_by_key = {}
    merged_records = []

    for record in records:
        key = (
            normalize_whitespace(record["context"]),
            normalize_whitespace(record["text"]),
        )
        if key not in merged_by_key:
            merged_record = record.copy()
            merged_record["sources"] = [merged_record.pop("source")]
            merged_by_key[key] = merged_record
            merged_records.append(merged_record)
            continue

        source = record["source"]
        sources = merged_by_key[key]["sources"]
        if source not in sources:
            sources.append(source)

    return merged_records


def token_jaccard_similarity(text_a, text_b):
    """计算两段文本的 token 集合 Jaccard 相似度。"""
    tokens_a = set(normalize_whitespace(text_a).split())
    tokens_b = set(normalize_whitespace(text_b).split())

    if not tokens_a and not tokens_b:
        return 1.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def find_near_duplicate_pairs(texts, threshold):
    """返回达到相似度阈值的无序文本索引对与分数。"""
    candidates = []

    for left_index in range(len(texts)):
        for right_index in range(left_index + 1, len(texts)):
            score = token_jaccard_similarity(
                texts[left_index], texts[right_index]
            )
            if score >= threshold:
                candidates.append((left_index, right_index, score))

    return candidates


def build_token_inverted_index(texts):
    """建立 token 到包含该 token 的文本编号列表的映射。"""
    inverted_index = {}
    for text_index, text in enumerate(texts):
        tokens = set(normalize_whitespace(text).split())
        for token in tokens:
            inverted_index.setdefault(token, []).append(text_index)
    return inverted_index


def candidate_pairs_from_inverted_index(
    texts,
    max_document_frequency=None,
    max_document_frequency_ratio=None,
):
    """只为至少共享一个 token 的文本生成无序候选对。"""
    if (
        max_document_frequency is not None
        and max_document_frequency_ratio is not None
    ):
        raise ValueError("文档频率计数与比例阈值不能同时设置")
    if (
        max_document_frequency_ratio is not None
        and not 0.0 <= max_document_frequency_ratio <= 1.0
    ):
        raise ValueError("文档频率比例必须位于 0 到 1 之间")

    inverted_index = build_token_inverted_index(texts)
    candidate_pairs = set()

    for text_indices in inverted_index.values():
        if (
            max_document_frequency is not None
            and len(text_indices) > max_document_frequency
        ):
            continue
        if (
            max_document_frequency_ratio is not None
            and len(text_indices) / len(texts)
            > max_document_frequency_ratio
        ):
            continue
        if len(text_indices) > 1:
            for index_1 in range(len(text_indices)):
                for index_2 in range(index_1 + 1, len(text_indices)):
                    candidate_pairs.add(
                        (text_indices[index_1], text_indices[index_2])
                    )
    return sorted(candidate_pairs)


def merge_candidate_pair_channels(*candidate_channels):
    """使用并集合并多个候选生成通道的无序文本对。"""
    merged_pairs = set()
    for candidate_pairs in candidate_channels:
        merged_pairs.update(candidate_pairs)
    return sorted(merged_pairs)


def find_near_duplicate_pairs_with_inverted_index(
    texts,
    threshold,
    max_document_frequency=None,
    max_document_frequency_ratio=None,
):
    """先用倒排索引缩小比较范围，再计算 Jaccard 相似度。"""
    candidates = []
    for left_index, right_index in candidate_pairs_from_inverted_index(
        texts,
        max_document_frequency=max_document_frequency,
        max_document_frequency_ratio=max_document_frequency_ratio,
    ):
        score = token_jaccard_similarity(
            texts[left_index],
            texts[right_index],
        )
        if score >= threshold:
            candidates.append((left_index, right_index, score))
    return candidates


def evaluate_inverted_index_prefilter(
    texts,
    duplicate_labels,
    max_document_frequency=None,
    max_document_frequency_ratio=None,
):
    """评估倒排索引预筛的召回率与实际比较缩减率。"""
    prefilter_pairs = set(candidate_pairs_from_inverted_index(
        texts,
        max_document_frequency=max_document_frequency,
        max_document_frequency_ratio=max_document_frequency_ratio,
    ))
    gold_duplicate_pairs = {
        pair for pair, is_duplicate in duplicate_labels.items()
        if is_duplicate
    }
    retained_gold_pairs = prefilter_pairs & gold_duplicate_pairs
    total_pair_count = pairwise_comparison_count(len(texts))

    if gold_duplicate_pairs:
        blocking_recall = (
            len(retained_gold_pairs) / len(gold_duplicate_pairs)
        )
    else:
        blocking_recall = None

    if total_pair_count == 0:
        comparison_reduction_rate = 0.0
    else:
        comparison_reduction_rate = (
            total_pair_count - len(prefilter_pairs)
        ) / total_pair_count

    return {
        "total_pair_count": total_pair_count,
        "prefilter_pair_count": len(prefilter_pairs),
        "gold_duplicate_pair_count": len(gold_duplicate_pairs),
        "retained_gold_pair_count": len(retained_gold_pairs),
        "blocking_recall": blocking_recall,
        "comparison_reduction_rate": comparison_reduction_rate,
    }


def pairwise_comparison_count(item_count):
    """返回无序两两比较所需的候选对数量。"""
    return item_count * (item_count - 1) // 2


def build_gold_pair_records(texts, duplicate_labels):
    """把全部无序文本对及其人工标签整理为统一记录。"""
    records = []
    for left_index in range(len(texts)):
        for right_index in range(left_index + 1, len(texts)):
            records.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_text": texts[left_index],
                    "right_text": texts[right_index],
                    "is_duplicate": duplicate_labels[(
                        left_index,
                        right_index,
                    )],
                }
            )
    return records


def evaluate_candidate_generation(texts, duplicate_labels, threshold):
    """用人工金标准评估近似重复候选生成效果。"""
    candidate_results = find_near_duplicate_pairs(texts, threshold)
    candidate_pairs = {
        (left_index, right_index)
        for left_index, right_index, _ in candidate_results
    }
    gold_duplicate_pairs = {
        pair for pair, is_duplicate in duplicate_labels.items()
        if is_duplicate
    }
    retrieved_gold_pairs = candidate_pairs & gold_duplicate_pairs

    total_pair_count = pairwise_comparison_count(len(texts))
    candidate_pair_count = len(candidate_pairs)
    gold_duplicate_pair_count = len(gold_duplicate_pairs)

    if gold_duplicate_pair_count == 0:
        candidate_recall = None
    else:
        candidate_recall = (
            len(retrieved_gold_pairs) / gold_duplicate_pair_count
        )

    if candidate_pair_count == 0:
        candidate_precision = None
    else:
        candidate_precision = (
            len(retrieved_gold_pairs) / candidate_pair_count
        )

    if total_pair_count == 0:
        candidate_reduction_rate = 0.0
    else:
        candidate_reduction_rate = (
            total_pair_count - candidate_pair_count
        ) / total_pair_count

    return {
        "total_pair_count": total_pair_count,
        "candidate_pair_count": candidate_pair_count,
        "gold_duplicate_pair_count": gold_duplicate_pair_count,
        "retrieved_gold_pair_count": len(retrieved_gold_pairs),
        "candidate_recall": candidate_recall,
        "candidate_precision": candidate_precision,
        "candidate_reduction_rate": candidate_reduction_rate,
    }


def candidate_threshold_report(texts, duplicate_labels, thresholds):
    """比较多个相似度阈值下的候选生成指标。"""
    return [
        {
            "threshold": threshold,
            **evaluate_candidate_generation(
                texts,
                duplicate_labels,
                threshold,
            ),
        }
        for threshold in thresholds
    ]


def select_threshold_by_minimum_recall(report, minimum_recall):
    """在满足最低召回率的方案中选择候选缩减率最高者。"""
    feasible_rows = [
        row for row in report
        if row["candidate_recall"] is not None
        and row["candidate_recall"] >= minimum_recall
    ]
    if not feasible_rows:
        return None
    return max(
        feasible_rows,
        key=lambda row: (
            row["candidate_reduction_rate"],
            row["threshold"],
        ),
    )


def candidate_error_pairs(texts, duplicate_labels, threshold):
    """返回候选生成中的假阳性与假阴性文本对。"""
    candidate_pairs = {
        (left_index, right_index)
        for left_index, right_index, _
        in find_near_duplicate_pairs(texts, threshold)
    }
    gold_duplicate_pairs = {
        pair for pair, is_duplicate in duplicate_labels.items()
        if is_duplicate
    }
    return {
        "false_positive_pairs": sorted(
            candidate_pairs - gold_duplicate_pairs
        ),
        "false_negative_pairs": sorted(
            gold_duplicate_pairs - candidate_pairs
        ),
    }


def calculate_removal_rate(original_count, kept_count):
    """计算去重过程中被删除文本所占的比例。"""
    if original_count == 0:
        return 0.0
    return (original_count - kept_count) / original_count


def weighted_source_error_rate(source_error_rates, source_weights):
    """按真实来源占比汇总各来源的误删率。"""
    if source_error_rates.keys() != source_weights.keys():
        raise ValueError("误删率与来源权重必须包含相同来源")

    total_weight = sum(source_weights.values())
    if total_weight == 0:
        return 0.0

    weighted_errors = sum(
        source_error_rates[source] * source_weights[source]
        for source in source_error_rates
    )
    return weighted_errors / total_weight


def compare_dedup_strategies(texts):
    """在同一输入上比较直接去重与规范化后去重。"""
    original_count = len(texts)
    raw_unique = deduplicate_exact(texts)
    normalized_unique = normalize_and_deduplicate(texts)
    return {
        "original_count": original_count,
        "raw_unique_count": len(raw_unique),
        "normalized_unique_count": len(normalized_unique),
        "raw_removal_rate": calculate_removal_rate(
            original_count, len(raw_unique)
        ),
        "normalized_removal_rate": calculate_removal_rate(
            original_count, len(normalized_unique)
        ),
    }


def curate_texts(texts):
    """依次执行空白规范化、空文本过滤和精确去重。"""
    normalized_texts = [normalize_whitespace(text) for text in texts]
    nonempty_texts = [text for text in normalized_texts if text]
    return deduplicate_exact(nonempty_texts)


def filter_by_min_length(texts, min_length):
    """保留长度不小于阈值的已规范化文本。"""
    return [text for text in texts if len(text) >= min_length]


def partition_by_min_length(texts, min_length):
    """按最小长度规则拆分保留样本与删除样本。"""
    kept_texts = []
    removed_texts = []

    for text in texts:
        if len(text) >= min_length:
            kept_texts.append(text)
        else:
            removed_texts.append(text)
    return kept_texts, removed_texts


def length_sensitivity_report(texts, thresholds):
    """记录多个最小长度阈值下的保留数量。"""
    report = {}
    for threshold in thresholds:
        report[threshold] = len(filter_by_min_length(texts, threshold))
    return report


def curate_with_min_length(texts, min_length):
    """清洗文本后，再执行最小长度过滤。"""
    return filter_by_min_length(curate_texts(texts), min_length)


def summarize_curation_stages(texts):
    """统计空文本过滤和精确去重两个阶段各自删除的数量。"""
    original_count = len(texts)
    normalized_texts = [normalize_whitespace(text) for text in texts]
    nonempty_texts = [text for text in normalized_texts if text]
    unique_texts = deduplicate_exact(nonempty_texts)

    return {
        "original_count": original_count,
        "nonempty_count": len(nonempty_texts),
        "unique_count": len(unique_texts),
        "empty_removed_count": len(normalized_texts) - len(nonempty_texts),
        "duplicate_removed_count": len(nonempty_texts) - len(unique_texts),
    }


def summarize_full_curation_stages(texts, min_length):
    """统计包含长度过滤在内的完整清洗阶段。"""
    original_count = len(texts)
    normalized_texts = [normalize_whitespace(text) for text in texts]
    nonempty_texts = [text for text in normalized_texts if text]
    unique_texts = deduplicate_exact(nonempty_texts)
    final_texts = filter_by_min_length(unique_texts, min_length)

    return {
        "original_count": original_count,
        "nonempty_count": len(nonempty_texts),
        "unique_count": len(unique_texts),
        "final_count": len(final_texts),
        "empty_removed_count": len(normalized_texts) - len(nonempty_texts),
        "duplicate_removed_count": len(nonempty_texts) - len(unique_texts),
        "length_removed_count": len(unique_texts) - len(final_texts),
    }


def is_consistent_curation_summary(summary):
    """检查分阶段清洗指标是否满足数量守恒关系。"""
    original_count = summary["original_count"]
    nonempty_count = summary["nonempty_count"]
    unique_count = summary["unique_count"]
    empty_removed_count = summary["empty_removed_count"]
    duplicate_removed_count = summary["duplicate_removed_count"]
    base_consistent = (
        original_count - empty_removed_count == nonempty_count
        and nonempty_count - duplicate_removed_count == unique_count
    )
    if not base_consistent:
        return False

    has_final_count = "final_count" in summary
    has_length_removed_count = "length_removed_count" in summary
    if has_final_count != has_length_removed_count:
        return False
    if not has_final_count:
        return True

    return (
        unique_count - summary["length_removed_count"]
        == summary["final_count"]
    )


def run_checks():
    assert normalize_whitespace("  Hello\t  world\n") == "Hello world"
    assert normalize_whitespace("machine   learning") == "machine learning"
    assert normalize_whitespace("clean") == "clean"
    print("Whitespace normalization checks passed.")

    assert deduplicate_exact(["A", "B", "A", "C", "B"]) == [
        "A",
        "B",
        "C",
    ]
    assert deduplicate_exact(["A", "a", "A ", "A"]) == [
        "A",
        "a",
        "A ",
    ]
    assert deduplicate_exact([]) == []
    print("Exact deduplication checks passed.")

    records = [
        {"context": "Confirm submission?", "text": "OK", "source": "forum"},
        {"context": "Is the server healthy?", "text": "OK", "source": "code"},
        {"context": " Confirm submission? ", "text": " OK ", "source": "news"},
    ]
    assert deduplicate_records_by_context_and_text(records) == records[:2]
    assert deduplicate_records_by_context_and_text([]) == []
    print("Context-aware deduplication checks passed.")

    merged_records = deduplicate_records_and_merge_sources(records)
    assert merged_records == [
        {
            "context": "Confirm submission?",
            "text": "OK",
            "sources": ["forum", "news"],
        },
        {
            "context": "Is the server healthy?",
            "text": "OK",
            "sources": ["code"],
        },
    ]
    assert records[0]["source"] == "forum"
    assert deduplicate_records_and_merge_sources([]) == []
    print("Source-merging deduplication checks passed.")

    assert token_jaccard_similarity("a b c", "c b a") == 1.0
    assert abs(
        token_jaccard_similarity(
            "machine learning is useful",
            "machine learning is very useful",
        )
        - 0.8
    ) < 1e-9
    assert abs(
        token_jaccard_similarity(
            "Python list keeps order",
            "Python set keeps order",
        )
        - 0.6
    ) < 1e-9
    assert token_jaccard_similarity("", "") == 1.0
    assert token_jaccard_similarity("", "text") == 0.0
    print("Token Jaccard checks passed.")

    candidate_pairs = find_near_duplicate_pairs(
        ["a b c", "a b c d", "x y"], 0.75
    )
    assert candidate_pairs == [(0, 1, 0.75)]
    assert find_near_duplicate_pairs(
        ["a b c", "a b c d", "x y"], 0.8
    ) == []
    assert find_near_duplicate_pairs([], 0.75) == []
    print("Near-duplicate candidate checks passed.")

    assert build_token_inverted_index(["a b", "b c", "x"]) == {
        "a": [0],
        "b": [0, 1],
        "c": [1],
        "x": [2],
    }
    assert candidate_pairs_from_inverted_index(
        ["a b", "b c", "x"]
    ) == [(0, 1)]
    assert candidate_pairs_from_inverted_index(
        ["a b", "a b", "b"]
    ) == [(0, 1), (0, 2), (1, 2)]
    assert candidate_pairs_from_inverted_index([]) == []
    print("Inverted-index candidate checks passed.")

    assert merge_candidate_pair_channels(
        {(0, 2), (3, 4)},
        {(0, 1), (3, 4)},
    ) == [(0, 1), (0, 2), (3, 4)]
    assert merge_candidate_pair_channels() == []
    print("Candidate-channel union checks passed.")

    assert len(candidate_pairs_from_inverted_index(
        GOLD_EVALUATION_TEXTS
    )) == 10
    filtered_inverted_pairs = candidate_pairs_from_inverted_index(
        GOLD_EVALUATION_TEXTS,
        max_document_frequency=3,
    )
    assert filtered_inverted_pairs == [
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
    ]
    assert candidate_pairs_from_inverted_index(
        GOLD_EVALUATION_TEXTS,
        max_document_frequency_ratio=0.6,
    ) == filtered_inverted_pairs
    filtered_near_duplicates = (
        find_near_duplicate_pairs_with_inverted_index(
            GOLD_EVALUATION_TEXTS,
            threshold=0.4,
            max_document_frequency=3,
        )
    )
    assert [
        (left_index, right_index)
        for left_index, right_index, _ in filtered_near_duplicates
    ] == [(0, 1), (0, 2), (3, 4)]
    print("Frequency-filtered inverted-index checks passed.")

    try:
        candidate_pairs_from_inverted_index(
            GOLD_EVALUATION_TEXTS,
            max_document_frequency_ratio=1.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("无效文档频率比例应触发 ValueError")
    print("Document-frequency ratio checks passed.")

    assert evaluate_inverted_index_prefilter(
        GOLD_EVALUATION_TEXTS,
        GOLD_DUPLICATE_LABELS,
        max_document_frequency=3,
    ) == {
        "total_pair_count": 10,
        "prefilter_pair_count": 4,
        "gold_duplicate_pair_count": 2,
        "retained_gold_pair_count": 2,
        "blocking_recall": 1.0,
        "comparison_reduction_rate": 0.6,
    }
    assert evaluate_inverted_index_prefilter(
        GOLD_EVALUATION_TEXTS,
        GOLD_DUPLICATE_LABELS,
        max_document_frequency_ratio=0.6,
    )["prefilter_pair_count"] == 4
    assert evaluate_inverted_index_prefilter([], {}) == {
        "total_pair_count": 0,
        "prefilter_pair_count": 0,
        "gold_duplicate_pair_count": 0,
        "retained_gold_pair_count": 0,
        "blocking_recall": None,
        "comparison_reduction_rate": 0.0,
    }
    print("Inverted-index prefilter evaluation checks passed.")

    gold_records = build_gold_pair_records(
        GOLD_EVALUATION_TEXTS,
        GOLD_DUPLICATE_LABELS,
    )
    assert len(gold_records) == pairwise_comparison_count(
        len(GOLD_EVALUATION_TEXTS)
    )
    assert all(
        record["left_index"] < record["right_index"]
        for record in gold_records
    )
    assert all(
        type(record["is_duplicate"]) is bool
        for record in gold_records
    ), "请完成 GOLD_DUPLICATE_LABELS 中的全部 TODO"
    print("Gold duplicate-pair labeling checks passed.")

    candidate_evaluation = evaluate_candidate_generation(
        GOLD_EVALUATION_TEXTS,
        GOLD_DUPLICATE_LABELS,
        threshold=0.5,
    )
    assert candidate_evaluation == {
        "total_pair_count": 10,
        "candidate_pair_count": 2,
        "gold_duplicate_pair_count": 2,
        "retrieved_gold_pair_count": 1,
        "candidate_recall": 0.5,
        "candidate_precision": 0.5,
        "candidate_reduction_rate": 0.8,
    }
    assert evaluate_candidate_generation([], {}, threshold=0.5) == {
        "total_pair_count": 0,
        "candidate_pair_count": 0,
        "gold_duplicate_pair_count": 0,
        "retrieved_gold_pair_count": 0,
        "candidate_recall": None,
        "candidate_precision": None,
        "candidate_reduction_rate": 0.0,
    }
    print("Candidate-generation evaluation checks passed.")

    threshold_report = candidate_threshold_report(
        GOLD_EVALUATION_TEXTS,
        GOLD_DUPLICATE_LABELS,
        thresholds=[0.4, 0.5, 0.6, 0.7],
    )
    assert threshold_report == [
        {
            "threshold": 0.4,
            "total_pair_count": 10,
            "candidate_pair_count": 3,
            "gold_duplicate_pair_count": 2,
            "retrieved_gold_pair_count": 2,
            "candidate_recall": 1.0,
            "candidate_precision": 2 / 3,
            "candidate_reduction_rate": 0.7,
        },
        {
            "threshold": 0.5,
            "total_pair_count": 10,
            "candidate_pair_count": 2,
            "gold_duplicate_pair_count": 2,
            "retrieved_gold_pair_count": 1,
            "candidate_recall": 0.5,
            "candidate_precision": 0.5,
            "candidate_reduction_rate": 0.8,
        },
        {
            "threshold": 0.6,
            "total_pair_count": 10,
            "candidate_pair_count": 1,
            "gold_duplicate_pair_count": 2,
            "retrieved_gold_pair_count": 0,
            "candidate_recall": 0.0,
            "candidate_precision": 0.0,
            "candidate_reduction_rate": 0.9,
        },
        {
            "threshold": 0.7,
            "total_pair_count": 10,
            "candidate_pair_count": 0,
            "gold_duplicate_pair_count": 2,
            "retrieved_gold_pair_count": 0,
            "candidate_recall": 0.0,
            "candidate_precision": None,
            "candidate_reduction_rate": 1.0,
        },
    ]
    print("Candidate-threshold report checks passed.")

    selected_threshold = select_threshold_by_minimum_recall(
        threshold_report,
        minimum_recall=1.0,
    )
    assert selected_threshold == threshold_report[0]
    assert select_threshold_by_minimum_recall(
        threshold_report,
        minimum_recall=1.1,
    ) is None
    print("Constrained threshold-selection checks passed.")

    assert candidate_error_pairs(
        GOLD_EVALUATION_TEXTS,
        GOLD_DUPLICATE_LABELS,
        threshold=0.5,
    ) == {
        "false_positive_pairs": [(0, 2)],
        "false_negative_pairs": [(0, 1)],
    }
    print("Candidate-error analysis checks passed.")

    assert pairwise_comparison_count(0) == 0
    assert pairwise_comparison_count(1) == 0
    assert pairwise_comparison_count(4) == 6
    assert pairwise_comparison_count(1000) == 499500
    print("Pairwise scaling checks passed.")

    assert normalize_and_deduplicate(
        ["Hello   world", " Hello world "]
    ) == ["Hello world"]
    assert normalize_and_deduplicate(
        ["A\tB", "A B", "a b", "A  B"]
    ) == ["A B", "a b"]
    assert normalize_and_deduplicate([]) == []
    print("Normalization-then-deduplication checks passed.")

    assert calculate_removal_rate(8, 6) == 0.25
    assert calculate_removal_rate(10, 10) == 0.0
    assert calculate_removal_rate(0, 0) == 0.0
    print("Removal-rate checks passed.")

    weighted_error_rate = weighted_source_error_rate(
        {"forum": 0.1, "code": 0.4, "news": 0.2},
        {"forum": 0.9, "code": 0.05, "news": 0.05},
    )
    assert abs(weighted_error_rate - 0.12) < 1e-9
    assert weighted_source_error_rate({}, {}) == 0.0
    print("Source-weighted error-rate checks passed.")

    comparison = compare_dedup_strategies(
        ["A B", "A  B", "C", "C", " D "]
    )
    assert comparison == {
        "original_count": 5,
        "raw_unique_count": 4,
        "normalized_unique_count": 3,
        "raw_removal_rate": 0.2,
        "normalized_removal_rate": 0.4,
    }
    assert compare_dedup_strategies([]) == {
        "original_count": 0,
        "raw_unique_count": 0,
        "normalized_unique_count": 0,
        "raw_removal_rate": 0.0,
        "normalized_removal_rate": 0.0,
    }
    print("Deduplication comparison checks passed.")

    assert curate_texts(["   ", "Hello", "\t"]) == ["Hello"]
    assert curate_texts(["A  B", "A B", "", " C ", "C"]) == [
        "A B",
        "C",
    ]
    assert curate_texts([]) == []
    print("Empty-text filtering checks passed.")

    assert filter_by_min_length(["AI", "OK", "model"], 3) == ["model"]
    assert filter_by_min_length(["a", "abc", "abcd"], 3) == [
        "abc",
        "abcd",
    ]
    assert filter_by_min_length([], 3) == []
    print("Minimum-length filtering checks passed.")

    kept_texts, removed_texts = partition_by_min_length(
        ["AI", "OK", "model", "Transformer"], 3
    )
    assert kept_texts == ["model", "Transformer"]
    assert removed_texts == ["AI", "OK"]
    assert partition_by_min_length([], 3) == ([], [])
    print("Length-partition checks passed.")

    assert length_sensitivity_report(
        ["AI", "cat", "model", "Transformer"], [1, 3, 5]
    ) == {1: 4, 3: 3, 5: 2}
    assert length_sensitivity_report([], [1, 3]) == {1: 0, 3: 0}
    print("Length-sensitivity checks passed.")

    assert curate_with_min_length(
        [" A ", "AI", "model", " model "], 3
    ) == ["model"]
    assert curate_with_min_length(["   ", "cat", "cat"], 3) == ["cat"]
    assert curate_with_min_length([], 3) == []
    print("Ordered curation checks passed.")

    full_summary = summarize_full_curation_stages(
        ["A", " ", "A ", "\t", "model", "long text"], 3
    )
    assert full_summary == {
        "original_count": 6,
        "nonempty_count": 4,
        "unique_count": 3,
        "final_count": 2,
        "empty_removed_count": 2,
        "duplicate_removed_count": 1,
        "length_removed_count": 1,
    }
    assert summarize_full_curation_stages([], 3) == {
        "original_count": 0,
        "nonempty_count": 0,
        "unique_count": 0,
        "final_count": 0,
        "empty_removed_count": 0,
        "duplicate_removed_count": 0,
        "length_removed_count": 0,
    }
    print("Full stage-summary checks passed.")

    stage_summary = summarize_curation_stages(
        ["A", " ", "A ", "\t", "B", " C "]
    )
    assert stage_summary == {
        "original_count": 6,
        "nonempty_count": 4,
        "unique_count": 3,
        "empty_removed_count": 2,
        "duplicate_removed_count": 1,
    }
    assert summarize_curation_stages([]) == {
        "original_count": 0,
        "nonempty_count": 0,
        "unique_count": 0,
        "empty_removed_count": 0,
        "duplicate_removed_count": 0,
    }
    print("Stage-wise curation checks passed.")

    assert is_consistent_curation_summary(stage_summary) is True
    invalid_summary = stage_summary.copy()
    invalid_summary["unique_count"] = 4
    assert is_consistent_curation_summary(invalid_summary) is False
    assert is_consistent_curation_summary(
        summarize_curation_stages([])
    ) is True
    assert is_consistent_curation_summary(full_summary) is True
    invalid_full_summary = full_summary.copy()
    invalid_full_summary["length_removed_count"] = 0
    assert is_consistent_curation_summary(invalid_full_summary) is False
    print("Curation consistency checks passed.")


if __name__ == "__main__":
    run_checks()
