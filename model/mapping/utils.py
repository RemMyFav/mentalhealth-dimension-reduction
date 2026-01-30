import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
def load_questions(processed_dir="../../question_database/processed/"):
    """
    Load the canonical question table.

    Returns
    -------
    df : pandas.DataFrame
        Columns: [qid, dataset, text]
    """
    processed_dir = Path(processed_dir)
    parquet_path = processed_dir / "questions_master.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df

    raise FileNotFoundError(
        "No questions_master.parquet found. "
        "Please run the build step first."
    )


def load_dimension_sets(csv_path: str) -> Dict[str, List[str]]:
    """
    Load dimension definitions and group them by model_name.

    Returns
    -------
    Dict[str, List[str]]
        Key   : model_name (e.g., 'Llama-4')
        Value : list of dimension definitions
                ['Emotional: ...', 'Environmental: ...', ...]
    """
    df = pd.read_csv(Path(csv_path))

    dimension_sets = {}

    for model_name, group in df.groupby("model_name"):
        dimension_sets[model_name] = [
            f"{row.dim_name}: {row.dim_text}"
            for _, row in group.iterrows()
        ]

    return dimension_sets

from itertools import combinations

def jaccard(set_a, set_b):
    set_a, set_b = set(set_a), set(set_b)
    if len(set_a | set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def mean_pairwise_jaccard(dim_sets):
    """
    dim_sets: List[Iterable[str]]
              e.g. [
                  ['Emotional', 'Social'],
                  ['Emotional', 'Social'],
                  ['Emotional'],
                  ...
              ]
    """
    scores = [
        jaccard(a, b)
        for a, b in combinations(dim_sets, 2)
    ]
    return sum(scores) / len(scores) if scores else 0.0



from collections import Counter
from itertools import combinations

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def mean_pairwise_jaccard(dim_sets):
    pairs = list(combinations(dim_sets, 2))
    if not pairs:
        return 1.0
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)

def compute_cross_model_agreement(all_results, consensus_k=3):
    """
    Compute cross-model agreement + interpretability fields.

    Args:
        all_results: dict[str, pd.DataFrame]
            key   = model name
            value = df_map with columns:
                    ['qid','dataset','text','dimensions', ...]
            NOTE: df_map['dimensions'] should be a list[str] per row.
        consensus_k: int
            threshold for consensus dimensions (e.g., >=3 out of 5)

    Returns:
        pd.DataFrame with columns:
            ['qid','dataset','text',
             'mean_pairwise_jaccard',
             'union_dimensions',
             'consensus_dimensions']
    """
    model_names = list(all_results.keys())

    # Build base table from the first model
    base = all_results[model_names[0]][['qid', 'dataset', 'text']].copy()

    mean_scores = []
    union_dims_out = []
    consensus_dims_out = []

    # For fast lookup by qid in each model
    lookup = {}
    for m in model_names:
        df = all_results[m].set_index('qid')
        lookup[m] = df

    for _, row in base.iterrows():
        qid = row['qid']

        dim_sets = []
        all_dims_flat = []

        for m in model_names:
            dims = lookup[m].loc[qid, 'dimensions']
            dims = list(dims)  # ensure list
            dim_sets.append(dims)
            all_dims_flat.extend(dims)

        # mean pairwise Jaccard
        mean_scores.append(mean_pairwise_jaccard(dim_sets))

        # union dimensions
        union_set = sorted(set(all_dims_flat))
        union_dims_out.append(union_set)

        # consensus dimensions (freq >= consensus_k)
        freq = Counter(all_dims_flat)
        consensus_set = sorted([d for d, c in freq.items() if c >= consensus_k])
        consensus_dims_out.append(consensus_set)

    base['mean_pairwise_jaccard'] = mean_scores
    base['union_dimensions'] = union_dims_out
    base['consensus_dimensions'] = consensus_dims_out

    return base

def compute_consensus_spectrum(all_results):
    """
    Compute a mutually-exclusive consensus spectrum (exact-count bins).

    For each question, each dimension is assigned to exactly ONE bucket
    based on how many models selected it:
        exact_5of5, exact_4of5, ..., exact_1of5

    Args:
        all_results: dict[str, pd.DataFrame]
            key   = model name
            value = df_map with columns:
                    ['qid','dataset','text','dimensions']
            NOTE: df_map['dimensions'] is list[str] per row.

    Returns:
        pd.DataFrame with columns:
            ['qid','dataset','text',
             'exact_5of5','exact_4of5','exact_3of5','exact_2of5','exact_1of5']
    """
    model_names = list(all_results.keys())
    K = len(model_names)

    base = all_results[model_names[0]][['qid', 'dataset', 'text']].copy()

    # fast lookup by qid
    lookup = {m: all_results[m].set_index('qid') for m in model_names}

    # containers for each exact bucket
    buckets = {k: [] for k in range(1, K + 1)}  # 1..K

    for _, row in base.iterrows():
        qid = row['qid']

        # count how many models selected each dimension
        freq = Counter()
        for m in model_names:
            dims = lookup[m].loc[qid, 'dimensions']
            freq.update(set(dims))  # set() just in case one model has duplicates

        # put each dimension into exactly one bucket by its exact count
        dims_by_count = {k: [] for k in range(1, K + 1)}
        for dim, c in freq.items():
            if 1 <= c <= K:
                dims_by_count[c].append(dim)

        # sort for stable output
        for k in range(1, K + 1):
            buckets[k].append(sorted(dims_by_count[k]))

    # attach columns (from strict to loose is easier to read)
    for k in range(K, 0, -1):
        base[f'exact_{k}of{K}'] = buckets[k]

    return base