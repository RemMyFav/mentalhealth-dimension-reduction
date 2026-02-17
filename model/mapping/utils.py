import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import numpy as np
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
import numpy as np

def compute_distribution_entropy(all_results, *, dims_vocab=None, normalize=True, return_mean_dist=False):
    """
    Compute per-question distribution entropy H(M) across models/annotators.

    For each qid:
      - Convert each model's (possibly multi-label) output dims into a distribution Pi over dims_vocab:
          If dims=[d1,d2,...], assign 1/len(dims) to each.
          If empty, assign uniform over V.
      - Compute mean distribution M = average_i Pi
      - Entropy = H(M) = -sum_j M(j) log2 M(j)

    Args:
      all_results: dict[str, pd.DataFrame]
        key=model name, value has columns ['qid','dataset','text','dimensions'] (dimensions is list-like).
      dims_vocab: list[str] | None
        If None, use union of all dimensions appearing in all_results.
      normalize: bool
        If True, return entropy_norm = entropy / log2(V) in [0,1].
      return_mean_dist: bool
        If True, include meanprob_<dim> columns.

    Returns:
      pd.DataFrame with columns:
        ['qid','dataset','text','entropy','entropy_norm'(optional)]
      plus mean distribution columns if return_mean_dist=True.
    """
    model_names = list(all_results.keys())
    K = len(model_names)
    if K < 1:
        raise ValueError("Need at least 1 model to compute entropy.")

    base = all_results[model_names[0]][["qid", "dataset", "text"]].copy()
    lookup = {m: all_results[m].set_index("qid") for m in model_names}

    # Build vocabulary
    if dims_vocab is None:
        vocab = set()
        for m in model_names:
            for dims in all_results[m]["dimensions"]:
                if isinstance(dims, (list, tuple, set)):
                    vocab.update(dims)
        dims_vocab = sorted(vocab)

    V = len(dims_vocab)
    if V == 0:
        raise ValueError("dims_vocab is empty (no dimensions found).")

    dim2idx = {d: i for i, d in enumerate(dims_vocab)}

    def entropy(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    ent_list = []
    mean_dists = []

    for _, row in base.iterrows():
        qid = row["qid"]

        # Build per-model distributions Pi over vocab
        P = np.zeros((K, V), dtype=float)

        for i, m in enumerate(model_names):
            dims = lookup[m].loc[qid, "dimensions"]

            # handle edge cases
            if dims is None or (isinstance(dims, float) and np.isnan(dims)):
                dims = []
            if not isinstance(dims, (list, tuple, set)):
                dims = [dims]

            dims = [d for d in set(dims) if d in dim2idx]  # unique + in vocab

            if len(dims) == 0:
                P[i, :] = 1.0 / V
            else:
                w = 1.0 / len(dims)
                for d in dims:
                    P[i, dim2idx[d]] += w

        # mean distribution
        M = P.mean(axis=0)

        ent_list.append(entropy(M))
        if return_mean_dist:
            mean_dists.append(M)

    base["entropy"] = ent_list

    if normalize:
        base["entropy_norm"] = base["entropy"] / np.log2(V) if V > 1 else 0.0

    if return_mean_dist:
        md = np.vstack(mean_dists)
        for j, d in enumerate(dims_vocab):
            base[f"meanprob_{d}"] = md[:, j]

    return base