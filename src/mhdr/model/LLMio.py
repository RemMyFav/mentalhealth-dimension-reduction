
from pathlib import Path
from typing import Dict, List
import pandas as pd

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