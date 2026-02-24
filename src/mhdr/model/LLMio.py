"""I/O utilities for loading LLM-generated dimension definitions."""
from pathlib import Path
from typing import Dict, List
import pandas as pd


# -------------------------------------------------
# Load Dimension Sets
# -------------------------------------------------

def load_dimension_sets(csv_path: str) -> Dict[str, List[str]]:
    """Load dimension definitions from CSV and group by model name.

    Args:
        csv_path: Path to CSV file with columns: model_name, dim_name, dim_text.

    Returns:
        Dict mapping model_name to list of dimension definitions.
        Each definition has format "dim_name: dim_text".
    """
    df = pd.read_csv(Path(csv_path))

    dimension_sets = {}

    for model_name, group in df.groupby("model_name"):
        dimension_sets[model_name] = [
            f"{row.dim_name}: {row.dim_text}"
            for _, row in group.iterrows()
        ]

    return dimension_sets