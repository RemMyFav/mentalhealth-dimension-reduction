from pathlib import Path
import pandas as pd

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