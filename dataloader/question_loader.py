from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class QuestionDataset(Dataset):
    """
    Load all survey question CSVs from preprocess/ directory.
    Each CSV must contain columns: question, source
    """

    def __init__(self, preprocess_dir="question_database/preprocess"):
        self.preprocess_dir = Path(preprocess_dir)
        assert self.preprocess_dir.exists(), f"{preprocess_dir} not found"

        rows = []


        for csv_path in sorted(self.preprocess_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)

            if not {"question", "source"}.issubset(df.columns):
                raise ValueError(f"{csv_path.name} must contain columns: question, source")

            df = df[["question", "source"]].copy()
            df.rename(columns={"question": "text", "source": "qid"}, inplace=True)

            df["dataset"] = csv_path.stem

            df["text"] = df["text"].astype(str).str.strip()
            df["qid"] = df["qid"].astype(str).str.strip()
            df = df[df["text"].str.len() > 0]

            rows.append(df)

        if not rows:
            raise RuntimeError("No CSV files found in preprocess directory")

        self.df = (
            pd.concat(rows, ignore_index=True)
              .drop_duplicates(subset=["qid"])
              .reset_index(drop=True)
        )

        print(f"[QuestionDataset] Loaded {len(self.df)} questions from {len(rows)} files")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "qid": row["qid"],
            "dataset": row["dataset"],
            "text": row["text"],
        }