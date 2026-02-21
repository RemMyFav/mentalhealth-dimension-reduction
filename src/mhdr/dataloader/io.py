from __future__ import annotations
from pathlib import Path
import pandas as pd

# -------------------------------------------------
# Core CSV Reader
# -------------------------------------------------

def read_csv(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """Read CSV with safe defaults.

    Args:
        path: Path to CSV file.
        encoding: File encoding. Defaults to "utf-8".
        **kwargs: Additional arguments passed to pd.read_csv.

    Returns:
        pd.DataFrame: Contents of the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded DataFrame is empty.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"[read_csv] File not found: {path}")

    df = pd.read_csv(path, encoding=encoding, **kwargs)

    if df.empty:
        raise ValueError(f"[read_csv] Loaded empty DataFrame: {path}")

    return df


# -------------------------------------------------
# Save CSV
# -------------------------------------------------

def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    create_dir: bool = True
) -> None:
    """Save DataFrame to CSV safely.

    Args:
        df: DataFrame to save.
        path: Output path for the CSV file.
        index: Whether to write row indices. Defaults to False.
        create_dir: Whether to create parent directories. Defaults to True.
    """

    path = Path(path)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=index)


# -------------------------------------------------
# Parquet (future-proof)
# -------------------------------------------------

def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read Parquet file.

    Args:
        path: Path to Parquet file.

    Returns:
        pd.DataFrame: Contents of the Parquet file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[read_parquet] File not found: {path}")
    return pd.read_parquet(path)

def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    create_dir: bool = True
) -> None:
    """Save DataFrame to Parquet.

    Args:
        df: DataFrame to save.
        path: Output path for the Parquet file.
        create_dir: Whether to create parent directories. Defaults to True.
    """
    path = Path(path)
    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path)

# -------------------------------------------------
#  Excel
# -------------------------------------------------
def read_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    engine: str = "openpyxl",
    **kwargs
) -> pd.DataFrame:
    """Read Excel file safely.

    Args:
        path: Path to Excel file.
        sheet_name: Sheet name or index. Defaults to 0 (first sheet).
        engine: Excel engine. Defaults to "openpyxl".
        **kwargs: Additional arguments passed to pd.read_excel.

    Returns:
        pd.DataFrame: Contents of the Excel sheet.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded DataFrame is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[read_excel] File not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, engine=engine, **kwargs)

    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError(f"[read_excel] Loaded empty DataFrame: {path} (sheet={sheet_name})")

    return df