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

AUX_LEMMAS = {"be", "do", "have", "can", "could", "would", "should", "may", "might", "will", "shall"}
PRONOUN_LEMMAS = {"i", "you", "we", "they", "he", "she", "it", "me", "him", "her", "us", "them", "my", "your", "our", "their"}

QUESTION_TEMPLATE = {
    "how", "often", "much", "many", "extent", "time", "month", "usually",
    "past", "together", "compared", "overall", "current", "same", "other",
    "what", "which", "when", "where", "why"
}

def unique_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def clean_backbone(text, nlp):
    doc = nlp(text)

    keep = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue

        lemma = tok.lemma_.lower()

        # drop empties / stopwords-like tokens
        if not lemma:
            continue

        # drop pronouns
        if lemma in PRONOUN_LEMMAS:
            continue

        # drop question template words
        if lemma in QUESTION_TEMPLATE:
            continue

        # keep negation explicitly
        if tok.dep_ == "neg" or tok.lower_ in {"no", "not", "never"}:
            keep.append(tok.lower_)
            continue

        # keep content: NOUN/PROPN/ADJ + content VERB (exclude auxiliaries)
        if tok.pos_ in {"NOUN", "PROPN", "ADJ"}:
            keep.append(lemma)
        elif tok.pos_ == "VERB" and lemma not in AUX_LEMMAS:
            keep.append(lemma)

    keep = unique_keep_order(keep)
    return " ".join(keep)

def replace_text_with_nvo(df, nlp, text_col="text"):
    df = df.copy()
    df[text_col] = [clean_backbone(s, nlp) for s in df[text_col].astype(str)]
    return df