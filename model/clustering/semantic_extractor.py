
import pandas as pd
import stanza
from pathlib import Path
from typing import Optional


class SemanticExtractor:
    def __init__(self, nlp: stanza.Pipeline):
        self.model = nlp
        self.df: Optional[pd.DataFrame] = None

        # ❌ drop grammatical shells + cop + cc
        self.DROP_DEPS = {"aux", "det", "punct"}

    # ==================================================
    # Token-level filtering (SOURCE OF TRUTH)
    # NOTE: token may contain multiple words (e.g., "cannot" -> ["can","not"])
    # We keep the whole token if ANY sub-word should be kept.
    # ==================================================
    def _keep_token(self, token) -> bool:
        ner = (getattr(token, "ner", "O") or "O").upper()

        # ❌ drop ANY NER span (DATE/TIME/NUMBER/etc.)
        if ner != "O":
            return False

        # ✅ HARD RULE: if ANY sub-word has PronType=Neg, keep the whole token
        for w in token.words:
            feats = getattr(w, "feats", "") or ""
            if "PronType=Neg" in feats:
                return True

        for w in token.words:
            dep = (getattr(w, "deprel", "") or "").lower()
            upos = (getattr(w, "upos", "") or "").upper()
            feats = getattr(w, "feats", "") or ""  # ✅ always defined

            # ❌ drop ONLY subject personal pronouns (I/you/he/she/we/they)
            # ✅ keep object/reflexive/etc. (me/him/her/us/them/myself...)
            if upos == "PRON" and "PronType=Prs" in feats:
                if "Case=Nom" in feats:
                    continue
                else:
                    return True

            # ❌ drop shells + cop + cc
            if dep in self.DROP_DEPS or dep in {"cop", "cc"}:
                continue

            # ❌ advmod: keep ONLY negation
            if dep == "advmod":
                if "Polarity=Neg" in feats:
                    return True
                continue

            # ✅ keep everything else
            return True

        return False

    # ==================================================
    # Backbone extraction (NO recursion / DFS)
    # ==================================================
    def extract_backbone_from_doc(self, doc) -> str:
        out = []
        for sent in doc.sentences:
            for token in sent.tokens:
                if self._keep_token(token):
                    t = (token.text or "").lower().strip()
                    if t:
                        out.append(t)
        return " ".join(out)

    # ==================================================
    # Public API
    # ==================================================
    def extract_backbone(self, text: str) -> str:
        if text is None:
            return ""
        return self.extract_backbone_from_doc(self.model(str(text)))

    def transform_texts_safe(self, texts: list[str]) -> list[str]:
        texts = ["" if t is None else str(t) for t in texts]
        docs = self.model.bulk_process(texts)
        return [self.extract_backbone_from_doc(d) for d in docs]

    def transform_df(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        raw_col: str = "text_raw"  
    ) -> pd.DataFrame:
        df = df.copy()

        df[raw_col] = df[text_col]

        df[text_col] = self.transform_texts_safe(df[text_col].tolist())

        self.df = df
        self.save_questions_nvo()
        return df

    def save_questions_nvo(self) -> Path:
        if self.df is None:
            raise ValueError("DataFrame is not set.")
        out_path = Path("./temp_result/questions_nvo.csv")
        self.df.to_csv(out_path, index=False)
        return out_path