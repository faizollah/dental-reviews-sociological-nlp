from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

LEXICON_PATH = Path(__file__).resolve().parent.parent / "lexicons" / "walther_sip.json"
TEXT_COL = "text"
IMD_COL = "Index of Multiple Deprivation Decile"


class SIPAnalyser:
    """Scores the seven SIP cue categories across a review corpus."""

    def __init__(self, lexicon_path: Path = LEXICON_PATH, model: str = "en_core_web_sm"):
        self.lexicon = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))["categories"]
        self.nlp = spacy.load(model, disable=["ner", "parser"])
        self.matchers = {}
        for concept, subcats in self.lexicon.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            for sub, phrases in subcats.items():
                matcher.add(f"{concept}::{sub}", [self.nlp.make_doc(p) for p in phrases])
            self.matchers[concept] = matcher

    @staticmethod
    def _clean(raw: str) -> str:
        if not isinstance(raw, str):
            return ""
        text = raw.replace("face-to-face", "face to face").replace("e-mail", "email")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _per_1k(count: int, n_words: int) -> float:
        return (count / n_words) * 1000 if n_words > 0 else 0.0

    @staticmethod
    def _word_count(doc) -> int:
        return sum(1 for t in doc if t.is_alpha)

    def _match(self, doc, concept: str) -> Tuple[int, Dict[str, int], List[str]]:
        counts, hits = Counter(), []
        for match_id, start, end in self.matchers[concept](doc):
            _, sub = self.nlp.vocab.strings[match_id].split("::", 1)
            counts[sub] += 1
            hits.append(doc[start:end].text.lower())
        for sub in self.lexicon[concept]:
            counts.setdefault(sub, 0)
        return sum(counts.values()), dict(counts), hits

    def analyse_frame(self, df: pd.DataFrame, text_column: str = TEXT_COL) -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"column '{text_column}' not found")

        df = df.copy()
        texts = df[text_column].fillna("").astype(str).map(self._clean).tolist()
        docs = list(self.nlp.pipe(texts, batch_size=100))
        word_counts = np.array([self._word_count(d) for d in docs])

        for concept in self.lexicon:
            totals, by_cats, all_hits = [], [], []
            for doc in docs:
                total, by_cat, hits = self._match(doc, concept)
                totals.append(total)
                by_cats.append(by_cat)
                all_hits.append(hits)
            key = concept.lower()
            df[f"{key}_total"] = totals
            df[f"{key}_by_cat"] = [json.dumps(x) for x in by_cats]
            df[f"{key}_hits"] = [json.dumps(x) for x in all_hits]
            df[f"{key}_per_1k"] = [self._per_1k(c, n) for c, n in zip(totals, word_counts)]

        df["word_count"] = word_counts
        return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def prevalence(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.endswith("_per_1k")]
    return df[cols].mean().sort_values(ascending=False)


def category_breakdown(df: pd.DataFrame, concept: str) -> pd.Series:
    """Mean per-1,000-word rate for each subcategory of a concept."""
    key = concept.lower()
    rows = pd.DataFrame([json.loads(x) for x in df[f"{key}_by_cat"]])
    words = df["word_count"].replace(0, np.nan).values
    return (rows.div(words, axis=0) * 1000).mean().sort_values(ascending=False)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations between SIP scores and outcome measures.

    Where the Parsons pipeline has been run first, cooperation_score_norm is
    included, which is the cross-framework link: it tests whether relational
    language accompanies cooperative stances.
    """
    cols = [c for c in df.columns if c.endswith("_per_1k")]
    outcomes = [c for c in ("stars", "cooperation_score_norm", "sentiment_vader")
                if c in df.columns]
    if not outcomes:
        return pd.DataFrame()
    matrix = df[cols + outcomes].corr(numeric_only=True)
    outcomes = [c for c in outcomes if c in matrix.columns]  # drop non-numeric
    return matrix.loc[cols, outcomes] if outcomes else pd.DataFrame()


def by_decile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean SIP scores by IMD decile.

    Descriptive only. The decile-level differences for this framework were not
    significant, and this table is reported to show the distribution rather than
    to support a gradient claim.
    """
    if IMD_COL not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["imd_decile"] = pd.to_numeric(d[IMD_COL], errors="coerce")
    d = d.dropna(subset=["imd_decile"])
    cols = [c for c in d.columns if c.endswith("_per_1k")]
    out = d.groupby("imd_decile")[cols].mean()
    out["n_reviews"] = d.groupby("imd_decile").size()
    return out


def by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Mean SIP scores by organisation subtype, where NHS and private practices
    can be distinguished."""
    col = next((c for c in ("Organisation SubType Code", "categoryName") if c in df.columns), None)
    if col is None:
        return pd.DataFrame()
    cols = [c for c in df.columns if c.endswith("_per_1k")]
    out = df.groupby(col)[cols].mean()
    out["n_reviews"] = df.groupby(col).size()
    return out


def report(df: pd.DataFrame) -> None:
    print(f"n reviews = {len(df):,}\n")
    print("SIP concept prevalence (mean per 1,000 words):")
    print(prevalence(df).round(3).to_string(), "\n")

    for concept in ("RELATIONAL_DEVELOPMENT", "HYPERPERSONAL_COMMUNICATION", "WARRANTING"):
        if f"{concept.lower()}_by_cat" in df.columns:
            print(f"{concept} subcategories (mean per 1,000 words):")
            print(category_breakdown(df, concept).round(3).to_string(), "\n")

    corr = correlations(df)
    if not corr.empty:
        print("Pearson correlations:")
        print(corr.round(3).to_string(), "\n")

    dec = by_decile(df)
    if not dec.empty:
        print("Mean scores by IMD decile (descriptive):")
        print(dec.round(3).to_string(), "\n")

    sec = by_sector(df)
    if not sec.empty:
        print("Mean scores by organisation subtype:")
        print(sec.round(3).to_string(), "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of reviews")
    ap.add_argument("--output", help="where to write the scored CSV")
    ap.add_argument("--text-col", default=TEXT_COL)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    analyser = SIPAnalyser()
    df = analyser.analyse_frame(df, text_column=args.text_col)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output}\n")

    report(df)


if __name__ == "__main__":
    main()
