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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import PhraseMatcher

LEXICON_PATH = Path(__file__).resolve().parent.parent / "lexicons" / "foucault_biopower_gaze.json"
TEXT_COL = "text"
IMD_COL = "Index of Multiple Deprivation Decile"


class FoucauldianAnalyser:
    """Scores biopower and medical-gaze language, agency, and topic structure."""

    def __init__(self, lexicon_path: Path = LEXICON_PATH, model: str = "en_core_web_sm"):
        self.lexicon = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))["categories"]
        # The parser is needed for the passivity score; NER is not.
        self.nlp = spacy.load(model, disable=["ner"])
        self.matchers = {}
        for concept, subcats in self.lexicon.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            for sub, phrases in subcats.items():
                matcher.add(f"{concept}::{sub}", [self.nlp.make_doc(p) for p in phrases])
            self.matchers[concept] = matcher
        self.terms = {c: set(sum(s.values(), [])) for c, s in self.lexicon.items()}

    # -- utilities ---------------------------------------------------------
    @staticmethod
    def _clean(raw: str) -> str:
        """Fold hyphenated variants that appear unhyphenated in the lexicon."""
        if not isinstance(raw, str):
            return ""
        text = raw.replace("x-ray", "x ray").replace("check-up", "check up")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _per_1k(count: int, n_words: int) -> float:
        return (count / n_words) * 1000 if n_words > 0 else 0.0

    @staticmethod
    def _word_count(doc) -> int:
        return sum(1 for t in doc if t.is_alpha)

    @staticmethod
    def _lemmas(doc) -> str:
        return " ".join(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop)

    # -- scoring -----------------------------------------------------------
    def _match(self, doc, concept: str) -> Tuple[int, Dict[str, int], List[str]]:
        counts, hits = Counter(), []
        for match_id, start, end in self.matchers[concept](doc):
            _, sub = self.nlp.vocab.strings[match_id].split("::", 1)
            counts[sub] += 1
            hits.append(doc[start:end].text.lower())
        for sub in self.lexicon[concept]:
            counts.setdefault(sub, 0)
        return sum(counts.values()), dict(counts), hits

    @staticmethod
    def _passivity(doc) -> float:
        """Passive constructions as a share of passive plus active."""
        passive = active = 0
        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            voice = token.morph.get("Voice")
            is_passive = ("Pass" in voice
                          or any(c.dep_ == "auxpass" for c in token.children)
                          or any(c.dep_ == "nsubjpass" for c in token.children))
            if is_passive:
                passive += 1
            elif any(c.dep_ == "nsubj" for c in token.children):
                active += 1
        total = passive + active
        return passive / total if total else 0.0

    # -- topic modelling ---------------------------------------------------
    def topic_model(self, texts: pd.Series, n_topics: int = 10, n_top_words: int = 10):
        """LDA over unigrams and bigrams, each topic scored for lexicon overlap."""
        valid = texts[texts.str.len() > 0]
        if valid.empty:
            return {}, pd.Series(index=texts.index, dtype="float")

        vec = CountVectorizer(max_df=0.95, min_df=5, ngram_range=(1, 2), stop_words="english")
        dtm = vec.fit_transform(valid)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)

        features = np.array(vec.get_feature_names_out())
        masks = {c: np.isin(features, list(terms)) for c, terms in self.terms.items()}

        topics = {}
        for k, comp in enumerate(lda.components_):
            top_idx = comp.argsort()[::-1][:n_top_words]
            topics[k] = {"top_words": features[top_idx].tolist()}
            for concept, mask in masks.items():
                # Relevance is the share of the topic's mass sitting on lexicon terms.
                topics[k][f"{concept.lower()}_relevance"] = float(comp[mask].sum() / comp.sum())

        dominant = pd.Series(index=texts.index, dtype="float")
        dominant.loc[valid.index] = lda.transform(dtm).argmax(axis=1)
        return topics, dominant

    # -- pipeline ----------------------------------------------------------
    def analyse_frame(self, df: pd.DataFrame, text_column: str = TEXT_COL) -> Tuple[pd.DataFrame, Dict]:
        if text_column not in df.columns:
            raise ValueError(f"column '{text_column}' not found")

        df = df.copy()
        texts = df[text_column].fillna("").astype(str).map(self._clean).tolist()
        docs = list(self.nlp.pipe(texts, batch_size=100))
        word_counts = np.array([self._word_count(d) for d in docs])

        scores = {c: {"total": [], "by_cat": [], "hits": []} for c in self.lexicon}
        passivity, lemmas = [], []

        for doc in docs:
            for concept in self.lexicon:
                total, by_cat, hits = self._match(doc, concept)
                scores[concept]["total"].append(total)
                scores[concept]["by_cat"].append(by_cat)
                scores[concept]["hits"].append(hits)
            passivity.append(self._passivity(doc))
            lemmas.append(self._lemmas(doc))

        for concept, s in scores.items():
            key = concept.lower()
            df[f"{key}_hits_total"] = s["total"]
            df[f"{key}_hits_by_cat"] = [json.dumps(x) for x in s["by_cat"]]
            df[f"{key}_hit_strings"] = [json.dumps(x) for x in s["hits"]]
            df[f"{key}_per_1k"] = [self._per_1k(c, n) for c, n in zip(s["total"], word_counts)]

        df["word_count"] = word_counts
        df["passivity_score"] = passivity
        topics, dominant = self.topic_model(pd.Series(lemmas, index=df.index))
        df["dominant_topic"] = dominant
        return df, topics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def prevalence(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.endswith("_per_1k")] + ["passivity_score"]
    return df[cols].mean()


def category_breakdown(df: pd.DataFrame, concept: str) -> pd.Series:
    """Mean per-1,000-word rate for each subcategory of a concept."""
    key = concept.lower()
    rows = pd.DataFrame([json.loads(x) for x in df[f"{key}_hits_by_cat"]])
    words = df["word_count"].replace(0, np.nan).values
    return (rows.div(words, axis=0) * 1000).mean().sort_values(ascending=False)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations between the lexicon scores and patient experience."""
    cols = [c for c in df.columns if c.endswith("_per_1k")] + ["passivity_score"]
    outcomes = [c for c in ("stars", "sentiment_vader") if c in df.columns]
    if not outcomes:
        return pd.DataFrame()
    matrix = df[cols + outcomes].corr(numeric_only=True)
    outcomes = [c for c in outcomes if c in matrix.columns]  # drop non-numeric
    return matrix.loc[cols, outcomes] if outcomes else pd.DataFrame()


def by_decile(df: pd.DataFrame) -> pd.DataFrame:
    
    if IMD_COL not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["imd_decile"] = pd.to_numeric(d[IMD_COL], errors="coerce")
    d = d.dropna(subset=["imd_decile"])
    agg = {c: "mean" for c in d.columns if c.endswith("_per_1k")}
    agg["passivity_score"] = "mean"
    if "stars" in d.columns:
        agg["stars"] = "mean"
    out = d.groupby("imd_decile").agg(agg)
    out["n_reviews"] = d.groupby("imd_decile").size()
    return out


def report(df: pd.DataFrame, topics: Dict | None = None) -> None:
    print(f"n reviews = {len(df):,}\n")
    print("Prevalence (mean per 1,000 words):")
    print(prevalence(df).round(3).to_string(), "\n")

    for concept in ("BIOPOWER", "MEDICAL_GAZE"):
        if f"{concept.lower()}_hits_by_cat" in df.columns:
            print(f"{concept} subcategories (mean per 1,000 words):")
            print(category_breakdown(df, concept).round(3).to_string(), "\n")

    corr = correlations(df)
    if not corr.empty:
        print("Pearson correlations with patient experience:")
        print(corr.round(3).to_string(), "\n")

    dec = by_decile(df)
    if not dec.empty:
        print("Mean scores by IMD decile (descriptive):")
        print(dec.round(3).to_string(), "\n")

    if topics:
        for concept in self_keys(topics):
            ranked = sorted(topics.items(), key=lambda kv: kv[1][concept], reverse=True)[:3]
            print(f"Top 3 topics by {concept.replace('_relevance', '')}:")
            for k, data in ranked:
                print(f"  topic {k} (relevance {data[concept]:.4f}): {', '.join(data['top_words'])}")
            print()


def self_keys(topics: Dict) -> List[str]:
    first = next(iter(topics.values()))
    return [k for k in first if k.endswith("_relevance")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of reviews")
    ap.add_argument("--output", help="where to write the scored CSV")
    ap.add_argument("--text-col", default=TEXT_COL)
    ap.add_argument("--topics", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    analyser = FoucauldianAnalyser()
    df, topics = analyser.analyse_frame(df, text_column=args.text_col)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output}\n")

    report(df, topics)


if __name__ == "__main__":
    main()
