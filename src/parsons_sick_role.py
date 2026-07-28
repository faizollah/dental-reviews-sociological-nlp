from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import DependencyMatcher, Matcher, PhraseMatcher
from spacy.tokens import Doc

LEXICON_PATH = Path(__file__).resolve().parent.parent / "lexicons" / "parsons_sick_role.json"
TEXT_COL = "text"
IMD_COL = "Index of Multiple Deprivation Decile"

NORMALISE_PATTERNS = True


def normalise_text(s: str) -> str:
    """Fold smart quotes and dashes and collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-"))
    return re.sub(r"\s+", " ", s).strip()


class ParsonsAnalyser:
    """Rule-based detector for upholding and challenging language."""

    def __init__(self, lexicon_path: Path = LEXICON_PATH, model: str = "en_core_web_sm"):
        self.lexicon = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        self.upholding = set(self.lexicon["polarity"]["upholding"])
        self.challenging = set(self.lexicon["polarity"]["challenging"])

        self.nlp = spacy.load(model)
        self.phraser = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher = Matcher(self.nlp.vocab)
        self.dep = DependencyMatcher(self.nlp.vocab)
        self._build()

    def _build(self) -> None:
        for label, spec in self.lexicon["categories"].items():
            phrases = spec.get("phrases", [])
            if NORMALISE_PATTERNS:
                phrases = sorted({normalise_text(p).lower() for p in phrases})
            if phrases:
                self.phraser.add(label, [self.nlp.make_doc(p) for p in phrases])
            if spec.get("token_patterns"):
                self.matcher.add(label, spec["token_patterns"])

        for name, patterns in self.lexicon.get("dependency_patterns", {}).items():
            self.dep.add(name, patterns)

    def analyse_doc(self, doc: Doc, return_details: bool = False) -> Dict:
        """Score a single parsed review."""
        upholding: List[Dict] = []
        challenging: List[Dict] = []

        for sent in doc.sents:
            hits = [(self.nlp.vocab.strings[mid], s, e) for mid, s, e in self.matcher(sent)]
            hits += [(self.nlp.vocab.strings[mid], s, e) for mid, s, e in self.phraser(sent)]
            # Negated trust verbs are recorded as LACK_TRUST regardless of rule name.
            for _mid, token_ids in self.dep(sent):
                if token_ids:
                    hits.append(("LACK_TRUST", min(token_ids), max(token_ids) + 1))

            for rule, start, end in hits:
                rec = {"sentence": sent.text.strip(), "pattern": rule,
                       "matched_text": sent[start:end].text}
                if rule in self.upholding:
                    upholding.append(rec)
                elif rule in self.challenging:
                    challenging.append(rec)

        upholding = _dedup(upholding)
        challenging = _dedup(challenging)
        n_sents = max(len(list(doc.sents)), 1)

        result = {
            "upholding_count": len(upholding),
            "challenging_count": len(challenging),
            "cooperation_score": len(upholding) - len(challenging),
            "cooperation_score_norm": round((len(upholding) - len(challenging)) / n_sents, 3),
            "upholding_categories": sorted({m["pattern"] for m in upholding}),
            "challenging_categories": sorted({m["pattern"] for m in challenging}),
        }
        if return_details:
            result["upholding_matches"] = upholding
            result["challenging_matches"] = challenging
        return result

    def analyse_frame(self, df: pd.DataFrame, text_column: str = TEXT_COL) -> pd.DataFrame:
        """Score every review in a dataframe and append the result columns."""
        if text_column not in df.columns:
            raise ValueError(f"column '{text_column}' not found")

        df = df.copy()
        df["normalized_text"] = df[text_column].fillna("").astype(str).map(normalise_text)
        docs = self.nlp.pipe(df["normalized_text"].tolist(), batch_size=100)
        results = pd.json_normalize([self.analyse_doc(d) for d in docs])

        for col in ("upholding_categories", "challenging_categories"):
            results[col] = results[col].apply(", ".join)

        out = pd.concat([df.reset_index(drop=True), results], axis=1)
        out["parsons_label"] = np.select(
            [out["cooperation_score"] > 0, out["cooperation_score"] < 0],
            ["UPHOLDING", "CHALLENGING"],
            default="MIXED/NEUTRAL",
        )
        return out


def _dedup(records: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in records:
        key = (r["sentence"], r["pattern"], r["matched_text"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

CHALLENGE_TYPES = ["SEEK_ALTERNATIVE", "POOR_COMMUNICATION", "LACK_TRUST",
                   "CHALLENGE_EXPERTISE", "NON_COMPLIANCE"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the analysis variables used by the statistics below."""
    df = df.copy()
    df["imd_decile"] = pd.to_numeric(df.get(IMD_COL), errors="coerce")
    df["stars"] = pd.to_numeric(df.get("stars"), errors="coerce")
    df["parsons_label"] = df["parsons_label"].fillna("MIXED/NEUTRAL")
    df["is_challenging"] = (df["parsons_label"] == "CHALLENGING").astype(int)

    def has(cell, key):
        if not isinstance(cell, str):
            return False
        return key in {s.strip() for s in cell.split(",") if s.strip()}

    for k in CHALLENGE_TYPES:
        df[f"has_{k}"] = df["challenging_categories"].apply(lambda s, k=k: has(s, k)).astype(int)
    return df


def label_by_stars(df: pd.DataFrame) -> Dict:
    """Chi-square and Cramer's V for the association between label and star rating.
    """
    from scipy import stats

    ct = pd.crosstab(df["parsons_label"], df["stars"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    return {"crosstab": ct, "chi2": chi2, "p": p, "dof": dof, "cramers_v": cramers_v}


def challenge_counts_by_decile(df: pd.DataFrame) -> pd.DataFrame:
    ch = df[df["is_challenging"] == 1]
    counts = ch.groupby("imd_decile")[[f"has_{k}" for k in CHALLENGE_TYPES]].sum()
    totals = ch.groupby("imd_decile").size().rename("n_challenging")
    return pd.concat([totals, counts], axis=1).fillna(0).astype(int)


def deprived_vs_affluent(df: pd.DataFrame) -> pd.DataFrame:
    ch = df[df["is_challenging"] == 1]
    lo = ch[ch["imd_decile"].between(1, 3)]
    hi = ch[ch["imd_decile"].between(8, 10)]
    rows = []
    for k in CHALLENGE_TYPES:
        p1, n1 = lo[f"has_{k}"].mean(), len(lo)
        p2, n2 = hi[f"has_{k}"].mean(), len(hi)
        se = np.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))
        rows.append({"category": k, "n_deprived": n1, "n_affluent": n2,
                     "prop_deprived": p1, "prop_affluent": p2,
                     "difference": p1 - p2, "ci95_halfwidth": 1.96 * se})
    return pd.DataFrame(rows)


def challenge_by_deprivation(df: pd.DataFrame):
    """Logistic regression: P(challenging) as a function of IMD decile."""
    import statsmodels.formula.api as smf

    try:
        model = smf.logit("is_challenging ~ imd_decile",
                          data=df.dropna(subset=["imd_decile"])).fit(disp=False)
    except Exception as exc:  # separation, or too few positive cases
        return None, f"not estimable: {exc}"
    return model, model.get_margeff(at="mean")


def subtype_by_deprivation(df: pd.DataFrame) -> Dict:
    """Logistic regression of each challenge subtype on IMD decile, among
    challenging reviews only."""
    import statsmodels.formula.api as smf

    sub = df[df["is_challenging"] == 1].dropna(subset=["imd_decile"])
    models = {}
    for k in CHALLENGE_TYPES:
        try:
            models[k] = smf.logit(f"has_{k} ~ imd_decile", data=sub).fit(disp=False)
        except Exception as exc:  # separation or too few positives
            models[k] = f"not estimable: {exc}"
    return models


def decile_gradient(df: pd.DataFrame, category: str = "POOR_COMMUNICATION") -> Dict:
    """Spearman rank correlation between IMD decile and a challenge type's share
    of challenge markers.

    Reported across all ten deciles. Restricting the range inflates the
    coefficient substantially and should not be done.
    """
    from scipy import stats

    ch = df[df["is_challenging"] == 1]
    flags = ch.groupby("imd_decile")[[f"has_{k}" for k in CHALLENGE_TYPES]].sum()
    share = (flags[f"has_{category}"] / flags.sum(axis=1) * 100).dropna()
    rho, p = stats.spearmanr(share.index.astype(float), share.values)
    return {"category": category, "share_by_decile": share, "spearman_rho": rho, "p": p}


def run_statistics(df: pd.DataFrame) -> None:
    """Print every statistic reported for this framework."""
    df = prepare(df)
    print(f"n reviews = {len(df):,}")
    print(f"label distribution:\n{df['parsons_label'].value_counts()}\n")

    res = label_by_stars(df)
    print(f"Label x stars: chi2 = {res['chi2']:.1f}, df = {res['dof']}, "
          f"p = {res['p']:.3g}, Cramer's V = {res['cramers_v']:.3f}\n")

    ch = df[df["is_challenging"] == 1]
    if len(ch):
        star5 = (ch["stars"] == 5).mean()
        print(f"challenging reviews = {len(ch):,} ({len(ch)/len(df)*100:.1f}% of corpus); "
              f"{star5*100:.0f}% carry five stars\n")

    print("Challenge markers by IMD decile (raw counts):")
    print(challenge_counts_by_decile(df), "\n")

    print("Deprived (1-3) vs affluent (8-10):")
    print(deprived_vs_affluent(df).to_string(index=False, float_format="%.3f"), "\n")

    model, margeff = challenge_by_deprivation(df)
    if model is None:
        print(f"P(challenging) ~ imd_decile: {margeff}\n")
    else:
        print(model.summary())
        print(margeff.summary(), "\n")

    for name, m in subtype_by_deprivation(df).items():
        print(f"--- {name} ~ imd_decile")
        print(m.get_margeff(at="mean").summary() if hasattr(m, "get_margeff") else m, "\n")

    grad = decile_gradient(df)
    print(f"{grad['category']} share of challenge markers by decile (%):")
    print(grad["share_by_decile"].round(1).to_string())
    print(f"Spearman rho = {grad['spearman_rho']:.3f}, p = {grad['p']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of reviews")
    ap.add_argument("--output", help="where to write the scored CSV")
    ap.add_argument("--text-col", default=TEXT_COL)
    ap.add_argument("--stats-only", action="store_true",
                    help="input is already scored; report statistics only")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    if not args.stats_only:
        analyser = ParsonsAnalyser()
        df = analyser.analyse_frame(df, text_column=args.text_col)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"wrote {args.output}")

    run_statistics(df)


if __name__ == "__main__":
    main()
