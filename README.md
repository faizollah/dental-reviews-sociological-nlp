# Sociological NLP for online dental patient feedback

Code and theoretical lexicons from a feasibility study testing whether established
sociological theory can be combined with natural language processing to read online
dental patient feedback as social texts rather than as consumer satisfaction data.

Three sociological frameworks are operationalised as transparent, inspectable
text-analysis pipelines and applied to a corpus of 6,172 Google Maps reviews of 100 
dental practices in England, sampled across seven NHS England regions and all ten
deciles of the Index of Multiple Deprivation (IMD).

| Framework | Concept operationalised | Module |
|---|---|---|
| Talcott Parsons, *The Social System* (1951) | The sick role: the obligation to seek and cooperate with technically competent help | `src/parsons_sick_role.py` |
| Michel Foucault, *The Birth of the Clinic* (1973); *The History of Sexuality* (1978) | Biopower and the medical gaze | `src/foucault_biopower_gaze.py` |
| Joseph Walther (1992, 1996) | Social Information Processing theory and the Hyperpersonal Model | `src/walther_sip.py` |

## What is here

```
lexicons/    the three theoretical lexicons as JSON
src/         one analysis module per framework
```

The lexicons are the reusable part. They translate abstract sociological concepts into
concrete keyword, phrase and grammatical patterns, and they can be applied to any
corpus of patient-generated text without using this code. Each was seeded from the
theoretical literature and then refined by reading the corpus, because patients do not
use the vocabulary of theory: they rarely say "bureaucracy" but frequently describe
appointments, waiting lists and paperwork.

## What is not here

The review corpus is not included and cannot be redistributed. The text is subject to
Google's terms of service, and reviews contain personal data. University of Manchester
ethical approval was obtained on that basis and the corpus was de-identified before
analysis using the MedCAT de-identification module.

The modules therefore expect a CSV supplied by the user. The required columns are a
free-text column (default `text`) and, for the deprivation analyses, an IMD decile
column (default `Index of Multiple Deprivation Decile`). A `stars` column enables the
correlation and validity checks.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Each module runs standalone and appends its scores to the input CSV. The Parsons
pipeline is run first if you want the cross-framework correlation, since the SIP module
will use `cooperation_score_norm` when it is present.

```bash
python src/parsons_sick_role.py       --input reviews.csv        --output reviews_parsons.csv
python src/foucault_biopower_gaze.py  --input reviews_parsons.csv --output reviews_foucault.csv
python src/walther_sip.py             --input reviews_foucault.csv --output reviews_sip.csv
```

To re-run only the statistics on an already scored file:

```bash
python src/parsons_sick_role.py --input reviews_parsons.csv --stats-only
```

Use `--text-col` to select the text column, for example `--text-col deid` to analyse
de-identified rather than raw review text.

## Method in brief

All three pipelines share one logic: translate theoretical concepts into keyword and
phrase lists, match them across the corpus, normalise the counts for review length, and
compare the results across IMD deciles.

**Parsons.** Combines three matching strategies: phrase matching for multi-word
expressions, token matching for grammatical constructions, and dependency matching for
negation, so that *I don't trust them* is scored as challenging rather than upholding.
Each review receives a cooperation score, normalised by sentence count, and a label of
`UPHOLDING`, `CHALLENGING` or `MIXED/NEUTRAL`. Statistics: chi-square and Cramér's V for
label against star rating; raw challenge-marker counts by decile; differences in
proportions between deciles 1 to 3 and 8 to 10 with Wald confidence intervals; logistic
regression of challenge on deprivation with marginal effects; and Spearman rank
correlation across deciles.

**Foucault.** Scores biopower and medical-gaze language per 1,000 alphabetic tokens
across fourteen subcategories. Adds a passivity score, the ratio of passive to active
constructions, as a rough proxy for the grammatical agency patients claim. Runs Latent
Dirichlet Allocation over unigrams and bigrams, then scores each topic for the share of
its mass falling on each lexicon. Convergence between unsupervised topics and
theory-derived lexicons is evidence that the lexicons track something in the data rather
than something projected onto it.

**Walther.** Scores seven categories of communicative cue per 1,000 tokens, capturing
how patients compensate in text for the absence of the non-verbal signals available
face to face. Correlates the scores with star ratings and with the Parsonian cooperation
measure.

## Lexicon format

`lexicons/parsons_sick_role.json` maps each of nine categories to its phrase list, its
spaCy token patterns and its polarity, alongside the dependency pattern used for
negation. `lexicons/foucault_biopower_gaze.json` and `lexicons/walther_sip.json` map
each concept to named subcategories and their phrase lists. All three are plain JSON and
can be loaded without spaCy.

## Citation

If you use these lexicons or this code, please cite the archived release (see the DOI
badge above) and the study:

> Feizollah A, Li M, Byrne M. [Title]. *Digital Health*. [Year].

Preliminary findings were presented at UK Public Health Science 2026 and published as a
conference abstract: Li M, Feizollah A, Byrne M. Socioeconomic inequalities in
patient-provider communication: a cross-sectional natural language processing (NLP)
study of UK dental care reviews. *Journal of Epidemiology & Community Health*
2026;80:A6. https://doi.org/10.1136/UKPHSC-2026-abstracts.13

## Funding

This work was supported by the British Academy/Leverhulme Small Research Grant
SRG2425\250409, "A Sociological Exploration of Online Dental Patient Feedback in the UK:
A Feasibility Study Using Natural Language Processing".

## Licence

Code is released under the MIT Licence. The lexicons in `lexicons/` are
released under CC BY 4.0, so they may be reused and adapted with attribution.
