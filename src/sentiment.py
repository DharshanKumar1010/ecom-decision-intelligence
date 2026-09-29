"""Sentiment analysis over synthetic call transcripts (CLAUDE.md section 7.2).

The pipeline is deliberately split into four separately-testable stages:
`preprocess` -> `extract_features` -> `classify` -> `score`. `classify` is
scored on the ORIGINAL, un-preprocessed text, not the tokens `preprocess`
produces. This looks inconsistent at first glance but is intentional: VADER
is a lexicon- and rule-based sentiment scorer that relies on punctuation
("!!!"), capitalization ("GREAT"), and degree modifiers to adjust intensity.
Feeding it the lowercased, punctuation-stripped token stream from
`preprocess` would throw away exactly the signal VADER is designed to use.
`preprocess`/`extract_features` exist to demonstrate the classic sentiment
process (tokenization, stop-word removal, lightweight lemmatization, simple
count/negation features) as a separate, inspectable stage; they are not fed
into `classify`.

Speech analytics here starts from TRANSCRIPTS. No audio and no ASR is
implemented or simulated. `CallTranscripts` is synthetic (see
`src.synthetic`); this module and everything downstream of it must be
labelled as such wherever it is shown.
"""

from __future__ import annotations

import re
import string

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SENTIMENT_NEGATIVE_THRESHOLD,
    SENTIMENT_POSITIVE_THRESHOLD,
    get_logger,
)

logger = get_logger(__name__)

# Small built-in stop-word list (no NLTK/external download per CLAUDE.md
# section 3: "No runtime downloads").
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "in", "on", "at", "for", "with",
        "about", "as", "by", "this", "that", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "my", "your", "his", "her", "its",
        "our", "their", "me", "him", "them", "us", "am", "do", "does", "did",
        "has", "have", "had", "will", "would", "can", "could", "just", "so",
    }
)

# Words/contractions that negate the sentiment of nearby tokens; used by
# `extract_features` for a simple negation flag, not by `classify` (VADER
# has its own, more sophisticated negation handling internally).
_NEGATION_WORDS: frozenset[str] = frozenset({"not", "no", "never", "n't"})

# Very small rule-based lemmatizer: strip common inflectional suffixes.
# Intentionally crude (no POS tagging, no dictionary) since the goal is to
# demonstrate the sentiment-process step, not to build a real lemmatizer.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)


def _lemmatize(token: str) -> str:
    """Strip a common inflectional suffix from a token, rule-based.

    Args:
        token: a lowercased, punctuation-free token.

    Returns:
        The token with at most one matching suffix rule applied. Tokens of
        length <= 3 are returned unchanged to avoid over-stemming short words.
    """
    if len(token) <= 3:
        return token
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def preprocess(text: str) -> list[str]:
    """Lowercase, strip punctuation, tokenize, remove stop words, lemmatize.

    Args:
        text: raw transcript text.

    Returns:
        List of cleaned, lemmatized tokens with stop words removed.
    """
    lowered = text.lower()
    stripped = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"[a-z0-9]+", stripped)
    kept = [token for token in tokens if token not in _STOPWORDS]
    return [_lemmatize(token) for token in kept]


def extract_features(tokens: list[str]) -> dict[str, int | bool]:
    """Extract simple count/negation features from preprocessed tokens.

    These features illustrate the "feature extraction" stage of the
    sentiment process; they are not consumed by `classify` (see module
    docstring for why VADER scores the original text instead).

    Args:
        tokens: tokens returned by `preprocess`.

    Returns:
        Dict with `token_count` (int), `negation_count` (int), and
        `has_negation` (bool).
    """
    negation_count = sum(1 for token in tokens if token in _NEGATION_WORDS)
    return {
        "token_count": len(tokens),
        "negation_count": negation_count,
        "has_negation": negation_count > 0,
    }


_analyzer = SentimentIntensityAnalyzer()


def classify(text: str) -> float:
    """Score the ORIGINAL text with VADER and return the compound score.

    See the module docstring for why this scores `text` directly rather than
    the output of `preprocess`.

    Args:
        text: raw transcript text (not preprocessed).

    Returns:
        VADER compound score in [-1.0, 1.0].
    """
    return float(_analyzer.polarity_scores(text)["compound"])


def score(compound: float) -> str:
    """Map a VADER compound score to a three-way sentiment label.

    Args:
        compound: VADER compound score in [-1.0, 1.0].

    Returns:
        `"Negative"` if `compound <= SENTIMENT_NEGATIVE_THRESHOLD`,
        `"Positive"` if `compound >= SENTIMENT_POSITIVE_THRESHOLD`, else
        `"Neutral"`.
    """
    if compound <= SENTIMENT_NEGATIVE_THRESHOLD:
        return "Negative"
    if compound >= SENTIMENT_POSITIVE_THRESHOLD:
        return "Positive"
    return "Neutral"


def classify_calls(transcripts: pd.DataFrame) -> pd.DataFrame:
    """Classify every call transcript, adding compound score and label.

    Args:
        transcripts: DataFrame with at least a `transcript` column (as
            produced by `src.synthetic.generate_call_transcripts`).

    Returns:
        A copy of `transcripts` with two added columns: `compound` (float)
        and `sentiment_label` (str, one of Negative/Neutral/Positive).
    """
    labeled = transcripts.copy()
    labeled["compound"] = labeled["transcript"].map(classify)
    labeled["sentiment_label"] = labeled["compound"].map(score)
    return labeled


def aggregate_by_agent(labeled_calls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-call sentiment/metrics to per-agent summary statistics.

    Args:
        labeled_calls: output of `classify_calls`, with columns `agent_id`,
            `compound`, `sentiment_label`, `hold_seconds`, `silence_pct`.

    Returns:
        DataFrame indexed by `agent_id` with columns: `mean_sentiment`
        (mean compound score), `mean_hold_seconds`, `mean_silence_pct`,
        `negative_call_share` (share of that agent's calls labeled
        Negative), and `call_count`.
    """
    grouped = labeled_calls.groupby("agent_id")
    summary = pd.DataFrame(
        {
            "mean_sentiment": grouped["compound"].mean(),
            "mean_hold_seconds": grouped["hold_seconds"].mean(),
            "mean_silence_pct": grouped["silence_pct"].mean(),
            "negative_call_share": grouped["sentiment_label"].apply(
                lambda labels: (labels == "Negative").mean()
            ),
            "call_count": grouped.size(),
        }
    )
    return summary.sort_index()


def main() -> None:
    """Run the sentiment pipeline over CallTranscripts and write outputs."""
    transcripts_path = PROCESSED_DATA_DIR / "CallTranscripts.parquet"
    transcripts = pd.read_parquet(transcripts_path)
    logger.info("Loaded %d call transcripts", len(transcripts))

    labeled = classify_calls(transcripts)
    label_counts = labeled["sentiment_label"].value_counts().to_dict()
    logger.info("Sentiment label counts: %s", label_counts)

    agent_summary = aggregate_by_agent(labeled)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(PROCESSED_DATA_DIR / "CallSentiment.parquet", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    agent_summary.to_csv(REPORTS_DIR / "sentiment_agent_summary.csv")
    logger.info(
        "Wrote CallSentiment.parquet (%d rows) and sentiment_agent_summary.csv (%d agents)",
        len(labeled),
        len(agent_summary),
    )


if __name__ == "__main__":
    main()
