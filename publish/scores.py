"""Scoring utilities used by the publish pipeline.

This module is intentionally self-contained and avoids expensive work at import time.
Heavy dependencies (SpaCy pipeline, SBERT model) are lazy-loaded when first used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional


@lru_cache(maxsize=1)
def _spacy_nlp():
    """Create a lightweight English pipeline for tokenization + lemmatization."""
    from spacy.lang.en import English

    nlp = English()
    # Rule-based lemmatizer; requires spacy-lookups-data (via spacy[lookups]).
    nlp.add_pipe("lemmatizer", config={"mode": "rule"})
    nlp.initialize()
    return nlp


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    from spacy.lang.en.stop_words import STOP_WORDS

    return {w.lower() for w in STOP_WORDS}


@lru_cache(maxsize=200_000)
def _lemmatize_cached(text: str) -> tuple[str, ...]:
    nlp = _spacy_nlp()
    stopwords = _stopwords()

    doc = nlp(text)
    lemmas: list[str] = []
    for token in doc:
        lemma = (token.lemma_ or "").strip().lower()
        if not lemma or token.is_punct or lemma in stopwords:
            continue
        lemmas.append(lemma)
    return tuple(lemmas)


def lemmatize(text: object) -> list[str]:
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    return list(_lemmatize_cached(s))


@lru_cache(maxsize=1)
def _sbert_model():
    """Lazy-load SBERT model and pick an available device."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'torch'. Install it to compute SBERT similarities."
        ) from exc

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'sentence-transformers'. Install it to compute SBERT similarities."
        ) from exc

    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def semantic_similarity_score_sbert(string_one: object, string_two: object) -> Optional[float]:
    """Cosine similarity between two strings using SBERT embeddings."""
    if not string_one or not string_two:
        return None
    s1 = str(string_one)
    s2 = str(string_two)
    if not s1.strip() or not s2.strip():
        return None

    from sentence_transformers import util

    model = _sbert_model()
    embeddings = model.encode([s1, s2], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    # Convert to python float
    score = float(cosine_score.cpu().numpy()[0][0])
    return score


def semantic_similarity_score_word_overlap(string_one: object, string_two: object) -> Optional[float]:
    """Word overlap score after lemmatization + stopword removal.

    Defined as: |intersection| / min(|set1|, |set2|)
    """
    if not string_one or not string_two:
        return None

    tokens_one = lemmatize(string_one)
    tokens_two = lemmatize(string_two)

    set_one = {t.lower() for t in tokens_one if t}
    set_two = {t.lower() for t in tokens_two if t}

    if not set_one or not set_two:
        return 0.0
    return len(set_one.intersection(set_two)) / min(len(set_one), len(set_two))


def citation_overlap_score(
    patent_citation_ids: object, paper_citation_ids: object
) -> Optional[float]:
    """Fraction of patent citations that also appear in paper citations."""
    if not isinstance(patent_citation_ids, list) or not isinstance(paper_citation_ids, list):
        return None

    pat = [x for x in patent_citation_ids if x]
    pap = [x for x in paper_citation_ids if x]
    if not pat or not pap:
        return None

    pat_set = {x.lower() if isinstance(x, str) else x for x in pat}
    pap_set = {x.lower() if isinstance(x, str) else x for x in pap}
    if not pat_set:
        return None

    return len(pat_set.intersection(pap_set)) / len(pat_set)

