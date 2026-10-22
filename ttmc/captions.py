"""Short caption variants, so the model also learns the casual prompts players type.

Cap3D captions are formal and long ("A porcelain figurine of an elderly man with a white
beard, dressed in a blue robe..."). Players type "a figurine" or "a red sword". From each
caption we take the first noun phrase with spaCy and make two variants:
    phrase  the subject with its adjectives      "a porcelain figurine", "a red and black sword"
    noun    the subject without adjectives       "a figurine", "a sword"
"pair of X" / "set of X" style subjects use X. Variants are lowercase with an article.
"""
from __future__ import annotations

import re

# Subjects that only say how many: "a pair of vases" -> the vases are the subject.
QUANTITY = {"pair", "set", "group", "collection", "stack", "pile", "row", "series", "cluster",
            "bunch", "couple", "variety", "assortment", "array", "number", "trio", "lot"}
# Nouns that describe form, not what the thing is ("a 3D model of a house").
VAGUE = {"model", "object", "render", "rendering", "image", "scene", "representation",
         "design", "item", "piece", "structure", "shape", "figure", "form", "version"}


def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _article(words: str) -> str:
    return ("an " if words[:1] in "aeiou" else "a ") + words


def _subject_chunk(doc):
    """First noun chunk that names a thing, skipping 'pair of' / 'model of' wrappers."""
    chunks = list(doc.noun_chunks)
    for i, ch in enumerate(chunks):
        root = ch.root.text.lower()
        nxt = doc[ch.end] if ch.end < len(doc) else None
        if (root in QUANTITY or root in VAGUE) and nxt is not None and nxt.text.lower() == "of" \
                and i + 1 < len(chunks):
            continue
        return ch
    return None


def short_variants(doc) -> tuple[str, ...] | None:
    """(phrase, noun) for a parsed caption, (phrase,) for vague nouns, or None."""
    ch = _subject_chunk(doc)
    if ch is None:
        return None

    def hyphenated(t) -> bool:                  # "two-level", "medieval-style"
        return t.i + 1 < len(doc) and doc[t.i + 1].tag_ == "HYPH"

    def after_hyphen(t) -> bool:
        return t.i > 0 and doc[t.i - 1].tag_ == "HYPH"

    def kept(t) -> bool:
        if t.pos_ == "PUNCT":
            return t.tag_ == "HYPH"
        return t.pos_ not in ("DET", "PRON", "NUM") or hyphenated(t)

    toks = [t for t in ch if kept(t)]
    while toks and toks[0].pos_ == "PUNCT":
        toks = toks[1:]
    if not toks or toks[-1].pos_ not in ("NOUN", "PROPN"):
        return None
    root = ch.root

    def words(ts) -> str:
        text = " ".join(t.text for t in ts)
        return re.sub(r"\s*-\s*", "-", text).lower()

    # noun: the root plus compound nouns in front of it ("kitchen counter", "tree house")
    noun_toks = [t for t in toks if t.i == root.i or (t.dep_ == "compound" and t.head.i == root.i
                                                           and not hyphenated(t) and not after_hyphen(t))]
    out = [words(toks)]
    if root.text.lower() not in VAGUE:          # "an object" says nothing
        out.append(words(noun_toks))
    if root.tag_ == "NN":                       # plurals and names read fine without one
        out = [_article(w) for w in out]
    return tuple(out)


def make_variants(captions: list[str], batch_size: int = 2000) -> list[list[str]]:
    """For each caption: [phrase, noun] (duplicates of the full caption dropped), maybe []."""
    nlp = _nlp()
    out = []
    for cap, doc in zip(captions, nlp.pipe(captions, batch_size=batch_size)):
        v = short_variants(doc)
        full = cap.strip().rstrip(".").lower()
        out.append([] if v is None else list(dict.fromkeys(s for s in v if s != full)))
    return out
