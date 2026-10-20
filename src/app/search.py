"""Closest-match suggestions for what the user types (planta codes and SKU names)."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

MIN_RATIO = 0.6
FUZZY_CEILING = 600.0  # fuzzy scores stay below 500; every contained-text tier is 600 or more


def normalize(text: str) -> str:
    r"""Lowercase, no accents, runs of anything that is not a letter or digit collapsed to one space."""
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip()


def _score(query: str, tokens: list[str], name: str) -> float:
    r"""0 when `name` does not match; higher is closer. Tiers: exact, prefix, every typed word inside, substring, typos."""
    n = normalize(name)
    if n == query:
        return 1000.0
    if n.startswith(query):
        return 900.0 - len(n) / 1000
    name_tokens = n.split()
    if all(any(t in w for w in name_tokens) for t in tokens):
        return 700.0 - len(n) / 1000
    if query in n:
        return 600.0 - len(n) / 1000
    whole = SequenceMatcher(None, query, n).ratio()
    per_word = sum(max((SequenceMatcher(None, t, w).ratio() for w in name_tokens), default=0.0) for t in tokens) / len(tokens)
    best = max(whole, per_word)
    return best * 500 if best >= MIN_RATIO else 0.0


def suggest(query: str, options: list[str], limit: int = 8) -> list[str]:
    r"""The `limit` options closest to `query`, best first; an empty query returns the first options in their order."""
    query_n = normalize(query)
    if not query_n:
        return list(options[:limit])
    tokens = query_n.split()
    scored = [(s, name) for name in options if (s := _score(query_n, tokens, name)) > 0]
    if any(s >= FUZZY_CEILING for s, _ in scored):
        scored = [(s, name) for s, name in scored if s >= FUZZY_CEILING]  # typo matches only help when nothing contains the text
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:limit]]


def resolve(query: str, options: list[str]) -> str | None:
    r"""The option the text names exactly (ignoring case, accents and punctuation), or None."""
    query_n = normalize(query)
    for name in options:
        if normalize(name) == query_n:
            return name
    return None
