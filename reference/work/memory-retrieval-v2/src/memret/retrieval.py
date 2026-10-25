"""Local lexical retrieval: CJK-aware tokenizer + BM25 + exact entity/phrase
and term-coverage signals. No embedding service, no external calls.

Design notes (fixes the old "SQL LIKE OR, newest-8, no ranking" defect):
  * ranking is BM25 over the whole pre-`now` selected-branch corpus;
  * exact multi-char phrases (typically names/aliases) get a strong boost;
  * recency is only a tiny tie-break, never the primary order;
  * there is NO truncation of the returned JSON downstream.
"""
from __future__ import annotations

import math
import re
from collections import Counter

CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_WORD_RE = re.compile(rf"[A-Za-z0-9_]{{2,}}|[{CJK}]+")
_PHRASE_RE = re.compile(rf"[{CJK}]{{2,}}|[A-Za-z0-9_]{{2,}}")

_CJK_STOP = {"什么", "怎么", "这个", "那个", "一个", "就是", "然后", "因为", "所以", "可以",
             "我们", "你们", "他们", "自己", "现在", "已经", "还是", "但是", "如果", "不是",
             "没有", "知道", "觉得", "可能", "应该", "时候", "东西", "一下", "这样", "那样"}


def tokenize(text: str) -> list[str]:
    """CJK -> overlapping bigrams (plus single chars); latin/digits -> lowercased words."""
    terms: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        tok = match.group(0)
        if re.fullmatch(rf"[{CJK}]+", tok):
            if len(tok) == 1:
                terms.append(tok)
            else:
                terms.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            terms.append(tok.lower())
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def content_terms(text: str) -> list[str]:
    """Tokenizer for the index: keeps duplicates so term-frequency is meaningful."""
    terms: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        tok = match.group(0)
        if re.fullmatch(rf"[{CJK}]+", tok):
            if len(tok) == 1:
                terms.append(tok)
            else:
                terms.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            terms.append(tok.lower())
    return terms


def query_terms(*parts: str, keep_stop: bool = False) -> list[str]:
    """Distinct, de-noised query terms drawn from keywords/entities/gaps/question."""
    out, seen = [], set()
    for p in parts:
        if not p:
            continue
        for m in _WORD_RE.finditer(p):
            tok = m.group(0)
            if re.fullmatch(rf"[{CJK}]+", tok):
                grams = [tok] if len(tok) == 1 else [tok[i:i + 2] for i in range(len(tok) - 1)]
            else:
                grams = [tok.lower()]
            for g in grams:
                if not keep_stop and g in _CJK_STOP:
                    continue
                if g not in seen:
                    seen.add(g)
                    out.append(g)
    return out


def phrases(*parts: str) -> list[str]:
    """Exact substrings that carry entity/phrase signal (CJK >=2 or latin >=2)."""
    out, seen = [], set()
    for p in parts:
        for m in _PHRASE_RE.finditer(p or ""):
            s = m.group(0)
            if s in _CJK_STOP:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


class Corpus:
    """In-memory BM25 index over a frozen set of documents."""

    K1, B = 1.4, 0.72

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        df: Counter = Counter()
        for d in docs:
            toks = content_terms(d["text"])
            c = Counter(toks)
            self.tf.append(c)
            self.dl.append(len(toks) or 1)
            for t in c:
                df[t] += 1
        self.df = df
        self.n = len(docs)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, idx: int, qterms: list[str], qphrases: list[str]) -> tuple[float, int]:
        tf, dl = self.tf[idx], self.dl[idx]
        bm = 0.0
        matched = 0
        for t in qterms:
            f = tf.get(t, 0)
            if not f:
                continue
            matched += 1
            bm += self._idf(t) * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
        if not qterms:
            return 0.0, 0
        # exact phrase / entity boost (names, aliases, quoted phrases)
        low = self.docs[idx]["text"].lower()
        phrase_boost = sum(2.5 * min(len(p), 6) for p in qphrases if p and p.lower() in low)
        coverage = matched / len(qterms)
        # coverage multiplies BM25 so a doc that matches more distinct query terms wins,
        # then size-independent phrase signal is added.
        return bm * (0.5 + 0.5 * coverage) + phrase_boost, matched


def rank(corpus: Corpus, doc_idx: list[int], qterms: list[str], qphrases: list[str],
         recency_tiebreak: bool = True) -> list[tuple[int, float]]:
    scored: list[tuple[int, float]] = []
    for idx in doc_idx:
        s, matched = corpus.score(idx, qterms, qphrases)
        if matched == 0 and s <= 0.0:
            continue
        scored.append((idx, s))
    if recency_tiebreak:
        # tiny recency nudge only to break exact ties; never dominates relevance.
        scored.sort(key=lambda x: (-x[1], -corpus.docs[x[0]]["t"]))
    else:
        scored.sort(key=lambda x: -x[1])
    return scored
