"""Local lexical retrieval for the trial (no embeddings, no network).

Two searchable sources, with namespaced source ids so they can never be confused:
  * imported export history  ->  source_id  "msg:<conversation_id>#<node_id>"
  * prototype chat_log turns ->  source_id  "chat#<id>"

Every lookup is time-guarded: only rows with message time strictly BEFORE the
request's single simulated `now` (Asia/Shanghai) are visible, selected branch only,
and assistant reasoning rows are never returned.

Ranking is BM25 (CJK bigrams + exact phrase/entity boost); recency is only a
tie-breaker. Results are paginated with explicit `has_more`; nothing is a sliced
JSON string.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3
import time
from collections import Counter
from typing import Any

from .util import looks_like_tool_serialization

SHANGHAI = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")

CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_WORD_RE = re.compile(rf"[A-Za-z0-9_]{{2,}}|[{CJK}]+")
_PHRASE_RE = re.compile(rf"[{CJK}]{{2,}}|[A-Za-z0-9_]{{2,}}")
_CJK_STOP = {"什么", "怎么", "这个", "那个", "一个", "就是", "然后", "因为", "所以", "可以",
             "我们", "你们", "他们", "自己", "现在", "已经", "还是", "但是", "如果", "不是",
             "没有", "知道", "觉得", "可能", "应该", "时候", "东西", "一下", "这样", "那样"}

MAX_TEXT = 4000


def now_epoch() -> float:
    return time.time()


def now_shanghai(epoch: float | None = None) -> str:
    return dt.datetime.fromtimestamp(epoch if epoch is not None else time.time(), SHANGHAI).isoformat()


def epoch_of(iso_str: str) -> float:
    return dt.datetime.fromisoformat(iso_str).timestamp()


def _tokenize(text: str) -> list[str]:
    terms: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        tok = match.group(0)
        if re.fullmatch(rf"[{CJK}]+", tok):
            terms.extend([tok] if len(tok) == 1 else (tok[i:i + 2] for i in range(len(tok) - 1)))
        else:
            terms.append(tok.lower())
    return terms


def _query_terms(*parts: str, keep_stop: bool = True) -> list[str]:
    out, seen = [], set()
    for p in parts:
        for g in _tokenize(p or ""):
            if not keep_stop and g in _CJK_STOP:
                continue
            if g not in seen:
                seen.add(g)
                out.append(g)
    return out


def _phrases(*parts: str) -> list[str]:
    out, seen = [], set()
    for p in parts:
        for m in _PHRASE_RE.finditer(p or ""):
            s = m.group(0)
            if s not in _CJK_STOP and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def excerpt(memory_text: str, query: str, max_chars: int = 2500) -> str:
    """Pick the memory lines most related to `query` (bounded known-facts excerpt)."""
    if not memory_text:
        return ""
    terms = _query_terms(query, keep_stop=False) or _query_terms(query)
    lines = [ln for ln in memory_text.splitlines() if ln.strip()]
    scored = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        hits = sum(1 for t in terms if t and t in low)
        if hits:
            scored.append((hits, -i, i))
    scored.sort(reverse=True)
    picked, total = [], 0
    for _, _, i in scored:
        if total + len(lines[i]) > max_chars:
            continue
        picked.append((i, lines[i]))
        total += len(lines[i])
        if total >= max_chars * 0.85:
            break
    picked.sort()
    return "\n".join(ln for _, ln in picked)


class _Corpus:
    K1, B = 1.4, 0.72

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        df: Counter = Counter()
        for d in docs:
            c = Counter(_tokenize(d["text"]))
            self.tf.append(c)
            self.dl.append(sum(c.values()) or 1)
            for t in c:
                df[t] += 1
        self.df, self.n = df, len(docs)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0

    def _idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, i: int, terms: list[str], phrases: list[str]) -> tuple[float, int]:
        if not terms:
            return 0.0, 0
        tf, dl = self.tf[i], self.dl[i]
        bm, matched = 0.0, 0
        for t in terms:
            f = tf.get(t, 0)
            if not f:
                continue
            matched += 1
            bm += self._idf(t) * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
        low = self.docs[i]["text"].lower()
        phrase_boost = sum(2.5 * min(len(p), 6) for p in phrases if p and p.lower() in low)
        return bm * (0.5 + 0.5 * matched / len(terms)) + phrase_boost, matched


def _snippet(text: str, terms: list[str], width: int = 220) -> str:
    low = text.lower()
    pos = min((low.find(t) for t in terms if t and low.find(t) >= 0), default=0)
    start = max(0, pos - width // 3)
    seg = text[start:start + width]
    return ("…" if start > 0 else "") + seg + ("…" if start + width < len(text) else "")


def _cap(text: str, n: int = MAX_TEXT) -> tuple[str, bool, int]:
    return (text, False, len(text)) if len(text) <= n else (text[:n], True, len(text))


def _export_sid(conv: str, node: str) -> str:
    return f"msg:{conv}#{node}"


def _chat_sid(cid: int) -> str:
    return f"chat#{cid}"


class TrialRetrieval:
    """One frozen view of the searchable past, bounded by a single `now`."""

    def __init__(self, conn: sqlite3.Connection, instance: str, now: float,
                 exclude_chat_ids: tuple[Any, ...] = ()):
        self.conn = conn
        self.instance = instance
        self.now = float(now)
        self.exclude = {str(x) for x in exclude_chat_ids}
        self.docs = self._load()
        self.by_sid = {d["sid"]: d for d in self.docs}
        self.corpus = _Corpus(self.docs)
        self._thread_cache: dict[str, list[dict]] = {}

    def _load(self) -> list[dict]:
        docs: list[dict] = []
        for r in self.conn.execute(
            "SELECT conversation_id, node_id, message_id, role, create_time, text FROM messages "
            "WHERE is_selected_branch=1 AND is_assistant_reasoning=0 "
            "AND text IS NOT NULL AND LENGTH(text)>0 AND create_time < ? ORDER BY create_time, node_id",
            (self.now,),
        ).fetchall():
            docs.append({"sid": _export_sid(r["conversation_id"], r["node_id"]), "kind": "export",
                         "role": r["role"], "t": float(r["create_time"]), "text": r["text"],
                         "conv": r["conversation_id"], "mid": r["message_id"], "node": r["node_id"],
                         "chat_id": None})
        for r in self.conn.execute(
            "SELECT id, created_at, role, content, conversation_id FROM chat_log "
            "WHERE instance=? AND role IN ('user','assistant') AND content IS NOT NULL "
            "AND LENGTH(content)>0 AND created_at < ? ORDER BY id",
            (self.instance, self.now),
        ).fetchall():
            if str(r["id"]) in self.exclude:
                continue
            # Never surface a failed/leaked assistant turn as retrievable evidence;
            # the row (and its id) stays intact in chat_log.
            if r["role"] == "assistant" and looks_like_tool_serialization(r["content"]):
                continue
            docs.append({"sid": _chat_sid(r["id"]), "kind": "chat", "role": r["role"],
                         "t": float(r["created_at"]), "text": r["content"],
                         "conv": r["conversation_id"], "mid": None, "node": None,
                         "chat_id": r["id"]})
        return docs

    def stats(self) -> dict:
        return {"now": now_shanghai(self.now), "instance": self.instance,
                "export_docs": sum(1 for d in self.docs if d["kind"] == "export"),
                "chat_docs": sum(1 for d in self.docs if d["kind"] == "chat")}

    def search(self, query: str = "", *, keywords: Any = (), entities: Any = (),
               time_start: str | None = None, time_end: str | None = None,
               conversation_id: str | None = None, kind: str | None = None,
               limit: int = 8) -> dict:
        kw = [keywords] if isinstance(keywords, str) else list(keywords or [])
        en = [entities] if isinstance(entities, str) else list(entities or [])
        terms = _query_terms(query, *kw, *en)
        ph = _phrases(query, *kw, *en)
        try:
            limit = max(1, min(int(limit or 8), 20))
        except (TypeError, ValueError):
            limit = 8
        lo = hi = None
        for raw, which in ((time_start, "lo"), (time_end, "hi")):
            if raw:
                try:
                    if which == "lo":
                        lo = epoch_of(str(raw))
                    else:
                        hi = epoch_of(str(raw))
                except (ValueError, TypeError):
                    pass
        cand = []
        for i, d in enumerate(self.docs):
            if lo is not None and d["t"] < lo:
                continue
            if hi is not None and d["t"] >= min(hi, self.now):
                continue
            if conversation_id and d["conv"] != conversation_id:
                continue
            if kind and d["kind"] != kind:
                continue
            cand.append(i)
        scored = []
        for i in cand:
            s, matched = self.corpus.score(i, terms, ph)
            if matched == 0 and s <= 0:
                continue
            scored.append((i, s))
        scored.sort(key=lambda x: (-x[1], -self.docs[x[0]]["t"]))
        hits = []
        for i, s in scored[:limit]:
            d = self.docs[i]
            hits.append({"source_id": d["sid"], "source": d["kind"], "role": d["role"],
                         "message_at": now_shanghai(d["t"]), "score": round(s, 4),
                         "text_len": len(d["text"]), "snippet": _snippet(d["text"], terms),
                         "conversation_id": d["conv"]})
        dropped = max(0, len(scored) - len(hits))
        return {"tool": "search_history", "query": query, "terms_used": terms[:24],
                "phrases_used": ph[:12], "now": now_shanghai(self.now),
                "results": hits, "matched_total": len(scored), "returned": len(hits),
                "has_more": dropped > 0,
                "more_hint": (f"{dropped} more pre-now matches ranked below the cut" if dropped else ""),
                "note": "BM25 + exact phrase boost; recency only a tie-break; all rows strictly before "
                        "now, selected branch, non-reasoning."}

    def read(self, source_id: str, *, max_chars: int = MAX_TEXT) -> dict:
        d = self._resolve(source_id)
        if d is None:
            return {"tool": "read_message", "source_id": source_id, "found": False,
                    "error": "not found before now (or not a visible user/assistant message)"}
        text, trunc, total = _cap(d["text"], max_chars)
        return {"tool": "read_message", "found": True, "item": {
            "source_id": d["sid"], "source": d["kind"], "role": d["role"],
            "message_at": now_shanghai(d["t"]), "conversation_id": d["conv"],
            "text": text, "truncated": trunc, "total_chars": total}}

    def expand(self, source_id: str, *, before: int = 2, after: int = 2, offset: int = 0) -> dict:
        target = self._resolve(source_id)
        if target is None:
            return {"tool": "expand", "source_id": source_id, "found": False,
                    "error": "target not found before now"}
        thread = self._thread(target["conv"], target["kind"])
        idx = next((i for i, d in enumerate(thread) if d["sid"] == target["sid"]), None)
        if idx is None:
            return {"tool": "expand", "source_id": source_id, "found": False,
                    "error": "target vanished from its own thread"}
        try:
            before = max(0, min(int(before), 12))
            after = max(0, min(int(after), 12))
            offset = int(offset)
        except (TypeError, ValueError):
            before = after = 2
            offset = 0
        lo = max(0, idx - before + offset)
        hi = min(len(thread), idx + after + 1 + offset)
        if lo > idx or hi <= idx:
            lo, hi = max(0, idx - before), min(len(thread), idx + after + 1)
        items = []
        for j in range(lo, hi):
            d = thread[j]
            text, trunc, total = _cap(d["text"])
            items.append({"source_id": d["sid"], "source": d["kind"], "role": d["role"],
                          "message_at": now_shanghai(d["t"]), "conversation_id": d["conv"],
                          "text": text, "truncated": trunc, "total_chars": total,
                          "is_target": j == idx})
        return {"tool": "expand", "found": True, "target": target["sid"], "target_index": idx,
                "thread_len": len(thread), "window": [lo, hi - 1], "items": items, "n_items": len(items),
                "has_more_before": lo > 0, "has_more_after": hi < len(thread),
                "note": "Same-source neighbours; the target is always included."}

    def _thread(self, conv: str, kind: str) -> list[dict]:
        key = f"{kind}:{conv}"
        if key not in self._thread_cache:
            self._thread_cache[key] = [d for d in self.docs if d["kind"] == kind and d["conv"] == conv]
        return self._thread_cache[key]

    def _resolve(self, source_id: str) -> dict | None:
        sid = str(source_id or "").strip()
        if not sid:
            return None
        if sid in self.by_sid:
            return self.by_sid[sid]
        if sid.startswith("chat#"):
            return None
        bare = sid[4:] if sid.startswith("msg:") else sid
        if "#" in bare:
            conv, node = bare.split("#", 1)
            if conv and node:
                return self.by_sid.get(_export_sid(conv, node))
        for d in self.docs:
            if d["mid"] == bare or d["node"] == bare:      # exact id (any length)
                return d
        if len(bare) >= 6:
            for d in self.docs:
                if (d["mid"] or "").startswith(bare):
                    return d
        return None
