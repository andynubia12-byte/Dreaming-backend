"""Render the derived readable files.

`render_memory` produces the exact "memory material" text that is written to
`memory.md` and passed to the summary-generation model. Because the summary is a
function of THIS material (never of any previous summary), the current
understanding is reconstructable from evidence + sourced claims alone.
"""

from __future__ import annotations

import json

from .store import dir_name

# Statuses shown as "in force" / planned / historical / unconfirmed.
BUCKETS: list[tuple[str, set[str]]] = [
    ("当前有效", {"active", "temporary"}),
    ("计划（含进行中）", {"planned", "in_progress"}),
    ("计划已到期（无完成证据）", {"plan_due_unresolved"}),
    ("存在冲突（待核验，未定论）", {"contested"}),
    ("历史与已变更", {"historical", "superseded", "retracted", "completed", "abandoned", "expired"}),
    ("未确认", {"unconfirmed"}),
]
# 'rejected' claims are audit-only and are never rendered into the material.
HIDDEN = {"rejected"}

# Kinds where the SAME source sentence can bind two mentions together, so the
# verbatim user source must be visible for the summary to resolve co-reference
# instead of silently splitting one person/thing into two.
CONTEXT_KINDS = {"relation"}
CONTEXT_CLIP = 320


def empty_summary(uid: str) -> str:
    return f"# 常驻摘要（尚未形成）\n\n用户 `{uid}` 暂无任何已核验的记忆材料。\n"


def _clip(text: str, limit: int = CONTEXT_CLIP) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def render_evidence(uid: str, messages: list[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(json.dumps({
            "user_id": uid,
            "message_id": m["message_id"],
            "conversation_id": m["conversation_id"],
            "role": m["role"],
            "message_at": m["message_at"],
            "text": m["text"],
            "ingested_at": m["ingested_at"],
            "batch_id": m["batch_id"],
        }, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _source_token(s: dict) -> str:
    return f"{s['message_id']}({s['role']})@{s['message_at']}"


def _claim_line(c: dict, sources: list[dict], msg_text: dict[str, str] | None = None) -> str:
    parts = [f"- [{c['status']}][{c['kind']}][{c.get('importance') or 'medium'}] {c['text']}"]
    meta: list[str] = []
    if c.get("relative_phrase"):
        # A relative phrase is anchored to the SOURCE MESSAGE's own time (never to
        # the current now nor to an event_at), so "昨天/这次/月底" keep their scope.
        src_anchor = sources[0]["message_at"] if sources else ""
        anchored = (f"，锚点 {src_anchor}" if src_anchor else "")
        if c.get("resolved_from_phrase"):
            meta.append(f"原话『{c['relative_phrase']}』→ {c['resolved_from_phrase']}{anchored}")
        else:
            meta.append(f"原话『{c['relative_phrase']}』（未换算为具体日期{anchored}）")
    window = []
    if c.get("valid_from"):
        window.append(f"自 {c['valid_from']}")
    if c.get("valid_to"):
        window.append(f"至 {c['valid_to']}")
    if window:
        meta.append("有效窗口（系统/模型推定的有效期，非用户原话）：" + " ".join(window))
    if c.get("event_at"):
        meta.append(f"事件时点 {c['event_at']}")
    if c.get("replaced_by"):
        meta.append(f"已被 {c['replaced_by']} 取代")
    if c.get("replaces"):
        if c.get("edge") == "replaces":
            meta.append(f"取代 {c['replaces']}（替换关系已证明）")
        else:
            # The link to the target exists but the replacement relation itself is
            # NOT proven: say so, so the material never reads as a settled
            # replacement even after the new content is confirmed.
            meta.append(f"指向 {c['replaces']}，但替换关系未证明（待核验）")
    if c.get("verification"):
        meta.append(f"核验 {c['verification']}")
    if c.get("note"):
        meta.append(f"备注 {c['note']}")
    src = "; ".join(_source_token(s) for s in sources) if sources else "（无来源）"
    meta.append(f"来源 {src}")
    quotes = [f"{s.get('role') or '?'}『{s['quote']}』" for s in sources if s.get("quote")]
    if quotes:
        meta.append("来源原句 " + " / ".join(quotes))
    if msg_text and c["kind"] in CONTEXT_KINDS:
        # Verbatim USER source messages only (assistant analysis is never context
        # for a user fact), so a single sentence that binds two mentions stays
        # together instead of being split into two people.
        seen: set[str] = set()
        for s in sources:
            mid = s.get("message_id")
            if s.get("role") != "user" or not mid or mid in seen:
                continue
            text = msg_text.get(mid)
            if text:
                seen.add(mid)
                meta.append(f"来源原消息({mid})『{_clip(text)}』")
    meta.append(f"更新 {c.get('last_updated')}")
    parts.append("（" + "；".join(meta) + "）")
    return "".join(parts)


def render_memory(uid: str, version: int, now_text: str, claims: list[dict],
                  sources: dict[str, list[dict]],
                  msg_text: dict[str, str] | None = None) -> str:
    shown = [c for c in claims if c["status"] not in HIDDEN]
    lines: list[str] = [
        f"# 记忆材料（用户 {dir_name(uid)}）",
        "",
        f"用户ID：`{uid}`",
        f"版本：v{version}",
        f"最近 now：{now_text or '(未设置)'}",
        f"主张总数：{len(shown)}（其中被判为未采纳的不在此列）",
        "",
        "> 本文件是带来源与状态的主张集合，是生成常驻摘要的唯一材料。",
        "> 旧摘要不参与本文件的形成；每条主张都必须能回溯到原始证据。",
        "> `有效窗口` 是系统/模型依原话推定的有效期，用于判过期，不是用户事实本身。",
        "> `存在冲突`/`未确认`/`未定` 表示尚未定论，不得当作已确定的当前事实。",
        "",
    ]
    for title, statuses in BUCKETS:
        group = [c for c in shown if c["status"] in statuses]
        lines.append(f"## {title}")
        if not group:
            lines.append("（无）")
        else:
            for c in group:
                lines.append(_claim_line(c, sources.get(c["claim_id"], []), msg_text))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
