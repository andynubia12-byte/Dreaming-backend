"""history-bootstrap-v1 core: read-only ChatGPT export -> evidence / batches / retrieval import.

Minimal, no framework. Reads `conversations-*.json` read-only, follows the
`current_node` parent chain to the SELECTED branch only, and produces:
  * evidence.jsonl  — selected-branch user/assistant (non-reasoning) messages
  * manifest.json   — source hashes, counts, branch issues, batch plan
  * batches/batch-000001.json ...  — fixed time-boundary slices compatible with
                       the memory-maintenance CLI batch format
  * retrieval_import.sqlite  — a NEW sqlite matching a copy of the retrieval schema

Never writes into the export dir; never opens/modifies another sqlite.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

SH = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")
REASONING_CONTENT_TYPES = {"thoughts", "reasoning_recap"}
FACT_ROLES = ("user", "assistant")
SKIP_ROLES = ("system", "tool")
# Text assembly policy (recorded in manifest/plan): each surviving part is used
# VERBATIM, parts are joined by "\n", empty-string parts are skipped, and no outer
# whitespace strip is applied. Dict parts carrying a string `text` (e.g. speech
# transcripts inside multimodal_text) are preserved; dict parts without text
# (image/audio/metadata) contribute NO text and are never rendered as str(dict).
TEXT_POLICY = ("parts joined by '\\n'; empty-string parts skipped; dict.text parts "
               "(e.g. audio transcripts) preserved verbatim; dict parts without text "
               "contribute no content; no outer strip")
# Oversize is REPORTED ONLY (never split or truncated): a batch at/over either
# threshold is listed in plan.oversized_batches so a finer --window can be chosen.
OVERSIZE_MESSAGES = 800
OVERSIZE_CHARS = 400_000
RETRIEVAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations(
  id TEXT PRIMARY KEY, title TEXT, create_time REAL, update_time REAL, current_node TEXT,
  node_count INTEGER, message_count INTEGER, source_file TEXT);
CREATE TABLE IF NOT EXISTS messages(
  conversation_id TEXT, node_id TEXT, message_id TEXT, parent_id TEXT, role TEXT,
  content_type TEXT, text TEXT, create_time REAL, is_selected_branch INTEGER,
  is_assistant_reasoning INTEGER, has_attachment INTEGER,
  PRIMARY KEY(conversation_id, node_id));
CREATE INDEX IF NOT EXISTS idx_msg_sel ON messages(conversation_id, is_selected_branch);
CREATE INDEX IF NOT EXISTS idx_msg_role_time ON messages(role, create_time);
"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def approx_tokens(text: str) -> int:
    """Same mixed CJK/latin heuristic as the demo util (an estimate, not a count)."""
    cjk = other = 0
    for ch in text:
        if "\u3040" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef":
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def iso_sh(epoch: float) -> str:
    return dt.datetime.fromtimestamp(float(epoch), SH).isoformat()


def _split_parts(parts: Any) -> tuple[str, int]:
    """Return (joined_text, omitted_non_text_parts).

    str parts and dict parts carrying a string `text` (e.g. speech transcripts in
    multimodal_text) are preserved; any other part (image/audio/metadata) is counted
    as omitted and contributes no text — never rendered as str(dict).
    """
    out: list[str] = []
    omitted = 0
    for part in parts or []:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                out.append(text)
            else:
                omitted += 1
        else:
            omitted += 1
    return "\n".join(p for p in out if p), omitted


def _attachments(message: dict) -> list[dict]:
    """Record attachment/media GAPS only; never fabricate content or emit str(dict)."""
    gaps: list[dict] = []
    content = message.get("content") or {}
    media = ("image", "audio", "video")
    for part in content.get("parts") or []:
        if not isinstance(part, dict):
            continue
        pointer = part.get("asset_pointer")
        ctype = part.get("content_type")
        if pointer:
            fid = str(pointer).replace("sediment://", "").split("/")[-1]
            gaps.append({"file_id": fid, "asset_pointer": pointer, "name": None,
                         "mime_type": ctype, "size": part.get("size_bytes"),
                         "width": part.get("width"), "height": part.get("height"),
                         "content": None, "gap": True})
        elif isinstance(ctype, str) and any(m in ctype for m in media):
            gaps.append({"file_id": None, "asset_pointer": None, "name": None,
                         "mime_type": ctype, "size": None, "width": None, "height": None,
                         "content": None, "gap": True})
    for asset in content.get("assets") or []:
        if isinstance(asset, dict) and asset.get("asset_pointer"):
            pointer = asset["asset_pointer"]
            fid = str(pointer).replace("sediment://", "").split("/")[-1]
            gaps.append({"file_id": fid, "asset_pointer": pointer, "name": asset.get("name"),
                         "mime_type": asset.get("mime_type"), "size": asset.get("size"),
                         "width": asset.get("width"), "height": asset.get("height"),
                         "content": None, "gap": True})
    for att in (message.get("metadata") or {}).get("attachments") or []:
        gaps.append({"file_id": att.get("id"), "asset_pointer": None, "name": att.get("name"),
                     "mime_type": att.get("mime_type"), "size": att.get("size"),
                     "width": att.get("width"), "height": att.get("height"),
                     "content": None, "gap": True})
    return gaps


def _paste_suspect(role: str, text: str) -> bool:
    """Heuristic flag only — never a fact assertion."""
    if role != "user" or not text:
        return False
    return len(text) > 800 or "```" in text or text.lstrip().startswith(">")


def scan_sources(export_dir: Path) -> list[dict]:
    files = []
    for path in sorted(export_dir.glob("conversations-*.json")):
        if path.name.startswith("."):
            continue
        files.append({"path": str(path), "name": path.name, "size": path.stat().st_size,
                      "sha256": sha256_file(path)})
    return files


def load_conversations(files: list[dict]) -> list[dict]:
    """Each conversation tagged with its source file name."""
    out = []
    for sf in files:
        data = json.loads(Path(sf["path"]).read_text("utf-8"))
        for conv in data:
            conv = dict(conv)
            conv["_source_file"] = sf["name"]
            out.append(conv)
    return out


def selected_chain(mapping: dict, current_node: Any) -> tuple[list[str], list[dict]]:
    """Ordered node ids root->leaf along the current_node parent chain, plus issues."""
    issues: list[dict] = []
    if not current_node or current_node not in mapping:
        issues.append({"kind": "missing_current_node", "current_node": current_node})
        return [], issues
    chain: list[str] = []
    seen: set[str] = set()
    node = current_node
    prev: str | None = None
    while node is not None:
        if node in seen:
            issues.append({"kind": "cycle", "node": node})
            break
        if node not in mapping:
            issues.append({"kind": "broken_parent", "child": prev, "parent": node})
            break
        seen.add(node)
        chain.append(node)
        prev = node
        node = (mapping.get(node) or {}).get("parent")
    chain.reverse()
    return chain, issues


def extract_conversation(conv: dict) -> dict:
    cid = conv.get("id") or conv.get("conversation_id")
    mapping = conv.get("mapping") or {}
    chain, issues = selected_chain(mapping, conv.get("current_node"))
    selected = set(chain)
    skipped = {"reasoning": 0, "system": 0, "tool": 0, "empty": 0, "other_role": 0}
    records: list[dict] = []
    attachments = 0
    for index, node_id in enumerate(chain):
        node = mapping.get(node_id) or {}
        message = node.get("message")
        if not message:
            continue
        role = (message.get("author") or {}).get("role") or "unknown"
        content = message.get("content") or {}
        ctype = content.get("content_type") or "text"
        if ctype in REASONING_CONTENT_TYPES:
            skipped["reasoning"] += 1
            continue
        if role in SKIP_ROLES:
            skipped[role] += 1
            continue
        if role not in FACT_ROLES:
            skipped["other_role"] += 1
            continue
        text, omitted = _split_parts(content.get("parts"))
        atts = _attachments(message)
        attachments += len(atts)
        if not text.strip():
            skipped["empty"] += 1
            continue
        raw_time = message.get("create_time")
        has_time = isinstance(raw_time, (int, float))
        records.append({
            "evidence_id": f"msg:{cid}#{node_id}",
            "conversation_id": cid,
            "conversation_title": conv.get("title"),
            "node_id": node_id,
            "message_id": message.get("id"),
            "parent_id": node.get("parent"),
            "branch_index": index,
            "role": role,
            "content_type": ctype,
            "message_at": iso_sh(raw_time) if has_time else None,
            "message_at_epoch": float(raw_time) if has_time else None,
            "time_missing": not has_time,
            "time_source": "message.create_time" if has_time else None,
            "text": text,
            "attachments": atts,
            "omitted_non_text_parts": omitted,
            "flags": ["pasted_text_suspect"] if _paste_suspect(role, text) else [],
        })
    # branch-order vs real timestamp: record non-monotonic neighbours explicitly
    out_of_order = []
    dated = [r for r in records if r["message_at_epoch"] is not None]
    for prev, cur in zip(dated, dated[1:]):
        if cur["message_at_epoch"] < prev["message_at_epoch"]:
            out_of_order.append({
                "conversation_id": cid, "node_id": cur["node_id"], "prev_node_id": prev["node_id"],
                "message_at": cur["message_at"], "prev_message_at": prev["message_at"],
                "note": "branch order advances but timestamp goes backwards",
            })
    return {
        "conversation_id": cid, "title": conv.get("title"), "source_file": conv.get("_source_file"),
        "current_node": conv.get("current_node"), "chain": chain, "issues": issues,
        "nodes_total": len(mapping), "nodes_selected": len(selected),
        "nodes_not_selected": len(mapping) - len(selected),
        "records": records, "skipped": skipped, "attachments": attachments,
        "out_of_order": out_of_order,
        "create_time": conv.get("create_time"), "update_time": conv.get("update_time"),
    }


def _window(epoch: float, kind: str) -> tuple[str, float]:
    """Return (window_key, window_end_epoch) for a fixed calendar boundary."""
    d = dt.datetime.fromtimestamp(float(epoch), SH)
    if kind == "day":
        start = dt.datetime(d.year, d.month, d.day, tzinfo=SH)
        return d.strftime("%Y-%m-%d"), (start + dt.timedelta(days=1)).timestamp()
    if kind == "year":
        return f"{d.year:04d}", dt.datetime(d.year + 1, 1, 1, tzinfo=SH).timestamp()
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return f"{d.year:04d}-{d.month:02d}", dt.datetime(y, m, 1, tzinfo=SH).timestamp()


def dedup_records(records: list[dict]) -> tuple[list[dict], int, list[dict]]:
    """Same message id + identical content -> dedup; different content -> conflict."""
    kept: list[dict] = []
    seen: dict[str, dict] = {}
    deduped = 0
    conflicts: list[dict] = []
    for rec in records:
        key = rec.get("message_id") or rec["evidence_id"]
        sig = (rec["role"], rec["text"], rec["message_at"])
        if key in seen:
            prev = seen[key]
            if (prev["role"], prev["text"], prev["message_at"]) == sig:
                deduped += 1
                continue
            conflicts.append({"message_id": key, "kept": prev["evidence_id"],
                              "conflict": rec["evidence_id"], "note": "same id, different content"})
            continue
        seen[key] = rec
        kept.append(rec)
    return kept, deduped, conflicts


def collect(export_dir: Path, *, user_id: str, window: str = "month",
            as_of: float | None = None) -> dict:
    sources = scan_sources(export_dir)
    conversations = load_conversations(sources)
    convs = [extract_conversation(c) for c in conversations]
    all_records: list[dict] = []
    issues = {"missing_current_node": [], "broken_parent": [], "cycle": []}
    skipped = {"reasoning": 0, "system": 0, "tool": 0, "empty": 0, "other_role": 0}
    out_of_order: list[dict] = []
    attachments = nodes_total = nodes_selected = 0
    for c in convs:
        all_records.extend(c["records"])
        nodes_total += c["nodes_total"]
        nodes_selected += c["nodes_selected"]
        attachments += c["attachments"]
        out_of_order.extend(c["out_of_order"])
        for k in skipped:
            skipped[k] += c["skipped"].get(k, 0)
        for issue in c["issues"]:
            bucket = issues.get(issue["kind"])
            if bucket is not None:
                bucket.append({"conversation_id": c["conversation_id"], **issue})
    all_kept, deduped, conflicts = dedup_records(all_records)
    excluded_after_as_of = 0
    if as_of is not None:
        kept = []
        for r in all_kept:
            t = r["message_at_epoch"]
            if t is None:
                kept.append(r)                        # retained (flagged time_missing); never batched
            elif t < as_of:
                kept.append(r)
            else:
                excluded_after_as_of += 1
    else:
        kept = all_kept
    undated = [r for r in kept if r["message_at_epoch"] is None]
    dated = [r for r in kept if r["message_at_epoch"] is not None]

    windows: dict[str, dict] = {}
    for r in dated:
        key, end_epoch = _window(r["message_at_epoch"], window)
        w = windows.setdefault(key, {"window": key, "end_epoch": end_epoch, "records": []})
        w["records"].append(r)
    for w in windows.values():
        w["now_epoch"] = min(w["end_epoch"], as_of) if as_of is not None else w["end_epoch"]
        w["clamped"] = as_of is not None and w["end_epoch"] > as_of
    ordered = sorted(windows, key=lambda k: windows[k]["now_epoch"])
    batch_plan = []
    prior_chars = 0
    prior_tokens = 0
    for i, key in enumerate(ordered, start=1):
        w = windows[key]
        recs = sorted(w["records"], key=lambda r: (r["message_at_epoch"], r["branch_index"]))
        new_chars = sum(len(r["text"]) for r in recs)
        new_tokens = sum(approx_tokens(r["text"]) for r in recs)
        batch_plan.append({
            "batch_id": f"b{i:06d}", "file": f"batch-{i:06d}.json", "window": key,
            "now": iso_sh(w["now_epoch"]), "clamped_by_as_of": w["clamped"],
            "messages": len(recs), "new_chars": new_chars, "est_new_tokens": new_tokens,
            "prior_batches": i - 1, "prior_evidence_chars": prior_chars,
            # coarse estimate of new messages + accumulated prior evidence (NOT a token upper bound)
            "est_input_tokens_coarse": prior_tokens + new_tokens,
            "cumulative_evidence_chars_after": prior_chars + new_chars,
            "max_message_at": max(r["message_at"] for r in recs),
            "oversize": len(recs) >= OVERSIZE_MESSAGES or new_chars >= OVERSIZE_CHARS,
        })
        prior_chars += new_chars
        prior_tokens += new_tokens

    total_chars = sum(len(r["text"]) for r in kept)
    plan = {
        "tool": "history-bootstrap-v1", "user_id": user_id, "window": window,
        "as_of": iso_sh(as_of) if as_of is not None else None,
        "text_policy": TEXT_POLICY,
        "source_files": sources,
        "counts": {
            "conversations": len(convs), "nodes_total": nodes_total, "nodes_selected": nodes_selected,
            "nodes_not_selected": nodes_total - nodes_selected,
            "evidence_messages": len(kept), "user_messages": sum(1 for r in kept if r["role"] == "user"),
            "assistant_messages": sum(1 for r in kept if r["role"] == "assistant"),
            "skipped_reasoning": skipped["reasoning"], "skipped_system": skipped["system"],
            "skipped_tool": skipped["tool"], "skipped_empty": skipped["empty"],
            "skipped_other_role": skipped["other_role"],
            "attachments_gap": attachments, "undated": len(undated),
            "time_out_of_order": len(out_of_order), "deduped": deduped,
            "duplicate_conflicts": len(conflicts),
            "excluded_after_as_of": excluded_after_as_of,
        },
        "branch_issues": issues, "out_of_order": out_of_order,
        "duplicate_conflicts": conflicts,
        "total_chars": total_chars, "total_est_tokens": sum(approx_tokens(r["text"]) for r in kept),
        "batches": batch_plan, "batch_count": len(batch_plan),
        "oversized_batches": [b["batch_id"] for b in batch_plan if b["oversize"]],
        "undated_not_batched": [r["evidence_id"] for r in undated],
        "notes": [
            "selected branch only (current_node parent chain); non-selected nodes are NOT evidence",
            "reasoning (thoughts/reasoning_cap) and system/tool roles are excluded from conversational facts",
            "attachments are recorded as GAPS only (no fabricated content)",
            "text policy: " + TEXT_POLICY,
            "message time comes from message.create_time ONLY; conversation.update_time is never used",
            "undated messages are kept in evidence.jsonl but never placed into a timed batch",
            "user text is stored verbatim; this tool asserts no user facts",
            "est_input_tokens_coarse is a COARSE heuristic estimate (new messages + accumulated "
            "prior evidence); it is NOT a token upper bound and excludes the engine's own "
            "claim/summary material",
            "prior_evidence_chars / cumulative_evidence_chars_after expose the cumulative growth",
            "with --as-of, only messages strictly before the cutoff are kept; the last batch's now "
            "= min(calendar boundary, as_of) and no future batch is produced",
            "oversized_batches are REPORTED ONLY; nothing is split or truncated",
            "full-scale adaptation is NOT demonstrated; do not claim it",
        ],
        "status": "failed" if conflicts else "ok",
    }
    return {"plan": plan, "kept": kept, "dated": dated, "undated": undated,
            "windows": windows, "ordered": ordered, "convs": convs, "raw": conversations}


def build_plan(export_dir: Path, *, user_id: str, window: str = "month",
               as_of: float | None = None) -> dict:
    return collect(export_dir, user_id=user_id, window=window, as_of=as_of)["plan"]
