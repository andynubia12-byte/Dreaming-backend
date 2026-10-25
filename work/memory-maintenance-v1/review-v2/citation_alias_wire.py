"""Summary-only reference alias wire; material and frozen validation stay unchanged."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
sys.path[:0] = [str(BASE / "repair-resolve-v26/src"), str(BASE / "subscription-sol")]

from driver_guards import input_metrics  # noqa: E402
from memmaint.engine import (  # noqa: E402
    Engine,
    _BRACKETED,
    _CITATION_SHAPE,
)
from memmaint.model import DeepSeekModel, ModelError, _append_log  # noqa: E402


PROTOCOL = """
引用传输协议（仅约束 summary wire 的引用写法）：
- 引用来源时只写 reference_aliases 中的完整短代号，并放在单独方括号中，例如 [R0001]。
- reference_aliases 的键是短代号，值是受信绑定的不透明 canonical ID。不要改写、截短、
  猜测或从输出反推 canonical ID；不同来源写成 [R0001] [R0002]，禁止合并进一个方括号。
- 短代号解码后会交给原 validator 查 canonical ID。原事实、不确定性、时间与人物对应约束
  继续有效；本协议不改变摘要内容或取舍。
""".strip()

_ALIAS_SHAPED = re.compile(
    r"(?<![A-Za-z0-9_.:\-#])[Rr][0-9]+(?![A-Za-z0-9_.:\-#])"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_source_path(obj: object) -> tuple[str, str]:
    path = Path(inspect.getsourcefile(obj)).resolve()
    try:
        relative = path.relative_to(PROJECT)
    except ValueError:
        relative = path
    return str(relative), _file_sha256(path)


_engine_source, _engine_hash = _relative_source_path(Engine)
_model_source, _model_hash = _relative_source_path(DeepSeekModel)

PROFILE = {
    "name": "citation-alias-wire-v1",
    "protocol_sha256": hashlib.sha256(PROTOCOL.encode()).hexdigest(),
    "summary_system": "unchanged frozen base prompt plus protocol suffix",
    "base_prompt_identical": False,
    "wire_payload": "shallow copy of original payload plus reference_aliases",
    "citation_id_source": "Engine._valid_source_ids(uid), filtered by material occurrence",
    "response_decoding": "standalone non-nested bound alias brackets only",
    "unbound_alias_policy": "reject unless an exact literal in current memory_material",
    "binding_identity": "Engine.model is the summarize receiver",
    "uid_normalization": "strip",
    "pre_dispatch_limit": "rough_input_tokens + max_output_tokens <= 900000",
    "engine_source": _engine_source,
    "engine_source_sha256": _engine_hash,
    "model_source": _model_source,
    "model_source_sha256": _model_hash,
    "citation_id_helper": "Engine._valid_source_ids",
}


@dataclass(frozen=True)
class _Binding:
    engine: Engine
    store: Any
    user_id: str
    batch_id: str


_BINDING: ContextVar[_Binding | None] = ContextVar("citation_alias_binding", default=None)


def _mapping_sha256(mapping: dict[str, str]) -> str:
    value = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def _contains_standalone_id(text: str, value: str) -> bool:
    start = 0
    while True:
        position = text.find(value, start)
        if position < 0:
            return False
        before = text[position - 1] if position else ""
        after_position = position + len(value)
        after = text[after_position] if after_position < len(text) else ""
        if not (before and _CITATION_SHAPE.match(before)) and not (
                after and _CITATION_SHAPE.match(after)):
            return True
        start = position + 1


def _reference_aliases(
    engine: Engine,
    uid: str,
    material: str,
    valid_ids: set[str] | None = None,
) -> dict[str, str]:
    """Bind only IDs proven by the current user's Store and present in material."""
    valid_ids = valid_ids if valid_ids is not None else engine._valid_source_ids(uid)
    canonical_ids = sorted(
        value for value in valid_ids
        if (isinstance(value, str) and value and _CITATION_SHAPE.fullmatch(value)
            and _contains_standalone_id(material, value))
    )
    material_folded = material.casefold()
    aliases: dict[str, str] = {}
    sequence = 1
    for canonical in canonical_ids:
        while True:
            if sequence > 9999:
                raise ModelError("summary wire exhausted deterministic four-digit alias namespace")
            alias = f"R{sequence:04d}"
            folded = alias.casefold()
            if (folded not in material_folded
                    and all(folded not in value.casefold() for value in valid_ids
                            if isinstance(value, str))):
                break
            sequence += 1
        aliases[alias] = canonical
        sequence += 1
    return aliases


def _bracket_layout(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    depths = [0] * (len(text) + 1)
    intervals: list[tuple[int, int]] = []
    stack: list[int] = []
    depth = 0
    for position, char in enumerate(text):
        depths[position] = depth
        if char == "[":
            depth += 1
            stack.append(position)
        elif char == "]" and depth:
            depth -= 1
            intervals.append((stack.pop(), position))
    depths[len(text)] = depth
    return depths, intervals


def _enclosing_bracket(
    intervals: list[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    containing = [
        interval for interval in intervals
        if interval[0] < start and end < interval[1]
    ]
    return min(containing, key=lambda interval: interval[0]) if containing else None


def _decode_output(
    output: str,
    aliases: dict[str, str],
    material: str,
    valid_ids: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Decode bound aliases and reject every non-material alias-shaped token."""
    brackets = list(_BRACKETED.finditer(output))
    depths, intervals = _bracket_layout(output)
    valid_ids = valid_ids or set()
    replacements: list[dict[str, str]] = []
    safe_spans: list[tuple[int, int]] = []
    decoded = output
    for match in reversed(brackets):
        body = match.group(1)
        if body not in aliases:
            continue
        if depths[match.start()] > 0:
            continue
        canonical = aliases[body]
        decoded = decoded[:match.start()] + f"[{canonical}]" + decoded[match.end():]
        replacements.append({"alias": body, "canonical_id": canonical})
        safe_spans.append(match.span())

    unresolved: list[str] = []
    invalid_positions: list[int] = []
    for match in _ALIAS_SHAPED.finditer(output):
        if any(start <= match.start() and match.end() <= end for start, end in safe_spans):
            continue
        token = match.group()
        containing = _enclosing_bracket(intervals, match.start(), match.end())
        if containing is not None:
            literal = output[containing[0]:containing[1] + 1]
            if literal in material:
                continue
            body = output[containing[0] + 1:containing[1]]
            if (body == token and token in valid_ids
                    and _contains_standalone_id(material, token)):
                continue
            unresolved.append(token)
            invalid_positions.append(match.start())
            continue
        if token in valid_ids and _contains_standalone_id(material, token):
            continue
        if token in material and _contains_standalone_id(material, token):
            continue
        unresolved.append(token)
        invalid_positions.append(match.start())

    audit = {
        "replacement_count": len(replacements),
        "replacements": list(reversed(replacements)),
        "unresolved_alias_tokens": sorted(set(unresolved)),
        "invalid_alias_positions": invalid_positions,
        "wire_failure": "unbound_or_illegal_alias_token" if invalid_positions else None,
    }
    if invalid_positions:
        error = ModelError(
            "summary has unbound or illegally positioned alias token(s): "
            + ", ".join(sorted(set(unresolved)))
        )
        error.audit = audit
        raise error
    return decoded, audit


def install() -> dict[str, Any]:
    """Install once in this process, chaining any existing run_batch wrapper."""
    if getattr(Engine.run_batch, "_citation_alias_wire_v1", False):
        return PROFILE

    original_run_batch = Engine.run_batch
    original_summarize = DeepSeekModel.summarize

    def run_batch(self: Engine, batch: dict) -> dict[str, Any]:
        raw_uid = (batch or {}).get("user_id") if isinstance(batch, dict) else None
        uid = raw_uid.strip() if isinstance(raw_uid, str) else raw_uid
        bid = (batch or {}).get("batch_id") if isinstance(batch, dict) else None
        binding = None
        if isinstance(uid, str) and uid and isinstance(bid, str) and bid:
            binding = _Binding(engine=self, store=self.store, user_id=uid, batch_id=bid)
        wire_batch = batch
        if isinstance(batch, dict) and uid != raw_uid:
            wire_batch = {**batch, "user_id": uid}
        token = _BINDING.set(binding)
        try:
            return original_run_batch(self, wire_batch)
        finally:
            _BINDING.reset(token)

    def summarize(self: DeepSeekModel, system: str, payload: dict[str, Any]) -> str:
        binding = _BINDING.get()
        if binding is None:
            raise ModelError("citation alias wire has no Engine.run_batch user/store binding")
        if binding.engine.store is not binding.store:
            raise ModelError("citation alias wire Store binding changed during run_batch")
        if binding.engine.model is not self:
            raise ModelError("citation alias wire model binding mismatch during run_batch")
        if not isinstance(binding.user_id, str) or not binding.user_id:
            raise ModelError("citation alias wire has an invalid user binding")
        if not isinstance(system, str) or not isinstance(payload, dict):
            raise ModelError("citation alias wire requires string system and object payload")
        if "reference_aliases" in payload:
            raise ModelError("summary payload already contains reserved reference_aliases")
        material = payload.get("memory_material")
        if not isinstance(material, str):
            raise ModelError("citation alias wire requires string memory_material")

        valid_ids = binding.engine._valid_source_ids(binding.user_id)
        aliases = _reference_aliases(
            binding.engine, binding.user_id, material, valid_ids
        )
        wire_payload = dict(payload)
        wire_payload["reference_aliases"] = aliases
        wire_system = system + "\n\n" + PROTOCOL
        metrics = input_metrics(wire_system, wire_payload)
        max_output_tokens = getattr(self, "max_output_tokens", None)
        if not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            raise ModelError("citation alias wire requires a positive max_output_tokens")

        log_path = Path(self.ledger_dir) / "citation-alias-wire.jsonl"
        metadata: dict[str, Any] = {
            "profile": PROFILE,
            "stage": "summarize",
            "user_id": binding.user_id,
            "batch_id": binding.batch_id,
            "material_sha256": hashlib.sha256(material.encode()).hexdigest(),
            "base_system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "wire_system_sha256": hashlib.sha256(wire_system.encode()).hexdigest(),
            "base_payload_sha256": hashlib.sha256(
                json.dumps(payload, ensure_ascii=False).encode()
            ).hexdigest(),
            "wire_payload_sha256": hashlib.sha256(
                json.dumps(wire_payload, ensure_ascii=False).encode()
            ).hexdigest(),
            "reference_aliases": aliases,
            "alias_mapping_sha256": _mapping_sha256(aliases),
            "bound_ids": sorted(aliases.values()),
            "metrics": metrics,
            "max_output_tokens": max_output_tokens,
            "rough_input_plus_output_tokens": metrics["rough_input_tokens"] + max_output_tokens,
        }
        if metrics["rough_input_tokens"] + max_output_tokens > 900000:
            _append_log(log_path, {**metadata, "ok": False, "error": "input_limit"})
            raise ModelError(
                "Complete summary wire input plus mapping exceeds safety margin; no truncation"
            )

        try:
            raw = original_summarize(self, wire_system, wire_payload)
        except BaseException as exc:
            _append_log(log_path, {
                **metadata,
                "ok": False,
                "dispatch_error": type(exc).__name__,
            })
            raise

        if not isinstance(raw, str):
            _append_log(log_path, {
                **metadata,
                "ok": False,
                "decode_error": "non_string_output",
            })
            raise ModelError("citation alias wire model output is not text")
        try:
            decoded, decode_audit = _decode_output(raw, aliases, material, valid_ids)
        except BaseException as exc:
            _append_log(log_path, {
                **metadata,
                "ok": False,
                "output_sha256_before_decode": hashlib.sha256(raw.encode()).hexdigest(),
                "decode_error": type(exc).__name__,
                "decode_audit": getattr(exc, "audit", None),
            })
            raise
        _append_log(log_path, {
            **metadata,
            "ok": True,
            "output_sha256_before_decode": hashlib.sha256(raw.encode()).hexdigest(),
            "output_sha256_after_decode": hashlib.sha256(decoded.encode()).hexdigest(),
            **decode_audit,
        })
        return decoded

    run_batch._citation_alias_wire_v1 = True
    summarize._citation_alias_wire_v1 = True
    Engine.run_batch = run_batch
    DeepSeekModel.summarize = summarize
    return PROFILE
