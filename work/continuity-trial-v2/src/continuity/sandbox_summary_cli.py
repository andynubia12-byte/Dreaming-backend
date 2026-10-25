"""Sandbox-only adapter around the admitted CLI, including its final HTTP wire.

No prompts, claim rules, citation rules or publisher implementation live here.
The adapter disables transport retries; the engine's bounded semantic repairs
still run, and every resulting HTTP dispatch is recorded after common redaction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuity.providers import Profile, create_client, redact_record


def install_responses_bridge(profile, wire_path):
    """Replace only the admitted model's lazy client factory in this subprocess.

    The frozen initializer supplies its original usage_file and cumulative caps.
    The shared client owns that same ledger; no second account or transport tap
    runs. Its recorder already contains the sanitized, actual provider request.
    """
    if not isinstance(profile, dict) or profile.get('api_format') != 'responses':
        raise ValueError('summary profile requires Responses API')
    selected = Profile(profile)
    from lettaproto import deepseek as legacy_client
    from memmaint.model import DeepSeekModel
    original_factory = legacy_client.DeepSeekClient
    original_init = DeepSeekModel.__init__

    def record(event):
        with wire_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def factory(config):
        config.chat_model = selected.model
        return create_client(config, profile, recorder=record, role='summary')

    def initialize(self, *args, **kwargs):
        kwargs['model'] = selected.model
        kwargs['max_retries'] = 0
        original_init(self, *args, **kwargs)

    legacy_client.DeepSeekClient = factory
    DeepSeekModel.__init__ = initialize

    def restore():
        legacy_client.DeepSeekClient = original_factory
        DeepSeekModel.__init__ = original_init

    # Read only the explicitly configured environment variable, never discover or
    # re-resolve subscription credentials. Provider exceptions are already safe.
    key = os.environ.get(selected.api_key_env) if selected.api_key_env else None
    return restore, [key] if key else []


def _artifact_paths(state, output):
    return [output, *state.glob('ledger/*.jsonl'), *state.glob('users/*/trace/*.json'),
            *state.glob('users/*/diagnostics.jsonl')]


def sanitize_artifacts(state, output, secrets, baseline):
    """The legacy engine may echo HTTP errors into diagnostic artifacts."""
    def errors(value):
        if isinstance(value, dict):
            return {k: ('error details omitted; see sanitized summary wire'
                        if k.endswith('error') and isinstance(v, str) else errors(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [errors(item) for item in value]
        return value

    for path in _artifact_paths(state, output):
        if not path.exists():
            continue
        stat = path.stat()
        previous = baseline.get(path)
        if previous == (stat.st_size, stat.st_mtime_ns):
            continue
        # Preserve existing log prefixes and untouched historical artifacts.
        offset = previous[0] if previous and path.suffix == '.jsonl' else 0
        with path.open('r+b') as stream:
            stream.seek(offset)
            raw = stream.read().decode('utf-8')
            lines = raw.splitlines() if path.suffix == '.jsonl' else [raw]
            safe = [json.dumps(redact_record(errors(json.loads(line)), secrets), ensure_ascii=False) for line in lines]
            stream.seek(offset)
            stream.write(('\n'.join(safe) + '\n').encode('utf-8'))
            stream.truncate()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--publish-user')
    parser.add_argument('--engine-cli', type=Path)
    args, rest = parser.parse_known_args(argv)
    if args.publish_user:
        cli = args.engine_cli
    else:
        metadata = json.loads((args.state_dir / 'sandbox-summary.json').read_text('utf-8'))
        cli = Path(metadata['engine_cli'])
    # Loading the admitted entry point sets exactly its normal dependency paths,
    # but does not execute __main__, instantiate a model, or read credentials.
    sys.path.insert(0, str(cli.parent))
    entry = runpy.run_path(str(cli), run_name='sandbox_admitted_cli')
    if args.publish_user:
        from memmaint.store import Store
        store = Store(args.state_dir)
        try:
            if not store.ensure_user(args.publish_user):
                store.publish(args.publish_user)
        finally:
            store.close()
        return 0
    entry['install']()  # admitted summary-only citation alias protocol
    if os.environ.get('CONTINUITY_OBSERVER_FD'):
        from realtime_observer import install
        install()
    # The existing runner writes each immutable batch beside its output.
    batch_path = Path(rest[rest.index('--batch') + 1])
    output = Path(rest[rest.index('--output') + 1])
    baseline = {path: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in _artifact_paths(args.state_dir, output) if path.exists()}
    restore, secrets = install_responses_bridge(metadata['profile'], batch_path.with_suffix('.wire.jsonl'))
    try:
        return entry['main'](['--state-dir', str(args.state_dir), *rest])
    finally:
        try:
            sanitize_artifacts(args.state_dir, output, secrets, baseline)
        finally:
            restore()


if __name__ == '__main__':
    raise SystemExit(main())
