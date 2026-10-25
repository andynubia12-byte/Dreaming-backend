#!/usr/bin/env python3
"""Verify the delivered source/document manifest without opening runtime data or calling APIs."""
from pathlib import Path
import ast
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'PACKAGE-MANIFEST.json').read_text('utf-8'))
errors = []
for item in manifest['files']:
    path = root / item['path']
    if not path.is_file() or path.is_symlink():
        errors.append(f"Missing or linked file: {item['path']}")
        continue
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != item['sha256']:
        errors.append(f"Changed since packaging: {item['path']}")
    try:
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8'), filename=item['path'])
        elif path.suffix == '.json':
            json.loads(data)
    except (ValueError, SyntaxError, UnicodeDecodeError) as exc:
        errors.append(f"Parse error: {item['path']}: {exc}")
print(json.dumps({'ok': not errors, 'checked_files': len(manifest['files']), 'errors': errors}, ensure_ascii=False, indent=2))
sys.exit(1 if errors else 0)
