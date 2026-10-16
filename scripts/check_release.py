"""Verify the release inventory and B2N/CD/DG configs without loading a model."""
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from spcp.config import load_config, validate_config

def main():
    identity = json.loads((ROOT / 'docs/source_identity.json').read_text(encoding='utf-8'))
    expected = {r['path']: r['sha256'] for r in identity['files']}
    actual = {p.relative_to(ROOT).as_posix() for folder in ('spcp', 'scripts', 'tests', 'configs')
              for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml')}
    if set(expected) != actual:
        raise ValueError('Release source inventory differs')
    for name, digest in expected.items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Release source SHA-256 differs: ' + name)
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=name)
    configs = list((ROOT / 'configs').glob('*.yaml'))
    checks = [validate_config(load_config(p, s)) for p in configs for s in (1, 2, 3)]
    print(json.dumps({'pass': True, 'verified_release_files': len(expected), 'full_configs': len(configs),
                      'config_seed_checks': len(checks), 'torch_loaded': 'torch' in sys.modules,
                      'real_training_executed': False}, indent=2))

if __name__=='__main__':main()
