"""Verify full-method source identities and B2N/CD/DG configs without a model."""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from t2methods.config import load_config,validate_config

def main():
    identity=json.loads((ROOT/'docs/source_identity.json').read_text(encoding='utf-8'))
    for r in identity['files']:
        if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Original source bytes changed: '+r['path'])
    configs=list((ROOT/'configs').glob('*.yaml'))
    checks=[validate_config(load_config(p,s)) for p in configs for s in [1,2,3]]
    print(json.dumps({'pass':True,'unchanged_source_files':len(identity['files']),'full_configs':len(configs),
                      'config_seed_checks':len(checks),'torch_loaded':'torch' in sys.modules,
                      'real_training_executed':False},indent=2))

if __name__=='__main__':main()
