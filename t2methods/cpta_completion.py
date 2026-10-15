"""Explicit human admission after exploratory DG; not a source selection receipt."""
from pathlib import Path
from .checkpoint import file_hash
from .selection_protocol_v2 import candidate_settings

SCHEMA = 't2methods.cpta_b2n_cd.v1'
ADMISSION_SHA = '40ca911add19e96781ca4fb1c9f1c4f7f7c5c8c7fc3559ac2f6d6752fbe06882'
GRID = tuple((task, ds, seed) for task, datasets in (
    ('b2n', ('patternnet','rsicd','resisc45','mlrsnet')), ('cd', ('patternnet',)))
    for ds in datasets for seed in (1,2,3))
DATASETS = dict(patternnet='PatternNet',rsicd='RSICD',resisc45='NWPU-RESISC45',mlrsnet='MLRSNet')

def validate_completion(config):
    from .config import require,read_json
    r=config['runtime']
    require(r.get('experiment_protocol')==SCHEMA and r.get('admission_sha256')==ADMISSION_SHA,
            'CPTA completion requires the sealed exploratory-DG admission')
    path=Path(__file__).absolute().parents[1]/'assets/admissions/cpta_eta050.json'
    require(file_hash(path)==ADMISSION_SHA,'Admission evidence changed')
    admission=read_json(path)
    require(admission['target_informed_selection'] and admission['source_selection_receipt'] is False,
            'Do not claim source-only selection')
    require(config['method']==candidate_settings('CPTA',1),'Admitted CPTA eta=0.50 settings changed')
    require(config['data']['task'] in ('b2n','cd'),'Existing DG is reused, not retrained')
    require(r.get('require_source_selection') is False and r.get('require_winner') is False
            and not r.get('selection_receipt') and r.get('candidate')==1,'Legacy selection flow is forbidden here')
    require(not config.get('source_validation'),'No new source validation budget in completion')
