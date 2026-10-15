"""Paper/internal summaries: dataset-first HM and seed-first target averages."""
import math
import statistics

B2N_DATASETS = ('PatternNet', 'RSICD', 'NWPU-RESISC45', 'MLRSNet')


def hm(base, new):
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in (base, new)):
        raise ValueError('Accuracy must be finite percentages')
    return 2 * base * new / (base + new) if base + new else 0.0


def summarize_seed_scores(scores):
    if set(scores) != {1, 2, 3} or not all(math.isfinite(v) and 0 <= v <= 100 for v in scores.values()):
        raise ValueError('Need exactly seeds 1/2/3 and finite scores')
    return dict(mean=statistics.mean(scores.values()), sample_sd=statistics.stdev(scores.values()),
                ddof=1, n=3, per_seed=scores)


def average_hm(rows):
    mapping = {(r['dataset'], r['seed']): hm(r['base'], r['new']) for r in rows}
    if len(rows) != 12 or set(mapping) != {(d,s) for d in B2N_DATASETS for s in (1,2,3)}:
        raise ValueError('Need four datasets x three seeds exactly')
    return summarize_seed_scores({s:statistics.mean(mapping[d,s] for d in B2N_DATASETS) for s in (1,2,3)})


def target_average(rows, domains):
    mapping = {(r['domain'], r['seed']): r['accuracy'] for r in rows}
    if len(domains) != 3 or len(set(domains)) != 3 or len(rows) != 9 or set(mapping) != {(d,s) for d in domains for s in (1,2,3)}:
        raise ValueError('Need exactly three target domains x seeds 1/2/3; source excluded')
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in mapping.values()):
        raise ValueError('Invalid target accuracy')
    return summarize_seed_scores({s:statistics.mean(mapping[d,s] for d in domains) for s in (1,2,3)})
