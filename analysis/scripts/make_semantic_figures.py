"""Plot semantic distortion from seed-level CSV metrics.

Mean bars and capless sample-SD intervals have separate raw-seed lanes.
The input CSV supplies the metric values; this script does not compute model
features or establish the provenance of those inputs. See ../README.md
for the expected analysis-workspace layout and output locations.
"""
from pathlib import Path
import os
import json
import argparse
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.transforms import ScaledTranslation
from PIL import Image
import pandas as pd
import numpy as np
from audit_panel_alignment import (require_matplotlib_panel_alignment, audit_layout_manifest,
                                   write_json_report, write_overlay_svg, exit_code)

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 3, 'svg.fonttype': 'none',
    'axes.linewidth': .55, 'axes.edgecolor': '#404040',
    'axes.spines.top': True, 'axes.spines.right': True,
    'xtick.major.size': 2.3, 'ytick.major.size': 2.3,
    'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.pad': 3.7, 'ytick.major.pad': 3.5,
    'text.color': '#242424', 'axes.labelcolor': '#242424',
    'xtick.color': '#242424', 'ytick.color': '#242424',
    'legend.frameon': False, 'savefig.facecolor': 'white',
})

DATASETS = ['PatternNet', 'RSICD', 'NWPU-RESISC45', 'MLRSNet']
LABELS = ['PatternNet', 'RSICD', 'NWPU45', 'MLRSNet']
STYLES = [('Full', -.24, '#C7D9EE', '#4B6F9B'), ('NoS', .24, '#EDC5BA', '#AC6656')]


def plot(df, metric, out, preview=False):
    length = metric == 'length_error'
    name = 'Fig4_length_distortion' if length else 'Fig3_direction_distortion'
    fig, axes = plt.subplots(1, 2, figsize=(160/25.4, 64/25.4), sharey=True)
    fig.subplots_adjust(left=15/160, right=158/160, bottom=12/64, top=55/64, wspace=13/65)
    rows, captions = [], []
    for j, (ax, split) in enumerate(zip(axes, ['base', 'new'])):
        ax.set_gid('ab'[j])
        label = ax.text(.5, -9.1/43, '(a) Base classes' if j == 0 else '(b) New classes',
                        transform=ax.transAxes, ha='center', va='center', fontsize=9.5,
                        fontweight='bold', clip_on=False)
        captions.append(label)
        for i, ds in enumerate(DATASETS):
            for variant, shift, fill, edge in STYLES:
                a = df[(df.dataset == ds) & (df.split == split) & (df.variant == variant)].sort_values('seed')
                assert a.seed.tolist() == [1, 2, 3]
                v = a[metric].to_numpy()
                mean, sd = float(v.mean()), float(v.std(ddof=1))
                assert np.isfinite(v).all()
                baseline = .001 if length else 0.
                if length:
                    assert (v > baseline).all() and mean-sd > baseline
                # Separate mean/SD and raw-seed lanes in x only. All y values
                # remain exact; an SD is never enlarged to meet a visual minimum.
                mean_x = i + shift - .10
                seed_x = i + shift + np.array([.06, .145, .23])
                ax.bar(mean_x, mean-baseline, bottom=baseline, width=.18,
                       color=fill, edgecolor=edge, linewidth=.5, zorder=2)
                ax.errorbar(mean_x, mean, yerr=sd, fmt='none', ecolor=edge,
                            elinewidth=.85, capsize=0, zorder=4)
                ax.plot(seed_x, v, linestyle='none', marker='o', markersize=2.1,
                        markerfacecolor='white', markeredgecolor=edge,
                        markeredgewidth=.60, zorder=5)
                rows.append(dict(dataset=ds, split=split, variant=variant, metric=metric,
                                 seeds=[1,2,3], values=v.tolist(), mean=mean, sample_sd=sd))
        ax.set_xlim(-.58, 3.60)
        ax.set_xticks(range(4), LABELS)
        ax.tick_params(axis='y', labelleft=True)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E2E2E2', linewidth=.35, linestyle=(0,(2,2)))
        if length:
            ax.set_yscale('log')
            ax.set_ylim(.001, 2)
            ax.yaxis.set_major_locator(FixedLocator([.001, .01, .1, 1]))
            ax.yaxis.set_major_formatter(FixedFormatter(['0.001', '0.01', '0.1', '1']))
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(0, .70)
            ax.yaxis.set_major_locator(FixedLocator([0,.2,.4,.6]))
            ax.yaxis.set_major_formatter(FixedFormatter(['0.0','0.2','0.4','0.6']))
    axes[0].set_ylabel('Length distortion (log scale)' if length else 'Direction distortion')
    axes[0].yaxis.set_label_coords(3.6/160,33.5/64,transform=fig.transFigure)
    handles=[Patch(facecolor=s[2],edgecolor=s[3],linewidth=.55,label='Full' if s[0]=='Full' else 'w/o S') for s in STYLES]

    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.997),ncol=2,
               handlelength=1.4,handleheight=.75,handletextpad=.55,columnspacing=2.1)
    fig.canvas.draw()
    # The generic Matplotlib label detector only recognizes top-left letters.
    # Measure the actual visible bottom-centered captions and audit them explicitly.
    report=require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=['a','b'],
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True)
    layout=report['layout']
    for panel,caption in zip(layout['panels'],captions):
        display=caption.get_transform().transform(caption.get_position())
        anchor=fig.dpi_scale_trans.inverted().transform(display)*72
        panel['panel_label_anchor_pt']=anchor.tolist()
        panel['panel_label']=caption.get_text()
    report=audit_layout_manifest(layout,tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True)
    assert exit_code(report,strict=True)==0, report
    write_json_report(report,out/'qa'/f'{name}.alignment.json')
    write_overlay_svg(report,out/'qa'/f'{name}.alignment.svg')
    (out/'qa'/f'{name}.plotted_values.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fig.savefig(out/'figures'/f'{name}.pdf',facecolor='white')
    fig.savefig(out/'qa'/f'{name}.preview.png',dpi=300,facecolor='white')
    if not preview:
        for ext in ['svg','eps']: fig.savefig(out/'figures'/f'{name}.{ext}',facecolor='white')
        with mpl.rc_context({'svg.fonttype':'path'}):
            fig.savefig(out/'figures'/f'{name}.outlined.svg',facecolor='white')
        fig.savefig(out/'figures'/f'{name}.png',dpi=600,facecolor='white')
        with Image.open(out/'figures'/f'{name}.png') as im:
            im.convert('RGB').save(out/'figures'/f'{name}.tiff',compression='raw',dpi=(600,600))
    plt.close(fig)
    print(name,report['verdict'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preview',action='store_true')
    args=p.parse_args()
    out=Path(__file__).resolve().parent.parent
    for folder in ['figures','qa']: (out/folder).mkdir(exist_ok=True)
    df=pd.read_csv(out/'data/semantic_seed_metrics.csv')
    assert len(df)==48 and set(df.edge_set)=={'local_k3'}
    for metric in ['direction_error','length_error']: plot(df,metric,out,args.preview)

if __name__=='__main__': main()
