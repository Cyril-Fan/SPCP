"""Plot new-class confusion differences and paired seed recall changes.

Input CSVs supply confusion counts and class-level recalls. Summary lanes show
a mean and a capless sample-SD interval; three ordered tile columns show seed
deltas. Every colour tile uses the same symmetric linear scale without clipping.
See ../README.md for the expected analysis-workspace layout.
"""
from pathlib import Path
import os
import json
import argparse
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 8.3, 'axes.labelsize': 8.3, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 3,
    'axes.linewidth': .45, 'lines.linewidth': .6,
    'xtick.major.width': .45, 'ytick.major.width': .45,
    'xtick.major.size': 2.3, 'ytick.major.size': 0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.unicode_minus': True, 'legend.frameon': False,
    'savefig.facecolor': 'white', 'text.color': '#30363D',
    'axes.labelcolor': '#30363D', 'xtick.color': '#30363D',
    'ytick.color': '#30363D',
})

BLUE, RED = '#4B6F9B', '#AC6656'


def panel_label(ax, label, title):
    offset = ScaledTranslation(0, 7 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, ha='left', va='bottom', weight='bold', fontsize=9,
            transform=ax.transAxes + offset, clip_on=False)
    title_offset = ScaledTranslation(11/72, 7/72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, title, ha='left', va='bottom', fontsize=9,
            transform=ax.transAxes + title_offset, clip_on=False)
    ax.set_gid(label)


def make_figure(cm, classes, out, preview=False):
    fig = plt.figure(figsize=(160 / 25.4, 178 / 25.4))
    grid = fig.add_gridspec(2, 2, left=34/160, right=145/160,
                           bottom=22/178, top=170/178,
                           width_ratios=[64, 39], wspace=8/51.5,
                           hspace=20/64)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(2)])
    cmap = plt.get_cmap('RdBu')
    norm = Normalize(vmin=-100, vmax=100, clip=False)
    tiles, rows, counts, labels = [], [], {}, []
    for row, dataset in enumerate(['NWPU-RESISC45', 'PatternNet']):
        block = cm[(cm.dataset == dataset) & (cm.split == 'new')]
        names = (block[['true_index', 'true_class']].drop_duplicates()
                 .sort_values('true_index').true_class.to_list())
        n = len(names)
        means = block.groupby(['variant', 'true_index', 'pred_index']).conditional_pct.mean()
        delta = means.loc['Full'].unstack().to_numpy() - means.loc['NoS'].unstack().to_numpy()
        assert delta.shape == (n, n) and np.isfinite(delta).all()
        assert np.allclose(delta.sum(1), 0, atol=1e-10)
        assert np.abs(delta).max() <= 100
        ax = axes[row, 0]
        panel_label(ax, 'ac'[row], dataset + ' (New)')
        mesh = ax.pcolormesh(np.arange(n+1)-.5, np.arange(n+1)-.5, delta,
                            cmap=cmap, norm=norm, shading='flat',
                            edgecolors='none', rasterized=False)
        for i in range(n):
            ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, fill=False,
                                  edgecolor='#A5ACB3', linewidth=.25))
        ax.set(xlim=(-.5, n-.5), ylim=(n-.5, -.5))
        xticks = np.arange(0, n, 2)
        ax.set_xticks(xticks, [str(i+1) for i in xticks])
        ax.set_yticks([])
        ax.set_xlabel('Predicted class index', labelpad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(Rectangle((-.5, -.5), n, n, fill=False,
                              edgecolor='#858D96', linewidth=.4, clip_on=False))
        # Separate index and name gutters give class names a common left edge.
        trans = blended_transform_factory(fig.transFigure, ax.transData)
        for i, name in enumerate(names):
            ax.text(4.5/160, i, str(i+1), transform=trans, ha='right', va='center',
                    fontsize=7.5, color='#7B838B', clip_on=False)
            text = ax.text(6.5/160, i, name[0].upper()+name[1:], transform=trans,
                          ha='left', va='center', fontsize=7.7,
                          color='#37414B', clip_on=False)
            labels.append(text)

        bx = axes[row, 1]
        panel_label(bx, 'bd'[row], 'Recall change')
        cb = classes[(classes.dataset == dataset) & (classes.split == 'new')]
        full = cb[cb.variant == 'Full'].pivot(index='class_index', columns='seed', values='recall').sort_index()
        nos = cb[cb.variant == 'NoS'].pivot(index='class_index', columns='seed', values='recall').sort_index()
        assert full.columns.to_list() == nos.columns.to_list() == [1, 2, 3]
        changes = full.to_numpy() - nos.to_numpy()
        assert changes.shape == (n, 3)
        assert np.allclose(changes.mean(1), np.diag(delta))
        assert np.isfinite(changes).all() and np.abs(changes).max() <= 100
        bx.set_axisbelow(True)
        bx.axvline(0, color='#8B939B', lw=.55, zorder=0)
        for x in [-100, -50, 50]:
            bx.axvline(x, color='#E8EBEE', lw=.35, zorder=0)
        for i, values in enumerate(changes):
            avg, sd = float(values.mean()), float(values.std(ddof=1))
            color = BLUE if avg >= 0 else RED
            # Only one summary point and one line per row. The raw replicates
            # are displayed separately, without jitter or a second glyph layer.
            bx.errorbar(avg, i, xerr=sd, fmt='o', color=color,
                        markersize=2.7, markeredgecolor='white', markeredgewidth=.25,
                        ecolor=color, elinewidth=.7, capsize=0, zorder=3)
            rows.append(dict(dataset=dataset, class_index=i, class_name=names[i],
                             seeds=[1,2,3], values=values.tolist(),
                             mean=avg, sample_sd=sd))
        bx.set(ylim=(n-.5, -.5), xlim=(-105,95))
        bx.set_xticks([-100, -50, 0, 50])
        bx.set_yticks([])
        bx.spines['left'].set_visible(False)
        bx.set_xlabel('Full − w/o S (pp)', labelpad=4)
        # An aligned seed matrix retains every raw replicate, yet each row of
        # the forest plot contains only the mean and uncertainty.
        box = bx.get_position()
        sx = fig.add_axes([148/160, box.y0, 10/160, box.height])
        sx.pcolormesh(np.arange(4)-.5, np.arange(n+1)-.5, changes,
                      cmap=cmap, norm=norm, shading='flat',
                      edgecolors='white', linewidth=.3, rasterized=False)
        sx.set(xlim=(-.5,2.5), ylim=(n-.5,-.5))
        sx.set_xticks([0,1,2], ['1','2','3'])
        sx.set_yticks([])
        sx.tick_params(axis='x', length=0, labelsize=7.7, pad=4)
        for spine in sx.spines.values():
            spine.set_visible(False)
        sx.text(.5,1,'Seeds',ha='center',va='bottom',fontsize=8,
                transform=sx.transAxes+ScaledTranslation(0,7/72,fig.dpi_scale_trans))
        tiles.append(sx)
        counts[dataset] = {'classes':n, 'raw_seed_deltas':int(changes.size),
                           'confusion_cells':int(delta.size), 'excluded':0,
                           'mean_sd':'three matched seeds; sample SD, ddof=1',
                           'raw_encoding':'three seed tile columns; same linear scale as confusion changes',
                           'colour_clip_count':int((np.abs(changes)>100).sum())}
    cax = fig.add_axes([34/160, 10/178, 64/160, 2.3/178])
    colorbar = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[-100,-50,0,50,100])
    # Matplotlib may otherwise rasterize a continuous colour bar at the
    # default figure DPI, even when the heatmaps themselves are vectors.
    colorbar.solids.set_rasterized(False)
    # Overlap same-colour vector edges to avoid PDF-viewer hairline seams
    # between adjacent colour-bar polygons. The normalization is unchanged.
    colorbar.solids.set_edgecolor('face')
    colorbar.solids.set_linewidth(.8)
    colorbar.set_label('Full − w/o S (pp)', fontsize=8.3, labelpad=3)
    colorbar.ax.tick_params(labelsize=8,pad=2,length=2)
    colorbar.outline.set_linewidth(.4)
    # Colour is signed change, not a universal improvement indicator:
    # positive off-diagonal confusion is worse, unlike positive recall.
    handles = [Line2D([],[],color=BLUE,marker='o',markersize=2.7,
                      linewidth=.7,label='Mean ± SD')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(126/160,11/178),
               fontsize=8.3, handlelength=1.8, handletextpad=.6)
    qa, figures = out/'qa', out/'figures'
    fig.canvas.draw()
    # Check the width of every actual class label, not a character estimate.
    renderer = fig.canvas.get_renderer()
    heat_left = axes[0,0].get_window_extent(renderer).x0
    label_right = max(t.get_window_extent(renderer).x1 for t in labels)
    gap_pt = (heat_left-label_right)*72/fig.dpi
    assert gap_pt >= 4, f'class name gutter too narrow: {gap_pt:.2f} pt'
    report = require_matplotlib_panel_alignment(fig, axes=axes.ravel().tolist(),
        panel_ids=['a','b','c','d'], exclude_axes=[cax]+tiles,
        json_out=qa/'Fig5_classwise_errors.alignment.json',
        overlay_svg=qa/'Fig5_classwise_errors.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    # Seed strips are sub-elements of b/d, not extra independent panels;
    # independently assert their row and physical-edge alignment.
    for bx,sx in zip(axes[:,1],tiles):
        assert abs(bx.get_position().y0-sx.get_position().y0)<1e-10
        assert abs(bx.get_position().height-sx.get_position().height)<1e-10
        assert bx.get_ylim()==sx.get_ylim()
    fig.savefig(figures/'Fig5_classwise_errors.pdf')
    fig.savefig(qa/'Fig5_classwise_errors.preview.png',dpi=300)
    if not preview:
        for ext in ['eps','svg']:
            fig.savefig(figures/f'Fig5_classwise_errors.{ext}')
        with plt.rc_context({'svg.fonttype':'path'}):
            fig.savefig(figures/'Fig5_classwise_errors.outlined.svg')
        png = figures/'Fig5_classwise_errors.png'
        fig.savefig(png,dpi=600)
        with Image.open(png) as image:
            image.convert('RGB').save(figures/'Fig5_classwise_errors.tiff',
                                      compression='raw',dpi=(600,600))
    plt.close(fig)
    (qa/'classwise_data_integrity.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    (qa/'Fig5_classwise_errors.plotted_values.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (qa/'classwise_label_spacing.json').write_text(json.dumps({'minimum_label_to_heatmap_gap_pt':gap_pt,'seed_strip_alignment':'PASS'},indent=2),encoding='utf-8')
    print('Fig5:',report['verdict'],'; all 41 classes and 123 paired seed deltas retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    make_figure(pd.read_csv(root/'data/confusion_counts.csv'),
                pd.read_csv(root/'data/class_seed_metrics.csv'),root,args.preview)
