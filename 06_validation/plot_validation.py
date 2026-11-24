"""Draw Figure 7 panels from a completed four-class classification run."""
from pathlib import Path
import argparse
import subprocess
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.ticker import LogFormatterMathtext
from sklearn.metrics import precision_recall_fscore_support

COLORS5 = ['#007C7C', '#1957AD', '#FF7900', '#70389F', '#818181']
COLORS4 = ['#FFC000', COLORS5[1], COLORS5[3], COLORS5[4]]
KEYS5 = ['microseismic', 'blasting', 'small_landslide', 'mechanical_mining', 'transport_vehicles']
LABELS4 = ['Natural slope events', 'Blasting', 'Mechanical mining', 'Transport vehicles']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})

def save(fig, out, panel):
    fig.savefig(out / f'fig7-{panel}.pdf', bbox_inches='tight', pad_inches=.035)
    fig.savefig(out / f'fig7-{panel}.png', dpi=600 if panel == 'd' else 180, bbox_inches='tight', pad_inches=.035)
    if panel == 'd':
        fig.savefig(out / f'fig7-{panel}.svg', bbox_inches='tight', pad_inches=.035)
    plt.close(fig)


def draw(results, out):
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((results/'summary.json').read_text('utf-8'))
    if info['train_schema'] != '4class':
        raise ValueError('Use a completed four-class run.')
    split = pd.read_csv(results/'splits.csv')
    valid = np.load(results/'valid_mask.npy', allow_pickle=False)
    if not valid.all():
        raise ValueError('Resolve invalid records before plotting the archive counts.')
    counts5 = [int((split.class_key == k).sum()) for k in KEYS5]
    counts4 = [counts5[0]+counts5[2], counts5[1], counts5[3], counts5[4]]
    for panel, counts, colors, names in [
        ('a', counts5, COLORS5, ['Micro-\nseismic', 'Blasting', 'Small\nlandslide', 'Mechanical\nmining', 'Transport\nvehicles']),
        ('b', counts4, COLORS4, ['Natural\nslope', 'Blasting', 'Mechanical\nmining', 'Transport\nvehicles'])]:
        fig, ax = plt.subplots(figsize=(4.3, 3.1))
        ax.bar(range(len(counts)), np.asarray(counts)-1, bottom=1, color=colors, width=.58, zorder=3)
        ax.set_yscale('log'); ax.set_ylim(1, 18000)
        ax.set_yticks([1, 10, 100, 1000, 10000]); ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_xticks(range(len(counts)), names); ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color='#c3c3c3', linestyle='--', linewidth=.65, zorder=0)
        for i, n in enumerate(counts):
            ax.text(i, n*1.12, f'{n:,}', ha='center', va='bottom', fontsize=11)
        fig.tight_layout(pad=.25); save(fig, out, panel)
    split['class4'] = np.array([0, 1, 0, 2, 3])[split.class_idx.to_numpy()]
    phases = ['train', 'val', 'test']
    totals = [int((split['split']==s).sum()) for s in phases]
    per_class = [[int(((split.class4 == c) & (split['split'] == s)).sum()) for s in phases] for c in range(4)]
    fig, ax = plt.subplots(figsize=(7.8, 3.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(.005, .98, f'{len(split):,} records in total', va='top')
    left = 0
    shares = np.asarray(totals, dtype=float) / len(split)
    for width, n, lab, color in zip(shares, totals, ['train','validation','test'], [COLORS5[0],COLORS5[1],COLORS5[3]]):
        ax.add_patch(Rectangle((left,.58),width,.3,facecolor=color,edgecolor='#bdbdbd',linewidth=.7))
        ax.text(left+width/2,.73,f'{n:,}\n{lab}',ha='center',va='center',color='white',fontsize=10)
        ax.text(left+width/2,.55,f'{width:.2%}',ha='center',va='top',color=color,fontsize=10)
        left += width
    ax.text(.005,.33,'Per-class counts (train / val / test)',va='bottom',fontsize=10)
    for c, (lab, color, ns) in enumerate(zip(['Natural slope','Blasting','Mechanical','Transport'],COLORS4,per_class)):
        x=c/4
        ax.add_patch(Rectangle((x,.03),.25,.27,facecolor='white',edgecolor='#bdbdbd',linewidth=.6))
        ax.scatter([x+.022],[.225],s=40,color=color)
        ax.text(x+.043,.225,lab,ha='left',va='center',fontsize=10)
        ax.text(x+.125,.11,' / '.join(map(str,ns)),ha='center',va='center',fontsize=9.5)
    save(fig,out,'c')
    panels = Path(__file__).resolve().parents[1] / 'figures' / 'figure07'
    for name in ('plot_fig7d_feature_space.py', 'plot_fig7e_baseline_sanity.py'):
        subprocess.run([sys.executable, str(panels / name), '--results', str(results), '--out', str(out)], check=True)
    printed = json.loads((out / 'fig7e_values.json').read_text('utf8'))

    report={'total':len(split),'five_class_counts':dict(zip(KEYS5,counts5)),
            'four_class_counts':dict(zip(LABELS4,counts4)),'split':dict(zip(phases,totals)),
            'four_class_split':dict(zip(LABELS4,[dict(zip(phases,v)) for v in per_class])),
            'split_shares':dict(zip(phases,shares.tolist())),
            'split_policy':info.get('split_policy','record'),
            'per_class_scores_percent':printed['per_class_scores_percent']}
    (out/'figure7_values.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--panel', choices=['all', 'd'], default='all',
                        help='Render all panels or only the saved feature-space projection.')
    args=parser.parse_args()
    if args.panel == 'd':
        plot = Path(__file__).resolve().parents[1] / 'figures' / 'figure07' / 'plot_fig7d_feature_space.py'
        subprocess.run([sys.executable, str(plot), '--results', str(args.results), '--out', str(args.out)], check=True)
    else:
        draw(args.results,args.out)
