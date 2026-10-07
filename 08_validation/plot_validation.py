"""Draw Figure 7 panels from a completed four-class classification run."""
from pathlib import Path
import argparse
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

def draw_feature_space(results, out):
    """Render saved t-SNE coordinates without changing the fitted representation."""
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((results / 'summary.json').read_text('utf-8'))
    if info['train_schema'] != '4class':
        raise ValueError('Panel d requires a completed four-class run.')
    proj = pd.read_csv(results / 'tsne_projection.csv')
    split = pd.read_csv(results / 'splits.csv')
    valid = np.load(results / 'valid_mask.npy', allow_pickle=False)
    record_indices = proj.record_index.to_numpy(dtype=int)
    labels = np.array([0, 1, 0, 2, 3])[split.class_idx.to_numpy()]
    if (not valid.all() or len(valid) != len(split)
            or not proj.record_index.is_unique
            or (record_indices < 0).any() or (record_indices >= len(split)).any()
            or not np.isfinite(proj[['tsne_x', 'tsne_y']].to_numpy()).all()
            or not np.array_equal(labels[record_indices], proj.class_index.to_numpy())):
        raise ValueError('The projection does not match the classification records.')
    stats = json.loads((results / 'feature_statistics.json').read_text('utf-8'))
    if len(proj) != stats['tsne_subsample']:
        raise ValueError('The projection sample count does not match its metadata.')
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for c, (label, color) in enumerate(zip(LABELS4, COLORS4)):
        v = proj[proj.class_index == c]
        ax.scatter(v.tsne_x, v.tsne_y, s=6, alpha=.65, color=color,
                   label=label, edgecolors='none')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=.035, y=.04)
    # Keep the key outside the point cloud so that no records are obscured.
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0,
              frameon=False, fontsize=10, markerscale=2.2, handletextpad=.45,
              labelspacing=1.05)
    fig.subplots_adjust(left=.08, right=.70, bottom=.12, top=.97)
    save(fig, out, 'd')

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
    draw_feature_space(results, out)
    y=np.load(results/'test_labels.npy',allow_pickle=False)
    p=np.load(results/'proba_Ensemble.npy',allow_pickle=False).argmax(axis=1)
    prec,rec,f1,_=precision_recall_fscore_support(y,p,labels=range(4),zero_division=0)
    scores=np.column_stack([prec,rec,f1]); scores=np.vstack([scores,scores.mean(axis=0)])*100
    fig,ax=plt.subplots(figsize=(8.0,3.7));ax.set_xlim(0,4);ax.set_ylim(0,6);ax.axis('off')
    edges=[0,1.4,2.2666667,3.1333333,4]
    for j,label in enumerate(['Class','Precision (%)','Recall (%)','F1-score (%)']):
        ax.add_patch(Rectangle((edges[j],5),edges[j+1]-edges[j],1,facecolor='white',edgecolor='#bdbdbd',lw=.6))
        ax.text((edges[j]+edges[j+1])/2,5.5,label,ha='center',va='center',fontsize=10)
    for i, lab in enumerate(LABELS4+['Overall (macro)']):
        yy=4-i
        ax.add_patch(Rectangle((0,yy),1.4,1,facecolor='white',edgecolor='#bdbdbd',lw=.6))
        if i<4:ax.scatter(.11,yy+.5,s=70,color=COLORS4[i])
        ax.text(.22 if i<4 else .7,yy+.5,lab,ha='left' if i<4 else 'center',va='center',fontsize=9.5,fontweight='bold' if i==4 else 'normal')
        for j,val in enumerate(scores[i]):
            ax.add_patch(Rectangle((edges[j+1],yy),edges[j+2]-edges[j+1],1,facecolor=plt.cm.Blues(val/100),edgecolor='#bdbdbd',lw=.6))
            ax.text((edges[j+1]+edges[j+2])/2,yy+.5,f'{val:.3f}',ha='center',va='center',color='white' if val>65 else 'black',fontweight='bold' if i==4 else 'normal',fontsize=12)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,100),cmap='Blues'),ax=ax,fraction=.025,pad=.02,shrink=.7)
    bar.ax.set_title('Score\n(%)',fontsize=10);bar.set_ticks([0,50,100]);bar.outline.set_edgecolor('#bdbdbd')
    fig.subplots_adjust(left=.01,right=.94,top=.98,bottom=.02);save(fig,out,'e')
    report={'total':len(split),'five_class_counts':dict(zip(KEYS5,counts5)),
            'four_class_counts':dict(zip(LABELS4,counts4)),'split':dict(zip(phases,totals)),
            'four_class_split':dict(zip(LABELS4,[dict(zip(phases,v)) for v in per_class])),
            'split_shares':dict(zip(phases,shares.tolist())),
            'split_policy':info.get('split_policy','record'),
            'per_class_scores_percent':dict(zip(LABELS4+['Overall (macro)'],scores.tolist()))}
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
        draw_feature_space(args.results, args.out)
    else:
        draw(args.results,args.out)
