"""Figure 1: errors per minute by version, with the y-axis cut at 130.

Participant means come from the same loader as modality_lmm.py. Stars come from
the FDR-corrected contrasts in modality_lmm_summary.csv, so run that script first.

Output: figure1_ylim130.png
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
from common import RESULTS  # noqa: E402
from load_tables import load_flower  # noqa: E402

OUT = 'mistakes_per_min'
ORDER = ['paper', 'tablet', 'mouse']
Y_CAP = 130
SUMMARY = RESULTS / 'modality_lmm_summary.csv'
OUTFILE = RESULTS / f'figure1_ylim{Y_CAP}.png'
# (contrast name, x1, x2, bracket height as a share of Y_CAP)
BRACKETS = [('Tablet_vs_Paper', 0, 1, 1.02), ('Mouse_vs_Tablet', 1, 2, 1.10), ('Mouse_vs_Paper', 0, 2, 1.18)]


def participant_means(flower):
    """Participant mean errors per minute in each modality."""
    grouped = flower.groupby(['sub_id', 'modality'], observed=True)[OUT]
    return grouped.mean().dropna().reset_index()


def to_stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def load_stars(path):
    """Significance stars for each errors per minute contrast."""
    table = pd.read_csv(path)
    table = table[table['outcome'] == OUT].set_index('contrast')
    return table['p_fdr'].map(to_stars)


def draw_bracket(ax, x1, x2, y, label):
    """Bracket drawn above the axis so it never covers data."""
    h = Y_CAP * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='black', clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom',
            fontsize=12, fontweight='bold', clip_on=False)


def draw_boxes(ax, subj):
    kw = dict(data=subj, x='modality', y=OUT, order=ORDER, hue='modality',
              palette='tab10', legend=False, ax=ax)
    sns.boxplot(showfliers=False, **kw)
    sns.stripplot(alpha=0.4, size=3, **kw)
    ax.set_ylim(0, Y_CAP)
    ax.set_ylabel('Errors per minute')
    ax.set_xlabel('')


def main():
    subj = participant_means(load_flower())
    stars = load_stars(SUMMARY)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    draw_boxes(ax, subj)
    for name, x1, x2, share in BRACKETS:
        draw_bracket(ax, x1, x2, Y_CAP * share, stars[name])
    fig.savefig(OUTFILE, dpi=300, bbox_inches='tight')
    print(subj.groupby('modality', observed=True)[OUT].agg(['count', 'median', 'max']).round(1))
    print('means above cap:', int((subj[OUT] > Y_CAP).sum()), '| file:', OUTFILE)


if __name__ == '__main__':
    main()
