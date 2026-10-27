"""Variant-effect (DART-Eval caQTL/dsQTL) summary and stratified plots from vep_metrics.tsv and vep_strata.tsv."""

import os
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']

V = '/pollard/data/projects/shzhang/gpn/vep'
OUT = '/pollard/home/szhang20/arsenal-chrombpnet/figs/vep'
DATE = date.today().strftime('%Y%m%d')
DATASETS = (('african', 'African caQTLs (ATAC)'), ('yoruba', 'Yoruba dsQTLs (DNase)'))
# Score column, display label, color; Yoruba ChromBPNet uses our hg38 rescoring of the authors' runs.
MODELS = {
	'authors_chrombpnet': ('ChromBPNet (paper)', '#7f8c8d'),
	'authors_arsenal': ('ARSENAL (paper)', '#8e44ad'),
	'GM12878_phylop-chrombpnet': ('phyloP channel', '#2980b9'),
	'gpnstar_GM12878_25m_last6': ('GPN-Star m447 25m last6', '#c0392b'),
	'gpnstar_GM12878_25m_last6_masked': ('GPN-Star m447 25m last6, species masked', '#f1948a'),
	'gpnstar_GM12878_25m_last1': ('GPN-Star m447 25m last1', '#e6b0aa'),
	'gpnstar_GM12878_85m_last6': ('GPN-Star m447 85m last6', '#7b241c'),
	'gpnstar_GM12878_85m_last1': ('GPN-Star m447 85m last1', '#a93226'),
	'distill_GM12878_final_shared': ('GPN-Star distill final', '#82e0aa'),
	'distill_GM12878_final_m447': ('GPN-Star distill final m447', '#45b39d'),
	'distill_GM12878_last6_shared': ('GPN-Star distill last6', '#239b56'),
	'distill_GM12878_last9_shared': ('GPN-Star distill last9', '#0b5345'),
}
METRICS = (('pearson', 'Pearson r (significant variants)'), ('spearman', 'Spearman r (significant variants)'),
	('auroc', 'AUROC (significant vs control)'), ('auprc', 'AUPRC (significant vs control)'))
# Strata shown in the stratified plot, top to bottom; blank strings become gaps between groups.
STRATA = ('all', '', 'site phyloP > 2', 'site phyloP 0-2', 'site phyloP < 0', '',
	'cCRE: promoter (PLS)', 'cCRE: proximal enhancer (pELS)', 'cCRE: distal enhancer (dELS)',
	'cCRE: other accessible (CA-*)', 'cCRE: Low-DNase cCRE', 'cCRE: no cCRE', '',
	'summit < 100 bp', 'summit 100-250 bp', 'summit > 250 bp', '',
	'|effect| small (sig)', '|effect| medium (sig)', '|effect| large (sig)')
STRATA_MODELS = ('authors_chrombpnet', 'authors_arsenal', 'GM12878_phylop-chrombpnet', 'gpnstar_GM12878_25m_last6',
	'gpnstar_GM12878_25m_last6_masked', 'gpnstar_GM12878_85m_last6', 'distill_GM12878_final_shared', 'distill_GM12878_last9_shared')


def load_metrics():
	m = pd.read_csv(f'{V}/vep_metrics.tsv', sep='\t')
	# Yoruba: drop the authors' hg19 scores and keep our hg38 rescoring under the same name.
	m = m[~((m['dataset'] == 'yoruba') & (m['model'] == 'authors_chrombpnet'))]
	m['model'] = m['model'].str.replace('_hg38', '')
	return m[m['model'].isin(MODELS)]


def plot_summary(m):
	"""Dot plot per model and metric: small dots are runs, large dots their mean, dotted line ChromBPNet."""
	fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(18, 4.2 * len(DATASETS)), squeeze=False)
	for i, (ds, ds_name) in enumerate(DATASETS):
		d = m[m['dataset'] == ds]
		order = d.groupby('model')['pearson'].mean().sort_values(ascending=False).index
		for j, (metric, title) in enumerate(METRICS):
			ax = axes[i, j]
			for k, model in enumerate(order):
				v = d.loc[d['model'] == model, metric]
				ax.axhline(k, color='#eeeeee', lw=0.8, zorder=0)
				if len(v) > 1:
					ax.scatter(v, [k] * len(v), color=MODELS[model][1], s=14, alpha=0.6, lw=0, zorder=2)
				ax.scatter(v.mean(), k, color=MODELS[model][1], s=60, edgecolor='black', lw=0.6, zorder=3)
				ax.text(v.max(), k, f'   {v.mean():.3f}', va='center', fontsize=8)
			ax.axvline(d.loc[d['model'] == 'authors_chrombpnet', metric].mean(), color='black', ls=':', lw=1, zorder=1)
			pad = (d[metric].max() - d[metric].min()) or 0.01
			ax.set_xlim(d[metric].min() - 0.1 * pad, d[metric].max() + 0.35 * pad)
			ax.set_yticks(range(len(order)), [MODELS[x][0] for x in order] if j == 0 else [''] * len(order), fontsize=8)
			ax.set_ylim(len(order) - 0.4, -0.6)
			n = d.iloc[0]
			ax.set_title(f'{ds_name}\n{title}' if j == 0 else title, fontsize=9)
			ax.set_xlabel(metric if metric in ('auroc', 'auprc') else 'r', fontsize=8)
		axes[i, 0].annotate(f'{int(n.n_sig):,} significant / {int(n.n_ctrl):,} control', (0, -0.16), xycoords='axes fraction', fontsize=7, color='#555555')
	fig.suptitle('Variant effect prediction: GM12878-trained models, no-bias log2 fold change (alt vs ref)', y=1.0)
	fig.tight_layout()
	fig.savefig(f'{OUT}/{DATE}_vep_summary.png', dpi=150, bbox_inches='tight')
	plt.close(fig)


def plot_strata():
	"""Pearson r per stratum (rows) and model (colored markers), one panel per dataset."""
	s = pd.read_csv(f'{V}/vep_strata.tsv', sep='\t')
	fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 9), sharey=True)
	for ax, (ds, ds_name) in zip(axes, DATASETS):
		d = s[s['dataset'] == ds]
		for y, stratum in enumerate(STRATA):
			if not stratum:
				continue
			r = d[d['stratum'] == stratum]
			if r.empty:
				continue
			ax.axhline(y, color='#eeeeee', lw=0.8, zorder=0)
			for model in STRATA_MODELS:
				v = r.loc[r['model'] == model, 'pearson']
				if len(v):
					ax.scatter(v.iloc[0], y, color=MODELS[model][1], s=45, edgecolor='black', lw=0.4, zorder=3, label=MODELS[model][0])
			ax.text(1.0, y, f"  n={int(r['n_sig'].iloc[0]):,}", transform=ax.get_yaxis_transform(), va='center', fontsize=7, color='#555555')
		ax.set_title(ds_name, fontsize=10)
		ax.set_xlabel('Pearson r (significant variants)')
		ax.spines[['top', 'right']].set_visible(False)
	axes[0].set_yticks(range(len(STRATA)), STRATA, fontsize=8)
	axes[0].set_ylim(len(STRATA) - 0.5, -0.5)
	# One legend entry per model.
	h, l = axes[0].get_legend_handles_labels()
	seen = dict(zip(l, h))
	fig.legend(seen.values(), seen.keys(), loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
	fig.suptitle('Variant effect prediction by variant site (ChromBPNet and ARSENAL: mean of 3 runs)', y=1.0)
	fig.tight_layout()
	fig.savefig(f'{OUT}/{DATE}_vep_strata.png', dpi=150, bbox_inches='tight')
	plt.close(fig)


def main():
	os.makedirs(OUT, exist_ok=True)
	plot_summary(load_metrics())
	plot_strata()
	print(f'plots in {OUT}')


if __name__ == '__main__':
	main()
