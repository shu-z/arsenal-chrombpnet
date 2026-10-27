"""Test-set predicted vs true log counts for every finished run, plus a per-cell-type summary of count Pearson."""

import glob
import json
import os
import re
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Nimbus Sans is the installed metric clone of Helvetica.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']

G = '/pollard/data/projects/shzhang/gpn'
OUT = '/pollard/home/szhang20/arsenal-chrombpnet/figs/test_correlation'
# Every saved file starts with the run date, e.g. 20261014_.
DATE = date.today().strftime('%Y%m%d')
CELLS = ('K562', 'GM12878')
# SCREEN v4 K562 Core cCREs (DNase + H3K4me3 + H3K27ac + CTCF).
CCRE = f'{G}/screen_v4/K562_cCREs_ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed'
# A region overlapping several classes takes the first in this order; 'none' means no cCRE at all.
CCRE_CLASSES = (
	('PLS', 'promoter-like (PLS)'), ('pELS', 'proximal enhancer-like\n(pELS)'), ('dELS', 'distal enhancer-like\n(dELS)'),
	('CA', 'other accessible\n(CA-CTCF/TF/H3K4me3/only)'), ('Low-DNase', 'Low-DNase\n(cCRE inactive in K562)'), ('none', 'no cCRE'),
)
# Panels with fewer windows than this show the count instead of a correlation.
MIN_WINDOWS = 30
COLORS = {'gpnstar': '#c0392b', 'distill': '#27ae60', 'phylop': '#2980b9', 'chrombpnet': '#7f8c8d', 'arsenal': '#8e44ad'}
# Distilled runs get their own shade of green, light (final layer) to dark (most layers averaged).
DISTILL_GREENS = {'final_shared': '#82e0aa', 'final_m447': '#45b39d', 'last6_shared': '#239b56', 'last9_shared': '#0b5345'}


def color(label, family):
	if family == 'distill':
		return next((c for k, c in DISTILL_GREENS.items() if label.endswith(k)), COLORS['distill'])
	return COLORS[family]


def run_info(run_dir):
	"""(cell, label, family) for one of our runs, from its directory name; seed repeats share a label."""
	name = re.sub(r'_seed\d+$', '', os.path.basename(run_dir))
	cell = next((c for c in CELLS if c in name), 'K562')
	label = name.replace(f'{cell}_', '').replace(f'_{cell}', '').replace('gpnstar_', 'GPN-Star ')
	label = label.replace('phylop-chrombpnet', 'phyloP channel').replace('chrombpnet', 'ChromBPNet (ours)')
	family = 'distill' if name.startswith('distill_') else 'gpnstar' if 'gpnstar' in name else 'phylop' if 'phylop' in name else 'chrombpnet'
	if family == 'distill':
		label = label.replace('distill_', 'GPN-Star distill ')
	# Mammal (447-species) runs have no alignment tag in their names; add it.
	if family == 'gpnstar' and not re.search(r'p243|v100', name):
		label = label.replace('GPN-Star ', 'GPN-Star m447_')
	return cell, label, family


def our_runs():
	runs = []
	for path in sorted(glob.glob(f'{G}/chrombpnet_*/runs/*/fold_0/predict/test/regions.csv')):
		run_dir = path.split('/fold_0/')[0]
		cell, label, family = run_info(run_dir)
		seed = re.search(r'_seed(\d+)$', run_dir)
		runs.append((cell, label, family, path, label + (f' (seed {seed[1]})' if seed else '')))
	return runs


def metrics_row(path, cell, label, family, run=None):
	"""Count Pearson and profile median JSD / normalized JSD per region set, keyed like pearson_peaks."""
	m = json.load(open(path))
	row = {'cell': cell, 'label': label, 'family': family, 'run': run}
	for region, key in (('all', 'peaks_and_nonpeaks'), ('peaks', 'peaks'), ('nonpeaks', 'nonpeaks')):
		row[f'pearson_{region}'] = m['counts_metrics'][key]['pearsonr']
		row[f'jsd_{region}'] = m['profile_metrics'][key]['median_jsd']
		row[f'normjsd_{region}'] = m['profile_metrics'][key]['median_norm_jsd']
	return row


def summary_table(runs):
	rows = [metrics_row(p.replace('regions.csv', 'metrics_all_regions.json'), cell, label, family) for cell, label, family, p, _ in runs]
	for folder, label, family in (('ARSENAL_ChromBPNet_Models', 'ARSENAL (paper)', 'arsenal'), ('ChromBPNet_Models', 'ChromBPNet (paper)', 'chrombpnet')):
		for cell in CELLS:
			for p in sorted(glob.glob(f'{G}/arsenal_download/{folder}/{cell}/run_*/fold_0/predict/test/metrics_all_regions.json')):
				rows.append(metrics_row(p, cell, label, family, run=p.split('/run_')[1].split('/')[0]))
	return pd.DataFrame(rows)


def scatter(ax, df, title):
	"""Hexbin density of predicted vs true log counts, with peaks and non-peaks Pearson in the corner."""
	lo, hi = np.percentile(np.r_[df['true_count'], df['pred_count']], [0.1, 99.9])
	ax.hexbin(df['true_count'], df['pred_count'], gridsize=80, extent=(lo, hi, lo, hi), bins='log', cmap='viridis', mincnt=1)
	ax.plot([lo, hi], [lo, hi], color='white', lw=0.8, ls='--')
	r_all = pearsonr(df['true_count'], df['pred_count'])[0]
	r = {k: pearsonr(g['true_count'], g['pred_count'])[0] for k, g in df.groupby('is_peak')}
	ax.text(0.03, 0.97, f"all r={r_all:.3f}\npeaks r={r[1]:.3f}\nnon-peaks r={r[0]:.3f}", transform=ax.transAxes,
		va='top', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, lw=0))
	ax.set(title=title, xlabel='true log counts', ylabel='predicted log counts', xlim=(lo, hi), ylim=(lo, hi))


def plot_scatters(runs):
	"""One predicted-vs-true scatter per run, in runs/."""
	for cell, label in sorted({(r[0], r[4]) for r in runs}):
		path = next(r[3] for r in runs if r[0] == cell and r[4] == label)
		df = pd.read_csv(path, sep='\t', usecols=['is_peak', 'pred_count', 'true_count'])
		f1, a1 = plt.subplots(figsize=(4.5, 4.5))
		scatter(a1, df, f'{cell}: {label}')
		f1.tight_layout()
		f1.savefig(f"{OUT}/runs/{DATE}_{cell}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=150)
		plt.close(f1)


# Metric prefix, title, x-axis label, whether lower is better.
SUMMARIES = (
	('pearson', 'count Pearson', 'Pearson r', False),
	('jsd', 'profile median JSD (lower is better)', 'median JSD', True),
	('normjsd', 'profile median normalized JSD (higher is better)', 'median normalized JSD', False),
)


def plot_summary(table, metric, title, xlabel, lower_better):
	"""Dot plot per model: small dots are runs, large dots their mean, dotted line the ChromBPNet baseline."""
	# Single-layer runs go in plot_layers instead.
	table = table[~table['label'].str.contains('_layer')]
	fig, axes = plt.subplots(len(CELLS), 3, figsize=(15, 4.5 * len(CELLS)), squeeze=False)
	for i, cell in enumerate(CELLS):
		t = table[table['cell'] == cell]
		# Best model on all regions at the top.
		order = t.groupby('label')[f'{metric}_all'].mean().sort_values(ascending=lower_better).index
		# Baseline is our 3 ChromBPNet seeds where they exist, else the authors' 3 runs.
		base_label = 'ChromBPNet (ours)' if (t['label'] == 'ChromBPNet (ours)').sum() >= 3 else 'ChromBPNet (paper)'
		for j, region in enumerate(('all', 'peaks', 'nonpeaks')):
			ax = axes[i, j]
			col = f'{metric}_{region}'
			means = t.groupby('label')[col].mean().reindex(order)
			fam = t.groupby('label')['family'].first().reindex(order)
			vals = t[col]
			pad = (vals.max() - vals.min()) or 0.01
			for k, label in enumerate(order):
				v = t.loc[t['label'] == label, col]
				ax.axhline(k, color='#eeeeee', lw=0.8, zorder=0)
				if len(v) > 1:
					ax.scatter(v, [k] * len(v), color=color(label, fam[label]), s=14, alpha=0.6, lw=0, zorder=2)
				ax.scatter(means[label], k, color=color(label, fam[label]), s=60, edgecolor='black', lw=0.6, zorder=3)
				ax.text(max(means[label], v.max()), k, f'   {means[label]:.3f}', va='center', fontsize=8)
			base = t.loc[t['label'] == base_label, col].mean()
			ax.axvline(base, color='black', ls=':', lw=1, zorder=1)
			ax.set_yticks(range(len(order)), order if j == 0 else [''] * len(order), fontsize=8)
			ax.set_xlim(vals.min() - 0.1 * pad, vals.max() + 0.3 * pad)
			ax.set_ylim(len(order) - 0.4, -0.6)
			ax.set_title(f'{cell}: {title.split(" (")[0]}, {region}')
			ax.set_xlabel(xlabel)
	fig.suptitle(title, y=1.0)
	fig.tight_layout()
	fig.savefig(f'{OUT}/{DATE}_summary_{metric}.png', dpi=150, bbox_inches='tight')
	plt.close(fig)


def plot_layers(table, cell='K562'):
	"""Test metrics for each single 85m block (block 12 alone is last1), with last-N averages and baselines as lines."""
	t = table[table['cell'] == cell]
	mean = lambda label, col: t.loc[t['label'] == label, col].mean()
	layers = sorted(int(re.search(r'_layer(\d+)', l).group(1)) for l in t['label'].unique() if '85m_layer' in l)
	labels = {L: f'GPN-Star m447_85m_layer{L}' for L in layers}
	labels[12] = 'GPN-Star m447_85m_last1'
	xs = sorted(labels)
	refs = (
		('85m last6 (avg of 7-12)', 'GPN-Star m447_85m_last6', '#e67e22', '--'),
		('85m last9 (avg of 4-12)', 'GPN-Star m447_85m_last9', 'black', '--'),
		('ARSENAL (paper)', 'ARSENAL (paper)', COLORS['arsenal'], ':'),
		('ChromBPNet (ours)', 'ChromBPNet (ours)', COLORS['chrombpnet'], ':'),
	)
	rows = (('pearson', 'count Pearson r'), ('normjsd', 'profile median normalized JSD'))
	fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)), squeeze=False)
	for i, (metric, ylabel) in enumerate(rows):
		for j, region in enumerate(('all', 'peaks', 'nonpeaks')):
			ax, col = axes[i, j], f'{metric}_{region}'
			ys = [mean(labels[L], col) for L in xs]
			ax.plot(xs, ys, color=COLORS['gpnstar'], marker='o', lw=1.5, label='85m single block')
			for L, y in zip(xs, ys):
				ax.annotate(f'{y:.3f}', (L, y), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7)
			for name, label, color, ls in refs:
				if (t['label'] == label).any():
					ax.axhline(mean(label, col), color=color, ls=ls, lw=1, label=name)
			ax.set_xticks(range(2, 13, 2))
			ax.set_xlabel('85m block (12 = last layer)')
			ax.set_ylabel(ylabel)
			ax.set_title(f'{cell}: {region}')
			ax.spines[['top', 'right']].set_visible(False)
	# Legend under the grid so it never covers points.
	h, l = axes[0, 0].get_legend_handles_labels()
	fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
	fig.suptitle(f'{cell} test set: GPN-Star m447 85m, one block at a time (single runs)', y=1.0)
	fig.tight_layout()
	fig.savefig(f'{OUT}/{DATE}_layers_85m.png', dpi=150, bbox_inches='tight')
	plt.close(fig)


def ccre_labels(regions):
	"""SCREEN class per region from its chr/start/end, by CCRE_CLASSES priority, else 'none'."""
	c = pd.read_csv(CCRE, sep='\t', header=None, usecols=[0, 1, 2, 9], names=['chr', 'start', 'end', 'cls'])
	c.loc[c['cls'].str.startswith('CA'), 'cls'] = 'CA'
	label = np.full(len(regions), 'none', dtype=object)
	for cls, _ in reversed(CCRE_CLASSES[:-1]):
		for chrom, g in c[c['cls'] == cls].groupby('chr'):
			g = g.sort_values('start')
			starts, ends = g['start'].values, g['end'].values
			m = (regions['chr'] == chrom).values
			r = regions.loc[m]
			# cCREs of one class do not overlap, so ends are sorted too: the first end past the region start decides.
			i = np.searchsorted(ends, r['start'].values, side='right')
			hit = (i < len(starts)) & (starts[np.minimum(i, len(starts) - 1)] < r['end'].values)
			idx = np.where(m)[0][hit]
			label[idx] = cls
	return label


def plot_ccre(runs, cell='K562'):
	"""Count Pearson per SCREEN class (rows) on test peaks and non-peaks (columns)."""
	paths = [(label, family, p) for c, label, family, p, _ in runs if c == cell and '_layer' not in label]
	for folder, label, family in (('ARSENAL_ChromBPNet_Models', 'ARSENAL (paper)', 'arsenal'), ('ChromBPNet_Models', 'ChromBPNet (paper)', 'chrombpnet')):
		paths += [(label, family, p) for p in sorted(glob.glob(f'{G}/arsenal_download/{folder}/{cell}/run_*/fold_0/predict/test/regions.csv'))]
	key = ['chr', 'start', 'end', 'is_peak']
	ref = pd.read_csv(paths[0][2], sep='\t')
	ref['ccre'] = ccre_labels(ref)
	ref[key + ['true_count', 'ccre']].to_csv(f'{G}/region_analysis/{cell.lower()}_test_ccre.tsv', sep='\t', index=False)
	sets = {'peaks': ref['is_peak'].values == 1, 'nonpeaks': ref['is_peak'].values == 0}
	rows = []
	for label, family, p in paths:
		d = pd.read_csv(p, sep='\t')
		# Every run lists the test regions in the same order.
		assert (d[key].values == ref[key].values).all()
		for cls, _ in CCRE_CLASSES:
			for region, rm in sets.items():
				m = rm & (ref['ccre'].values == cls)
				r = pearsonr(d.loc[m, 'pred_count'], d.loc[m, 'true_count'])[0] if m.sum() >= MIN_WINDOWS else np.nan
				rows.append({'label': label, 'family': family, 'ccre': cls, 'regions': region, 'n': m.sum(), 'pearson': r})
	t = pd.DataFrame(rows)
	t.to_csv(f'{OUT}/{DATE}_{cell}_ccre_pearson.tsv', sep='\t', index=False)
	fig, axes = plt.subplots(len(CCRE_CLASSES), len(sets), figsize=(11.5, 4.1 * len(CCRE_CLASSES)), squeeze=False)
	for i, (cls, name) in enumerate(CCRE_CLASSES):
		c = t[t['ccre'] == cls]
		# One model order per row, best on peaks at the top.
		order = c[c['regions'] == 'peaks'].groupby('label')['pearson'].mean().sort_values(ascending=False).index
		fam = c.groupby('label')['family'].first()
		for j, region in enumerate(sets):
			ax, s = axes[i, j], c[c['regions'] == region]
			n = int(s['n'].iloc[0])
			ax.set_title(f'{region} (n = {n:,})', fontsize=10)
			ax.set_yticks(range(len(order)), order if j == 0 else [''] * len(order), fontsize=8)
			ax.set_ylim(len(order) - 0.4, -0.6)
			# Too few windows for a correlation: leave the panel empty.
			if n < MIN_WINDOWS:
				ax.set_xticks([])
				continue
			for k, label in enumerate(order):
				v = s.loc[s['label'] == label, 'pearson']
				ax.axhline(k, color='#eeeeee', lw=0.8, zorder=0)
				if len(v) > 1:
					ax.scatter(v, [k] * len(v), color=color(label, fam[label]), s=14, alpha=0.6, lw=0, zorder=2)
				ax.scatter(v.mean(), k, color=color(label, fam[label]), s=60, edgecolor='black', lw=0.6, zorder=3)
				ax.text(v.max(), k, f'   {v.mean():.3f}', va='center', fontsize=8)
			ax.axvline(s.loc[s['label'] == 'ChromBPNet (ours)', 'pearson'].mean(), color='black', ls=':', lw=1, zorder=1)
			pad = (s['pearson'].max() - s['pearson'].min()) or 0.01
			ax.set_xlim(s['pearson'].min() - 0.1 * pad, s['pearson'].max() + 0.3 * pad)
			ax.set_xlabel('Pearson r')
		axes[i, 0].set_ylabel(name, fontsize=12, fontweight='bold', labelpad=12)
	fig.suptitle(f'{cell} test set: count Pearson by SCREEN v4 cCRE class', y=1.0)
	fig.tight_layout()
	fig.savefig(f'{OUT}/{DATE}_{cell}_ccre_pearson.png', dpi=150, bbox_inches='tight')
	plt.close(fig)
	return ref


def main():
	os.makedirs(f'{OUT}/runs', exist_ok=True)
	runs = our_runs()
	print(f'{len(runs)} runs with test predictions:', *[f'  {r[0]}: {r[4]}' for r in runs], sep='\n')
	table = summary_table(runs)
	table.to_csv(f'{OUT}/{DATE}_summary_metrics.tsv', sep='\t', index=False)
	for metric, title, xlabel, lower_better in SUMMARIES:
		plot_summary(table, metric, title, xlabel, lower_better)
	plot_layers(table)
	ref = plot_ccre(runs)
	print('K562 test windows by cCRE class (peak, class):', ref.groupby(['is_peak', 'ccre']).size().to_dict())
	plot_scatters(runs)
	print(f'plots in {OUT}')


if __name__ == '__main__':
	main()
