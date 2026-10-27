"""Per-stratum count accuracy of each K562 model on the test regions, and where GPN-Star gains over ChromBPNet."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

G = '/pollard/data/projects/shzhang/gpn'
MODELS = {
	'gpnstar_last6': f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6',
	'gpnstar_last1': f'{G}/chrombpnet_layers/runs/gpnstar_25m_last1',
	'phylop': f'{G}/chrombpnet_phylop/runs/K562_phylop-chrombpnet',
	'chrombpnet': f'{G}/chrombpnet_phylop/runs/K562_chrombpnet',
}
KEY = ['chr', 'start', 'end', 'is_peak']


def load():
	df = pd.read_csv(f'{G}/region_analysis/k562_test_features.tsv', sep='\t')
	for name, run in MODELS.items():
		p = pd.read_csv(f'{run}/fold_0/predict/test/regions.csv', sep='\t')
		assert (p[KEY].values == df[KEY].values).all() and np.allclose(p['true_count'], df['true_count'])
		df[name] = p['pred_count'].values
	return df


def strata(df):
	"""Named boolean masks over the regions."""
	s = {'all': df.index == df.index, 'peaks': df['is_peak'] == 1, 'nonpeaks': df['is_peak'] == 0}
	for col, label in (('phylop_mean', 'phyloP'), ('align_depth', 'align depth'), ('gc', 'GC'), ('true_count', 'true count')):
		q = pd.qcut(df[col], 4, labels=['Q1 low', 'Q2', 'Q3', 'Q4 high'])
		for k in q.cat.categories:
			s[f'{label} {k}'] = q == k
	s['TSS <1kb'] = df['tss_dist'] < 1_000
	s['TSS 1-10kb'] = df['tss_dist'].between(1_000, 10_000)
	s['TSS >10kb'] = df['tss_dist'] > 10_000
	s['no repeat at center'] = df['repeat_class'] == 'none'
	for cls in ('SINE', 'LINE', 'LTR', 'DNA', 'Simple_repeat', 'Low_complexity'):
		s[f'repeat: {cls}'] = df['repeat_class'] == cls
	return s


def summarize(df, mask):
	d = df[mask]
	row = {'n': len(d), 'frac_peaks': d['is_peak'].mean()}
	for m in MODELS:
		row[f'r_{m}'] = pearsonr(d[m], d['true_count'])[0] if len(d) > 50 else np.nan
		row[f'mae_{m}'] = (d[m] - d['true_count']).abs().mean()
	row['bias_gpn'] = (d['gpnstar_last6'] - d['true_count']).mean()
	row['bias_cbp'] = (d['chrombpnet'] - d['true_count']).mean()
	# Positive gain means GPN-Star's absolute error is smaller.
	gain = (d['chrombpnet'] - d['true_count']).abs() - (d['gpnstar_last6'] - d['true_count']).abs()
	row['gain_mae'] = gain.mean()
	row['frac_gpn_better'] = (gain > 0).mean()
	return row


def main():
	df = load()
	rows = {name: summarize(df, m) for name, m in strata(df).items() if m.sum() > 50}
	t = pd.DataFrame(rows).T
	t.to_csv(f'{G}/region_analysis/k562_strata.tsv', sep='\t')
	pd.set_option('display.width', 250)
	cols = ['n', 'frac_peaks', 'r_chrombpnet', 'r_phylop', 'r_gpnstar_last1', 'r_gpnstar_last6',
		'mae_chrombpnet', 'mae_gpnstar_last6', 'gain_mae', 'frac_gpn_better', 'bias_cbp', 'bias_gpn']
	print(t[cols].astype(float).round(3).to_string())
	# Worst-predicted regions for every model: largest absolute error under the best model.
	df['err_gpn'] = df['gpnstar_last6'] - df['true_count']
	df['err_cbp'] = df['chrombpnet'] - df['true_count']
	df.to_csv(f'{G}/region_analysis/k562_test_predictions.tsv', sep='\t', index=False)


if __name__ == '__main__':
	main()
