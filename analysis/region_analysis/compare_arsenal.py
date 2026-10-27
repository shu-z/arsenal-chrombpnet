"""Where GPN-Star 25m last6 predicts K562 test counts better than ARSENAL, by region stratum and feature."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

G = '/pollard/data/projects/shzhang/gpn'
TEST = 'fold_0/predict/test/regions.csv'
OUT = f'{G}/region_analysis'
KEY = ['chr', 'start', 'end', 'is_peak']


def load():
	"""Features plus count predictions: GPN-Star, ChromBPNet (ours) and each ARSENAL run with their mean."""
	df = pd.read_csv(f'{OUT}/k562_test_features.tsv', sep='\t')
	runs = {
		'gpn': f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6/{TEST}',
		'cbp': f'{G}/chrombpnet_phylop/runs/K562_chrombpnet/{TEST}',
		**{f'ars{i}': f'{G}/arsenal_download/ARSENAL_ChromBPNet_Models/K562/run_{i}/{TEST}' for i in (1, 2, 3)},
	}
	for name, path in runs.items():
		p = pd.read_csv(path, sep='\t')
		# The authors' regions come in the same order as ours.
		assert (p[KEY].values == df[KEY].values).all() and np.allclose(p['true_count'], df['true_count'], atol=1e-4)
		df[name] = p['pred_count'].values
	# Mean prediction is used only for bias; errors and r use single runs, since an ensemble beats any one run.
	df['ars'] = df[['ars1', 'ars2', 'ars3']].mean(axis=1)
	for m in ('gpn', 'cbp', 'ars1', 'ars2', 'ars3'):
		df[f'abserr_{m}'] = (df[m] - df['true_count']).abs()
	df['abserr_ars'] = df[['abserr_ars1', 'abserr_ars2', 'abserr_ars3']].mean(axis=1)
	# Positive gain means GPN-Star's log-count error is smaller than a typical single ARSENAL run's.
	df['gain'] = df['abserr_ars'] - df['abserr_gpn']
	# Seed-to-seed null: the same comparison between two ARSENAL runs.
	df['gain_null'] = df['abserr_ars1'] - df['abserr_ars2']
	return df


def strata(df):
	"""Named boolean masks over the regions."""
	s = {'all': df.index == df.index, 'peaks': df['is_peak'] == 1, 'nonpeaks': df['is_peak'] == 0}
	for col, label in (('true_count', 'true count'), ('align_depth', 'align depth'), ('phylop_mean', 'phyloP mean'), ('phylop_frac_gt2', 'frac phyloP>2'), ('gc', 'GC')):
		q = pd.qcut(df[col].rank(method='first'), 4, labels=['Q1 low', 'Q2', 'Q3', 'Q4 high'])
		for k in q.cat.categories:
			s[f'{label} {k}'] = (q == k).values
	s['phyloP <90% aligned'] = df['phylop_frac_aligned'] < 0.9
	s['TSS <1kb'] = df['tss_dist'] < 1_000
	s['TSS 1-10kb'] = df['tss_dist'].between(1_000, 10_000)
	s['TSS >10kb'] = df['tss_dist'] > 10_000
	s['no repeat at center'] = df['repeat_class'] == 'none'
	for cls in ('SINE', 'LINE', 'LTR', 'DNA', 'Simple_repeat', 'Low_complexity'):
		s[f'repeat: {cls}'] = df['repeat_class'] == cls
	return s


def summarize(d):
	r = lambda m: pearsonr(d[m], d['true_count'])[0]
	return {
		'n': len(d), 'frac_peaks': d['is_peak'].mean(),
		'r_gpn': r('gpn'), 'r_ars': np.mean([r(f'ars{i}') for i in (1, 2, 3)]), 'r_cbp': r('cbp'),
		'r_gpn_minus_ars': r('gpn') - np.mean([r(f'ars{i}') for i in (1, 2, 3)]),
		# Run-to-run spread of ARSENAL's r, to judge whether a difference is real.
		'r_ars_seed_sd': np.std([r(f'ars{i}') for i in (1, 2, 3)]),
		'mean_gain': d['gain'].mean(), 'frac_gpn_better': (d['gain'] > 0).mean(),
		'bias_gpn': (d['gpn'] - d['true_count']).mean(), 'bias_ars': (d['ars'] - d['true_count']).mean(),
	}


def main():
	df = load()
	t = pd.DataFrame({k: summarize(df[m]) for k, m in strata(df).items() if m.sum() > 50}).T.astype(float)
	t.to_csv(f'{OUT}/k562_gpn_vs_arsenal_strata.tsv', sep='\t')
	pd.set_option('display.width', 250)
	print(t.round(3).to_string())
	# Regions where GPN-Star beats ARSENAL by more than any ARSENAL seed pair usually differs.
	thr = df['gain_null'].abs().quantile(0.95)
	big = df['gain'] > thr
	print(f'\nseed-noise threshold (95th pct |gain| between ARSENAL runs 1 and 2): {thr:.3f} log counts')
	print(f'GPN-Star better beyond it: {big.sum()} regions ({big.mean():.1%}); ARSENAL better beyond it: {(df["gain"] < -thr).sum()} ({(df["gain"] < -thr).mean():.1%})')
	cols = ['is_peak', 'true_count', 'align_depth', 'phylop_frac_aligned', 'phylop_mean', 'phylop_frac_gt2', 'gc', 'tss_dist', 'repeat_frac']
	comp = pd.DataFrame({
		'GPN-Star much better': df.loc[big, cols].median(numeric_only=True),
		'ARSENAL much better': df.loc[df['gain'] < -thr, cols].median(numeric_only=True),
		'all regions': df[cols].median(numeric_only=True),
	})
	comp.loc['is_peak (mean)'] = [df.loc[big, 'is_peak'].mean(), df.loc[df['gain'] < -thr, 'is_peak'].mean(), df['is_peak'].mean()]
	comp.loc['ARSENAL over-predicts (frac)'] = [(df.loc[big, 'ars'] > df.loc[big, 'true_count']).mean(), (df.loc[df['gain'] < -thr, 'ars'] > df.loc[df['gain'] < -thr, 'true_count']).mean(), (df['ars'] > df['true_count']).mean()]
	print('\nmedian features:\n' + comp.round(3).to_string())
	df.to_csv(f'{OUT}/k562_gpn_vs_arsenal_regions.tsv', sep='\t', index=False)


if __name__ == '__main__':
	main()
