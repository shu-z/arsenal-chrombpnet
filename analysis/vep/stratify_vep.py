"""DART-Eval variant metrics per model within strata of conservation, alignment, effect size, peak position and repeats."""

import gzip
import sys

import numpy as np
import pandas as pd
import pyBigWig

sys.path.insert(0, '/pollard/home/szhang20/arsenal-chrombpnet/analysis/vep')
from eval_vep import G, african_truth, yoruba_truth, score_files, score_keys, metrics

PHYLOP = '/pollard/data/projects/shzhang/alus/data/annotations/hg38.phyloP447way.bw'
RMSK = '/pollard/data/projects/shzhang/alus/data/repeats/rmsk.txt.gz'
PEAKS = f'{G}/arsenal_download/ARSENAL_ChromBPNet_Models/GM12878/filtered.peaks.bed'
# SCREEN v4 GM12878 Core cCREs (DNase + H3K4me3 + H3K27ac + CTCF).
CCRE = f'{G}/screen_v4/GM12878_cCREs_ENCFF428XFI_ENCFF280PUF_ENCFF469WVA_ENCFF644EEX.bed'
# Variant-site cCRE classes, with every CA-* class pooled as other accessible.
CCRE_STRATA = (('PLS', 'promoter (PLS)'), ('pELS', 'proximal enhancer (pELS)'), ('dELS', 'distal enhancer (dELS)'),
	('CA', 'other accessible (CA-*)'), ('Low-DNase', 'Low-DNase cCRE'), ('none', 'no cCRE'))
POSITIONS = {
	'african': f'{G}/arsenal_download/african_caqtl_chrombpnet_schema.tsv',
	'yoruba': f'{G}/vep/data/yoruban_dsqtl_hg38_chrombpnet_schema.tsv',
}
# Score columns compared; ChromBPNet and ARSENAL metrics are averaged over their 3 runs.
MODELS = ('gpnstar_GM12878_25m_last6', 'gpnstar_GM12878_25m_last1', 'gpnstar_GM12878_85m_last6', 'gpnstar_GM12878_85m_last1', 'gpnstar_GM12878_25m_last6_masked', 'GM12878_phylop-chrombpnet', 'authors_chrombpnet', 'authors_arsenal',
	'distill_GM12878_final_shared', 'distill_GM12878_final_m447', 'distill_GM12878_last6_shared', 'distill_GM12878_last9_shared')
OUT = f'{G}/vep/vep_strata.tsv'


def positions(dataset):
	"""Variant key -> hg38 chr and 1-based pos."""
	d = pd.read_csv(POSITIONS[dataset], sep='\t', header=None, usecols=range(5), names=['chr', 'pos', 'allele1', 'allele2', 'variant_id'])
	d['key'] = score_keys(d, dataset)
	return d.drop_duplicates('key').set_index('key')[['chr', 'pos']]


def site_features(pos):
	"""phyloP at the site and over +-10 bp (unaligned bases, stored as 0, are missing), distance to peak summit, repeat class."""
	bw = pyBigWig.open(PHYLOP)
	site, win, aligned = [], [], []
	for c, p in zip(pos['chr'], pos['pos']):
		v = np.array(bw.values(c, p - 11, p + 10), dtype=float)
		v[v == 0] = np.nan
		site.append(v[10])
		win.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
		aligned.append(np.isfinite(v).mean())
	f = pd.DataFrame({'phylop_site': site, 'phylop_win': win, 'frac_aligned_win': aligned}, index=pos.index)
	peaks = pd.read_csv(PEAKS, sep='\t', header=None)
	summits = {c: np.sort((g[1] + g[9]).values) for c, g in peaks.groupby(0)}
	dist = []
	for c, p in zip(pos['chr'], pos['pos']):
		s = summits.get(c)
		if s is None:
			dist.append(np.inf)
			continue
		i = np.searchsorted(s, p - 1)
		dist.append(min(abs(s[j] - (p - 1)) for j in (i - 1, i) if 0 <= j < len(s)))
	f['summit_dist'] = dist
	f['repeat_class'] = repeat_class(pos)
	f['ccre'] = ccre_class(pos)
	return f


def ccre_class(pos):
	"""GM12878 cCRE class covering each variant (CA-* pooled as CA), or none."""
	c = pd.read_csv(CCRE, sep='\t', header=None, usecols=[0, 1, 2, 9], names=['chr', 'start', 'end', 'cls'])
	c.loc[c['cls'].str.startswith('CA'), 'cls'] = 'CA'
	out = pd.Series('none', index=pos.index)
	for chrom, g in pos.groupby('chr'):
		cc = c[c['chr'] == chrom].sort_values('start')
		starts, ends, cls = cc['start'].values, cc['end'].values, cc['cls'].values
		# cCREs do not overlap each other, so the last one starting at or before the site is the only candidate.
		i = np.searchsorted(starts, g['pos'].values - 1, side='right') - 1
		hit = (i >= 0) & (ends[np.clip(i, 0, None)] > g['pos'].values - 1)
		out.loc[g.index[hit]] = cls[i[hit]]
	return out


def repeat_class(pos):
	"""RepeatMasker class covering each variant, or none."""
	cols = [5, 6, 7, 11]
	r = pd.read_csv(RMSK, sep='\t', header=None, usecols=cols, names=['chr', 'start', 'end', 'cls'])
	out = pd.Series('none', index=pos.index)
	for c, g in pos.groupby('chr'):
		rc = r[r['chr'] == c].sort_values('start')
		starts, ends, cls = rc['start'].values, rc['end'].values, rc['cls'].values
		i = np.searchsorted(starts, g['pos'].values - 1, side='right') - 1
		# Nearest repeat starting at or before the site; good enough for non-overlapping RepeatMasker calls.
		hit = (i >= 0) & (ends[np.clip(i, 0, None)] > g['pos'].values - 1)
		out.loc[g.index[hit]] = cls[i[hit]]
	return out


def strata(f, t):
	"""Named masks over a dataset's variants; effect-size strata apply to significant variants only."""
	s = {'all': np.ones(len(f), bool)}
	s['site unaligned'] = f['phylop_site'].isna().values
	s['site aligned'] = f['phylop_site'].notna().values
	s['site phyloP > 2'] = (f['phylop_site'] > 2).values
	s['site phyloP 0-2'] = f['phylop_site'].between(0, 2).values
	s['site phyloP < 0'] = (f['phylop_site'] < 0).values
	s['+-10bp <50% aligned'] = (f['frac_aligned_win'] < 0.5).values
	s['summit < 100 bp'] = (f['summit_dist'] < 100).values
	s['summit 100-250 bp'] = f['summit_dist'].between(100, 250).values
	s['summit > 250 bp'] = (f['summit_dist'] > 250).values
	for cls, name in CCRE_STRATA:
		s[f'cCRE: {name}'] = (f['ccre'] == cls).values
	s['cCRE: enhancer (pELS + dELS)'] = f['ccre'].isin(['pELS', 'dELS']).values
	s['no repeat'] = (f['repeat_class'] == 'none').values
	for cls in ('SINE', 'LINE', 'LTR'):
		s[f'repeat: {cls}'] = (f['repeat_class'] == cls).values
	# Effect-size tertiles among significant variants; controls are kept so AUROC stays defined.
	q = pd.qcut(t.loc[t['sig'], 'effect'].abs(), 3, labels=['small', 'medium', 'large'])
	for k in q.cat.categories:
		m = ~t['sig'].values.copy()
		m[np.where(t['sig'].values)[0][(q == k).values]] = True
		s[f'|effect| {k} (sig)'] = m
	return s


def main():
	truth = {'african': african_truth(), 'yoruba': yoruba_truth()}
	scored = {}
	for model, run, dataset, path in score_files():
		if model in MODELS or model == 'authors_chrombpnet_hg38':
			s = pd.read_csv(path, sep='\t')
			scored[model, run, dataset] = pd.Series(s['logfc'].values, index=score_keys(s, dataset))
	rows = []
	for dataset, t in truth.items():
		runs = {k: v for k, v in scored.items() if k[2] == dataset}
		common = set(t['key'])
		for v in runs.values():
			common &= set(v.index)
		pos = positions(dataset)
		common &= set(pos.index)
		t = t[t['key'].isin(common)].reset_index(drop=True)
		f = site_features(pos.loc[t['key']])
		for name, mask in strata(f, t).items():
			sub = t[mask]
			if sub['sig'].sum() < 30 or (~sub['sig']).sum() < 30:
				continue
			for (model, run, _), v in runs.items():
				# Yoruba ChromBPNet uses our hg38 rescoring of the authors' runs, matching the hg38 GPN-Star scores.
				if dataset == 'yoruba' and model == 'authors_chrombpnet':
					continue
				m = metrics(sub.assign(logfc=v.loc[sub['key']].values))
				rows.append({'dataset': dataset, 'stratum': name, 'model': model.replace('_hg38', ''), 'run': run, **m})
	df = pd.DataFrame(rows)
	summ = df.groupby(['dataset', 'stratum', 'model'], sort=False)[['n_sig', 'n_ctrl', 'pearson', 'auroc', 'auprc']].mean().reset_index()
	summ.to_csv(OUT, sep='\t', index=False)
	wide = summ.pivot_table(index=['dataset', 'stratum'], columns='model', values=['pearson', 'auroc'], sort=False)
	short = {'gpnstar_GM12878_25m_last6': 'gpn6', 'gpnstar_GM12878_25m_last1': 'gpn1', 'gpnstar_GM12878_85m_last6': 'g85_6', 'gpnstar_GM12878_85m_last1': 'g85_1', 'gpnstar_GM12878_25m_last6_masked': 'gpn6m', 'GM12878_phylop-chrombpnet': 'phyloP', 'authors_chrombpnet': 'cbp', 'authors_arsenal': 'ars', 'distill_GM12878_final_shared': 'dfin', 'distill_GM12878_final_m447': 'dm447', 'distill_GM12878_last6_shared': 'd6', 'distill_GM12878_last9_shared': 'd9'}
	wide.columns = [f'{m}_{short[c]}' for m, c in wide.columns]
	n = summ.drop_duplicates(['dataset', 'stratum']).set_index(['dataset', 'stratum'])[['n_sig', 'n_ctrl']]
	pd.set_option('display.width', 250)
	print(n.join(wide)[['n_sig', 'pearson_cbp', 'pearson_ars', 'pearson_phyloP', 'pearson_gpn6', 'pearson_gpn6m', 'pearson_g85_6', 'pearson_d9', 'pearson_dfin', 'auroc_cbp', 'auroc_phyloP', 'auroc_gpn6']].round(3).to_string())


if __name__ == '__main__':
	main()
