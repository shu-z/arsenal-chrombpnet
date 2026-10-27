"""Per-region features (conservation, alignment depth, GC, repeats, TSS distance) for the K562 test regions."""

import gzip
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx

sys.path.insert(0, '/pollard/home/szhang20/arsenal-chrombpnet')
from chrombpnet.gpnstar import MSAReader, _TOKEN_TABLE

G = '/pollard/data/projects/shzhang/gpn'
REGIONS = f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6/fold_0/predict/test/regions.csv'
OUT = f'{G}/region_analysis/k562_test_features.tsv'
PHYLOP = '/pollard/data/projects/shzhang/alus/data/annotations/hg38.phyloP447way.bw'
RMSK = '/pollard/data/projects/shzhang/alus/data/repeats/rmsk.txt.gz'
GENES = '/pollard/data/projects/shzhang/alus/data/genes_oneper.gencode.v31.fixed.bed'
FASTA = '/pollard/home/szhang20/.cache/genome/hg38/hg38.fa'
MSA = '/pollard/data/projects/shzhang/s2f/cactus447way/447'
GAP = _TOKEN_TABLE[ord('-')]
# Alignment depth is measured over the central 100 bp of each 500 bp region.
CORE = (200, 300)


def phylop_and_gc(df):
	bw, fa = pyBigWig.open(PHYLOP), pyfaidx.Fasta(FASTA)
	mean, frac_cons, frac_aligned, gc = [], [], [], []
	for c, s, e in zip(df['chr'], df['start'], df['end']):
		v = np.array(bw.values(c, s, e), dtype=float)
		# The phyloP bigwig stores unaligned bases as exactly 0.
		v[v == 0] = np.nan
		mean.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
		frac_cons.append(np.mean(v > 2))
		frac_aligned.append(np.mean(np.isfinite(v)))
		seq = fa[c][s:e].seq.upper()
		gc.append((seq.count('G') + seq.count('C')) / len(seq))
	return pd.DataFrame({'phylop_mean': mean, 'phylop_frac_gt2': frac_cons, 'phylop_frac_aligned': frac_aligned, 'gc': gc}, index=df.index)


def align_depth(rows):
	"""Mean fraction of non-human species with a non-gap base in the core."""
	reader = MSAReader(MSA)
	out = []
	for c, s in rows:
		msa = reader.get(c, s + CORE[0], s + CORE[1], '+')
		out.append(float((msa[:, 1:] != GAP).mean()))
	return out


def repeat_overlap(df):
	"""Repeat class covering the region center, and repeat-covered fraction of the region."""
	cols = [5, 6, 7, 11]
	rm = pd.read_csv(RMSK, sep='\t', header=None, usecols=cols, names=['chr', 's', 'e', 'cls'])
	cls, frac = pd.Series('none', index=df.index), pd.Series(0.0, index=df.index)
	for chrom, sub in df.groupby('chr'):
		r = rm[rm['chr'] == chrom].sort_values('s')
		rs, re_, rc = r['s'].values, r['e'].values, r['cls'].values
		for i, s, e in zip(sub.index, sub['start'], sub['end']):
			lo = np.searchsorted(rs, s - 100_000)
			hi = np.searchsorted(rs, e)
			hit = np.nonzero(re_[lo:hi] > s)[0] + lo
			if len(hit) == 0:
				continue
			cover = np.zeros(e - s, bool)
			mid = (s + e) // 2
			for j in hit:
				cover[max(rs[j], s) - s:min(re_[j], e) - s] = True
				if rs[j] <= mid < re_[j]:
					cls[i] = rc[j]
			frac[i] = cover.mean()
	return cls, frac


def tss_distance(df):
	g = pd.read_csv(GENES, sep='\t', header=None, names=['chr', 's', 'e', 'name', 'score', 'strand'])
	g['tss'] = np.where(g['strand'] == '+', g['s'], g['e'])
	dist = pd.Series(np.nan, index=df.index)
	for chrom, sub in df.groupby('chr'):
		t = np.sort(g.loc[g['chr'] == chrom, 'tss'].values)
		if len(t) == 0:
			continue
		mid = ((sub['start'] + sub['end']) // 2).values
		k = np.clip(np.searchsorted(t, mid), 1, len(t) - 1)
		dist[sub.index] = np.minimum(np.abs(mid - t[k - 1]), np.abs(mid - t[k]))
	return dist


def main():
	df = pd.read_csv(REGIONS, sep='\t')[['chr', 'start', 'end', 'is_peak', 'true_count']]
	feats = phylop_and_gc(df)
	print('phylop/gc done', flush=True)
	rows = list(zip(df['chr'], df['start']))
	chunks = [rows[i:i + 1000] for i in range(0, len(rows), 1000)]
	with Pool(16) as pool:
		feats['align_depth'] = np.concatenate(pool.map(align_depth, chunks))
	print('alignment depth done', flush=True)
	feats['repeat_class'], feats['repeat_frac'] = repeat_overlap(df)
	feats['tss_dist'] = tss_distance(df)
	pd.concat([df, feats], axis=1).to_csv(OUT, sep='\t', index=False)
	print(f'wrote {OUT}')


if __name__ == '__main__':
	main()
