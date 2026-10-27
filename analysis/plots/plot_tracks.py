"""Observed vs predicted base-resolution ATAC profiles at a handful of K562 test loci, one panel per model."""

import os
import sys
import textwrap
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.spatial.distance import jensenshannon

# Nimbus Sans is the installed metric clone of Helvetica.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']

G = '/pollard/data/projects/shzhang/gpn'
A = f'{G}/arsenal_download'
OUT = '/pollard/home/szhang20/arsenal-chrombpnet/figs/example_tracks'
ALU_OUT = '/pollard/home/szhang20/alu_evo/figs'
DATE = date.today().strftime('%Y%m%d')
OBSERVED = f'{A}/ARSENAL_ChromBPNet_Models/K562/K562_unstranded.bw'
TEST = 'fold_0/predict/test'
# Label, prediction bigwig, color.
MODELS = (
	('ChromBPNet (paper)', f'{A}/ChromBPNet_Models/K562/run_1/{TEST}/pred.bw', '#7f8c8d'),
	('ChromBPNet (ours)', f'{G}/chrombpnet_phylop/runs/K562_chrombpnet/{TEST}/pred.bw', '#7f8c8d'),
	('ARSENAL (paper)', f'{A}/ARSENAL_ChromBPNet_Models/K562/run_1/{TEST}/pred.bw', '#8e44ad'),
	('phyloP channel', f'{G}/chrombpnet_phylop/runs/K562_phylop-chrombpnet/{TEST}/pred.bw', '#2980b9'),
	('GPN-Star m447 25m last6', f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6/{TEST}/pred.bw', '#c0392b'),
	('GPN-Star m447 85m last6', f'{G}/chrombpnet_layers/runs/gpnstar_85m_last6/{TEST}/pred.bw', '#c0392b'),
	('GPN-Star p243', f'{G}/chrombpnet_layers/runs/gpnstar_p243_25m_last6/{TEST}/pred.bw', '#c0392b'),
	('GPN-Star v100', f'{G}/chrombpnet_layers/runs/gpnstar_v100_25m_last6/{TEST}/pred.bw', '#c0392b'),
)
WIDTH = 1000
SMOOTH = 20
PHYLOP = '/pollard/data/projects/shzhang/alus/data/annotations/hg38.phyloP447way.bw'
# phyloP above this counts as conserved (about p < 0.01).
CONSERVED = 2
RMSK = '/pollard/data/projects/shzhang/alus/data/repeats/rmsk.txt.gz'
# The user's curated Alu calls, which replace the rmsk Alus.
ALUS = '/pollard/home/szhang20/alu/data/Alu_hg38_repeatmasker_all.txt'


def pick_loci():
	"""Eight test regions spanning typical, GPN-Star-better, GPN-Star-worse, repeat and non-peak cases."""
	regions = pd.read_csv(f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6/{TEST}/regions.csv', sep='\t')
	df = pd.read_csv(f'{G}/region_analysis/k562_test_predictions.tsv', sep='\t')
	df['center'] = regions['start'].values + regions['summit'].values
	df['gain'] = df['err_cbp'].abs() - df['err_gpn'].abs()
	peaks, nonpeaks = df[df['is_peak'] == 1], df[df['is_peak'] == 0]
	easy = peaks[(peaks['err_gpn'].abs() < 0.15) & (peaks['err_cbp'].abs() < 0.15)]
	picks = [
		('strongest peak, all models close on counts', peaks.nlargest(1, 'true_count')),
		('strong peak, all models close on counts', easy.nlargest(2, 'true_count').iloc[1:]),
		('typical peak, all models close on counts', easy[easy['true_count'].between(5.5, 6)].iloc[[len(easy[easy['true_count'].between(5.5, 6)]) // 2]]),
		('GPN-Star fixes a ChromBPNet count error', peaks[peaks['true_count'] > 5.5].nlargest(1, 'gain')),
		('GPN-Star worse than ChromBPNet on counts', peaks[peaks['true_count'] > 5.5].nsmallest(1, 'gain')),
		('LTR repeat peak, poorly aligned', peaks[(peaks['repeat_class'] == 'LTR') & (peaks['true_count'] > 5.5)].nsmallest(1, 'align_depth')),
		('non-peak, GPN-Star fixes over-prediction', nonpeaks[nonpeaks['true_count'] > 3].nlargest(1, 'gain')),
	]
	return [(why, row) for why, d in picks for _, row in d.iterrows()]


def pick_vs_arsenal(n=1):
	"""Top peak and non-peak where GPN-Star is near-exact and beats all three ARSENAL runs beyond seed noise."""
	regions = pd.read_csv(f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6/{TEST}/regions.csv', sep='\t')
	df = pd.read_csv(f'{G}/region_analysis/k562_gpn_vs_arsenal_regions.tsv', sep='\t')
	df['center'] = regions['start'].values + regions['summit'].values
	df['gpnstar_last6'], df['chrombpnet'] = df['gpn'], df['cbp']
	# 95th percentile of |error difference| between ARSENAL runs 1 and 2.
	thr = df['gain_null'].abs().quantile(0.95)
	worst_ars = df[['abserr_ars1', 'abserr_ars2', 'abserr_ars3']].min(axis=1)
	ok = (worst_ars - df['abserr_gpn'] > thr) & (df['abserr_gpn'] < 0.2) & (df['phylop_frac_aligned'] == 1)
	picks = [
		('peak, GPN-Star beats ARSENAL', df[ok & (df['is_peak'] == 1) & (df['true_count'] > 6)].nlargest(n, 'gain')),
		('non-peak, GPN-Star beats ARSENAL', df[ok & (df['is_peak'] == 0) & (df['true_count'] > 2)].nlargest(n, 'gain')),
	]
	return [(why, row) for why, d in picks for _, row in d.iterrows()]


def load_rmsk(chroms):
	"""Non-Alu RepeatMasker calls plus the curated Alus on the given chromosomes."""
	r = pd.read_csv(RMSK, sep='\t', header=None, usecols=[5, 6, 7, 9, 10, 11, 12], names=['chr', 'start', 'end', 'strand', 'name', 'cls', 'family'])
	a = pd.read_csv(ALUS, sep='\t', usecols=['genoName', 'genoStart', 'genoEnd', 'strand', 'repName', 'repClass', 'repFamily'])
	a.columns = ['chr', 'start', 'end', 'strand', 'name', 'cls', 'family']
	r = pd.concat([r[r['family'] != 'Alu'], a])
	return {c: g.sort_values('start').reset_index(drop=True) for c, g in r[r['chr'].isin(chroms)].groupby('chr')}


def repeats_in(rmsk, chrom, start, end):
	"""RepeatMasker calls overlapping [start, end)."""
	g = rmsk.get(chrom)
	if g is None:
		return pd.DataFrame(columns=['start', 'end', 'strand', 'name', 'cls', 'family'])
	# Repeats are under 10 kb, except a few satellites that do not matter here.
	i, j = np.searchsorted(g['start'].values, [start - 10_000, end])
	h = g.iloc[i:j]
	return h[h['end'] > start]


def alu_table(rmsk, obs_bw):
	"""Per test window: Alu bp, whether an Alu covers the summit and the fraction of observed reads inside Alus."""
	regions = pd.read_csv(f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6/{TEST}/regions.csv', sep='\t')
	df = pd.read_csv(f'{G}/region_analysis/k562_gpn_vs_arsenal_regions.tsv', sep='\t')
	df['center'] = regions['start'].values + regions['summit'].values
	df['gpnstar_last6'], df['chrombpnet'] = df['gpn'], df['cbp']
	alu_bp, at_summit, names, frac_reads = [], [], [], []
	for c, x in zip(df['chr'], df['center']):
		start = x - WIDTH // 2
		a = repeats_in(rmsk, c, start, start + WIDTH)
		a = a[a['family'] == 'Alu']
		inside = np.zeros(WIDTH, bool)
		for s, e in zip(a['start'], a['end']):
			inside[max(s - start, 0):max(e - start, 0)] = True
		alu_bp.append(int(inside.sum()))
		hit = a[(a['start'] <= x) & (a['end'] > x)]
		at_summit.append(len(hit) > 0)
		names.append(','.join(a['name']) if len(a) else '')
		obs = profile(obs_bw, c, start) if len(a) else np.zeros(WIDTH)
		frac_reads.append(obs[inside].sum() / obs.sum() if obs.sum() > 0 else 0.0)
	df['alu_bp'], df['alu_at_summit'], df['alus'], df['frac_reads_in_alu'] = alu_bp, at_summit, names, frac_reads
	df['n_alus'] = [len(n.split(',')) if n else 0 for n in names]
	return df


def pick_alu(df):
	"""Alu-driven windows (half the reads inside Alus) plus peaks with at least two Alus in the 1 kb window."""
	thr = df['gain_null'].abs().quantile(0.95)
	in_alu = df[df['frac_reads_in_alu'] >= 0.5]
	pk = in_alu[in_alu['is_peak'] == 1]
	strong = pk[pk['true_count'] > 5.5]
	typical = pk[pk['true_count'].between(5.5, 6) & (pk['abserr_gpn'] < 0.15) & (pk['abserr_ars'] < 0.15)]
	young = strong[strong['alus'].str.contains('AluY')]
	picks = [
		('strongest peak with most reads in an Alu', pk.nlargest(1, 'true_count')),
		('typical Alu peak, models close', typical.iloc[[len(typical) // 2]]),
		('GPN-Star beats ARSENAL on an Alu peak', strong[strong['gain'] > thr].nlargest(1, 'gain')),
		('ARSENAL beats GPN-Star on an Alu peak', strong[strong['gain'] < -thr].nsmallest(1, 'gain')),
		('strongest peak on a young AluY', young.nlargest(1, 'true_count')),
		('non-peak with most reads in an Alu', in_alu[in_alu['is_peak'] == 0].nlargest(1, 'true_count')),
	]
	# Skip windows within 1 kb of an earlier pick.
	near = np.zeros(len(df), bool)
	for _, d in picks:
		for _, r in d.iterrows():
			near |= ((df['chr'] == r['chr']) & ((df['center'] - r['center']).abs() < WIDTH)).values
	multi = df[(df['n_alus'] >= 3) & (df['is_peak'] == 1) & (df['alu_bp'] >= 600) & ~near]
	multi_ok = multi[(multi['abserr_gpn'] < 0.2) & (multi['abserr_ars'] < 0.2)]
	picks += [
		('strongest peak among 3+ Alus', multi.nlargest(1, 'true_count')),
		('peak among 3+ Alus, models close', multi_ok.nlargest(1, 'true_count')),
	]
	return [(why, row) for why, d in picks for _, row in d.iterrows()]


def alu_summary(df):
	"""Count Pearson per model for windows with and without Alus."""
	strata = {'all': df.index == df.index, '2+ Alus in 1 kb': df['n_alus'] >= 2, '>=50% reads in Alus': df['frac_reads_in_alu'] >= 0.5, 'Alu at summit': df['alu_at_summit'], 'Alu in 1 kb, not at summit': (df['alu_bp'] > 0) & ~df['alu_at_summit'], 'no Alu in 1 kb': df['alu_bp'] == 0}
	rows = []
	for name, m in strata.items():
		for peak, d in (('all', df[m]), ('peaks', df[m & (df['is_peak'] == 1)]), ('nonpeaks', df[m & (df['is_peak'] == 0)])):
			r = {k: np.corrcoef(d[k], d['true_count'])[0, 1] for k in ('cbp', 'ars1', 'ars2', 'ars3', 'gpn')}
			rows.append({'stratum': name, 'regions': peak, 'n': len(d), 'r_chrombpnet': r['cbp'], 'r_arsenal': np.mean([r[f'ars{i}'] for i in (1, 2, 3)]), 'r_gpnstar_25m_last6': r['gpn'], 'frac_gpn_better': (d['gain'] > 0).mean()})
	return pd.DataFrame(rows)


def repeat_row(ax, reps, start):
	"""RepeatMasker calls as boxes: Alus in orange with name and strand, other repeats in grey."""
	for k, (_, r) in enumerate(reps.sort_values('start').iterrows()):
		s, e = max(r['start'], start), min(r['end'], start + WIDTH)
		alu = r['family'] == 'Alu'
		ax.add_patch(plt.Rectangle((s, 0.25), e - s, 0.5, facecolor='#e67e22' if alu else '#bbbbbb', edgecolor='white', lw=1))
		if alu or e - s > 80:
			# Labels alternate between two heights so neighbours do not collide.
			ax.text((s + e) / 2, 0.85 + 0.4 * (k % 2), f"{r['name']} {'>' if r['strand'] == '+' else '<'}", ha='center', va='bottom', fontsize=6.5, color='#a04000' if alu else '#555555')
	ax.set_ylim(0, 1.6)
	ax.set_yticks([])
	ax.set_ylabel('RepeatMasker', rotation=0, ha='right', va='center', fontsize=8)


def profile(bw, chrom, start):
	return np.nan_to_num(np.array(bw.values(chrom, start, start + WIDTH)))


def smooth(x):
	return np.convolve(x, np.ones(SMOOTH) / SMOOTH, mode='same')


def noise_floor(shape, total, draws=50, seed=0):
	"""Mean JSD of a perfect model: reads sampled from the predicted shape itself, at the observed depth."""
	rng = np.random.default_rng(seed)
	return np.mean([jensenshannon(rng.multinomial(int(total), shape) / total, shape) for _ in range(draws)])


def conservation(phylop):
	"""Alignment coverage, mean phyloP and percent of bases above CONSERVED; unaligned bases count as not conserved."""
	parts = []
	for name, v in (('1 kb', phylop), ('central 500 bp', phylop[WIDTH // 4:3 * WIDTH // 4])):
		aligned = ~np.isnan(v)
		mean = f'{np.nanmean(v):.2f}' if aligned.any() else 'n/a'
		parts.append(f'{name}: {100 * aligned.mean():.0f}% aligned, mean phyloP {mean}, {100 * np.mean(v > CONSERVED):.0f}% bases >{CONSERVED}')
	return '   |   '.join(parts)


def plot_locus(name, why, row, bws, obs_bw, phylop_bw, rmsk=None, out=OUT):
	"""One figure: observed counts on top, then each model's prediction over a smoothed observed outline."""
	chrom, start = row['chr'], int(row['center']) - WIDTH // 2
	obs = profile(obs_bw, chrom, start)
	preds = [profile(bw, chrom, start) for bw in bws]
	x = np.arange(start, start + WIDTH)
	obs_p = obs / (obs.sum() + 1e-3)
	phylop = np.array(phylop_bw.values(chrom, start, start + WIDTH), dtype=float)
	# This bigwig stores unaligned bases as exactly 0.
	phylop[phylop == 0] = np.nan
	extra = rmsk is not None
	fig, axes = plt.subplots(len(MODELS) + 2 + extra, 1, figsize=(10, 1.1 * (len(MODELS) + 2 + extra)), sharex=True, gridspec_kw={'height_ratios': [1] * (len(MODELS) + 2) + [0.8] * extra})
	cons, axes = axes[0], axes[1:]
	if extra:
		reps, axes = axes[-1], axes[:-1]
		repeat_row(reps, repeats_in(rmsk, chrom, start, start + WIDTH), start)
		reps.spines[['top', 'right', 'left']].set_visible(False)
		reps.tick_params(labelsize=7)
	lo, hi = -4, max(8, np.nanmax(phylop, initial=0) + 1)
	cons.bar(x, np.clip(phylop, lo, hi), width=1, color=np.where(phylop > CONSERVED, '#c0392b', '#bbbbbb'))
	# Unaligned bases (no phyloP) as a black strip along the bottom.
	cons.fill_between(x, lo, lo + 0.08 * (hi - lo), where=np.isnan(phylop), color='black', lw=0, step='mid')
	cons.axhline(CONSERVED, color='black', ls=':', lw=0.8)
	cons.axvspan(start + WIDTH // 4, start + 3 * WIDTH // 4, color='#f7f1dc', zorder=0)
	cons.set_ylim(lo, hi)
	cons.set_ylabel('phyloP\n(447 mammals)', rotation=0, ha='right', va='center', fontsize=8)
	cons.text(1.01, 0.5, f'{100 * (~np.isnan(phylop)).mean():.0f}% aligned\n{100 * np.mean(phylop > CONSERVED):.0f}% bases >{CONSERVED}', transform=cons.transAxes, va='center', fontsize=8)
	axes[0].bar(x, obs, width=1, color='black')
	axes[0].set_ylabel('observed', rotation=0, ha='right', va='center', fontsize=8)
	mean_shape = np.mean([p / p.sum() for p in preds], axis=0)
	floor = noise_floor(mean_shape, obs.sum())
	axes[0].text(1.01, 0.5, f'total {obs.sum():.0f}\nperfect-model JSD {floor:.3f}', transform=axes[0].transAxes, va='center', fontsize=8)
	# Model rows share one y-scale with the smoothed observed signal; raw observed reads keep their own axis.
	ymax = max(max(p.max() for p in preds), smooth(obs).max()) * 1.05
	for ax, (label, _, color), p in zip(axes[1:], MODELS, preds):
		ax.fill_between(x, smooth(obs), color='#cccccc', lw=0, step='mid')
		ax.plot(x, p, color=color, lw=1)
		ax.set_ylim(0, ymax)
		ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8)
		j = jensenshannon(obs_p, p / p.sum())
		norm = max(0, 1 - j / jensenshannon(obs_p, np.full(WIDTH, 1 / WIDTH)))
		c = slice(WIDTH // 4, 3 * WIDTH // 4)
		r, rc = np.corrcoef(obs, p)[0, 1], np.corrcoef(obs[c], p[c])[0, 1]
		ax.text(1.01, 0.5, f'total {p.sum():.0f}\nJSD {j:.3f}  norm {norm:.3f}\nprofile r {r:.2f} (central {rc:.2f})', transform=ax.transAxes, va='center', fontsize=8)
	for ax in (cons, *axes):
		ax.spines[['top', 'right']].set_visible(False)
		ax.tick_params(labelsize=7)
	bottom = reps if extra else axes[-1]
	bottom.set_xlabel(f'{chrom} position (bp)', fontsize=8)
	bottom.ticklabel_format(axis='x', style='plain', useOffset=False)
	bottom.set_xlim(start, start + WIDTH)
	fig.suptitle(f'{chrom}:{start:,}-{start + WIDTH:,}  {why}\n{conservation(phylop)}', fontsize=10, x=0.02, ha='left')
	fig.text(0.02, 0.0, textwrap.fill(f'Top: phyloP per base (clipped at -4), red above {CONSERVED}; black strip = not in the alignment; shaded band = central 500 bp peak region. Raw observed reads are on their own axis; model rows share one axis with grey = observed smoothed over {SMOOTH} bp, which approximates expected reads per base. Lines: predicted reads per base (shape x total). JSD compares shapes only; profile r is the per-base Pearson of predicted vs observed reads over 1 kb (and the central 500 bp); perfect-model JSD is what an exact model would score from read sampling alone.' + (' Bottom: RepeatMasker calls, Alus in orange (> plus strand, < minus strand), other repeats grey.' if extra else ''), 150), fontsize=7, va='top')
	fig.tight_layout()
	fig.savefig(f'{out}/{DATE}_K562_{name}.png', dpi=150, bbox_inches='tight')
	plt.close(fig)


def main():
	os.makedirs(OUT, exist_ok=True)
	obs_bw = pyBigWig.open(OBSERVED)
	bws = [pyBigWig.open(p) for _, p, _ in MODELS]
	phylop_bw = pyBigWig.open(PHYLOP)
	# Pass vs_arsenal for GPN-Star's wins over ARSENAL, or alu for Alu windows with a RepeatMasker row.
	mode = sys.argv[1] if len(sys.argv) > 1 else ''
	rmsk = None
	if mode == 'alu':
		rmsk = load_rmsk(('chr5', 'chr10', 'chr14', 'chr18', 'chr20', 'chr22'))
		df = alu_table(rmsk, obs_bw)
		summ = alu_summary(df)
		summ.to_csv(f'{G}/region_analysis/k562_alu_strata.tsv', sep='\t', index=False)
		print(summ.round(3).to_string())
		df[df['alu_bp'] > 0].to_csv(f'{G}/region_analysis/k562_alu_windows.tsv', sep='\t', index=False)
		loci = pick_alu(df)
	else:
		loci = pick_vs_arsenal() if mode == 'vs_arsenal' else pick_loci()
	prefix = {'alu': 'alu', 'vs_arsenal': 'gpn_beats_arsenal'}.get(mode, 'locus')
	for i, (why, row) in enumerate(loci, 1):
		print(f"locus {i}: {row['chr']}:{int(row['center'])} true {row['true_count']:.2f} cbp {row['chrombpnet']:.2f} gpn {row['gpnstar_last6']:.2f}  {why}")
		tag = why.split(',')[0].replace(' ', '_').replace('-', '')
		plot_locus(f'{prefix}{i}_{tag}', why, row, bws, obs_bw, phylop_bw, rmsk, ALU_OUT if mode == 'alu' else OUT)
	print(f"plots in {ALU_OUT if mode == 'alu' else OUT}")


if __name__ == '__main__':
	main()
