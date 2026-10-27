"""DART-Eval task 5 metrics for every scored model, following the ARSENAL authors' supervised_variant_scoring notebooks."""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

G = '/pollard/data/projects/shzhang/gpn'
DL = f'{G}/arsenal_download'
OUT = f'{G}/vep/vep_metrics.tsv'


def african_truth():
	t = pd.read_csv(f'{G}/vep/data/caqtl_truth/Afr.CaQTLS.tsv', sep='\t')
	t = t[t['IsUsed'] & t['in_peaks']]
	return pd.DataFrame({
		'key': t['chr_hg38'] + ':' + t['pos_hg38'].astype(str) + ':' + t['allele1'] + ':' + t['allele2'],
		'sig': t['label'].astype(bool), 'effect': t['beta'],
	})


def yoruba_truth():
	t = pd.read_csv(f'{G}/vep/data/dsqtl_truth/yoruban.dsqtls.benchmarking.tsv', sep='\t')
	t = t[t['var.isused'] & t['var.label'].isin([1, -1])]
	# The notebook flips obs.estimate so it shares logfc's sign convention.
	return pd.DataFrame({'key': t['var.rsid'], 'sig': t['var.label'] == 1, 'effect': -t['obs.estimate']})


def score_keys(scores, dataset):
	if dataset == 'african':
		return scores['chr'] + ':' + scores['pos'].astype(str) + ':' + scores['allele1'] + ':' + scores['allele2']
	return scores['variant_id']


def score_files():
	"""(model, run, dataset, path) for our scores and the authors' published ones (Yoruba in hg19 context)."""
	files = []
	for path in sorted(glob.glob(f'{G}/vep/scores/*/*.tsv')):
		name, dataset = path.split('/')[-2], os.path.basename(path)[:-4]
		model, _, run = name.rpartition('_run_') if '_run_' in name else (name, '', '1')
		files.append((model + ('_hg38' if dataset == 'yoruba' and name.startswith('authors') else ''), run, dataset, path))
	for label, folder in (('authors_arsenal', 'ARSENAL_ChromBPNet_Models'), ('authors_chrombpnet', 'ChromBPNet_Models')):
		for path in sorted(glob.glob(f'{DL}/{folder}/GM12878/run_*/*_variants/variant_scores.tsv')):
			run = path.split('/run_')[1].split('/')[0]
			files.append((label, run, path.split('/')[-2].replace('_variants', ''), path))
	return files


def metrics(df):
	sig, ctrl = df[df['sig']], df[~df['sig']]
	labels = np.r_[np.ones(len(sig)), np.zeros(len(ctrl))]
	scores = np.r_[sig['logfc'].abs(), ctrl['logfc'].abs()]
	return {
		'n_sig': len(sig), 'n_ctrl': len(ctrl),
		'pearson': pearsonr(sig['effect'], sig['logfc'])[0], 'spearman': spearmanr(sig['effect'], sig['logfc'])[0],
		'auroc': roc_auc_score(labels, scores), 'auprc': average_precision_score(labels, scores),
	}


def main():
	truth = {'african': african_truth(), 'yoruba': yoruba_truth()}
	scored = {}
	for model, run, dataset, path in score_files():
		s = pd.read_csv(path, sep='\t')
		scored[model, run, dataset] = pd.Series(s['logfc'].values, index=score_keys(s, dataset))
	rows = []
	for dataset, t in truth.items():
		runs = {k: v for k, v in scored.items() if k[2] == dataset}
		# Every model is evaluated on the variants all of them scored, e.g. Yoruba variants that lifted to hg38.
		common = set(t['key'])
		for v in runs.values():
			common &= set(v.index)
		sub = t[t['key'].isin(common)]
		for (model, run, _), v in runs.items():
			rows.append({'dataset': dataset, 'model': model, 'run': run, **metrics(sub.assign(logfc=v.loc[sub['key']].values))})
	df = pd.DataFrame(rows).sort_values(['dataset', 'model', 'run'])
	df.to_csv(OUT, sep='\t', index=False)
	cols = ['n_sig', 'n_ctrl', 'pearson', 'spearman', 'auroc', 'auprc']
	summary = df.groupby(['dataset', 'model'])[cols].mean()
	summary['n_runs'] = df.groupby(['dataset', 'model']).size()
	pd.set_option('display.width', 200)
	print(summary.round(4).to_string())
	print(f'\nper-run metrics: {OUT}')


if __name__ == '__main__':
	main()
