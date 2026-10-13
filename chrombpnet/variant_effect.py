"""No-bias variant scores (log2 allele2/allele1 counts) for ChromBPNet, phyloP and GPN-Star checkpoints."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data_utils import dna_to_one_hot
from .model_wrappers import load_pretrained_model

INPUT_LEN = 2114
FLANK = INPUT_LEN // 2
MODEL_TYPES = ['chrombpnet', 'phylop-chrombpnet', 'gpnstar-chrombpnet']


def read_variants(path, chrom_sizes):
	"""Headerless chr, 1-based pos, allele1, allele2, variant_id; keeps SNVs whose window fits the chromosome."""
	df = pd.read_csv(path, sep='\t', header=None, usecols=range(5), names=['chr', 'pos', 'allele1', 'allele2', 'variant_id'])
	size = df['chr'].map(chrom_sizes)
	snv = df['allele1'].str.len().eq(1) & df['allele2'].str.len().eq(1)
	fits = size.notna() & (df['pos'] - 1 - FLANK >= 0) & (df['pos'] - 1 + FLANK <= size)
	print(f"{path}: {len(df)} variants, {(~snv).sum()} non-SNV and {(snv & ~fits).sum()} off-chromosome dropped")
	return df[snv & fits].reset_index(drop=True)


def softmax(x):
	x = x - x.max(axis=-1, keepdims=True)
	e = np.exp(x)
	return e / e.sum(axis=-1, keepdims=True)


def jsd(p, q):
	"""Row-wise Jensen-Shannon distance, base 2, as scipy's jensenshannon."""
	m = (p + q) / 2
	kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0), axis=-1)
	return np.sqrt(np.maximum((kl(p, m) + kl(q, m)) / 2, 0))


class VariantInputs(torch.utils.data.Dataset):
	"""Item i is batch i's (allele1, allele2) model inputs per strand, as numpy; file handles open lazily per worker."""

	def __init__(self, variants, model_type, fasta, batch_size=64, strands=('+', '-'), msa_path=None, phylop_bw=None):
		if model_type not in MODEL_TYPES:
			raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type}")
		self.variants, self.model_type, self.batch_size, self.strands = variants, model_type, batch_size, strands
		self.fasta_path, self.msa_path, self.phylop_bw = fasta, msa_path, phylop_bw
		self._handles = None

	def __len__(self):
		return -(-len(self.variants) // self.batch_size)

	def handles(self):
		if self._handles is None:
			import pyfaidx
			h = {'fasta': pyfaidx.Fasta(self.fasta_path)}
			if self.model_type == 'gpnstar-chrombpnet':
				from .gpnstar import MSAReader
				h['msa'] = MSAReader(self.msa_path)
			if self.model_type == 'phylop-chrombpnet':
				import pyBigWig
				h['phylop'] = pyBigWig.open(self.phylop_bw)
			self._handles = h
		return self._handles

	def __getstate__(self):
		state = self.__dict__.copy()
		state['_handles'] = None
		return state

	@staticmethod
	def window(pos):
		start = pos - 1 - FLANK
		return start, start + INPUT_LEN

	def onehots(self, batch):
		"""(allele1, allele2) forward-strand one-hots, each (n, 4, L)."""
		fasta = self.handles()['fasta']
		ref = [fasta[c][slice(*self.window(p))].seq.upper() for c, p in zip(batch['chr'], batch['pos'])]
		return [
			dna_to_one_hot([s[:FLANK] + a + s[FLANK + 1:] for s, a in zip(ref, batch[col])]).transpose(0, 2, 1).astype(np.float32)
			for col in ('allele1', 'allele2')
		]

	def msas(self, batch, strand):
		"""(allele1, allele2) token alignments, each (n, L, species), on the given strand."""
		from .gpnstar import _RC_TOKEN_TABLE, _TOKEN_TABLE
		table = _TOKEN_TABLE if strand == '+' else _RC_TOKEN_TABLE
		idx = FLANK if strand == '+' else INPUT_LEN - 1 - FLANK
		reader = self.handles()['msa']
		ref = np.stack([reader.get(c, *self.window(p), strand) for c, p in zip(batch['chr'], batch['pos'])])
		out = []
		for col in ('allele1', 'allele2'):
			msa = ref.copy()
			msa[:, idx, 0] = table[[ord(a) for a in batch[col]]]
			out.append(msa)
		return out

	def __getitem__(self, i):
		batch = self.variants.iloc[i * self.batch_size:(i + 1) * self.batch_size]
		if self.model_type == 'gpnstar-chrombpnet':
			return {s: self.msas(batch, s) for s in self.strands}
		x1, x2 = self.onehots(batch)
		if self.model_type == 'phylop-chrombpnet':
			bw = self.handles()['phylop']
			cons = np.stack([np.nan_to_num(np.array(bw.values(c, *self.window(p)), dtype=np.float32)) for c, p in zip(batch['chr'], batch['pos'])])
			x1, x2 = (np.concatenate([x, cons[:, None]], axis=1) for x in (x1, x2))
		# Reverse complement flips A<->T and C<->G rows, and the phyloP row is reversed in place.
		rc = lambda x: np.concatenate([x[:, 3::-1, ::-1], x[:, 4:, ::-1]], axis=1).copy()
		return {s: (x1, x2) if s == '+' else (rc(x1), rc(x2)) for s in self.strands}


class VariantScorer:
	"""Runs the accessibility model without its bias model on both alleles."""

	def __init__(self, wrapper, model_type, device='cuda'):
		self.net = wrapper.model.eval().to(device)
		self.model_type, self.device = model_type, device
		if model_type == 'gpnstar-chrombpnet':
			from .gpnstar import chunk_layout
			emb = self.net.gpnstar_model
			self.layout = chunk_layout(INPUT_LEN, emb.chunk_size)
			self.width = min(emb.chunk_size, INPUT_LEN)

	def gpnstar_embeddings(self, msa1, msa2, strand):
		"""Allele embeddings; for allele2 only chunks whose window holds the variant are re-embedded."""
		idx = FLANK if strand == '+' else INPUT_LEN - 1 - FLANK
		emb = self.net.gpnstar_model
		e1 = emb(msa1)
		e2 = e1.clone()
		for w, a, b in self.layout:
			if w <= idx < w + self.width:
				e2[:, a:b] = emb.embed_chunks(msa2[:, w:w + self.width])[:, a - w:b - w]
		return e1, e2

	@torch.no_grad()
	def predict(self, inputs, strand):
		"""Per allele (log counts, profile logits) for one strand, profiles in forward orientation."""
		x1, x2 = (torch.as_tensor(x).to(self.device) for x in inputs)
		if self.model_type == 'gpnstar-chrombpnet':
			x1, x2 = self.gpnstar_embeddings(x1, x2, strand)
		elif self.model_type == 'phylop-chrombpnet':
			# BPNet transposes any input whose dim 1 is not 4, so pass (n, L, 5).
			x1, x2 = x1.transpose(1, 2), x2.transpose(1, 2)
		out = []
		for x in (x1, x2):
			profile, counts = self.net.model(x)
			profile = profile.squeeze(1).float().cpu().numpy()
			out.append((counts.squeeze(-1).float().cpu().numpy(), profile[:, ::-1] if strand == '-' else profile))
		return out

	def score(self, dataset, num_workers=0):
		"""Allele counts, logfc and jsd per variant, averaging strands in count and logit space as snp_score does."""
		loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers, prefetch_factor=4 if num_workers else None)
		counts1, counts2, jsds = [], [], []
		for item in tqdm(loader):
			preds = [self.predict(inputs, strand) for strand, inputs in item.items()]
			counts1.append(np.mean([np.exp(p[0][0]) for p in preds], axis=0))
			counts2.append(np.mean([np.exp(p[1][0]) for p in preds], axis=0))
			p1 = np.mean([p[0][1] for p in preds], axis=0)
			p2 = np.mean([p[1][1] for p in preds], axis=0)
			jsds.append(jsd(softmax(p2), softmax(p1)))
		out = dataset.variants.copy()
		out['allele1_pred_counts'] = np.concatenate(counts1)
		out['allele2_pred_counts'] = np.concatenate(counts2)
		out['logfc'] = np.log2(out['allele2_pred_counts'] / out['allele1_pred_counts'])
		out['abs_logfc'] = out['logfc'].abs()
		out['jsd'] = np.concatenate(jsds)
		return out


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--checkpoint', required=True, help='best_model.ckpt of a training run')
	parser.add_argument('--model_type', required=True, choices=MODEL_TYPES)
	parser.add_argument('--variants', required=True, nargs='+', help='chrombpnet-schema variant tsvs on hg38')
	parser.add_argument('--out', required=True, nargs='+', help='one output tsv per --variants file')
	parser.add_argument('--fasta', required=True)
	parser.add_argument('--chrom_sizes', required=True)
	parser.add_argument('--msa_path', help="defaults to the run's training msa_path")
	parser.add_argument('--phylop_bw', help="defaults to the run's training phylop_bw")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--forward_only', action='store_true')
	parser.add_argument('--matmul_precision', default='highest', choices=['highest', 'high', 'medium'])
	args = parser.parse_args()
	if len(args.out) != len(args.variants):
		parser.error('need one --out per --variants')
	torch.set_float32_matmul_precision(args.matmul_precision)

	wrapper = load_pretrained_model(argparse.Namespace(model_type=args.model_type, bias_scaled=None, data_dir=''), args.checkpoint)
	train_args = getattr(wrapper.hparams, 'args', argparse.Namespace())
	scorer = VariantScorer(wrapper, args.model_type, device='cuda' if torch.cuda.is_available() else 'cpu')
	sizes = pd.read_csv(args.chrom_sizes, sep='\t', header=None, index_col=0)[1].to_dict()
	for variants_path, out_path in zip(args.variants, args.out):
		dataset = VariantInputs(
			read_variants(variants_path, sizes), args.model_type, args.fasta, args.batch_size,
			strands=('+',) if args.forward_only else ('+', '-'),
			msa_path=args.msa_path or getattr(train_args, 'msa_path', None),
			phylop_bw=args.phylop_bw or getattr(train_args, 'phylop_bw', None),
		)
		scores = scorer.score(dataset, args.num_workers)
		os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
		scores.to_csv(out_path, sep='\t', index=False)
		print(f"wrote {len(scores)} scores to {out_path}")


if __name__ == '__main__':
	main()
